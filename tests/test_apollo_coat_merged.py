"""Focused regression tests for APOLLO + COAT + single-buffer merged gradients (Step 3)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from colab.final.coat_spark import (
    CoatLinear,
    install_spark_coat,
    qualify_coat_kernels,
    verify_spark_mlp_not_silu,
)
from colab.final.common import write
from colab.final.engine import _build_scheduler, checkpoint, load_checkpoint, optimizer_for
from colab.final.optimizer import (
    ApolloMergedOptimizer,
    CoatDynamicRangeFP8Codec,
    MergedGradientAccumulator,
    build_apollo_param_groups,
    stochastic_round_fp32_to_bf16,
)
from colab.final.qualify import qualify_matched_profiles
from tests._spark_test_fakes import make_synthetic_spark_native


def test_apollo_param_partition_never_applies_lowrank_to_norms_gates_biases_or_coat_adamw():
    model = make_synthetic_spark_native({"visual_cross_attn_blocks": 3, "apollo_min_dim": 64})
    model.set_stage(False, stage_mode="full_spark")

    opt = optimizer_for(model, model.policy)
    assert isinstance(opt, ApolloMergedOptimizer)
    assert "CoatAdamW" not in type(opt).__name__

    apollo_ids = {id(p) for g in opt.param_groups if g["kind"] == "apollo" for p in g["params"]}
    highprec_ids = {id(p) for g in opt.param_groups if g["kind"] == "adamw" for p in g["params"]}
    assert apollo_ids and highprec_ids
    assert apollo_ids.isdisjoint(highprec_ids)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if any(tok in lname for tok in ("norm", "attn_gate", "headwise_attn_output_gate", "embed_tokens", "lm_head", "modality", "alpha_projector")) or p.ndim != 2:
            if "gate_proj" not in lname:
                assert id(p) in highprec_ids, f"Expected high-precision AdamW for {name}"
        elif "q_k_v_proj" in lname or "o_proj" in lname or "up_proj" in lname or "down_proj" in lname or "gate_proj" in lname:
            assert id(p) in apollo_ids, f"Expected APOLLO low-rank projection for {name}"


def test_stochastic_rounding_is_unbiased_and_preserves_bf16():
    target_val = 1.0 + (1.0 / 256.0) * 0.35
    x = torch.full((16384,), target_val, dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(99)
    rounded = stochastic_round_fp32_to_bf16(x, generator=gen)
    assert rounded.dtype == torch.bfloat16
    empirical_mean = float(rounded.float().mean())
    assert abs(empirical_mean - target_val) < 5e-4


def test_fp8_dynamic_range_codec_roundtrip_and_zero_safety():
    codec = CoatDynamicRangeFP8Codec(group_size=32)
    zeros = torch.zeros((4, 32), dtype=torch.float32)
    packed_z = codec.encode(zeros, signed=False)
    assert float(codec.decode(packed_z).abs().max()) == 0.0

    gen = torch.Generator().manual_seed(7)
    vals = torch.randn((8, 64), generator=gen, dtype=torch.float32) * 0.05
    packed = codec.encode(vals, signed=True)
    assert packed["q"].dtype == torch.float8_e4m3fn
    recon = codec.decode(packed)
    rel_err = float((recon - vals).norm() / vals.norm())
    assert rel_err < 0.08

    pos = vals.abs() + 1e-6
    packed_pos = codec.encode(pos, signed=False)
    recon_pos = codec.decode(packed_pos)
    assert bool((recon_pos >= 0).all())
    assert float((recon_pos - pos).norm() / pos.norm()) < 0.08
    assert codec.storage_bytes(packed_pos) < pos.numel() * pos.element_size()


@pytest.mark.parametrize(
    ("rank", "scale_type", "fp8_moments"),
    [
        (32, "channel", False),
        (16, "channel", True),
        (1, "tensor", False),  # APOLLO-Mini
    ],
)
def test_apollo_merged_optimizer_ranks_subspace_transfer_and_exact_state_dict_resume(rank, scale_type, fp8_moments):
    def _build_pair():
        torch.manual_seed(42)
        net = nn.Sequential(
            nn.Linear(96, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64, bias=True),
        )
        # Cast 2D Linear weights to BF16 to exercise stochastic-rounded BF16 weight updates!
        net[0].weight.data = net[0].weight.data.to(torch.bfloat16)
        net[3].weight.data = net[3].weight.data.to(torch.bfloat16)
        cfg = {
            "backbone_lr": 3e-3,
            "weight_decay": 0.01,
            "apollo_min_dim": 64,
            "apollo_update_proj_gap": 2,
            "apollo_scale": 1.0,
            "stochastic_rounding": True,
        }
        groups = build_apollo_param_groups(
            net,
            cfg,
            rank=rank,
            scale_type=scale_type,
            moment_codec="coat_fp8" if fp8_moments else "fp32",
        )
        opt = ApolloMergedOptimizer(groups, seed=777, coat_group_size=16)
        return net, opt

    net_a, opt_a = _build_pair()
    inputs = [torch.randn(6, 96, generator=torch.Generator().manual_seed(100 + s)) for s in range(5)]
    targets = [torch.randn(6, 64, generator=torch.Generator().manual_seed(200 + s)) for s in range(5)]

    losses = []
    saved_net_state = None
    saved_opt_state = None
    for step_idx in range(5):
        h = torch.nn.functional.linear(inputs[step_idx], net_a[0].weight.float())
        h = torch.nn.functional.gelu(net_a[1](h))
        pred = torch.nn.functional.linear(h, net_a[3].weight.float(), net_a[3].bias)
        loss = torch.nn.functional.mse_loss(pred, targets[step_idx])
        losses.append(float(loss.detach()))
        net_a.zero_grad(set_to_none=True)
        loss.backward()
        for p in net_a.parameters():
            if p.grad is not None:
                p.grad = p.grad.to(p.dtype)
        opt_a.step()
        if step_idx == 2:
            saved_net_state = {k: v.clone() for k, v in net_a.state_dict().items()}
            saved_opt_state = opt_a.state_dict()

    assert losses[-1] < losses[0], f"Expected loss decrease across 5 APOLLO steps; got {losses}"

    # Resume into a fresh model + fresh optimizer at step 2 (right across update_proj_gap=2 subspace rotation!)
    net_b, opt_b = _build_pair()
    net_b.load_state_dict(saved_net_state)
    opt_b.load_state_dict(saved_opt_state)
    for step_idx in range(3, 5):
        h = torch.nn.functional.linear(inputs[step_idx], net_b[0].weight.float())
        h = torch.nn.functional.gelu(net_b[1](h))
        pred = torch.nn.functional.linear(h, net_b[3].weight.float(), net_b[3].bias)
        loss = torch.nn.functional.mse_loss(pred, targets[step_idx])
        net_b.zero_grad(set_to_none=True)
        loss.backward()
        for p in net_b.parameters():
            if p.grad is not None:
                p.grad = p.grad.to(p.dtype)
        opt_b.step()

    for (k, pa), pb in zip(net_a.state_dict().items(), net_b.state_dict().values()):
        assert torch.equal(pa, pb), f"Exact resume mismatch on {k}"


@pytest.mark.parametrize("norm_mode", ["per_example", "per_token"])
def test_merged_gradient_accumulator_hook_clears_grad_uneven_batch_exactness_and_nan_rollback(norm_mode):
    torch.manual_seed(314)
    ref_net = nn.Sequential(nn.Linear(32, 48), nn.GELU(), nn.Linear(48, 16))
    acc_net = nn.Sequential(nn.Linear(32, 48), nn.GELU(), nn.Linear(48, 16))
    acc_net.load_state_dict(ref_net.state_dict())

    # Uneven microbatches: [2, 2, 1] -> total 5 examples, token counts [10, 14, 6] -> 30 tokens
    xs = [torch.randn(2, 32), torch.randn(2, 32), torch.randn(1, 32)]
    ys = [torch.randn(2, 16), torch.randn(2, 16), torch.randn(1, 16)]
    tok_counts = [10, 14, 6]

    # Reference full-batch exact update with SGD(lr=0.1)
    ref_opt = torch.optim.SGD(ref_net.parameters(), lr=0.1)
    ref_opt.zero_grad(set_to_none=True)
    if norm_mode == "per_example":
        x_all = torch.cat(xs, dim=0)
        y_all = torch.cat(ys, dim=0)
        (((ref_net(x_all) - y_all) ** 2).mean(dim=-1).mean()).backward()
    else:
        total_tok = float(sum(tok_counts))
        loss_tok = sum(((ref_net(xb) - yb) ** 2).mean() * ntok for xb, yb, ntok in zip(xs, ys, tok_counts)) / total_tok
        loss_tok.backward()
    ref_opt.step()

    acc_opt = torch.optim.SGD(acc_net.parameters(), lr=0.1)
    accumulator = MergedGradientAccumulator(acc_net, accum_mode="fp32")
    try:
        mbs = list(zip(xs, ys, tok_counts))

        def _compute_fn(item):
            xb, yb, ntok = item
            # Critical check after prior microbatch: `.grad` must always remain None!
            assert all(p.grad is None for p in acc_net.parameters())
            loss_mean = ((acc_net(xb) - yb) ** 2).mean()
            return loss_mean, ntok

        _, diag = accumulator.run_effective_batch(
            mbs,
            _compute_fn,
            acc_opt,
            normalization=norm_mode,
            microbatch_size_fn=lambda item: len(item[0]),
            max_grad_norm=0.0,
        )
        assert diag.finite is True and diag.applied is True
        assert all(p.grad is None for p in acc_net.parameters())
        for (k, p_acc), p_ref in zip(acc_net.state_dict().items(), ref_net.state_dict().values()):
            assert torch.allclose(p_acc, p_ref, atol=1e-6, rtol=1e-5), f"Uneven batch mismatch on {k}"

        # Inject NaN in microbatch 2 and verify clean rollback without mutating parameters!
        snap_before = {k: v.detach().clone() for k, v in acc_net.state_dict().items()}
        with accumulator.microbatch(weight=0.5, supervised_units=2):
            ((acc_net(xs[0]) - ys[0]) ** 2).mean().backward()
            assert all(p.grad is None for p in acc_net.parameters())
        with accumulator.microbatch(weight=0.5, supervised_units=2):
            (acc_net(xs[1]) * float("nan")).sum().backward()
        bad_diag = accumulator.step(acc_opt, max_grad_norm=1.0)
        assert bad_diag.finite is False and bad_diag.applied is False
        assert all(p.grad is None for p in acc_net.parameters())
        for k, v in acc_net.state_dict().items():
            assert torch.equal(v, snap_before[k])
    finally:
        accumulator.close()


def test_coat_spark_mlp_rejection_frozen_skip_eval_identity_and_weight_cache_rebuild():
    report = qualify_coat_kernels("cpu")
    assert report["passed"] is True
    assert report["telemetry"]["compression_ratio"] > 3.5

    # Must reject Llama-style SiLU MLPs explicitly!
    class BadSiLUMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(32, 64, bias=False)
            self.up_proj = nn.Linear(32, 64, bias=False)
            self.down_proj = nn.Linear(64, 32, bias=False)
            self.act_fn = nn.SiLU()

    with pytest.raises(ValueError, match="SiLU"):
        verify_spark_mlp_not_silu(BadSiLUMLP())

    model = make_synthetic_spark_native()
    ctrl = install_spark_coat(model.spark, group_size=16, min_features=64, use_fp8_matmul=False)
    assert len(ctrl.wrapped_linears) > 0
    assert len(ctrl.verified_mlp_activations) == len(model.spark.model.layers)

    # 1. Frozen Spark stage: CoatLinear skips saving `x`, yet propagates grad_x!
    model.set_stage(True, stage_mode="frozen_spark_connector")
    model.train()
    ctrl.telemetry.reset_step()
    x_in = torch.randn(2, 8, model.hidden, requires_grad=True)
    with ctrl.activation_context(training=True):
        out = model.spark(inputs_embeds=x_in).logits
    out.sum().backward()
    assert x_in.grad is not None and torch.isfinite(x_in.grad).all() and float(x_in.grad.abs().sum()) > 0.0
    assert ctrl.telemetry.skipped_frozen_linear_inputs > 0

    # 2. Eval mode: exact untouched F.linear parity (0.0 diff vs disabled controller)
    model.eval()
    x_eval = torch.randn(2, 6, model.hidden)
    ctrl.enabled = True
    y_on = model.spark(inputs_embeds=x_eval).logits
    ctrl.enabled = False
    y_off = model.spark(inputs_embeds=x_eval).logits
    assert float((y_on - y_off).abs().max()) == 0.0
    ctrl.enabled = True

    # 3. Weight cache invalidation & rebuild on mutation
    sample_coat_linear = next(m for m in model.spark.modules() if isinstance(m, CoatLinear))
    sample_coat_linear.rebuild_fp8_cache(force=True)
    assert "w_fp8_col_major" in sample_coat_linear._fp8_weight_cache
    old_q = sample_coat_linear._fp8_weight_cache["w_fp8_col_major"].clone()
    ctrl.invalidate_weight_caches()
    assert sample_coat_linear._fp8_weight_cache == {}
    with torch.no_grad():
        sample_coat_linear.weight.mul_(-2.5)
        sample_coat_linear.weight.add_(0.35)
    ctrl.rebuild_weight_caches(force=True)
    assert "w_fp8_col_major" in sample_coat_linear._fp8_weight_cache
    assert not torch.equal(sample_coat_linear._fp8_weight_cache["w_fp8_col_major"], old_q)


def test_qualify_matched_profiles_4_rung_ablation_and_end_to_end_checkpoint_restore(tmp_path: Path):
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :, 1] = 180
    arr[:, :, 3] = 255
    arr[16:48, 16:48, 3] = 0
    rgba_img = Image.fromarray(arr, mode="RGBA")
    examples = []
    for i in range(4):
        row = {
            "id": f"ex_{i}",
            "_family": "image",
            "language": "en",
            "task": "rgba_visual_pilot",
            "visual_kind": "uv_atlas",
            "enable_thinking": bool(i % 2 == 0),
            "prompt": f"Check cutout {i}",
            "answer": "<think>\nInspect alpha.\n</think>\ncutout" if i % 2 == 0 else "cutout",
        }
        media = {
            "images": [rgba_img.convert("RGB")],
            "alphas": [arr[:, :, 3].astype(np.float32) / 255.0],
            "visual_kind": "uv_atlas",
            "visual_units": [
                {
                    "visual_kind": "uv_atlas",
                    "visual_kind_index": 2,
                    "view_id": "uv_vanilla",
                    "view_id_index": 7,
                    "camera_right": [1.0, 0.0, 0.0],
                    "camera_up": [0.0, 1.0, 0.0],
                    "ortho_scale": 1.0,
                    "timestamp_s": 0.0,
                    "has_timestamp": False,
                    "capture_s": None,
                    "is_pixel_art_atlas": True,
                }
            ],
            "times": [0.0],
            "capture_times": None,
            "audio": None,
            "target_audio": None,
            "motion": None,
            "motion_times": None,
        }
        examples.append((row, media))

    base_cfg = make_synthetic_spark_native().policy
    report = qualify_matched_profiles(
        lambda cfg: make_synthetic_spark_native(cfg, seed=2026),
        examples,
        base_cfg,
        device="cpu",
        microbatch_size=1,
        accumulation_steps=2,
        num_updates=2,
    )
    profs = report["profiles"]
    assert set(profs) == {
        "bf16_adamw",
        "apollo_merged_r64",
        "apollo_merged_r32",
        "apollo_coat",
        "apollo_coat_fp8_moments",
    }
    for name, item in profs.items():
        assert item["status"] == "passed", f"{name} failed: {item}"
        assert item["frozen_parameters_unchanged"] is True

    # APOLLO rank-32 state must be strictly smaller than AdamW state, and FP8 moments smaller still!
    assert profs["apollo_merged_r32"]["optimizer_state_bytes"] < profs["bf16_adamw"]["optimizer_state_bytes"]
    assert profs["apollo_coat_fp8_moments"]["optimizer_state_bytes"] < profs["apollo_merged_r64"]["optimizer_state_bytes"]
    assert profs["apollo_coat"]["coat_telemetry"]["compressed_bytes"] > 0

    # End-to-end checkpoint() + load_checkpoint() round-trip with COAT weight cache rebuild
    model = make_synthetic_spark_native({"optimizer_profile": "apollo_coat", "apollo_rank": 16, "microbatch_size": 2}, seed=909)
    model.set_stage(False, stage_mode="full_spark")
    opt = optimizer_for(model, model.policy)
    sched = _build_scheduler(opt, kind="cosine", num_warmup_steps=1, num_training_steps=4)
    out = model(examples=examples[:2])
    out["loss"].backward()
    opt.step()
    sched.step()

    write(tmp_path / "compute-budget.json", {"used_cu": 0.0, "active_seconds": 1.0})
    cfg_save = {
        **model.policy,
        "initial_checkpoint_reservation_bytes": 50_000_000,
        "local_free_floor_bytes": 1_000_000,
    }
    state = {
        "stage": "unit_full_spark",
        "updates": 1,
        "alignment_only": False,
        "stage_mode": "full_spark",
        "unfreeze_vision_last_n_layers": 0,
        "planned_updates": 4,
        "warmup_steps": 1,
        "scheduler_kind": "cosine",
        "checkpoint_bytes": 0,
    }
    bound = {"source": "unit_test_binding"}
    store = SimpleNamespace(commit=lambda d: {"backend": "local_cold", "directory": str(d), "commit_sha256": "a" * 64})
    checkpoint(model, opt, sched, state, tmp_path, cfg_save, bound, store)

    fresh_model = make_synthetic_spark_native({"optimizer_profile": "apollo_coat", "apollo_rank": 16}, seed=111)
    loaded_model, loaded_opt, _loaded_sched, loaded_state = load_checkpoint(
        Path(state["last_checkpoint_local"]),
        cfg_save,
        bound,
        model=fresh_model,
        load_optimizer=True,
        device="cpu",
    )
    assert loaded_state["updates"] == 1
    for (k, p_orig), p_load in zip(model.state_dict().items(), loaded_model.state_dict().values()):
        assert torch.equal(p_orig, p_load), f"Checkpoint reload mismatch on {k}"
    assert len(loaded_opt.state) == len(opt.state)


def test_workstation_cuda_diagnostics_and_ablation_oom_recovery():
    import weakref
    from colab.final.run import _diagnose_cuda_unavailable

    diag = _diagnose_cuda_unavailable(torch)
    assert f"torch={torch.__version__}" in diag
    assert "torch_cuda_build=" in diag
    if torch.version.cuda is None:
        assert "CPU-only PyTorch wheel" in diag

    rgb = Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8), mode="RGB")
    examples = [
        (
            {
                "id": "oom_ex",
                "_family": "image",
                "language": "en",
                "task": "oom_test",
                "visual_kind": "ortho_render",
                "enable_thinking": False,
                "prompt": "Inspect",
                "answer": "ok",
            },
            {
                "images": [rgb],
                "alphas": [None],
                "visual_kind": "ortho_render",
                "visual_units": [{"visual_kind": "ortho_render", "visual_kind_index": 0, "view_id": "front", "view_id_index": 0}],
                "times": [0.0],
                "capture_times": None,
                "audio": None,
                "target_audio": None,
                "motion": None,
                "motion_times": None,
            },
        )
    ]
    oom_refs: list[weakref.ReferenceType] = []

    def builder(cfg):
        m = make_synthetic_spark_native(cfg, seed=77)
        if cfg["optimizer_profile"] == "bf16_adamw":
            oom_refs.append(weakref.ref(m))
            orig_fwd = m.forward

            def boom(*args, **kwargs):
                _ = orig_fwd(*args, **kwargs)
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 24.00 GiB")

            m.forward = boom
        return m

    base_cfg = make_synthetic_spark_native().policy
    report = qualify_matched_profiles(
        builder,
        examples,
        base_cfg,
        profiles=("bf16_adamw", "apollo_coat", "apollo_coat_fp8_moments"),
        rank_candidates=(32,),
        microbatch_size=1,
        accumulation_steps=1,
        num_updates=1,
        device="cpu",
    )
    assert report["profiles"]["bf16_adamw"]["status"] == "cuda_oom"
    assert report["profiles"]["apollo_coat"]["status"] == "passed"
    assert report["profiles"]["apollo_coat_fp8_moments"]["status"] == "passed"
    assert len(oom_refs) == 1 and oom_refs[0]() is None, "OOM model must be freed before subsequent profiles run"


def test_gradient_checkpointing_recomputation_with_visual_cross_attn_and_coat():
    """Regression test for non-reentrant gradient checkpointing + GatedVisualCrossAttentionBlock + COAT.

    Verifies mixed image + pure-text microbatches save identical tensor counts during forward
    and recomputation (no CheckpointError) under both `_gradient_checkpointing_func` and
    direct module-level `checkpoint(...)` styles, plus `apollo_include_embeddings=True`.
    """
    import tests._spark_test_fakes as fakes_mod
    from colab.final.engine import ensure_coat_for_model

    rgb = Image.fromarray(np.full((48, 48, 3), 90, dtype=np.uint8), mode="RGB")
    img_ex = (
        {
            "id": "ckpt_img",
            "_family": "image",
            "language": "en",
            "task": "ckpt_test",
            "visual_kind": "multiview_sheet",
            "enable_thinking": True,
            "prompt": "Inspect multiview",
            "answer": "<think>\nCheck box.\n</think>\nok",
        },
        {
            "images": [rgb],
            "alphas": [None],
            "visual_kind": "multiview_sheet",
            "visual_units": [{"visual_kind": "multiview_sheet", "visual_kind_index": 1, "view_id": "front", "view_id_index": 0}],
            "times": [0.0],
            "capture_times": None,
            "audio": None,
            "target_audio": None,
            "motion": None,
            "motion_times": None,
        },
    )
    txt_ex = (
        {
            "id": "ckpt_txt",
            "_family": "text",
            "language": "en",
            "task": "ckpt_test_txt",
            "enable_thinking": False,
            "prompt": "Hello",
            "answer": "world",
        },
        {
            "images": None,
            "alphas": None,
            "times": None,
            "capture_times": None,
            "audio": None,
            "target_audio": None,
            "motion": None,
            "motion_times": None,
        },
    )

    cfg = {
        "optimizer_profile": "apollo_coat",
        "apollo_rank": 16,
        "apollo_min_dim": 64,
        "apollo_include_embeddings": True,
        "visual_cross_attn_blocks": 3,
        "microbatch_size": 2,
    }
    model = make_synthetic_spark_native(cfg, seed=515)
    assert model.spark.model.gradient_checkpointing is True
    # Set nonzero cross-attn gates so gradients actively flow through both gate_attn/gate_ffn and vis_ctx!
    with torch.no_grad():
        for blk in model.visual_cross_attn:
            blk.gate_attn.fill_(0.25)
            blk.gate_ffn.fill_(0.25)

    model.set_stage(False, stage_mode="full_spark")
    model.train()
    coat_ctrl = ensure_coat_for_model(model, model.policy)
    opt = optimizer_for(model, model.policy)
    acc = MergedGradientAccumulator(
        model,
        accum_mode="bf16_apollo_fp32_small",
        min_bf16_accum_dim=64,
        allow_embeddings_in_apollo=True,
    )
    # Verify embed_tokens uses BF16 merged buffer and APOLLO low-rank group when apollo_include_embeddings=True
    emb_pid = id(model.spark.model.embed_tokens.weight)
    assert acc.buffers[emb_pid].dtype == torch.bfloat16
    apollo_pids = {id(p) for g in opt.param_groups if g.get("kind") == "apollo" for p in g["params"]}
    assert emb_pid in apollo_pids

    try:
        # 1. Mixed [image, pure-text] batch in a single autograd graph -> backward outside coat_ctx / scope!
        with coat_ctrl.activation_context(training=True):
            out = model(examples=[img_ex, txt_ex])
            loss_sum = out["per_example_loss"].sum()
        with acc.microbatch(weight=1.0, supervised_units=int(out["supervised_units"])):
            loss_sum.backward()

        xattn_grad_norm = sum(
            float(acc.buffers[id(p)].float().norm())
            for p in model.visual_cross_attn.parameters()
            if id(p) in acc.buffers
        )
        proj_grad_norm = sum(
            float(acc.buffers[id(p)].float().norm())
            for p in model.vision_projector.parameters()
            if id(p) in acc.buffers
        )
        assert xattn_grad_norm > 0.0
        assert proj_grad_norm > 0.0
        stats = acc.step(opt, invalidate_caches_fn=coat_ctrl.invalidate_weight_caches)
        assert stats.finite and stats.applied

        # 2. Also verify direct module-level `checkpoint(layer.__call__, ...)` path without `_gradient_checkpointing_func`
        model.spark.model._gradient_checkpointing_func = lambda fn, *a, **kw: fakes_mod.checkpoint(
            fn, *a, use_reentrant=False, **kw
        )
        with coat_ctrl.activation_context(training=True):
            out2 = model(examples=[img_ex, txt_ex])
            loss2 = out2["per_example_loss"].sum()
        with acc.microbatch(weight=1.0, supervised_units=int(out2["supervised_units"])):
            loss2.backward()
        stats2 = acc.step(opt, invalidate_caches_fn=coat_ctrl.invalidate_weight_caches)
        assert stats2.finite and stats2.applied
    finally:
        acc.close()
