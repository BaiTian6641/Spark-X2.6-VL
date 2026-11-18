"""Unit tests for Stage-7 `Spark-X2.6-4B-Loop48-En1B-VL` (`loop_engram.py` + `model.py` + `optimizer.py` + `engine.py`)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from colab.final.engine import _save_model_safetensors, optimizer_for
from colab.final.loop_engram import MultimodalTieredMoEEngram, SublayerDualMHC
from colab.final.model import (
    export_spark_native_fp8_safetensors,
    load_spark_native_safetensors,
)
from colab.final.optimizer import MergedGradientAccumulator
from tests._spark_test_fakes import make_synthetic_spark_native


def test_en1b_exact_capacity_and_2d_apollo_expert_shapes() -> None:
    # Verify exact full-scale Spark-X2.6-4B (`width=2560`) En1B-A256M parameter budget on meta device (0 RAM).
    with torch.device("meta"):
        eng = MultimodalTieredMoEEngram(
            width=2560,
            hot_buckets=8192,
            cold_buckets=32768,
            eng_dim=256,
            shared_rank=1536,
            num_experts=16,
            top_k=4,
            expert_rank=1368,
            route_dim=64,
            max_passes=2,
            offload_tier1b_cpu=False,
        )
        mhc = SublayerDualMHC(width=2560, pass_adapter_rank=16, max_passes=2)

    per_anchor_total = sum(p.numel() for p in eng.parameters())
    tier1b_per_anchor = 32768 * 3 * 256
    gpu_per_anchor = per_anchor_total - tier1b_per_anchor
    total_8_anchors = 8 * per_anchor_total
    total_12_mhc = 12 * sum(p.numel() for p in mhc.parameters())
    grand_total = total_8_anchors + total_12_mhc

    # Must hit `+1.0B` target cleanly (~1.008B total: 201.3M Host DRAM Tier-1B + ~806.7M GPU).
    assert 0.99e9 <= grand_total <= 1.03e9, f"Expected ~1.00B added params, got {grand_total:,}"
    assert 8 * tier1b_per_anchor == 201_326_592
    assert 790_000_000 <= 8 * gpu_per_anchor <= 815_000_000

    # All 48 expert weight tensors per anchor must be 2D with min(shape) >= 256 for rank-64 APOLLO projection.
    for plist in (eng.expert_gates, eng.expert_ups, eng.expert_downs):
        assert len(plist) == 16
        for p in plist:
            assert p.ndim == 2 and min(p.shape) >= 256


def test_step0_zero_gate_equivalence_and_kv_cache_parity() -> None:
    base_model = make_synthetic_spark_native(
        {"visual_cross_attn_blocks": 3, "detail_tiles": 2},
        seed=77,
    )
    loop_model = make_synthetic_spark_native(
        {
            "visual_cross_attn_blocks": 3,
            "detail_tiles": 2,
            "enable_loop_engram": True,
            "loop_start": 2,
            "loop_end": 4,
            "loop_passes": 2,
            "engram_anchors": [1, 3, 5],
            "engram_hot_buckets": 64,
            "engram_cold_buckets": 128,
            "engram_dim": 64,
            "engram_shared_rank": 96,
            "engram_num_experts": 4,
            "engram_top_k": 2,
            "engram_expert_rank": 64,
            "min_tile_source_dim": 0,
        },
        seed=77,
    )
    base_model.eval()
    loop_model.eval()

    # Copy exact backbone + vision + connector weights from `base_model` into `loop_model` (leaving loop/engram at Step-0 init).
    base_sd = base_model.state_dict()
    loop_sd = loop_model.state_dict()
    for k, v in base_sd.items():
        assert k in loop_sd
        loop_sd[k].copy_(v)

    text_row = {
        "id": "t0",
        "_family": "text",
        "task": "general_reasoning",
        "language": "en",
        "prompt": "Compute 17 + 25 step by step.",
        "answer": "<think>\n17 + 25 = 42.\n</think>\n42",
        "enable_thinking": True,
    }
    empty_media = {"images": [], "times": [], "capture_times": [], "audio": None, "motion": None}

    loss_base, u_base = base_model.text_loss(text_row, empty_media)
    loss_loop, u_loop = loop_model.text_loss(text_row, empty_media)
    assert u_base == u_loop
    assert abs(float(loss_base.item()) - float(loss_loop.item())) / max(1.0, abs(float(loss_base.item()))) < 1e-5

    # Multimodal Step-0 equivalence (including active non-zero visual cross-attention gates!).
    for blk_b, blk_l in zip(base_model.visual_cross_attn, loop_model.visual_cross_attn):
        blk_b.gate_attn.data.fill_(0.35)
        blk_l.gate_attn.data.fill_(0.35)

    img = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8), mode="RGB")
    mm_media = {"images": [img], "times": [0.0], "capture_times": [None], "audio": None, "motion": None}
    mm_row = {
        "id": "m0",
        "_family": "image",
        "task": "vqa",
        "language": "en",
        "prompt": "Point to the center object.",
        "answer": '<think>\n<point x="500" y="500" label="center"/>\n</think>\ncenter',
        "enable_thinking": True,
    }
    mm_loss_base, _ = base_model.text_loss(mm_row, mm_media)
    mm_loss_loop, _ = loop_model.text_loss(mm_row, mm_media)
    assert abs(float(mm_loss_base.item()) - float(mm_loss_loop.item())) / max(1.0, abs(float(mm_loss_base.item()))) < 1e-5

    # Verify KV-cached greedy decoding matches full-sequence logits & produces valid output on both pure-text and VL.
    ans_txt_base = base_model.answer(text_row, empty_media, max_new_tokens=8)
    ans_txt_loop = loop_model.answer(text_row, empty_media, max_new_tokens=8)
    assert ans_txt_base == ans_txt_loop

    ans_mm_base = base_model.answer(mm_row, mm_media, max_new_tokens=8)
    ans_mm_loop = loop_model.answer(mm_row, mm_media, max_new_tokens=8)
    assert ans_mm_base == ans_mm_loop


def test_min_tile_source_dim_guard_and_tool_response_masking() -> None:
    loop_model = make_synthetic_spark_native(
        {
            "detail_tiles": 4,
            "enable_loop_engram": True,
            "loop_start": 2,
            "loop_end": 4,
            "loop_passes": 2,
            "engram_anchors": [1, 3],
            "engram_hot_buckets": 32,
            "engram_cold_buckets": 64,
            "engram_dim": 64,
            "engram_shared_rank": 64,
            "engram_num_experts": 4,
            "engram_top_k": 2,
            "engram_expert_rank": 64,
        },
        seed=91,
    )
    # Default min_tile_source_dim == 224 when enable_loop_engram is True:
    # 1) Tiny 96x40 OCR word crop must NOT be sliced into 4 blurry detail tiles.
    tiny_ocr = Image.fromarray(np.zeros((40, 96, 3), dtype=np.uint8), mode="RGB")
    assert len(loop_model.tile_views(tiny_ocr, video=False)) == 1
    # 2) Pixel-art UV atlas remains exempt and still gets nearest-neighbor detail tiles.
    assert len(loop_model.tile_views(tiny_ocr, video=False, is_pixel_art_atlas=True)) == 5
    # 3) Large 320x320 diagram still gets 1 thumbnail + 4 detail tiles = 5 views.
    large_diag = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8), mode="RGB")
    assert len(loop_model.tile_views(large_diag, video=False)) == 5

    # Verify `<tool_response>...</tool_response>` tokens are masked with `-100` while `<think>` and `<tool_call>` stay supervised.
    interleaved = (
        '<think>\nCheck file.\n</think>\n<tool_call>\n{"name":"bash","cmd":"ls"}\n</tool_call>\n'
        "<tool_response>\nSECRET_ENV_OUTPUT_DO_NOT_SUPERVISE\n</tool_response>\n"
        "<think>\nFound file.\n</think>\nDone.<|im_end|>\n"
    )
    ids, labels, sup_cnt = loop_model._encode_answer_ids_and_labels(interleaved)
    assert ids.shape == labels.shape
    masked_positions = (labels == -100).nonzero(as_tuple=True)[1].tolist()
    assert len(masked_positions) > 10
    assert sup_cnt == int((labels != -100).sum().item())
    masked_decoded = loop_model.tokenizer.decode([int(ids[0, i].item()) for i in masked_positions], skip_special_tokens=False)
    assert "SECRET_ENV_OUTPUT_DO_NOT_SUPERVISE" in masked_decoded


def test_defrost_phase_a_and_full_phase_b_with_tier1b_sparse_adam_and_safetensors_roundtrip(tmp_path: Path) -> None:
    stage6_model = make_synthetic_spark_native({"visual_cross_attn_blocks": 3}, seed=105)
    stage6_sf = tmp_path / "stage6_final.safetensors"
    _save_model_safetensors(stage6_model, stage6_sf)

    stage7_cfg = {
        "visual_cross_attn_blocks": 3,
        "enable_loop_engram": True,
        "loop_start": 2,
        "loop_end": 4,
        "loop_passes": 2,
        "engram_anchors": [1, 3, 5],
        "engram_hot_buckets": 16,
        "engram_cold_buckets": 64,
        "engram_dim": 64,
        "engram_shared_rank": 64,
        "engram_num_experts": 4,
        "engram_top_k": 2,
        "engram_expert_rank": 64,
        "optimizer_profile": "apollo_coat",
        "apollo_rank": 16,
        "apollo_min_dim": 64,
        "backbone_lr": 1.2e-4,
        "connector_learning_rate": 2e-4,
        "engram_lr": 5e-4,
        "vision_lr": 3e-5,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
    }
    model = make_synthetic_spark_native(stage7_cfg, seed=205)

    # Warm-start from Stage-6 safetensors must succeed ONLY when allow_missing_loop_engram=True.
    with pytest.raises(RuntimeError):
        load_spark_native_safetensors(model, stage6_sf, allow_missing_loop_engram=False)
    load_spark_native_safetensors(model, stage6_sf, allow_missing_loop_engram=True)

    for mod in model.engram_anchors.values():
        assert mod.tier1b_cold_weight is not None
        assert mod.tier1b_cold_weight.device.type == "cpu"

    img = Image.fromarray(np.full((240, 240, 3), 90, dtype=np.uint8), mode="RGB")
    mm_ex = (
        {
            "id": "ex_mm",
            "_family": "image",
            "task": "vqa",
            "language": "en",
            "prompt": "Inspect region and count.",
            "answer": '<think>\n<box x1="100" y1="120" x2="400" y2="450" label="panel"/>\n<point x="250" y="280" label="bolt_1"/>\n</think>\n1',
            "enable_thinking": True,
        },
        {"images": [img], "times": [0.0], "capture_times": [None], "audio": None, "motion": None},
    )
    txt_ex = (
        {
            "id": "ex_txt",
            "_family": "text",
            "task": "deepswe_agent",
            "language": "en",
            "prompt": "Fix bug in parser.py",
            "answer": '<think>\nInspect AST.\n</think>\n<tool_call>\n{"cmd":"pytest"}\n</tool_call>\n<tool_response>\n1 passed\n</tool_response>\n<think>\nVerified.\n</think>\nFixed.',
            "enable_thinking": True,
        },
        {"images": [], "times": [], "capture_times": [], "audio": None, "motion": None},
    )

    # Phase A (`defrost_loop_engram`): Spark and SigLIP2 stay frozen; Engram + Dual-mHC + Connectors + Tier-1B update!
    model.set_stage(False, stage_mode="defrost_loop_engram")
    frozen_before = model.frozen_parameter_hashes()
    opt_a = optimizer_for(model, stage7_cfg)
    acc_a = MergedGradientAccumulator(model, accum_mode="bf16_apollo_fp32_small", min_bf16_accum_dim=64)
    try:
        for _step in range(2):
            out = model(examples=[mm_ex, txt_ex])
            loss = out["per_example_loss"].sum()
            with acc_a.microbatch(weight=1.0, supervised_units=out["supervised_units"]):
                loss.backward()
            diag = acc_a.step(opt_a, None, max_grad_norm=1.0, post_scale=0.5)
            assert diag.applied and diag.finite
            model.step_tier1b_sparse_adam(lr=5e-4, denom=2.0)
    finally:
        acc_a.close()

    assert model.frozen_parameter_hashes() == frozen_before
    tel = model.loop_engram_telemetry()
    assert tel["enabled"] is True
    assert tel["effective_layers"] == 8  # 6 physical + (4 - 2) * (2 - 1) == 8
    assert any(v["project_weight_norm"] > 0.0 for v in tel["engrams"].values())
    assert any(v["tier1b_nonzero_rows"] > 0 for v in tel["engrams"].values())

    # Phase B (`full_spark` + `unfreeze_vision_last_n_layers=2`): Spark + top-2 SigLIP2 + Connectors + Engram all update!
    model.set_stage(False, stage_mode="full_spark", unfreeze_vision_last_n_layers=2)
    frozen_early_vis = model.frozen_parameter_hashes()
    assert any("vision.vision_model.encoder.layers.0." in k for k in frozen_early_vis)
    assert not any("vision.vision_model.encoder.layers.3." in k for k in frozen_early_vis)

    opt_b = optimizer_for(model, stage7_cfg)
    acc_b = MergedGradientAccumulator(model, accum_mode="bf16_apollo_fp32_small", min_bf16_accum_dim=64)
    spark_embed_before = model.spark.model.embed_tokens.weight.detach().clone()
    vis_top_before = model.vision.vision_model.encoder.layers[-1].fc1.weight.detach().clone()
    try:
        out = model(examples=[mm_ex, txt_ex])
        loss = out["per_example_loss"].sum()
        with acc_b.microbatch(weight=1.0, supervised_units=out["supervised_units"]):
            loss.backward()
        diag = acc_b.step(opt_b, None, max_grad_norm=1.0, post_scale=0.5)
        assert diag.applied and diag.finite
        model.step_tier1b_sparse_adam(lr=1.2e-4, denom=2.0)
    finally:
        acc_b.close()

    assert model.frozen_parameter_hashes() == frozen_early_vis
    assert not torch.equal(spark_embed_before, model.spark.model.embed_tokens.weight.detach())
    assert not torch.equal(vis_top_before, model.vision.vision_model.encoder.layers[-1].fc1.weight.detach())

    # Verify full Stage-7 BF16 + FP8 safetensors roundtrip (requiring 100% of loop + engram + tier1b keys).
    stage7_bf16_sf = tmp_path / "stage7_bf16.safetensors"
    stage7_fp8_sf = tmp_path / "stage7_fp8.safetensors"
    _save_model_safetensors(model, stage7_bf16_sf)
    fp8_meta = export_spark_native_fp8_safetensors(model, stage7_fp8_sf)
    assert fp8_meta["quantized_linear_tensors"] > 0

    reloaded_bf16 = make_synthetic_spark_native(stage7_cfg, seed=999)
    load_spark_native_safetensors(reloaded_bf16, stage7_bf16_sf, allow_missing_loop_engram=False)
    for k in model.engram_anchors:
        assert torch.allclose(
            model.engram_anchors[k].tier1b_cold_weight,
            reloaded_bf16.engram_anchors[k].tier1b_cold_weight,
        )

    reloaded_fp8 = make_synthetic_spark_native(stage7_cfg, seed=888)
    load_spark_native_safetensors(reloaded_fp8, stage7_fp8_sf, allow_missing_loop_engram=False)
    model.eval()
    reloaded_fp8.eval()
    l_orig, _ = model.text_loss(*txt_ex)
    l_fp8, _ = reloaded_fp8.text_loss(*txt_ex)
    assert abs(float(l_orig.item()) - float(l_fp8.item())) < 1e-2


def test_stage7_campaign_config_and_rsi_verifiers() -> None:
    from colab.final.common import HERE, read, validate_config
    from colab.final.vision_rsi import score_rsi_candidate
    from colab.final.visual_primitives_corpus import render_arc_task_sheet

    cfg = validate_config(read(HERE / "campaign_rtx_pro_4000_stage7_loop48_en1b_vl.json"))
    assert cfg["enable_loop_engram"] is True
    assert cfg["loop_start"] == 12 and cfg["loop_end"] == 24 and cfg["loop_passes"] == 2
    assert cfg["max_visual_tokens"] == 1536 and cfg["min_tile_source_dim"] == 224
    assert len(cfg["stages"]) == 2
    assert cfg["stages"][0]["stage_mode"] == "defrost_loop_engram"
    assert cfg["stages"][1]["stage_mode"] == "full_spark"
    assert cfg["stages"][1]["unfreeze_vision_last_n_layers"] == 6

    png_b, cmeta = render_arc_task_sheet(
        [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
        [[0, 2], [2, 0]],
    )
    assert len(png_b) > 500 and len(cmeta["demo_boxes_1000"]) == 1

    arc_row = {
        "task": "arc_agi_2_visual_primitives",
        "enable_thinking": True,
        "metadata": {"oracle_grid": [[2, 0], [0, 2]]},
    }
    assert score_rsi_candidate(
        arc_row,
        '<think>\n<box x1="10" y1="10" x2="400" y2="400" label="demo"/>\n</think>\n```json\n[[2, 0], [0, 2]]\n```',
    )["verified"]
    assert not score_rsi_candidate(
        arc_row,
        "<think>\nNo visual primitives used here.\n</think>\n```json\n[[2, 0], [0, 2]]\n```",
    )["verified"]

    vp_row = {
        "task": "visual_primitives_point_count",
        "enable_thinking": True,
        "metadata": {"oracle_count": 2},
    }
    assert score_rsi_candidate(
        vp_row,
        '<think>\n<point x="120" y="340" label="obj_1"/> <point x="560" y="780" label="obj_2"/>\n</think>\n2',
    )["verified"]

    yn_row = {
        "task": "visual_primitives_existence_yesno",
        "enable_thinking": True,
        "metadata": {"oracle_yesno": "No"},
    }
    assert score_rsi_candidate(
        yn_row,
        '<think>\n<box x1="0" y1="0" x2="500" y2="500" label="nw"/>\n</think>\nNo',
    )["verified"]
