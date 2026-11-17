"""End-to-end CPU verification for Spark-X2.6-VL vision architecture, ablations, and SyncCraft v3.2 specialization.

Covers Plan Steps 1, 2, 4, 5, and 6:
- Step 1: Pure-text logit & greedy-decode parity (`<think>` and non-thinking), custom system prompts,
  sampling controls, zero visual overhead on text-only calls, `vision_text` audio/motion deferral,
  and bit-exact SHA256 invariance of frozen parameters across training steps.
- Step 2: Explicit `visual_kind` routing (`independent_image`, `multiview_render`, `uv_atlas`,
  `state_pair`, `temporal_frame`) so static multiviews/UVs/state-pairs NEVER become video,
  detail-tile preservation, nearest-neighbor crisp texel magnification on pixel-art UV islands,
  and RGBA alpha transparency preservation (`0.0` exact output on opaque RGB vs distinct tokens on cutouts).
- Steps 4 & 5: Optional `0` / `3` / `6` Flamingo-style zero-init `tanh`-gated visual cross-attention
  adapters (exact step-0 identity + nonzero multimodal gradients), gated late SigLIP2 layer unfreezing,
  and blank-image + shuffled-view/donor-image evaluation controls.
- Step 6: SyncCraft v3.2 clean-room provenance / `NoAI` / Mojang / `-ND` / `research_nc` license gates,
  zero-leakage asset/family split grouping, and compiler / 15-axis SAT / UV-alpha / WorldAction /
  `<0.02u` stance-slip `@behavior` oracle verification.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from colab.final.assets import prepare_local_synccraft_data
from colab.final.common import read
from colab.final.data import (
    Catalogue,
    MediaReader,
    decode_visual_image,
    family,
    infer_visual_kind,
    prepare_media,
)
from colab.final.engine import optimizer_for
from colab.final.evaluate import evaluate, verify_text_only_parity
from colab.final.inference import empty_media, generate, visual_media
from colab.final.model import compute_patch_alpha_transparency_features
from colab.final.optimizer import MergedGradientAccumulator
from colab.final.synccraft_bridge import (
    LicenseGateError,
    _inject_collision_defect,
    build_synccraft_curriculum_records,
    evaluate_synccraft_completion,
    make_clean_room_provenance,
    validate_synccraft_license_record,
)
from synccraft.dsl.ast_nodes import Attach, Bone
from synccraft.dsl.parser import format_program, parse_program
from tests._spark_test_fakes import make_synthetic_spark_native


def _make_png_bytes(arr: np.ndarray, mode: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def test_step1_pure_text_parity_dual_thinking_system_prompts_and_sampling() -> None:
    model = make_synthetic_spark_native({"visual_cross_attn_blocks": 3}, seed=11)
    reference_spark = copy.deepcopy(model.spark)

    # Deferred audio & motion towers must stay unloaded in vision_text mode.
    assert model.modalities == "vision_text"
    assert model.speech_in is None
    assert model.sound_in is None
    assert model.motion is None
    assert model.speech_out is None
    assert model.sound_out is None

    model.vision.forward_calls = 0
    parity = verify_text_only_parity(
        model,
        reference_spark,
        prompts=[
            "Explain rigid SE(3) bone pivots concisely.",
            "List the 15 separating axes between two 3D oriented boxes.",
        ],
        enable_thinking_modes=(False, True),
        max_new_tokens=10,
        atol=0.0,
    )
    assert parity["passed"] is True
    assert parity["cases_checked"] == 4
    assert parity["max_abs_logit_diff"] == 0.0
    assert parity["exact_generation_match"] is True
    assert model.vision.forward_calls == 0

    # Pure-text training loss & batched text loss also bypass all visual towers and cross-attn hooks.
    text_rows = [
        {
            "_family": "text",
            "language": "en",
            "task": "text_replay",
            "prompt": "What is inflate in Block-DSL?",
            "answer": "Inflate is symmetric visual shell dilation separate from integer box size.",
            "enable_thinking": False,
        },
        {
            "_family": "text",
            "language": "en",
            "task": "text_replay",
            "prompt": "Derive box placement.",
            "answer": "<think>\nCheck parent pivot frame.\n</think>\nUse attach_point minus local anchor.",
            "enable_thinking": True,
        },
    ]
    loss_single, units_single = model.loss(text_rows[0], empty_media())
    loss_batch, units_batch = model.text_batch_loss([(r, empty_media()) for r in text_rows])
    assert torch.isfinite(loss_single) and units_single > 0
    assert len(loss_batch) == 2 and all(torch.isfinite(x) for x in loss_batch)
    assert sum(units_batch) > units_single
    assert model.vision.forward_calls == 0
    assert model._active_visual_context is None

    # Unified generate() supports custom system prompt + sampling controls without invoking vision.
    torch.manual_seed(123)
    sampled = generate(
        model,
        "Summarize Block-DSL integer dimensions.",
        enable_thinking=True,
        max_new_tokens=12,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=16,
        repetition_penalty=1.15,
        system_prompt="You are a clean-room 3D CAD geometry assistant.",
    )
    assert isinstance(sampled, str)
    assert sampled.startswith("<think>\n")
    assert model.vision.forward_calls == 0


def test_step1_frozen_parameter_hashes_unchanged_during_connector_and_cross_attn_stages() -> None:
    model = make_synthetic_spark_native({"visual_cross_attn_blocks": 3}, seed=19)
    rgb = np.zeros((56, 56, 3), dtype=np.uint8)
    rgb[12:44, 12:44] = (210, 90, 45)
    media = visual_media([Image.fromarray(rgb, mode="RGB")], visual_kind="independent_image")
    row = {
        "_family": "image",
        "language": "en",
        "task": "caption",
        "prompt": "Describe the centered orange square.",
        "answer": "An orange square centered on a dark canvas.",
        "enable_thinking": False,
    }

    for stage_mode in ("frozen_spark_connector", "frozen_spark_cross_attn"):
        model.set_stage(stage_mode=stage_mode)
        before_frozen = model.frozen_parameter_hashes()
        assert any(k.startswith("spark.") for k in before_frozen)
        assert any(k.startswith("vision.") for k in before_frozen)
        if stage_mode == "frozen_spark_connector":
            assert any(k.startswith("visual_cross_attn.") for k in before_frozen)
        else:
            assert not any(k.startswith("visual_cross_attn.") for k in before_frozen)

        before_proj = model.vision_projector[1].weight.detach().clone()
        opt_cfg = {
            "optimizer_profile": "bf16_adamw",
            "adapter_lr": 5e-3,
            "backbone_lr": 1e-4,
            "vision_lr": 5e-5,
            "output_lr": 5e-6,
            "weight_decay": 0.01,
        }
        optimizer = optimizer_for(model, opt_cfg)
        accum = MergedGradientAccumulator(model, accum_mode="fp32")
        try:
            model.train()
            _, diag = accum.run_effective_batch(
                [[(row, media)]],
                lambda mb: model.loss(mb[0][0], mb[0][1]),
                optimizer,
                normalization="per_token",
                max_grad_norm=1.0,
            )
            assert diag.applied is True
        finally:
            accum.close()

        after_frozen = model.frozen_parameter_hashes()
        assert before_frozen == after_frozen, f"Frozen parameters mutated during {stage_mode}!"
        assert not torch.equal(before_proj, model.vision_projector[1].weight.detach())


def test_step2_explicit_visual_kinds_never_conflate_multiview_uv_or_state_pair_with_video() -> None:
    model = make_synthetic_spark_native({"detail_tiles": 4}, seed=23)
    png_a = _make_png_bytes(np.full((48, 48, 3), 80, dtype=np.uint8), "RGB")
    png_b = _make_png_bytes(np.full((48, 48, 3), 160, dtype=np.uint8), "RGB")
    store = {"a.png": png_a, "b.png": png_b, "c.png": png_a, "d.png": png_b}
    cfg = {"modalities": "omni", "motion_frames": 4, "video_semantic_frames": 4}

    multiview_row = {
        "task": "synccraft_multiview_inverse_dsl",
        "visual_kind": "multiview_render",
        "frames": ["a.png", "b.png", "c.png", "d.png"],
        "media_times_s": [0.0, 0.0, 0.0, 0.0],
    }
    uv_row = {
        "task": "synccraft_uv_alpha_diagnosis",
        "visual_kind": "uv_atlas",
        "frames": ["a.png", "b.png"],
        "media_times_s": [0.0, 0.0],
    }
    state_pair_row = {
        "task": "synccraft_typed_scene_edit",
        "visual_kind": "state_pair",
        "frames": ["a.png", "b.png"],
        "media_times_s": [0.0, 0.0],
    }
    temporal_row = {
        "task": "synccraft_behavior_animation",
        "visual_kind": "temporal_frame",
        "frames": ["a.png", "b.png", "c.png", "d.png"],
        "media_times_s": [0.0, 0.25, 0.5, 0.75],
    }

    for static_row, expected_kind in (
        (multiview_row, "multiview_render"),
        (uv_row, "uv_atlas"),
        (state_pair_row, "state_pair"),
    ):
        assert infer_visual_kind(static_row) == expected_kind
        assert family(static_row) == "image"
        static_row["_family"] = "image"
        m = prepare_media(static_row, store.__getitem__, cfg)
        assert m["visual_kind"] == expected_kind
        assert m["motion"] is None, f"{expected_kind} must NEVER populate the video motion tower"
        # Static multiviews / UVs / state-pairs retain 1 thumbnail + 4 detail tiles = 5 views per image.
        views = model.tile_views(m["images"][0], video=False, alpha=m["alphas"][0])
        assert len(views) == 5

    assert infer_visual_kind(temporal_row) == "temporal_frame"
    assert family(temporal_row) == "video"
    temporal_row["_family"] = "video"
    m_temp = prepare_media(temporal_row, store.__getitem__, {"modalities": "vision_text"})
    assert m_temp["visual_kind"] == "temporal_frame"
    assert m_temp["motion"] is None  # V-JEPA2 deferred in vision_text mode
    assert len(model.tile_views(m_temp["images"][0], video=True)) == 1

    # Crisp nearest-neighbor magnification on tiny (<64px) pixel-art UV islands preserves exact palette colors.
    checker = np.zeros((32, 32, 3), dtype=np.uint8)
    checker[::2, ::2] = (255, 0, 0)
    checker[1::2, 1::2] = (0, 255, 0)
    uv_views = model.tile_views(Image.fromarray(checker, mode="RGB"), video=False, is_pixel_art_atlas=True)
    assert len(uv_views) == 5
    detail_crop_im = uv_views[1][0]
    assert detail_crop_im.size == (64, 64)
    unique_colors = {tuple(px) for px in np.asarray(detail_crop_im).reshape(-1, 3).tolist()}
    assert unique_colors.issubset({(0, 0, 0), (255, 0, 0), (0, 255, 0)})


def test_step2_rgba_alpha_preservation_and_zero_bias_opaque_vs_cutout_distinction() -> None:
    model = make_synthetic_spark_native({"detail_tiles": 2}, seed=29)
    model.train()

    rgb_arr = np.full((56, 56, 3), 140, dtype=np.uint8)
    rgb_arr[8:48, 8:48] = (60, 130, 220)
    rgba_arr = np.concatenate([rgb_arr, np.full((56, 56, 1), 255, dtype=np.uint8)], axis=-1)
    # Punch an alpha=0 cutout window while keeping RGB pixels identical!
    rgba_arr[16:40, 16:40, 3] = 0

    opaque_png = _make_png_bytes(rgb_arr, "RGB")
    cutout_png = _make_png_bytes(rgba_arr, "RGBA")

    im_opaque, alpha_opaque = decode_visual_image(opaque_png)
    im_cutout, alpha_cutout = decode_visual_image(cutout_png)
    assert alpha_opaque is None
    assert alpha_cutout is not None and alpha_cutout.shape == (56, 56)
    assert float(alpha_cutout.min()) == 0.0 and float(alpha_cutout.max()) == 1.0

    # Zero-bias alpha_projector guarantees exact 0.0 contribution on opaque RGB inputs.
    zero_feat = compute_patch_alpha_transparency_features(None, 4, 4, model.device)
    assert float(zero_feat.abs().max().item()) == 0.0
    zero_proj = model.alpha_projector(zero_feat)
    assert float(zero_proj.abs().max().item()) == 0.0

    m_opaque = visual_media(
        [im_opaque],
        visual_kind="uv_atlas",
        visual_units=[
            {
                "visual_kind": "uv_atlas",
                "view_id": "uv_vanilla",
                "camera_right": [1.0, 0.0, 0.0],
                "camera_up": [0.0, 1.0, 0.0],
                "ortho_scale": 1.0,
                "is_pixel_art_atlas": True,
            }
        ],
    )
    m_cutout = visual_media(
        [im_cutout],
        visual_kind="uv_atlas",
        visual_units=m_opaque["visual_units"],
    )

    tok_opaque = model.image_tokens(
        m_opaque["images"],
        m_opaque["times"],
        m_opaque["capture_times"],
        m_opaque["alphas"],
        m_opaque["visual_units"],
    )
    tok_cutout = model.image_tokens(
        m_cutout["images"],
        m_cutout["times"],
        m_cutout["capture_times"],
        m_cutout["alphas"],
        m_cutout["visual_units"],
    )
    assert tok_opaque is not None and tok_cutout is not None
    assert tok_opaque.shape == tok_cutout.shape
    diff = float((tok_cutout - tok_opaque).abs().max().item())
    assert diff > 1e-4, f"RGBA cutout and opaque RGB produced indistinguishable visual tokens (diff={diff})"

    loss_proxy = (tok_cutout ** 2).mean()
    loss_proxy.backward()
    assert model.alpha_projector[0].weight.grad is not None
    assert float(model.alpha_projector[0].weight.grad.abs().max().item()) > 0.0
    assert model.camera_pose_encoder[0].weight.grad is not None
    assert model.visual_kind_embedding.weight.grad is not None
    assert model.view_id_embedding.weight.grad is not None


def test_step4_and_step5_gated_cross_attention_zero_init_late_vision_unfreeze_and_visual_ablations(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="visual_cross_attn_blocks"):
        make_synthetic_spark_native({"visual_cross_attn_blocks": 2})

    m0 = make_synthetic_spark_native({"visual_cross_attn_blocks": 0, "max_sequence": 8192, "max_answer_tokens": 4096}, seed=41)
    m3 = make_synthetic_spark_native({"visual_cross_attn_blocks": 3, "max_sequence": 8192, "max_answer_tokens": 4096}, seed=41)
    m6 = make_synthetic_spark_native({"visual_cross_attn_blocks": 6, "max_sequence": 8192, "max_answer_tokens": 4096}, seed=41)
    assert len(m0.visual_cross_attn) == 0
    assert len(m3.visual_cross_attn) == 3
    assert len(m6.visual_cross_attn) == 6

    # Copy exact connector & backbone weights from m0 into m3 and m6 so we isolate tanh(0) cross-attn identity.
    m0_sd = m0.state_dict()
    m3.load_state_dict(m0_sd, strict=False)
    m6.load_state_dict(m0_sd, strict=False)
    m0.eval()
    m3.eval()
    m6.eval()

    canvas = np.zeros((56, 56, 3), dtype=np.uint8)
    canvas[10:46, 14:42] = (180, 220, 90)
    med = visual_media([Image.fromarray(canvas, mode="RGB")], visual_kind="independent_image")
    sample_row = {
        "_family": "image",
        "language": "en",
        "task": "caption",
        "prompt": "Identify the green rectangle.",
        "answer": "A centered green rectangle.",
        "enable_thinking": False,
    }
    with torch.no_grad():
        l0, _ = m0.loss(sample_row, med)
        l3, _ = m3.loss(sample_row, med)
        l6, _ = m6.loss(sample_row, med)
    assert torch.allclose(l0, l3, atol=0.0, rtol=0.0)
    assert torch.allclose(l0, l6, atol=0.0, rtol=0.0)

    # Multimodal backward at step 0 drives nonzero gradient into the Flamingo tanh gates (`gate_attn`),
    # and once gates step off zero, step 1 drives nonzero gradient into cross-attention projections!
    m3.train()
    m3.set_stage(stage_mode="frozen_spark_cross_attn")
    loss_xattn, _ = m3.loss(sample_row, med)
    loss_xattn.backward()
    gate_grads = [
        float(blk.gate_attn.grad.abs().max().item())
        for blk in m3.visual_cross_attn
        if blk.gate_attn.grad is not None
    ]
    assert len(gate_grads) == 3 and any(g > 0.0 for g in gate_grads)
    with torch.no_grad():
        for blk in m3.visual_cross_attn:
            blk.gate_attn.fill_(0.1)
            blk.gate_ffn.fill_(0.1)
    m3.zero_grad(set_to_none=True)
    loss_xattn_open, _ = m3.loss(sample_row, med)
    loss_xattn_open.backward()
    attn_proj_grads = [
        float(blk.attn.out_proj.weight.grad.abs().max().item())
        for blk in m3.visual_cross_attn
        if blk.attn.out_proj.weight.grad is not None
    ]
    assert len(attn_proj_grads) == 3 and any(g > 0.0 for g in attn_proj_grads)

    # Late SigLIP2 layer unfreezing (`vision_late_unfreeze`, last_n=2 of 4 synthetic vision layers).
    m3.zero_grad(set_to_none=True)
    m3.set_stage(stage_mode="vision_late_unfreeze", unfreeze_vision_last_n_layers=2)
    enc_layers = m3.vision.vision_model.encoder.layers
    assert len(enc_layers) == 4
    assert all(not p.requires_grad for layer in enc_layers[:2] for p in layer.parameters())
    assert all(p.requires_grad for layer in enc_layers[2:] for p in layer.parameters())
    assert all(p.requires_grad for p in m3.vision.vision_model.post_layernorm.parameters())
    assert all(not p.requires_grad for p in m3.spark.parameters())

    # Build local SyncCraft dataset and run evaluate() to verify blank-image & shuffled-view controls.
    ws_root = tmp_path / "workstation"
    cfg = {
        "seed": 20260419,
        "modalities": "vision_text",
        "enable_thinking_default": True,
        "allow_synccraft_research_nc": False,
        "split_percent": {"train": 70, "dev": 10, "promotion": 10, "audit": 10},
        "evaluation_per_family_language": 3,
        "evaluation_max_new_tokens": 24,
        "cache_bytes_per_component": 200_000_000,
        "initial_checkpoint_reservation_bytes": 0,
        "local_free_floor_bytes": 0,
    }
    prepare_local_synccraft_data(ws_root, cfg, panel_size=(64, 64))
    catalogue = Catalogue(ws_root / "metadata" / "corpus.sqlite")
    handoff = read(ws_root / "metadata" / "HANDOFF.json")
    reader = MediaReader(ws_root, handoff, cfg)
    eval_report = evaluate(
        m3,
        catalogue,
        reader,
        cfg,
        split="dev",
        path=ws_root / "eval_dev",
        generate=True,
    )
    assert eval_report["human_audio_review"] == "not_applicable_vision_text"
    image_rows = [r for r in eval_report["rows"] if r["family"] == "image"]
    assert len(image_rows) > 0
    assert all("ablated_loss" in r and "media_loss_advantage" in r for r in image_rows)
    assert all("shuffled_loss" in r and "shuffled_loss_advantage" in r for r in image_rows)
    assert eval_report["synccraft_summary"] is not None
    assert eval_report["synccraft_summary"]["evaluated_records"] >= 1


def test_step6_synccraft_provenance_gate_family_split_isolation_and_compiler_sat_uv_motion_oracles(
    tmp_path: Path,
) -> None:
    # 1. Clean-room permissive license passes; Mojang, NoAI, -ND, AI-generated, and unquarantined NC fail!
    clean_prov = make_clean_room_provenance("test_asset_01", title="Clean Procedural Timber")
    clean_rec = {"prompt": "Inspect timber.", "answer": "OK", "provenance": clean_prov}
    assert validate_synccraft_license_record(clean_rec) == "clean_room_permissive"

    mojang_rec = copy.deepcopy(clean_rec)
    mojang_rec["provenance"]["source_url"] = "https://minecraft.net/assets/creeper.png"
    with pytest.raises(LicenseGateError):
        validate_synccraft_license_record(mojang_rec)

    noai_rec = copy.deepcopy(clean_rec)
    noai_rec["metadata"] = {"notes": "Artist tag: NoAI / do-not-train"}
    with pytest.raises(LicenseGateError):
        validate_synccraft_license_record(noai_rec)

    nd_rec = copy.deepcopy(clean_rec)
    nd_rec["provenance"]["license_spdx"] = "CC-BY-ND-4.0"
    with pytest.raises(LicenseGateError):
        validate_synccraft_license_record(nd_rec)

    ai_gen_rec = copy.deepcopy(clean_rec)
    ai_gen_rec["provenance"]["is_ai_generated"] = True
    with pytest.raises(LicenseGateError):
        validate_synccraft_license_record(ai_gen_rec)

    nc_rec = copy.deepcopy(clean_rec)
    nc_rec["provenance"]["license_spdx"] = "CC-BY-NC-4.0"
    with pytest.raises(LicenseGateError):
        validate_synccraft_license_record(nc_rec, allow_research_nc=False)
    assert validate_synccraft_license_record(nc_rec, allow_research_nc=True) == "research_nc"

    # 2. Asset & family lineage split isolation: every derivative task of an asset_id stays in ONE split.
    ws_root = tmp_path / "split_audit"
    cfg = {
        "seed": 20260419,
        "modalities": "vision_text",
        "enable_thinking_default": True,
        "allow_synccraft_research_nc": False,
        "split_percent": {"train": 70, "dev": 10, "promotion": 10, "audit": 10},
        "evaluation_per_family_language": 2,
    }
    prepare_local_synccraft_data(ws_root, cfg, panel_size=(64, 64))
    cat = Catalogue(ws_root / "metadata" / "corpus.sqlite")
    splits_by_asset: dict[str, set[str]] = {}
    for split_name in ("train", "dev", "promotion", "audit"):
        for rid, _fam, _lang, _comp, _shard in cat.select(split_name):
            row, _pkt = cat.row(rid)
            base_asset = row.get("metadata", {}).get("base_asset_id")
            if base_asset:
                splits_by_asset.setdefault(base_asset, set()).add(split_name)
    assert len(splits_by_asset) >= 4
    for aid, asset_splits in splits_by_asset.items():
        assert len(asset_splits) == 1, f"Cross-split leakage on {aid}: appeared in {asset_splits}"

    # 3. SyncCraft v3.2 compiler, 15-axis SAT, RGBA UV/alpha, WorldAction, and <0.02u stance-slip oracles.
    records_and_media = build_synccraft_curriculum_records(
        include_animation=True,
        enable_thinking=True,
        panel_size=(80, 80),
    )
    seen_tasks: set[str] = set()
    multiview_record = None
    for rec, _payloads in records_and_media:
        seen_tasks.add(rec["task"])
        res = evaluate_synccraft_completion(rec, rec["answer"])
        assert res["has_think_trace"] is True
        assert res["parse_valid"] is True
        assert res["passed"] is True, f"Oracle completion failed for {rec['id']}: {res}"
        if rec["task"] == "synccraft_behavior_animation":
            assert res["swept_collision_free"] is True
            assert res["stance_slip_passed"] is True
            assert res["max_stance_slip_units"] <= 0.02
        if rec["task"] == "synccraft_multiview_inverse_dsl" and multiview_record is None:
            multiview_record = rec

    assert seen_tasks == {
        "synccraft_part_grounding",
        "synccraft_multiview_inverse_dsl",
        "synccraft_uv_alpha_diagnosis",
        "synccraft_typed_scene_edit",
        "synccraft_behavior_animation",
    }

    # Inject a deliberate volumetric 3D box overlap (unexempted sibling bone at identical pose) and confirm 15-axis SAT rejects it!
    assert multiview_record is not None
    prog = parse_program(multiview_record["metadata"]["canonical_dsl"])
    first_box = next(item for item in prog.root.items if type(item).__name__ == "Box")
    rogue_bone = Bone(
        name="rogue_overlap_bone",
        attach=Attach(first_box.name, anchor=(0.0, 1.0, 0.0), align=(0.0, -1.0, 0.0)),
        items=[replace(first_box, name="rogue_volumetric_slab", place=None)],
    )
    bad_prog = replace(prog, root=replace(prog.root, items=[*prog.root.items, rogue_bone]))
    bad_completion = f"<think>\nTesting SAT rejection.\n</think>\n```blockdsl\n{format_program(bad_prog)}\n```"
    bad_eval = evaluate_synccraft_completion(multiview_record, bad_completion)
    assert bad_eval["parse_valid"] is True
    assert bad_eval["sat_passed"] is False
    assert bad_eval["collision_count"] >= 1
    assert bad_eval["passed"] is False


def test_synccraft_v32_sculpted_spin_and_4way_uv_curriculum():
    """Verify ADR 0007 sculpted + blend_with + spin(...) exemplars, anti-cheat profile gate, and 4-way UV balance."""
    from dataclasses import replace
    from colab.final.synccraft_bridge import (
        build_animation_clip_record,
        build_artist_assistant_record,
        build_multiview_consistency_record,
        build_parametric_creature_world,
        build_parametric_sculpted_wyvern_world,
        build_parametric_walking_citadel_world,
        build_uv_seam_alpha_diagnosis_record,
        ensure_synccraft_importable,
        evaluate_synccraft_completion,
        make_clean_room_provenance,
    )

    ensure_synccraft_importable()
    from synccraft.dsl.parser import format_program, parse_program

    # 1. Exact 4-way UV/alpha diagnosis (`a0_o0`, `a1_o0`, `a0_o1`, `a1_o1`)
    bio_ws = build_parametric_creature_world(variant=0, subdivisions_per_unit=2)
    for inject_alpha in (False, True):
        for inject_overlap in (False, True):
            rec, blobs = build_uv_seam_alpha_diagnosis_record(
                bio_ws,
                asset_id=f"bio_{int(inject_alpha)}_{int(inject_overlap)}",
                family_id="bio_fam",
                inject_alpha_cutout=inject_alpha,
                inject_uv_overlap=inject_overlap,
                vary_target_faces=True,
                panel_size=(128, 128),
            )
            assert len(blobs) == 2
            ev = evaluate_synccraft_completion(rec, rec["answer"])
            assert ev["passed"] is True, f"4-way UV failed for ({inject_alpha}, {inject_overlap}): {ev}"
            # Flipping `has_alpha_cutout` must fail evaluation!
            wrong_ans = rec["answer"].replace(
                f'"has_alpha_cutout":{str(inject_alpha).lower()}',
                f'"has_alpha_cutout":{str(not inject_alpha).lower()}',
            )
            assert evaluate_synccraft_completion(rec, wrong_ans)["passed"] is False

    # 2. ADR 0007 Sculpted Wyvern & Colossal Walking Citadel (`spin(...)` + `blend_with`)
    wyvern_ws = build_parametric_sculpted_wyvern_world(variant=1, subdivisions_per_unit=1)
    citadel_ws = build_parametric_walking_citadel_world(variant=1, subdivisions_per_unit=1)
    assert getattr(wyvern_ws.program.meta, "audit_profile", None) == "sculpted"
    assert getattr(citadel_ws.program.meta, "audit_profile", None) == "sculpted"

    mv_rec, mv_blobs = build_multiview_consistency_record(
        citadel_ws,
        asset_id="citadel_mv_01",
        family_id="citadel_fam",
        panel_size=(96, 96),
    )
    assert len(mv_blobs) == 4
    mv_eval = evaluate_synccraft_completion(mv_rec, mv_rec["answer"])
    assert mv_eval["passed"] is True
    assert mv_eval["audit_profile"] == "sculpted"
    assert mv_eval["sculpted_overlap_count"] >= 1

    # Anti-cheat: downgrading `audit_profile` from `sculpted` to `authored` MUST fail!
    parsed_citadel = parse_program(format_program(citadel_ws.program))
    cheated_prog = replace(parsed_citadel, meta=replace(parsed_citadel.meta, audit_profile="authored"))
    cheated_ans = f"<think>\nDowngrade profile.\n</think>\n```blockdsl\n{format_program(cheated_prog)}\n```"
    cheat_eval = evaluate_synccraft_completion(mv_rec, cheated_ans)
    assert cheat_eval["audit_profile_ok"] is False
    assert cheat_eval["passed"] is False

    # Spin-track animation record evaluation
    anim_rec, anim_blobs = build_animation_clip_record(
        world_state=citadel_ws,
        clip_name="fortress_march",
        asset_id="citadel_anim_01",
        family_id="citadel_fam",
        num_keyframes=4,
        panel_size=(96, 96),
    )
    assert len(anim_blobs) == 4
    anim_eval = evaluate_synccraft_completion(anim_rec, anim_rec["answer"])
    assert anim_eval["passed"] is True
    assert anim_eval["spin_tracks_ok"] is True

    # Artist-assistant record builder & evaluator
    prov = make_clean_room_provenance("bio_assist_01", title="Bio assist")
    assist_rec, _ = build_artist_assistant_record(
        bio_ws.program,
        mode="coarse_to_fine_upgrade",
        asset_id="bio_assist_01",
        family_id="bio_fam",
        provenance_override=prov,
    )
    assist_eval = evaluate_synccraft_completion(assist_rec, assist_rec["answer"])
    assert assist_eval["passed"] is True


def test_world_pipeline_bounded_repair_and_multi_engine_export(tmp_path):
    """Verify SyncCraftWorldPipeline PGD/model bounded repair (`<=3` re-audits, CostLedger) + multi-format exports."""
    from dataclasses import replace
    from colab.final.synccraft_bridge import build_animated_quadruped_program, ensure_synccraft_importable
    from colab.final.world_pipeline import SyncCraftWorldPipeline

    ensure_synccraft_importable()
    from synccraft.dsl.ast_nodes import Attach, Bone
    from synccraft.dsl.parser import format_program

    clean_prog = build_animated_quadruped_program(0)
    first_box = next(item for item in clean_prog.root.items if type(item).__name__ == "Box")
    # Attempt 0 intentionally emits an overlapping sibling box; Attempt 1 repairs it when prompted with SAT feedback
    colliding_bone = Bone(
        name="colliding_horn_stub",
        attach=Attach(first_box.name, anchor=(0.0, 0.2, 0.0), align=(0.0, -1.0, 0.0)),
        items=[replace(first_box, name="colliding_slab", place=None)],
    )
    broken_prog = replace(clean_prog, root=replace(clean_prog.root, items=[*clean_prog.root.items, colliding_bone]))

    calls = []

    def mock_vlm(prompt: str, images, attempt: int) -> str:
        calls.append((attempt, prompt))
        if attempt == 0:
            return f"<think>\nInitial draft.\n</think>\n```blockdsl\n{format_program(broken_prog)}\n```"
        assert "SAT collision" in prompt or "audit" in prompt.lower()
        return f"<think>\nRemoving colliding slab while preserving discrete sizes.\n</think>\n```blockdsl\n{format_program(clean_prog)}\n```"

    pipeline = SyncCraftWorldPipeline(mock_vlm)
    out_dir = tmp_path / "exported_quad"
    res = pipeline.generate_3d_asset(
        "Create an articulated quadruped creature with gait animation.",
        prompt_spec={"min_boxes": 5},
        asset_name="iron_hound",
        output_dir=out_dir,
        export_formats=("bbmodel", "bedrock", "geckolib", "gltf_pbr", "labpbr"),
        max_reaudits=3,
        seed=101,
    )
    assert res["passed"] is True
    assert 1 <= res["reaudits_used"] <= 3
    assert res["cost_ledger"]["initial_audit_calls"] == 1
    assert res["cost_ledger"]["reaudit_calls"] == res["reaudits_used"]
    assert res["cost_ledger"]["compiler_calls"] == 1 + res["reaudits_used"]

    bundle = res["bundle"]
    assert bundle is not None
    assert bundle["exports"]["bbmodel"]["roundtrip_box_count_match"] is True
    assert bundle["exports"]["bedrock"]["serialized_vertex_max_error"] <= 1e-6
    assert bundle["exports"]["geckolib"]["serialized_vertex_max_error"] <= 1e-6
    assert (out_dir / "iron_hound.bbmodel").is_file()
    assert (out_dir / "iron_hound.geo.json").is_file()
    assert (out_dir / "iron_hound.animation.json").is_file()
    assert (out_dir / "iron_hound_gltf" / "iron_hound.gltf").is_file()
    assert (out_dir / "iron_hound_labpbr_n.png").is_file()

    # Verify artist `.bbmodel` ingestion + critique + upgrade roundtrip
    bb_doc = bundle["exports"]["bbmodel"]["document"]
    upg = pipeline.critique_and_upgrade_asset(
        bb_doc,
        format_kind="bbmodel",
        use_model_first=False,
        asset_name="iron_hound_upgraded",
        output_dir=tmp_path / "upgraded_quad",
        export_formats=("bbmodel", "bedrock"),
        seed=101,
    )
    assert upg["license_policy"] == "local_workspace_unverified"
    assert upg["commercial_weights_eligible"] is False
    assert upg["result"]["passed"] is True


def test_dynamic_epoch_stratified_slicing_and_campaign_config(tmp_path):
    """Verify task-stratified cyclic per-epoch subsampling draws fresh slices per epoch & validates v3.2 config."""
    import sqlite3
    from colab.final.common import HERE, read, validate_config
    from colab.final.data import Catalogue, CommittedCursor

    cfg = validate_config(read(HERE / "campaign_rtx_pro_4000_world_v32.json"))
    assert cfg["dataset_mode"] == "hybrid_world_v32"
    assert cfg["cache_bytes_per_component"] == 4_000_000_000

    db_path = tmp_path / "slice_test.sqlite"
    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE packets(pid INTEGER PRIMARY KEY, component TEXT, shard TEXT, body TEXT, group_id TEXT);
        CREATE TABLE rows(rid TEXT PRIMARY KEY, pid INTEGER, variant INTEGER, unit INTEGER, family TEXT, language TEXT, task TEXT, split TEXT, group_id TEXT);
        """
    )
    pid = 1
    for task_name, count in (("factual_caption", 40), ("sat_3d_perspective", 40), ("synccraft_uv_alpha_diagnosis", 12)):
        for i in range(count):
            rid = f"{task_name}_{i:03d}"
            grp = f"grp_{task_name}_{i // 2:03d}"
            body = f'{{"component":"c1","record":{{"id":"{rid}","task":"{task_name}","language":"en","prompt":"p","answer":"a"}},"transport":{{"shard":"s0.tar","media":[]}}}}'
            db.execute("INSERT INTO packets VALUES(?,?,?,?,?)", (pid, "c1", "s0.tar", body, grp))
            db.execute("INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?)", (rid, pid, 0, 0, "image", "en", task_name, "train", grp))
            pid += 1
    db.commit()
    db.close()

    cat = Catalogue(db_path, modalities="vision_text")
    stage = {
        "name": "test_dyn_stage",
        "families": ["image"],
        "epochs": 2,
        "dynamic_epoch_records_per_family": {"image": 24},
    }
    order1 = cat.plan(stage, seed=39526)
    order2 = cat.plan(stage, seed=39526)
    assert len(order1) == 48
    assert order1 == order2  # Deterministic across checkpoint resumes!
    ep0 = set(order1[:24])
    ep1 = set(order1[24:])
    assert len(ep0) == 24 and len(ep1) == 24
    # Large pools (`factual_caption`, `sat_3d_perspective`) advance cyclically so Epoch 1 introduces fresh unseen rows!
    assert ep0 != ep1
    cursor = CommittedCursor(order1)
    assert CommittedCursor(order2, cursor.state()).committed == 0


def test_aesthetic_toy_rejection_and_pipeline_upgrade(tmp_path):
    """Verify Stage-4 9-box headless dog is caught as toy_blockout and upgraded to 27-box artist_grade."""
    from colab.final.synccraft_bridge import SENTINEL_HOUND_V32_DSL, build_animated_quadruped_program
    from colab.final.world_pipeline import SyncCraftWorldPipeline, evaluate_3d_asset_aesthetics
    from synccraft.dsl.parser import format_program

    # Variant 0 is the legacy 9-box headless quadruped; Variant 1 is the 27-box Sentinel Hound
    toy_prog = build_animated_quadruped_program(variant=0)
    toy_dsl = format_program(toy_prog)
    toy_aes = evaluate_3d_asset_aesthetics(toy_prog, prompt="armored sentinel hound quadruped")
    assert toy_aes["is_toy_like"] is True
    assert toy_aes["grade"] == "toy_blockout"
    assert toy_aes["landmarks"]["head_or_focal"] is False

    hound_aes = evaluate_3d_asset_aesthetics(SENTINEL_HOUND_V32_DSL, prompt="armored sentinel hound quadruped")
    assert hound_aes["static_audit_passed"] is True
    assert hound_aes["is_toy_like"] is False
    assert hound_aes["grade"] == "artist_grade"
    assert hound_aes["box_count"] == 27
    assert hound_aes["bone_count"] == 15
    assert hound_aes["aesthetic_score"] >= 0.85

    calls = []

    def _mock_model(prompt_txt, images, attempt):
        calls.append((prompt_txt, images is not None and len(images) == 4, attempt))
        if len(calls) == 1:
            return f"<think>\nQuick rough draft.\n</think>\n```blockdsl\n{toy_dsl}\n```", 120
        return f"<think>\nUpgrading with full head, neck, pelvis, tail, and paws.\n</think>\n```blockdsl\n{SENTINEL_HOUND_V32_DSL}\n```", 420

    pipe = SyncCraftWorldPipeline(_mock_model)
    res = pipe.generate_3d_asset(
        "Design an armored sentinel hound quadruped.",
        asset_name="upgraded_hound",
        output_dir=tmp_path / "aes_hound",
        export_formats=("bbmodel", "bedrock"),
        min_aesthetic_score=0.72,
    )
    assert res["passed"] is True
    assert res["aesthetic_upgraded"] is True
    assert res["boxes"] == 27
    assert res["bones"] == 15
    assert res["aesthetic_evaluation"]["grade"] == "artist_grade"
    # Second call received the rendered 4-view preview of the initial 9-box draft!
    assert len(calls) == 2 and calls[1][1] is True


def test_visual_aesthetic_judgment_record_and_oracle():
    """Verify 4-view `synccraft_visual_aesthetic_judgment` distinguishes toy blockouts from artist-grade rigs."""
    from colab.final.synccraft_bridge import (
        build_parametric_sentinel_hound_world,
        build_visual_aesthetic_judgment_record,
        evaluate_synccraft_completion,
    )

    ws = build_parametric_sentinel_hound_world(variant=0, subdivisions_per_unit=1)
    toy_row, toy_blobs = build_visual_aesthetic_judgment_record(
        ws,
        asset_id="hound_judge_test",
        family_id="sentinel_hound",
        inject_toy_coarsening=True,
    )
    art_row, art_blobs = build_visual_aesthetic_judgment_record(
        ws,
        asset_id="hound_judge_test",
        family_id="sentinel_hound",
        inject_toy_coarsening=False,
    )
    assert len(toy_blobs) == 4 and len(art_blobs) == 4
    assert toy_row["metadata"]["oracle_json"]["is_toy_like"] is True
    assert art_row["metadata"]["oracle_json"]["is_toy_like"] is False
    assert art_row["metadata"]["oracle_json"]["grade"] == "artist_grade"

    assert evaluate_synccraft_completion(toy_row, toy_row["answer"])["passed"] is True
    assert evaluate_synccraft_completion(art_row, art_row["answer"])["passed"] is True
    # Flipping toy prediction onto artist ground truth must fail!
    assert evaluate_synccraft_completion(art_row, toy_row["answer"])["passed"] is False


def test_anti_collapse_think_sanitizer_and_stage5_config():
    """Verify `<think>` anti-collapse bounds, repetition detector, Stage-5 task families, and Stage-5 JSON config."""
    from colab.final.common import HERE, read, validate_config
    from colab.final.data import family
    from colab.final.reasoning_agentic_corpus import (
        STAGE5_PURE_TEXT_TASKS,
        check_generation_collapse_health,
        sanitize_and_bound_think_trace,
        sanitize_workspace_paths,
    )

    cfg = validate_config(read(HERE / "campaign_rtx_pro_4000_stage5_rsi.json"))
    assert cfg["rsi_mode"] == "vision_self_improvement"
    assert cfg["vision_rsi_rounds"] == 2
    assert cfg["enable_stage5_reasoning_mix"] is True
    assert cfg["init_from_release_dir"] == "./work/spark-world-v32-24gb/final-model"

    for tname in STAGE5_PURE_TEXT_TASKS:
        assert family({"task": tname, "frames": [], "target": {"kind": "text", "text": "ok"}}) == "text"
    assert family({"task": "hle_multimodal_reasoning", "frames": ["img.png"], "target": {"kind": "text", "text": "ok"}}) == "image"

    assert sanitize_workspace_paths("File at /Users/alice_dev/repo/main.py and /home/bob/app.py") == "File at /workspace/repo/main.py and /workspace/app.py"

    clean = sanitize_and_bound_think_trace(
        "Inspect router segments and verify wildcard remainder capture.",
        "Implemented `Match(path)` in `router.go` and verified `go test ./...`.",
        enable_thinking=True,
    )
    assert clean is not None and clean.startswith("<think>\n") and "</think>\nImplemented" in clean

    # Unclosed `<think>` or circular repetition loop must be rejected!
    assert check_generation_collapse_health("<think>\nUnclosed monologue without end tag", expect_thinking=True)["healthy"] is False
    loop_str = " ".join(["Wait, let me double check the exact value again."] * 18)
    assert sanitize_and_bound_think_trace(loop_str, "Answer: 42", enable_thinking=True) is None


def test_vision_rsi_verifier_and_self_critique_no_leakage():
    """Verify `score_rsi_candidate` + `build_self_critique_turn` enforce aesthetics & never leak ground truth."""
    from colab.final.synccraft_bridge import (
        SENTINEL_HOUND_V32_DSL,
        build_animated_quadruped_program,
        make_clean_room_provenance,
    )
    from colab.final.vision_rsi import build_self_critique_turn, score_rsi_candidate
    from synccraft.dsl.parser import format_program

    secret_marker = "SUPER_SECRET_ORACLE_MARKER_DO_NOT_LEAK"
    row = {
        "id": "rsi_test_001",
        "task": "synccraft_artist_assistant",
        "language": "en",
        "enable_thinking": True,
        "frames": [],
        "prompt": "Design an armored sentinel hound quadruped.",
        "answer": f"<think>\n{secret_marker}\n</think>\n```blockdsl\n{SENTINEL_HOUND_V32_DSL.strip()}\n```",
        "target": {"kind": "text", "text": f"```blockdsl\n{SENTINEL_HOUND_V32_DSL.strip()}\n```"},
        "provenance": make_clean_room_provenance("rsi_test_001"),
        "metadata": {
            "assistant_mode": "create_full_asset",
            "expected_audit_profile": "strict",
        },
    }
    toy_cand = f"<think>\nSimple draft body and legs.\n</think>\n```blockdsl\n{format_program(build_animated_quadruped_program(0))}\n```"
    diag_toy = score_rsi_candidate(row, toy_cand, min_aesthetic_score=0.70)
    assert diag_toy["verified"] is False
    assert "toy_aesthetic_rejected" in diag_toy["reason"]

    turn = build_self_critique_turn(row, {"images": []}, toy_cand, diag_toy)
    assert turn is not None
    crit_row, crit_media = turn
    assert secret_marker not in crit_row["prompt"]
    assert len(crit_media["images"]) == 4

    rich_cand = f"<think>\nComplete 27-box hound with skull, muzzle, ears, pelvis, tail, and 3-part legs.\n</think>\n```blockdsl\n{SENTINEL_HOUND_V32_DSL.strip()}\n```"
    diag_rich = score_rsi_candidate(row, rich_cand, min_aesthetic_score=0.70)
    assert diag_rich["verified"] is True
    assert diag_rich["score"] >= 0.85


def test_adr002_rsi_collapse_scope_think_budget_and_neutral_grammar():
    """Verify ADR-002 instruction-scoped collapse gate, KV-cached `<think>` budget forcing, and neutral 3D grammar."""
    from colab.final.inference import empty_media
    from colab.final.vision_rsi import audit_promotion_generation_health
    from colab.final.world_pipeline import SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT, _build_repair_prompt

    # 1. Neutral grammar must NOT contain hound-specific identifiers (`pelvis_root` / `ribcage`)
    assert "pelvis_root" not in SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT
    assert "ribcage" not in SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT
    assert "primary_mat" in SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT

    # 2. Targeted `duplicate identifier` & `mirror_x` repair prompt guidance + Terminal-Bench 4.0 pin
    from colab.final.reasoning_agentic_corpus import TERMINAL_BENCH4_REPO_ID, TERMINAL_BENCH4_REVISION

    assert TERMINAL_BENCH4_REPO_ID == "harborframework/terminal-bench"
    assert len(TERMINAL_BENCH4_REVISION) == 40
    rep_p = _build_repair_prompt(
        "Upgrade hound",
        "@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16 }",
        {"error": "syntax_or_validation_error: duplicate identifier: chest_armor"},
        1,
    )
    assert "chest_armor_mat" in rep_p
    rep_mir = _build_repair_prompt(
        "Upgrade hound",
        "bone fl_upper attach(parent_box=trunk, anchor=[-1,0,0], align=[1,0,0]) mirror_x(fr_upper) {}\nbone fr_upper {}",
        {"error": "syntax_or_validation_error: duplicate identifier: fr_upper"},
        1,
    )
    assert "CRITICAL MIRROR_X FIX" in rep_mir

    # 3. `audit_promotion_generation_health` ignores raw Wikipedia `text_replay` loops while catching real instruction collapse
    mock_report = {
        "rows": [
            {"id": "w1", "task": "text_replay", "prediction": " ".join(["wiki continuation phrase"] * 30)},
            {"id": "v1", "task": "video_box_tracking", "prediction": '[{"timestamp_seconds": 0.0, "box_xyxy": [0,0,1,1], "visible": true}] ' * 8},
            {"id": "i1", "task": "gpqa_extended_science", "prediction": "<think>\nDerive orbital parity step by step.\n</think>\nAnswer: (B)"},
            {"id": "i2", "task": "factual_caption", "prediction": "A stone sculpture of two white birds sits beside a sunlit grassy hillside."},
            {"id": "i3", "task": "sat_3d_perspective", "prediction": "<think>\nCamera rotates right 30 degrees.\n</think>\nleft of the table"},
        ]
    }
    health = audit_promotion_generation_health(mock_report)
    assert health["generated_count"] == 5
    assert health["instruction_count"] == 3
    assert health["healthy_rate"] == 1.0
    assert health["think_close_rate"] == 1.0
    assert health["anti_collapse_passed"] is True

    # 4. KV-cached `max_think_tokens` budget forcing closes `</think>` cleanly
    model = make_synthetic_spark_native({"visual_cross_attn_blocks": 3}, seed=17)
    forced_out = model.answer(
        {"_family": "text", "language": "en", "task": "gpqa_extended_science", "prompt": "Solve orbital energy."},
        empty_media(),
        max_new_tokens=28,
        enable_thinking=True,
        max_think_tokens=6,
    )
    assert "<think>" in forced_out and "</think>" in forced_out



