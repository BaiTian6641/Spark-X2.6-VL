#!/usr/bin/env python3
"""Stage-10 Visual Projector RMS-Match Alignment Surgery (`53.06 -> 0.035` matching `Spark-X2.6-4B` `txt_rms = 0.025`)
+ High-Res 6-Mode Direct OCR / Chart / Table & Strictly `50/50 Yes/No` Visual Grounding
+ `Blockbench_sculpted_style-v28` Block-DSL v4.1 Grammar-Error Contrastive Self-Repair
+ Dual-Track Closed-Loop Verified **RSI-4** (`apollo_coat` on Blackwell `cuda:0` + `muon8bit` on Arc Pro B50 `xpu:0`).

Root-Cause Breakthrough Addressed:
- Live layerwise profiling of Stage-9 (`task-8018`) proved visual tokens entered Spark Layer 0 at
  `vis_rms = 53.0602` vs `txt_rms = 0.0245` (`2,165.8x` mismatch!) and stayed frozen (`53.06 -> 53.01`)
  across all 48 virtual layers while text tokens evolved `47x` (`0.0245 -> 1.1615`).
- Stage-10 activates `radio_output_rms_match=True` (`RMSNorm` + learnable `out_scale=0.035` + normalized
  `meta_scale=0.008`) on `RadioMultiLayerSwiGLUProjector` while warm-starting 100% of Stage-9 weights:
  1. **Phase 10A (`750` updates, `frozen_spark_cross_attn`, `C-RADIOv4-H` frozen on B50)**: Fast 100%-visual
     connector alignment onto Spark's exact `0.025-0.035` residual manifold (`0%` text/frontier forgetting!).
  2. **Phase 10B (`450` updates, `full_spark` `apollo_coat` on Blackwell + top-6 `muon8bit` on B50)**: Joint
     high-res OCR/Chart/Table + `50/50 Yes/No` + 3D 4-Primitives + `v28` Block-DSL v4.1 grammar repair + Frontier.
  3. **Phase 10C (`140` updates, Dual-Track Verified RSI-4 + `3x` `En1B` LR + 5-axis promotion/rollback gate)**:
     Consolidates verified `v28` compiler-repaired DSL + `Terminal-Pivot`/`DAPO-Math`/STEM breakthroughs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.aesthetic_v28_biomech_corpus import build_aesthetic_v28_biomech_records  # noqa: E402
from colab.final.common import emit, read, write  # noqa: E402
from colab.final.data import Catalogue, MediaReader  # noqa: E402
from colab.final.frontier_stage9_corpus import (  # noqa: E402
    build_dapo_math_rsi3_pool,
    build_stage9_frontier_training_records,
    build_synthetic_scene_ocr_pairs,
    build_v28_grammar_repair_records,
    select_stage9_direct_ocr_and_yesno_rids,
)
from colab.final.inference import visual_media  # noqa: E402
from colab.final.model import SparkNative, load_spark_native_safetensors  # noqa: E402
from colab.final.spatial_3d_primitives_corpus import (  # noqa: E402
    build_nemotron_v3_spatial_records,
    build_shaded_3d_spatial_primitives_records,
)
from scripts.run_stage75_visual_alignment_and_aesthetic_polish import (  # noqa: E402
    _select_eval_image_rids,
    measure_visual_alignment_diagnostics,
    select_stage75a_pixel_indispensable_rids,
    select_stage75b_balanced_rids,
    v25_record_to_example,
)
from scripts.run_stage8_cradiov4_deepstack_3d_spatial import (  # noqa: E402
    _export_stage8_releases,
    _extract_stage8_gate_stats,
    _run_stage8_phase,
)
from scripts.run_stage9_frontier_v28_and_rsi3 import (  # noqa: E402
    _fetch_tb41_excluded_task_ids,
    _text_record_to_pair,
    audit_stage9_records_strict,
    evaluate_rsi3_promotion_gate,
    mine_rsi3_dual_track_trajectories,
)


@torch.no_grad()
def measure_layerwise_visual_text_rms(
    model: SparkNative,
    sample_image: Image.Image | None = None,
    *,
    tag: str = "rms_audit",
) -> dict[str, Any]:
    """Measure exact Layer-0..35 visual-vs-text hidden RMS inside Spark-X2.6-4B to verify residual-stream alignment."""
    was_train = model.training
    model.eval()
    if sample_image is None:
        sample_image = Image.new("RGB", (384, 384), (72, 132, 198))
    row = {
        "_family": "image",
        "language": "en",
        "task": "document_ocr",
        "prompt": "Read the visible text and describe the key object.",
        "enable_thinking": False,
    }
    med = visual_media([sample_image])
    embeds, vis_ctx, pref_tids, pref_tmask = model.prefix_with_visual_context_and_ids(row, med, for_generation=True)
    vis_mask = ~pref_tmask[0]
    txt_mask = pref_tmask[0]
    v0 = float(embeds[0, vis_mask].detach().float().pow(2).mean().sqrt().item()) if bool(vis_mask.any()) else 0.0
    t0 = float(embeds[0, txt_mask].detach().float().pow(2).mean().sqrt().item()) if bool(txt_mask.any()) else 1e-6

    layers = getattr(getattr(model, "spark", None), "model", None)
    layer_list = getattr(layers, "layers", []) if layers is not None else []
    probe_indices = [i for i in (0, 5, 11, 23, 35) if i < len(layer_list)]
    layer_stats: dict[str, dict[str, float]] = {}
    hooks = []

    for idx in probe_indices:
        def _make_hook(li: int):
            def _hk(_mod: Any, _inp: Any, out: Any) -> None:
                key = f"layer_{li}"
                if key in layer_stats:
                    return
                h = out[0] if isinstance(out, tuple) else out
                vr = float(h[0, vis_mask].detach().float().pow(2).mean().sqrt().item()) if bool(vis_mask.any()) else 0.0
                tr = float(h[0, txt_mask].detach().float().pow(2).mean().sqrt().item()) if bool(txt_mask.any()) else 1e-6
                layer_stats[key] = {
                    "vis_rms": round(vr, 4),
                    "txt_rms": round(tr, 4),
                    "ratio": round(vr / max(1e-6, tr), 2),
                }
            return _hk
        hooks.append(layer_list[idx].register_forward_hook(_make_hook(idx)))

    try:
        use_amp = torch.cuda.is_available() and str(model.device).startswith("cuda")
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = model.forward_looped_model(
                    inputs_embeds=embeds,
                    token_ids_for_engram=pref_tids,
                    text_token_mask=pref_tmask,
                    visual_context=vis_ctx,
                )
        else:
            _ = model.forward_looped_model(
                inputs_embeds=embeds,
                token_ids_for_engram=pref_tids,
                text_token_mask=pref_tmask,
                visual_context=vis_ctx,
            )
    finally:
        for hk in hooks:
            hk.remove()
        if was_train:
            model.train()

    out = {
        "tag": tag,
        "vis_tokens": int(vis_mask.sum().item()),
        "input_vis_rms": round(v0, 4),
        "input_txt_rms": round(t0, 4),
        "input_ratio": round(v0 / max(1e-6, t0), 2),
        "layers": layer_stats,
    }
    emit("stage10_layerwise_rms_audit", **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--init-release-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage9-frontier-rsi3/final-model-stage9-rsi3-bf16"),
    )
    ap.add_argument(
        "--radio-vision-dir",
        type=Path,
        default=Path("cache/c-radiov4-h"),
    )
    ap.add_argument(
        "--stage7-work-root",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50"),
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage10-rms-v28-rsi4"),
    )
    ap.add_argument(
        "--nemotron-cache-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage8-radiov4/nemotron_v3_cache"),
    )
    ap.add_argument("--stage10a-updates", type=int, default=750)
    ap.add_argument("--stage10b-updates", type=int, default=450)
    ap.add_argument("--rsi4-updates", type=int, default=140)
    ap.add_argument("--grad-accum", type=int, default=12)
    ap.add_argument("--stage10a-connector-lr", type=float, default=4.0e-4)
    ap.add_argument("--stage10b-backbone-lr", type=float, default=1.5e-5)
    ap.add_argument("--stage10b-connector-lr", type=float, default=6.0e-5)
    ap.add_argument("--stage10b-engram-lr", type=float, default=4.5e-5)
    ap.add_argument("--stage10b-vision-lr", type=float, default=2.0e-6)
    ap.add_argument("--unfreeze-vision-layers", type=int, default=6)
    ap.add_argument("--out-scale-init", type=float, default=0.035)
    ap.add_argument("--meta-scale-init", type=float, default=0.008)
    ap.add_argument("--vision-optimizer", type=str, default="muon8bit", choices=("muon8bit", "apollo8bit", "adamw"))
    ap.add_argument("--intel-vision", type=str, default="xpu", choices=("xpu", "none"))
    ap.add_argument("--synth-ocr-max", type=int, default=600)
    ap.add_argument("--v28-max-records", type=int, default=96)
    ap.add_argument("--v28-repair-max", type=int, default=96)
    ap.add_argument("--nemotron-max-gqa", type=int, default=900)
    ap.add_argument("--nemotron-max-geo", type=int, default=600)
    ap.add_argument("--nemotron-max-aokvqa", type=int, default=600)
    ap.add_argument("--nemotron-max-clevr", type=int, default=600)
    ap.add_argument("--nemotron-max-zhwiki", type=int, default=300)
    ap.add_argument("--nemotron-max-plotqa", type=int, default=300)
    ap.add_argument("--max-nemotron-terminal", type=int, default=1200)
    ap.add_argument("--max-ot-agent", type=int, default=800)
    ap.add_argument("--max-terminal-pivot", type=int, default=600)
    ap.add_argument("--max-ot3-stem", type=int, default=1200)
    ap.add_argument("--max-nemotron-science", type=int, default=900)
    ap.add_argument("--rsi4-v28-explore", type=int, default=16)
    ap.add_argument("--rsi4-frontier-explore", type=int, default=36)
    ap.add_argument("--eval-images", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261151)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    db_path = args.stage7_work_root / "metadata/corpus.sqlite"
    handoff_path = args.stage7_work_root / "metadata/HANDOFF.json"

    # 1. Build & Strictly Audit Stage-10 Visual + OCR/Chart/Table + v28 Grammar Repair + Frontier Corpora
    tb41_excluded = _fetch_tb41_excluded_task_ids(hf_token)
    emit("stage10_build_corpora_start", excluded_tb41_tasks=len(tb41_excluded))

    v28_records = build_aesthetic_v28_biomech_records(seed=args.seed, max_records=args.v28_max_records)
    v28_pixel_only = [r for r in v28_records if (r.get("metadata") or {}).get("pixel_indispensable") and r.get("images")]
    v28_repair_records = build_v28_grammar_repair_records(v28_records, seed=args.seed + 5, max_items=args.v28_repair_max)
    spatial3d_pairs = build_shaded_3d_spatial_primitives_records(seed=args.seed + 11, panel_size=(384, 384), angles_per_asset=6)
    synth_ocr_pairs = build_synthetic_scene_ocr_pairs(
        seed=args.seed + 17, max_items=args.synth_ocr_max, include_charts_and_tables=True
    )
    nemotron_pairs = build_nemotron_v3_spatial_records(
        cache_dir=args.nemotron_cache_dir,
        hf_token=hf_token,
        max_gqa=args.nemotron_max_gqa,
        max_geometry3k=args.nemotron_max_geo,
        max_aokvqa=args.nemotron_max_aokvqa,
        max_clevr=args.nemotron_max_clevr,
        max_zhwiki=args.nemotron_max_zhwiki,
        max_plotqa=args.nemotron_max_plotqa,
        seed=args.seed + 29,
    )
    frontier_records = build_stage9_frontier_training_records(
        token=hf_token,
        max_nemotron_terminal=args.max_nemotron_terminal,
        max_openthoughts_agent=args.max_ot_agent,
        max_terminal_pivot=args.max_terminal_pivot,
        max_openthoughts3_stem=args.max_ot3_stem,
        max_nemotron_science=args.max_nemotron_science,
        seed=args.seed + 37,
        excluded_tb_task_ids=tb41_excluded,
    )

    all_text_recs = list(frontier_records) + list(v28_repair_records)
    all_visual_pairs = list(spatial3d_pairs) + list(synth_ocr_pairs) + list(nemotron_pairs)
    audit_rep = audit_stage9_records_strict(
        all_text_recs, v28_records, all_visual_pairs, excluded_tb41_ids=tb41_excluded
    )
    write(args.output_root / "STAGE10_DATASET_AUDIT.json", audit_rep)
    emit(
        "stage10_dataset_audit_done",
        passed=audit_rep["passed"],
        failures=audit_rep["failures_count"],
        total=audit_rep["total_checked"],
        v28_repair=len(v28_repair_records),
        synth_ocr=len(synth_ocr_pairs),
        nemotron_v3=len(nemotron_pairs),
    )
    if not audit_rep["passed"]:
        raise RuntimeError(f"Stage-10 pre-training dataset quality gate FAILED: {audit_rep['failures'][:5]}")

    rng = random.Random(args.seed)

    # 2A. Phase 10A Mix (100% Visual RMS-Matched Projector Alignment, `9,000` samples):
    #     ~28% Direct OCR (Cached + 600 6-mode Synthetic OCR/Chart/Table)
    #     + ~18% Strictly 50/50 Yes/No Direct Existence (via select_stage9_direct_ocr_and_yesno_rids)
    #     + ~22% Shaded 3D 4-Primitives & Nemotron-v3 Spatial/Geometry
    #     + ~12% v28 Pixel-Indispensable Visual
    #     + ~20% Broad Stage-7 Pixel-Indispensable Visual Replay
    total_10a = args.stage10a_updates * args.grad_accum
    n10a_direct_cat = int(total_10a * 0.34)  # ~40% of this = strictly 50/50 Yes/No, ~60% = cached TextVQA/DocVQA/ChartQA/LaTeX/RegionOCR
    n10a_synth_ocr = min(int(total_10a * 0.12), len(synth_ocr_pairs) * 2)
    n10a_s3d = min(int(total_10a * 0.10), len(spatial3d_pairs) * 2)
    n10a_nem = min(int(total_10a * 0.14), len(nemotron_pairs))
    n10a_v28 = min(int(total_10a * 0.10), max(0, len(v28_pixel_only) * 16))
    n10a_cat_vis = max(1, total_10a - n10a_direct_cat - n10a_synth_ocr - n10a_s3d - n10a_nem - n10a_v28)

    direct_10a_rids, direct_10a_summary = select_stage9_direct_ocr_and_yesno_rids(
        db_path, target_samples=n10a_direct_cat, seed=args.seed + 101
    )
    cat_10a_rids, cat_10a_summary = select_stage75a_pixel_indispensable_rids(
        db_path, target_catalogue_samples=n10a_cat_vis, seed=args.seed + 102
    )
    order_10a: list[tuple[str, Any]] = [("cat_direct", rid) for rid in direct_10a_rids]
    order_10a.extend(("cat", rid) for rid in cat_10a_rids)
    for i in range(n10a_synth_ocr):
        order_10a.append(("pair", synth_ocr_pairs[i % len(synth_ocr_pairs)]))
    for i in range(n10a_s3d):
        order_10a.append(("pair", spatial3d_pairs[i % len(spatial3d_pairs)]))
    for i in range(n10a_nem):
        order_10a.append(("pair", nemotron_pairs[i % len(nemotron_pairs)]))
    for i in range(n10a_v28):
        order_10a.append(("v25", v28_pixel_only[i % len(v28_pixel_only)]))
    rng.shuffle(order_10a)
    order_10a = order_10a[:total_10a]

    # 2B. Phase 10B Mix (Full Joint Fine-Tuning, `5,400` samples):
    #     ~26% Direct OCR + 50/50 Yes/No + ~20% Shaded 3D & Nemotron-v3 + ~18% v28 + Grammar Repair
    #     + ~22% 2026 Frontier STEM/Terminal + ~14% Broad Visual/Text Replay
    total_10b = args.stage10b_updates * args.grad_accum
    n10b_direct_cat = int(total_10b * 0.18)
    n10b_synth_ocr = min(int(total_10b * 0.08), len(synth_ocr_pairs))
    n10b_s3d = min(int(total_10b * 0.10), len(spatial3d_pairs))
    n10b_nem = min(int(total_10b * 0.10), len(nemotron_pairs))
    n10b_v28 = min(int(total_10b * 0.11), len(v28_records) * 8)
    n10b_v28_rep = min(int(total_10b * 0.07), max(0, len(v28_repair_records) * 8))
    n10b_front = min(int(total_10b * 0.22), len(frontier_records))
    n10b_cat = max(1, total_10b - n10b_direct_cat - n10b_synth_ocr - n10b_s3d - n10b_nem - n10b_v28 - n10b_v28_rep - n10b_front)

    direct_10b_rids, direct_10b_summary = select_stage9_direct_ocr_and_yesno_rids(
        db_path, target_samples=n10b_direct_cat, seed=args.seed + 201
    )
    cat_10b_rids, cat_10b_summary = select_stage75b_balanced_rids(
        db_path, target_catalogue_samples=n10b_cat, vision_fraction=0.65, seed=args.seed + 202
    )
    order_10b: list[tuple[str, Any]] = [("cat_direct", rid) for rid in direct_10b_rids]
    order_10b.extend(("cat", rid) for rid in cat_10b_rids)
    for i in range(n10b_synth_ocr):
        order_10b.append(("pair", synth_ocr_pairs[i % len(synth_ocr_pairs)]))
    for i in range(n10b_s3d):
        order_10b.append(("pair", spatial3d_pairs[i % len(spatial3d_pairs)]))
    for i in range(n10b_nem):
        order_10b.append(("pair", nemotron_pairs[i % len(nemotron_pairs)]))
    for i in range(n10b_v28):
        order_10b.append(("v25", v28_records[i % len(v28_records)]))
    for i in range(n10b_v28_rep):
        order_10b.append(("pair", _text_record_to_pair(v28_repair_records[i % len(v28_repair_records)])))
    for i in range(n10b_front):
        order_10b.append(("pair", _text_record_to_pair(frontier_records[i % len(frontier_records)])))
    rng.shuffle(order_10b)
    order_10b = order_10b[:total_10b]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "audit": audit_rep,
                    "order_10a": len(order_10a),
                    "order_10b": len(order_10b),
                    "breakdown_10a": {
                        "direct_ocr_yesno_cat": len(direct_10a_rids),
                        "synth_ocr_6mode": n10a_synth_ocr,
                        "spatial3d": n10a_s3d,
                        "nemotron_v3": n10a_nem,
                        "v28_visual": n10a_v28,
                        "cat_visual_replay": len(cat_10a_rids),
                    },
                    "breakdown_10b": {
                        "direct_ocr_yesno_cat": len(direct_10b_rids),
                        "synth_ocr_6mode": n10b_synth_ocr,
                        "spatial3d": n10b_s3d,
                        "nemotron_v3": n10b_nem,
                        "v28_all": n10b_v28,
                        "v28_grammar_repair": n10b_v28_rep,
                        "frontier": n10b_front,
                        "cat_replay": len(cat_10b_rids),
                    },
                    "direct_10a_summary": direct_10a_summary,
                },
                indent=2,
            )
        )
        return

    # 3. Warm-Start Stage-9 Release with `radio_output_rms_match=True` (`out_scale=0.035`, `meta_scale=0.008`)!
    src_cfg = read(args.init_release_dir / "config.json")
    cfg = dict(src_cfg)
    cfg.update(
        {
            "campaign_id": "spark-x26-4b-loop48-en1b-vl-stage10-rms-v28-rsi4",
            "init_from_release_dir": str(args.init_release_dir.resolve()),
            "allow_warmstart_vision_swap": False,
            "allow_warmstart_missing_loop_engram": False,
            "allow_warmstart_projector_rms_match": True,
            "radio_output_rms_match": True,
            "radio_out_scale_init": float(args.out_scale_init),
            "radio_meta_scale_init": float(args.meta_scale_init),
            "intel_arc_vision_offload": (args.intel_vision == "xpu"),
            "intel_arc_openvino_eval": False,
            "intel_arc_vision_optimizer": str(args.vision_optimizer),
            "vision_dtype": "bfloat16",
            "spark_dtype": "bfloat16",
        }
    )
    paths = {
        "spark": args.init_release_dir / "reload-assets/spark",
        "vision": args.radio_vision_dir.resolve(),
    }
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SparkNative(paths, cfg, device=device, pretrained=True)
    model.gradient_checkpointing_enable()

    catalogue = Catalogue(db_path)
    handoff = read(handoff_path)
    reader = MediaReader(args.stage7_work_root, handoff, cfg, token=hf_token, reserve=0)
    eval_rids = _select_eval_image_rids(db_path, max_items=args.eval_images)
    probe_ex = v25_record_to_example(v28_records[0]) if v28_records else None
    sample_im = v28_pixel_only[0]["images"][0] if v28_pixel_only else None

    try:
        rms_pre = measure_layerwise_visual_text_rms(model, sample_im, tag="pre_stage10a")
        cal_pre = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="pre_stage10a"
        )
        emit("stage10_pre_calibration", **cal_pre)

        # Phase 10A: Frozen Spark + Frozen C-RADIOv4-H -> High-Speed RMS-Matched Projector & Connector Alignment
        stage10a_cfg = {
            "stage_mode": "frozen_spark_cross_attn",
            "unfreeze_vision_last_n_layers": 0,
            "optimizer_profile": "bf16_adamw",
            "connector_learning_rate": args.stage10a_connector_lr,
            "adapter_lr": args.stage10a_connector_lr,
            "learning_rate": args.stage10a_connector_lr,
            "warmup_ratio": 0.08,
            "clip_norm": 1.0,
            "weight_decay": 0.01,
        }
        res_10a = _run_stage8_phase(
            stage_name="stage10a_visual_rms_match_alignment",
            model=model,
            base_cfg=cfg,
            stage_cfg=stage10a_cfg,
            order_items=order_10a,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage10a",
            seed=args.seed + 300,
        )
        rms_post_10a = measure_layerwise_visual_text_rms(model, sample_im, tag="post_stage10a")
        cal_post_10a = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_stage10a"
        )
        emit("stage10_post_10a_calibration", **cal_post_10a)

        # Phase 10B: Full Joint Spark-4B + Loop48 + En1B + RMS-Matched Projector + Top-6 C-RADIOv4-H (`muon8bit` on B50)
        stage10b_cfg = {
            "stage_mode": "full_spark",
            "unfreeze_vision_last_n_layers": int(args.unfreeze_vision_layers),
            "optimizer_profile": "apollo_coat",
            "learning_rate": args.stage10b_backbone_lr,
            "backbone_lr": args.stage10b_backbone_lr,
            "apollo_learning_rate": args.stage10b_backbone_lr,
            "connector_learning_rate": args.stage10b_connector_lr,
            "adapter_lr": args.stage10b_connector_lr,
            "apollo_connector_learning_rate": args.stage10b_connector_lr,
            "engram_lr": args.stage10b_engram_lr,
            "vision_lr": args.stage10b_vision_lr,
            "warmup_ratio": 0.10,
            "clip_norm": 0.8,
            "weight_decay": 0.01,
        }
        res_10b = _run_stage8_phase(
            stage_name="stage10b_full_joint_rms_ocr_v28_frontier",
            model=model,
            base_cfg=cfg,
            stage_cfg=stage10b_cfg,
            order_items=order_10b,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage10b",
            seed=args.seed + 400,
        )
        rms_post_10b = measure_layerwise_visual_text_rms(model, sample_im, tag="post_stage10b")
        cal_post_10b = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_stage10b"
        )
        emit("stage10_post_10b_calibration", **cal_post_10b)

        # Phase 10C: Dual-Track Verified RSI-4 (`v28` Compiler-Repair + Terminal/DAPO-Math/STEM + 3x `En1B` LR)
        dapo_math_pool = build_dapo_math_rsi3_pool(token=hf_token, max_items=120, seed=args.seed + 499)
        rsi4_signal_items, rsi4_mining_summary = mine_rsi3_dual_track_trajectories(
            model,
            v28_records=v28_records,
            frontier_records=list(frontier_records) + list(dapo_math_pool),
            v28_explore_n=args.rsi4_v28_explore,
            frontier_explore_n=args.rsi4_frontier_explore,
            candidates_k=2,
            seed=args.seed + 500,
        )
        total_rsi4 = args.rsi4_updates * args.grad_accum
        n_sig = min(total_rsi4 // 2, max(1, len(rsi4_signal_items) * 6))
        n_rsi_direct = min(len(direct_10b_rids), int(total_rsi4 * 0.20))
        n_rsi_rep = min(len(v28_repair_records) * 2, int(total_rsi4 * 0.10))
        n_rep_cat = max(1, total_rsi4 - n_sig - n_rsi_direct - n_rsi_rep)
        cat_rsi_rids, _ = select_stage75b_balanced_rids(
            db_path, target_catalogue_samples=n_rep_cat, vision_fraction=0.65, seed=args.seed + 501
        )
        order_rsi4: list[tuple[str, Any]] = [("cat", rid) for rid in cat_rsi_rids]
        for i in range(n_rsi_direct):
            order_rsi4.append(("cat_direct", direct_10b_rids[i % len(direct_10b_rids)]))
        for i in range(n_rsi_rep):
            order_rsi4.append(("pair", _text_record_to_pair(v28_repair_records[i % len(v28_repair_records)])))
        for i in range(n_sig):
            order_rsi4.append(rsi4_signal_items[i % len(rsi4_signal_items)])
        rng.shuffle(order_rsi4)
        order_rsi4 = order_rsi4[:total_rsi4]

        rsi4_bb_lr = min(args.stage10b_backbone_lr * 0.45, 7.0e-6)
        rsi4_cfg = {
            "stage_mode": "full_spark",
            "unfreeze_vision_last_n_layers": 0,
            "optimizer_profile": "apollo_coat",
            "learning_rate": rsi4_bb_lr,
            "backbone_lr": rsi4_bb_lr,
            "apollo_learning_rate": rsi4_bb_lr,
            "connector_learning_rate": min(args.stage10b_connector_lr * 0.45, 2.5e-5),
            "adapter_lr": min(args.stage10b_connector_lr * 0.45, 2.5e-5),
            "apollo_connector_learning_rate": min(args.stage10b_connector_lr * 0.45, 2.5e-5),
            "engram_lr": max(min(args.stage10b_engram_lr * 0.65, 3.0e-5), rsi4_bb_lr * 3.0),
            "vision_lr": 0.0,
            "warmup_ratio": 0.12,
            "clip_norm": 0.6,
            "weight_decay": 0.01,
        }
        res_rsi4 = _run_stage8_phase(
            stage_name="stage10c_dual_track_rsi4_candidate",
            model=model,
            base_cfg=cfg,
            stage_cfg=rsi4_cfg,
            order_items=order_rsi4,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage10c-rsi4",
            seed=args.seed + 600,
        )
        rms_post_rsi4 = measure_layerwise_visual_text_rms(model, sample_im, tag="post_rsi4")
        cal_post_rsi4 = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_rsi4"
        )
        promo_gate = evaluate_rsi3_promotion_gate(cal_post_10b, cal_post_rsi4)
        emit("stage10_rsi4_promotion_gate", **promo_gate)

        selected_phase = "stage10c_rsi4"
        if not promo_gate["accepted"]:
            emit("stage10_rsi4_rolled_back_to_stage10b")
            load_spark_native_safetensors(model, args.output_root / "ckpt-stage10b/model.safetensors")
            if getattr(model, "_intel_vision_bridge", None) is not None:
                model._intel_vision_bridge.sync_from_host(model.vision)
            selected_phase = "stage10b_parent_rollback"

        # Clear warm-start-only flag before exporting standalone portable releases
        model.policy["allow_warmstart_projector_rms_match"] = False
        report = {
            "stage10_architecture": {
                "warmstart": str(args.init_release_dir),
                "vision_backbone": "nvidia/C-RADIOv4-H (L24+L32 RMSNorm SwiGLU DeepStack + Exit RMSNorm Match)",
                "radio_output_rms_match": True,
                "out_scale_init": args.out_scale_init,
                "meta_scale_init": args.meta_scale_init,
                "vision_optimizer_b50": args.vision_optimizer,
                "selected_phase": selected_phase,
            },
            "dataset_audit": audit_rep,
            "direct_10a_summary": direct_10a_summary,
            "direct_10b_summary": direct_10b_summary,
            "stage10a_cat_summary": cat_10a_summary,
            "stage10b_cat_summary": cat_10b_summary,
            "layerwise_rms": {
                "pre_10a": rms_pre,
                "post_10a": rms_post_10a,
                "post_10b": rms_post_10b,
                "post_rsi4": rms_post_rsi4,
            },
            "calibration": {
                "pre_10a": cal_pre,
                "post_10a": cal_post_10a,
                "post_10b": cal_post_10b,
                "post_rsi4": cal_post_rsi4,
            },
            "stage10a": res_10a,
            "stage10b": res_10b,
            "rsi4_mining": rsi4_mining_summary,
            "stage10c_rsi4": res_rsi4,
            "rsi4_promotion_gate": promo_gate,
            "final_gates": _extract_stage8_gate_stats(model),
        }
        write(args.output_root / "STAGE10_RMS_RSI4_REPORT.json", report)

        out_bf16 = args.output_root / "final-model-stage10-rms-rsi4-bf16"
        out_fp8 = args.output_root / "final-model-stage10-rms-rsi4-fp8"
        _export_stage8_releases(
            model,
            src_release_dir=args.init_release_dir,
            radio_dir=args.radio_vision_dir,
            out_bf16_dir=out_bf16,
            out_fp8_dir=out_fp8,
            report=report,
        )
        write(out_bf16 / "STAGE10_RMS_RSI4_REPORT.json", report)
        write(out_fp8 / "STAGE10_RMS_RSI4_REPORT.json", report)
    finally:
        model.close_coprocessors()


if __name__ == "__main__":
    main()
