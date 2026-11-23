#!/usr/bin/env python3
"""Stage-11 Visual-Attention Awakening + Differentiable Projector Anti-Collapse Decorrelation (`L_decorr`)
+ Real-vs-Blank Visual-Dependency Hard-Mining + Split Early-Layer Visual LR (`layers 0..11` @ `2.2e-5` vs `12..35` @ `7.5e-6`)
+ Real-Photo `50/50` GQA POPE Grounding + 7-Mode Vivid OCR + `(A)/(B)` 3D Spatial MCQ
+ SyncCraft `6031e58` (`v38.8`) 64-Archetype Verified Corpus + Modular `--extra-jsonl` Hook + Verified RSI-5.

Addresses the exact Stage-10.5 X-Ray bottlenecks (`STAGE105_CLEAN_PROJ_OCR_REPORT.json`) while protecting
Stage-10.5's 6 all-time highs (`AI2D 60.83%`, `HLE 7.50%`, `MMMU-Pro 25.00%`, `MMStar 29.44%`, `SimpleQA 4.00%`, `Gate D 100%`):
1. Differentiable Projector Spatial-Decorrelation Loss (`L_decorr`): penalizes `dc_ratio > 0.65` and off-diagonal
   spatial patch cosine `> 0.35`, preventing `vision_projector` from collapsing `C-RADIOv4-H` into a `0.93`-cosine DC stub.
2. Visual-Dependency Hard-Mining (`visual_margin_every_n=4`): compares `loss_real` vs detached `no_grad` `loss_blank`
   on pixel-indispensable visual microbatches, concentrating up to `2.25x` gradient weight on examples where Spark still ignores the image.
3. Split Layerwise LR (`early_spark_layers=12` @ `2.2e-5` vs deep reasoning `layers 12..35` @ `7.5e-6`): awakens
   early visual-reading attention heads without disturbing deep STEM/Terminal/Math representations.
4. Scales real OCR (`3,600` `DocVQA` + `3,000` `SynthDoG-EN` + `1,200` 7-mode vivid synth OCR including giant posters),
   adds `1,400` strictly `50/50 Yes/No` real-photo GQA POPE pairs, adds `480` explicit `(A)/(B)` 4-primitive 3D MCQs,
   and adds `64` live-verified SyncCraft `6031e58` (`v38.8`) archetypes (`sculpt_group`, `inset_carve`, `uv_scale`, `bio_joint`,
   `swim`, `flight`) + `140` 14-mutator `v38.8` compiler-repair pairs + modular `--extra-jsonl` / `--extra-dataset-dir` hook.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.aesthetic_v28_biomech_corpus import build_aesthetic_v28_biomech_records  # noqa: E402
from colab.final.common import emit, read, write  # noqa: E402
from colab.final.data import Catalogue, MediaReader  # noqa: E402
from colab.final.frontier_stage9_corpus import (  # noqa: E402
    build_3d_spatial_mcq_pairs,
    build_dapo_math_rsi3_pool,
    build_hf_real_ocr_pairs,
    build_real_photo_pope_yesno_pairs,
    build_stage9_frontier_training_records,
    build_synthetic_scene_ocr_pairs,
    build_v28_grammar_repair_records,
    load_extra_user_dataset_records,
    select_stage9_direct_ocr_and_yesno_rids,
)
from colab.final.model import SparkNative, load_spark_native_safetensors  # noqa: E402
from colab.final.reasoning_agentic_corpus import check_generation_collapse_health  # noqa: E402
from colab.final.spatial_3d_primitives_corpus import (  # noqa: E402
    build_nemotron_v3_spatial_records,
    build_shaded_3d_spatial_primitives_records,
)
from colab.final.v38_sculpted_corpus import build_v38_64_archetype_corpus  # noqa: E402
from scripts.run_stage75_visual_alignment_and_aesthetic_polish import (  # noqa: E402
    _select_eval_image_rids,
    measure_visual_alignment_diagnostics,
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
from scripts.run_stage10_visual_rms_surgery_and_v28_rsi4 import (  # noqa: E402
    measure_layerwise_visual_text_rms,
)
from scripts.run_stage105_clean_projector_ocr_v28_rsi import (  # noqa: E402
    measure_visual_grounding_xray,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--init-release-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage105-clean-proj-ocr-v28/final-model-stage105-bf16"),
    )
    ap.add_argument("--radio-vision-dir", type=Path, default=Path("cache/c-radiov4-h"))
    ap.add_argument("--stage7-work-root", type=Path, default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50"))
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage11-visual-awakening-v38"),
    )
    ap.add_argument(
        "--nemotron-cache-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage8-radiov4/nemotron_v3_cache"),
    )
    ap.add_argument("--stage11a-updates", type=int, default=960)
    ap.add_argument("--rsi5-updates", type=int, default=140)
    ap.add_argument("--grad-accum", type=int, default=12)
    ap.add_argument("--early-spark-layers", type=int, default=12)
    ap.add_argument("--early-spark-lr", type=float, default=2.2e-5)
    ap.add_argument("--backbone-lr", type=float, default=7.5e-6)
    ap.add_argument("--connector-lr", type=float, default=8.0e-5)
    ap.add_argument("--engram-lr", type=float, default=2.8e-5)
    ap.add_argument("--vision-lr", type=float, default=3.0e-6)
    ap.add_argument("--unfreeze-vision-layers", type=int, default=6)
    ap.add_argument("--decorr-weight", type=float, default=0.15)
    ap.add_argument("--decorr-target-dc", type=float, default=0.65)
    ap.add_argument("--decorr-target-cos", type=float, default=0.35)
    ap.add_argument("--visual-margin-every-n", type=int, default=4)
    ap.add_argument("--visual-margin-target", type=float, default=0.25)
    ap.add_argument("--visual-margin-boost", type=float, default=1.25)
    ap.add_argument("--vision-optimizer", type=str, default="muon8bit", choices=("muon8bit", "apollo8bit", "adamw"))
    ap.add_argument("--intel-vision", type=str, default="xpu", choices=("xpu", "none"))
    ap.add_argument("--hf-docvqa-max", type=int, default=3600)
    ap.add_argument("--hf-synthdog-max", type=int, default=3000)
    ap.add_argument("--real-pope-max", type=int, default=1400)
    ap.add_argument("--synth-ocr-max", type=int, default=1200)
    ap.add_argument("--s3d-mcq-max", type=int, default=480)
    ap.add_argument("--v38-visual-renders", type=int, default=16)
    ap.add_argument("--v38-repair-max", type=int, default=140)
    ap.add_argument("--v28-max-records", type=int, default=80)
    ap.add_argument("--v28-repair-max", type=int, default=80)
    ap.add_argument("--nemotron-max-gqa", type=int, default=700)
    ap.add_argument("--nemotron-max-geo", type=int, default=500)
    ap.add_argument("--nemotron-max-aokvqa", type=int, default=500)
    ap.add_argument("--nemotron-max-clevr", type=int, default=500)
    ap.add_argument("--nemotron-max-zhwiki", type=int, default=250)
    ap.add_argument("--nemotron-max-plotqa", type=int, default=250)
    ap.add_argument("--max-nemotron-terminal", type=int, default=900)
    ap.add_argument("--max-ot-agent", type=int, default=600)
    ap.add_argument("--max-terminal-pivot", type=int, default=450)
    ap.add_argument("--max-ot3-stem", type=int, default=900)
    ap.add_argument("--max-nemotron-science", type=int, default=700)
    ap.add_argument("--extra-jsonl", action="append", type=Path, default=[], help="Optional extra JSONL dataset(s) to blend.")
    ap.add_argument("--extra-dataset-dir", action="append", type=Path, default=[], help="Optional extra dataset directory(ies).")
    ap.add_argument("--rsi5-v38-explore", type=int, default=16)
    ap.add_argument("--rsi5-frontier-explore", type=int, default=32)
    ap.add_argument("--eval-images", type=int, default=48)
    ap.add_argument("--cleanup-intermediate-ckpts", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=20261170)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    db_path = args.stage7_work_root / "metadata/corpus.sqlite"
    handoff_path = args.stage7_work_root / "metadata/HANDOFF.json"

    # 1. Build & Strictly Audit Stage-11 Corpora (v38.8 64-Archetypes + Real OCR + Real 50/50 POPE + 3D MCQ + Frontier + Extra)
    tb41_excluded = _fetch_tb41_excluded_task_ids(hf_token)
    emit("stage11_build_corpora_start", excluded_tb41_tasks=len(tb41_excluded))

    v38_text_recs, v38_vis_pairs, v38_repair_recs = build_v38_64_archetype_corpus(
        seed=args.seed,
        max_visual_renders=args.v38_visual_renders,
        max_repairs=args.v38_repair_max,
    )
    v28_records = build_aesthetic_v28_biomech_records(seed=args.seed + 3, max_records=args.v28_max_records)
    v28_repair_records = build_v28_grammar_repair_records(v28_records, seed=args.seed + 5, max_items=args.v28_repair_max)

    raw_spatial3d_pairs = build_shaded_3d_spatial_primitives_records(seed=args.seed + 11, panel_size=(384, 384), angles_per_asset=6)
    spatial3d_pairs = [
        pair
        for pair in raw_spatial3d_pairs
        if check_generation_collapse_health(
            str(pair[0].get("answer") or ""),
            expect_thinking=bool(pair[0].get("enable_thinking", True)),
            max_rep_4gram=0.140,
            max_think_chars=4800,
        )["healthy"]
    ]
    s3d_mcq_pairs = build_3d_spatial_mcq_pairs(spatial3d_pairs, max_items=args.s3d_mcq_max, seed=args.seed + 12)
    hf_real_ocr_pairs = build_hf_real_ocr_pairs(
        token=hf_token,
        max_docvqa=args.hf_docvqa_max,
        max_synthdog=args.hf_synthdog_max,
        seed=args.seed + 13,
    )
    real_pope_pairs = build_real_photo_pope_yesno_pairs(
        token=hf_token,
        max_pairs=args.real_pope_max,
        seed=args.seed + 15,
    )
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
    extra_text_recs, extra_vis_pairs = load_extra_user_dataset_records(
        jsonl_paths=args.extra_jsonl,
        dataset_dirs=args.extra_dataset_dir,
    )

    all_text_recs = (
        list(frontier_records)
        + list(v38_text_recs)
        + list(v38_repair_recs)
        + list(v28_repair_records)
        + list(extra_text_recs)
    )
    all_visual_pairs = (
        list(spatial3d_pairs)
        + list(s3d_mcq_pairs)
        + list(hf_real_ocr_pairs)
        + list(real_pope_pairs)
        + list(synth_ocr_pairs)
        + list(nemotron_pairs)
        + list(v38_vis_pairs)
        + list(extra_vis_pairs)
    )
    audit_rep = audit_stage9_records_strict(
        all_text_recs, v28_records, all_visual_pairs, excluded_tb41_ids=tb41_excluded
    )
    write(args.output_root / "STAGE11_DATASET_AUDIT.json", audit_rep)
    emit(
        "stage11_dataset_audit_done",
        passed=audit_rep["passed"],
        failures=audit_rep["failures_count"],
        total=audit_rep["total_checked"],
        v38_archetypes=len(v38_text_recs),
        v38_vis=len(v38_vis_pairs),
        v38_repair=len(v38_repair_recs),
        hf_real_ocr=len(hf_real_ocr_pairs),
        real_pope_yesno=len(real_pope_pairs),
        s3d_mcq=len(s3d_mcq_pairs),
        synth_ocr_7mode=len(synth_ocr_pairs),
        extra_text=len(extra_text_recs),
        extra_vis=len(extra_vis_pairs),
    )
    if not audit_rep["passed"]:
        raise RuntimeError(f"Stage-11 pre-training dataset quality gate FAILED: {audit_rep['failures'][:5]}")

    rng = random.Random(args.seed)

    # 2. Construct Phase 11A Joint Visual-Awakening Mix (`11,520` samples)
    total_a = args.stage11a_updates * args.grad_accum
    n_a_hf_ocr = min(int(total_a * 0.22), len(hf_real_ocr_pairs))
    n_a_direct_cat = int(total_a * 0.13)
    n_a_pope = min(int(total_a * 0.12), len(real_pope_pairs) * 2) if real_pope_pairs else 0
    n_a_synth_ocr = min(int(total_a * 0.08), len(synth_ocr_pairs))
    n_a_s3d = min(int(total_a * 0.05), len(spatial3d_pairs))
    n_a_s3d_mcq = min(int(total_a * 0.05), len(s3d_mcq_pairs) * 2) if s3d_mcq_pairs else 0
    n_a_nem = min(int(total_a * 0.09), len(nemotron_pairs))
    n_a_v38_dsl = min(int(total_a * 0.04), len(v38_text_recs) * 8)
    n_a_v38_vis = min(int(total_a * 0.02), len(v38_vis_pairs) * 16) if v38_vis_pairs else 0
    n_a_v38_rep = min(int(total_a * 0.03), len(v38_repair_recs) * 3) if v38_repair_recs else 0
    n_a_v28 = min(int(total_a * 0.02), len(v28_records) * 4) if v28_records else 0
    n_a_v28_rep = min(int(total_a * 0.01), len(v28_repair_records) * 2) if v28_repair_records else 0
    n_a_extra_t = min(int(total_a * 0.06), len(extra_text_recs)) if extra_text_recs else 0
    n_a_extra_v = min(int(total_a * 0.06), len(extra_vis_pairs)) if extra_vis_pairs else 0
    n_a_front = min(int(total_a * 0.10), len(frontier_records))
    used_a = (
        n_a_hf_ocr + n_a_direct_cat + n_a_pope + n_a_synth_ocr + n_a_s3d + n_a_s3d_mcq
        + n_a_nem + n_a_v38_dsl + n_a_v38_vis + n_a_v38_rep + n_a_v28 + n_a_v28_rep
        + n_a_extra_t + n_a_extra_v + n_a_front
    )
    n_a_cat_rep = max(1, total_a - used_a)

    direct_a_rids, direct_a_summary = select_stage9_direct_ocr_and_yesno_rids(
        db_path, target_samples=n_a_direct_cat, seed=args.seed + 101
    )
    cat_a_rids, cat_a_summary = select_stage75b_balanced_rids(
        db_path, target_catalogue_samples=n_a_cat_rep, vision_fraction=0.65, seed=args.seed + 102
    )

    order_a: list[tuple[str, Any]] = [("cat_direct", rid) for rid in direct_a_rids]
    order_a.extend(("cat", rid) for rid in cat_a_rids)
    for i in range(n_a_hf_ocr):
        order_a.append(("pair", hf_real_ocr_pairs[i % len(hf_real_ocr_pairs)]))
    for i in range(n_a_pope):
        order_a.append(("pair", real_pope_pairs[i % len(real_pope_pairs)]))
    for i in range(n_a_synth_ocr):
        order_a.append(("pair", synth_ocr_pairs[i % len(synth_ocr_pairs)]))
    for i in range(n_a_s3d):
        order_a.append(("pair", spatial3d_pairs[i % len(spatial3d_pairs)]))
    for i in range(n_a_s3d_mcq):
        order_a.append(("pair", s3d_mcq_pairs[i % len(s3d_mcq_pairs)]))
    for i in range(n_a_nem):
        order_a.append(("pair", nemotron_pairs[i % len(nemotron_pairs)]))
    for i in range(n_a_v38_dsl):
        order_a.append(("pair", _text_record_to_pair(v38_text_recs[i % len(v38_text_recs)])))
    for i in range(n_a_v38_vis):
        order_a.append(("pair", v38_vis_pairs[i % len(v38_vis_pairs)]))
    for i in range(n_a_v38_rep):
        order_a.append(("pair", _text_record_to_pair(v38_repair_recs[i % len(v38_repair_recs)])))
    for i in range(n_a_v28):
        order_a.append(("v25", v28_records[i % len(v28_records)]))
    for i in range(n_a_v28_rep):
        order_a.append(("pair", _text_record_to_pair(v28_repair_records[i % len(v28_repair_records)])))
    for i in range(n_a_extra_t):
        order_a.append(("pair", _text_record_to_pair(extra_text_recs[i % len(extra_text_recs)])))
    for i in range(n_a_extra_v):
        order_a.append(("pair", extra_vis_pairs[i % len(extra_vis_pairs)]))
    for i in range(n_a_front):
        order_a.append(("pair", _text_record_to_pair(frontier_records[i % len(frontier_records)])))
    rng.shuffle(order_a)
    order_a = order_a[:total_a]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "audit": audit_rep,
                    "order_11a": len(order_a),
                    "breakdown_11a": {
                        "hf_real_ocr": n_a_hf_ocr,
                        "direct_ocr_yesno_cat": len(direct_a_rids),
                        "real_photo_pope_yesno": n_a_pope,
                        "synth_ocr_7mode": n_a_synth_ocr,
                        "spatial3d": n_a_s3d,
                        "spatial3d_mcq": n_a_s3d_mcq,
                        "nemotron_v3": n_a_nem,
                        "v38_64_archetypes": n_a_v38_dsl,
                        "v38_visual_renders": n_a_v38_vis,
                        "v38_grammar_repair": n_a_v38_rep,
                        "v28_all": n_a_v28,
                        "v28_repair": n_a_v28_rep,
                        "extra_user_text": n_a_extra_t,
                        "extra_user_vis": n_a_extra_v,
                        "frontier": n_a_front,
                        "cat_replay": len(cat_a_rids),
                    },
                    "direct_11a_summary": direct_a_summary,
                },
                indent=2,
            )
        )
        return

    # 3. Warm-Start Intact Stage-10.5 Anchor (preserving RMS-matched 0.035 projector + 6 all-time highs!)
    src_cfg = read(args.init_release_dir / "config.json")
    cfg = dict(src_cfg)
    cfg.update(
        {
            "campaign_id": "spark-x26-4b-loop48-en1b-vl-stage11-visual-awakening-v38",
            "init_from_release_dir": str(args.init_release_dir.resolve()),
            "allow_warmstart_vision_swap": False,
            "allow_warmstart_missing_loop_engram": False,
            "allow_warmstart_projector_rms_match": False,
            "reinit_vision_projector": False,
            "radio_output_rms_match": True,
            "radio_default_max_patches": 1536,
            "radio_highres_max_patches": 1600,
            "radio_highres_max_side": 896,
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
    sample_im = (
        next((r["images"][0] for r in v28_records if r.get("images")), None)
        if v28_records
        else None
    )

    try:
        rms_pre = measure_layerwise_visual_text_rms(model, sample_im, tag="pre_stage11a")
        xray_pre = measure_visual_grounding_xray(model, tag="pre_stage11a")
        cal_pre = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="pre_stage11a"
        )
        emit("stage11_pre_calibration", **cal_pre)

        # Phase 11A: Joint Visual-Attention Awakening (Split Early-Layer LR + L_decorr + Visual-Margin Hard-Mining + B50 muon8bit!)
        stage11a_cfg = {
            "stage_mode": "full_spark",
            "unfreeze_vision_last_n_layers": int(args.unfreeze_vision_layers),
            "optimizer_profile": "apollo_coat",
            "early_spark_layers": int(args.early_spark_layers),
            "early_spark_lr": float(args.early_spark_lr),
            "learning_rate": float(args.backbone_lr),
            "backbone_lr": float(args.backbone_lr),
            "apollo_learning_rate": float(args.backbone_lr),
            "connector_learning_rate": float(args.connector_lr),
            "adapter_lr": float(args.connector_lr),
            "apollo_connector_learning_rate": float(args.connector_lr),
            "engram_lr": float(args.engram_lr),
            "vision_lr": float(args.vision_lr),
            "visual_decorr_loss_weight": float(args.decorr_weight),
            "visual_decorr_target_dc_ratio": float(args.decorr_target_dc),
            "visual_decorr_target_cos": float(args.decorr_target_cos),
            "visual_margin_every_n": int(args.visual_margin_every_n),
            "visual_margin_target": float(args.visual_margin_target),
            "visual_margin_boost": float(args.visual_margin_boost),
            "warmup_ratio": 0.10,
            "clip_norm": 0.6,
            "weight_decay": 0.01,
        }
        res_a = _run_stage8_phase(
            stage_name="stage11a_visual_awakening_decorr_v38",
            model=model,
            base_cfg=cfg,
            stage_cfg=stage11a_cfg,
            order_items=order_a,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage11a",
            seed=args.seed + 300,
        )
        rms_post_a = measure_layerwise_visual_text_rms(model, sample_im, tag="post_stage11a")
        xray_post_a = measure_visual_grounding_xray(model, tag="post_stage11a")
        cal_post_a = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_stage11a"
        )
        emit("stage11_post_a_calibration", **cal_post_a)

        # Phase 11B: Dual-Track Verified RSI-5 (v38.8 + v28 + Frontier + DAPO-Math) + 3x En1B LR + 5-Axis Promotion Gate
        dapo_math_pool = build_dapo_math_rsi3_pool(token=hf_token, max_items=120, seed=args.seed + 499)
        rsi_v38_pool = list(v38_text_recs) + list(v28_records)
        rsi_signal_items, rsi_mining_summary = mine_rsi3_dual_track_trajectories(
            model,
            v28_records=rsi_v38_pool,
            frontier_records=list(frontier_records) + list(dapo_math_pool),
            v28_explore_n=args.rsi5_v38_explore,
            frontier_explore_n=args.rsi5_frontier_explore,
            candidates_k=2,
            seed=args.seed + 500,
        )
        total_rsi = args.rsi5_updates * args.grad_accum
        n_sig = min(int(total_rsi * 0.42), max(1, len(rsi_signal_items) * 6))
        n_rsi_ocr = min(len(hf_real_ocr_pairs), int(total_rsi * 0.14))
        n_rsi_pope = min(len(real_pope_pairs), int(total_rsi * 0.12)) if real_pope_pairs else 0
        n_rsi_rep = min(len(v38_repair_recs) * 2, int(total_rsi * 0.10)) if v38_repair_recs else 0
        n_rsi_direct = min(len(direct_a_rids), int(total_rsi * 0.10))
        n_rep_cat = max(1, total_rsi - n_sig - n_rsi_ocr - n_rsi_pope - n_rsi_rep - n_rsi_direct)
        cat_rsi_rids, _ = select_stage75b_balanced_rids(
            db_path, target_catalogue_samples=n_rep_cat, vision_fraction=0.60, seed=args.seed + 501
        )
        order_rsi: list[tuple[str, Any]] = [("cat", rid) for rid in cat_rsi_rids]
        for i in range(n_rsi_direct):
            order_rsi.append(("cat_direct", direct_a_rids[i % len(direct_a_rids)]))
        for i in range(n_rsi_ocr):
            order_rsi.append(("pair", hf_real_ocr_pairs[(n_a_hf_ocr + i) % len(hf_real_ocr_pairs)]))
        for i in range(n_rsi_pope):
            order_rsi.append(("pair", real_pope_pairs[i % len(real_pope_pairs)]))
        for i in range(n_rsi_rep):
            order_rsi.append(("pair", _text_record_to_pair(v38_repair_recs[i % len(v38_repair_recs)])))
        for i in range(n_sig):
            order_rsi.append(rsi_signal_items[i % len(rsi_signal_items)])
        rng.shuffle(order_rsi)
        order_rsi = order_rsi[:total_rsi]

        rsi_bb_lr = min(args.backbone_lr * 0.55, 4.5e-6)
        rsi_cfg = {
            "stage_mode": "full_spark",
            "unfreeze_vision_last_n_layers": 0,
            "optimizer_profile": "apollo_coat",
            "early_spark_layers": int(args.early_spark_layers),
            "early_spark_lr": min(args.early_spark_lr * 0.45, 1.0e-5),
            "learning_rate": rsi_bb_lr,
            "backbone_lr": rsi_bb_lr,
            "apollo_learning_rate": rsi_bb_lr,
            "connector_learning_rate": min(args.connector_lr * 0.35, 2.8e-5),
            "adapter_lr": min(args.connector_lr * 0.35, 2.8e-5),
            "apollo_connector_learning_rate": min(args.connector_lr * 0.35, 2.8e-5),
            "engram_lr": max(min(args.engram_lr * 0.75, 2.2e-5), rsi_bb_lr * 3.0),
            "vision_lr": 0.0,
            "visual_decorr_loss_weight": float(args.decorr_weight) * 0.5,
            "visual_decorr_target_dc_ratio": float(args.decorr_target_dc),
            "visual_decorr_target_cos": float(args.decorr_target_cos),
            "warmup_ratio": 0.12,
            "clip_norm": 0.5,
            "weight_decay": 0.01,
        }
        res_rsi = _run_stage8_phase(
            stage_name="stage11b_dual_track_rsi5_candidate",
            model=model,
            base_cfg=cfg,
            stage_cfg=rsi_cfg,
            order_items=order_rsi,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage11b-rsi5",
            seed=args.seed + 600,
        )
        rms_post_rsi = measure_layerwise_visual_text_rms(model, sample_im, tag="post_rsi5")
        xray_post_rsi = measure_visual_grounding_xray(model, tag="post_rsi5")
        cal_post_rsi = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_rsi5"
        )
        promo_gate = evaluate_rsi3_promotion_gate(cal_post_a, cal_post_rsi)
        emit("stage11_rsi5_promotion_gate", **promo_gate)

        selected_phase = "stage11b_rsi5"
        if not promo_gate["accepted"]:
            emit("stage11_rsi5_rolled_back_to_stage11a")
            load_spark_native_safetensors(model, args.output_root / "ckpt-stage11a/model.safetensors")
            if getattr(model, "_intel_vision_bridge", None) is not None:
                model._intel_vision_bridge.sync_from_host(model.vision)
            selected_phase = "stage11a_parent_rollback"

        # Strip training-only decorrelation policy flags before standalone export
        for k in ("visual_decorr_loss_weight", "visual_decorr_target_dc_ratio", "visual_decorr_target_cos"):
            model.policy.pop(k, None)

        report = {
            "stage11_architecture": {
                "warmstart": str(args.init_release_dir),
                "early_spark_layers": args.early_spark_layers,
                "early_spark_lr": args.early_spark_lr,
                "backbone_lr": args.backbone_lr,
                "connector_lr": args.connector_lr,
                "decorr_weight": args.decorr_weight,
                "visual_margin_every_n": args.visual_margin_every_n,
                "vision_optimizer_b50": args.vision_optimizer,
                "selected_phase": selected_phase,
            },
            "dataset_audit": audit_rep,
            "direct_11a_summary": direct_a_summary,
            "stage11a_cat_summary": cat_a_summary,
            "layerwise_rms": {
                "pre_11a": rms_pre,
                "post_11a": rms_post_a,
                "post_rsi5": rms_post_rsi,
            },
            "visual_grounding_xray": {
                "pre_11a": xray_pre,
                "post_11a": xray_post_a,
                "post_rsi5": xray_post_rsi,
            },
            "calibration": {
                "pre_11a": cal_pre,
                "post_11a": cal_post_a,
                "post_rsi5": cal_post_rsi,
            },
            "stage11a": res_a,
            "rsi5_mining": rsi_mining_summary,
            "stage11b_rsi5": res_rsi,
            "rsi5_promotion_gate": promo_gate,
            "final_gates": _extract_stage8_gate_stats(model),
        }
        write(args.output_root / "STAGE11_VISUAL_AWAKENING_V38_REPORT.json", report)

        out_bf16 = args.output_root / "final-model-stage11-bf16"
        out_fp8 = args.output_root / "final-model-stage11-fp8"
        _export_stage8_releases(
            model,
            src_release_dir=args.init_release_dir,
            radio_dir=args.radio_vision_dir,
            out_bf16_dir=out_bf16,
            out_fp8_dir=out_fp8,
            report=report,
        )
        write(out_bf16 / "STAGE11_VISUAL_AWAKENING_V38_REPORT.json", report)
        write(out_fp8 / "STAGE11_VISUAL_AWAKENING_V38_REPORT.json", report)

        if (
            args.cleanup_intermediate_ckpts
            and (out_bf16 / "model.safetensors").is_file()
            and (out_fp8 / "model.safetensors").is_file()
        ):
            for ck in ("ckpt-stage11a", "ckpt-stage11b-rsi5"):
                ck_p = args.output_root / ck
                if ck_p.is_dir():
                    shutil.rmtree(ck_p, ignore_errors=True)
            emit("stage11_intermediate_ckpts_cleaned")
    finally:
        model.close_coprocessors()


if __name__ == "__main__":
    main()
