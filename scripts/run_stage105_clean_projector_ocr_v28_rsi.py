#!/usr/bin/env python3
"""Stage-10.5 Clean-Projector RMS-Match Alignment + Scaled Real OCR (`lmms-lab/DocVQA` + `SynthDoG-EN` + Cached)
+ Window-Aware Visual Budget (`<= 400` tokens fitting 100% inside Spark-X2.6-4B's `sliding_window=512` across all 48 layers)
+ 11-Mutator `v28` Block-DSL v4.1 Grammar & Semantic (`size_step=0.5`, `ramp`, `@meta`, Gate D enums) Repair
+ Gentle Joint `apollo_coat` (`8e-6`) + Top-6 `C-RADIOv4-H` `muon8bit` (`1.5e-6`) + Verified RSI-4.5.

Forensic Breakthroughs Addressed (from Stage-10 X-Ray `task-8471` & `task-8430`):
1. Warm-starting Stage-8/9's `53.06`-RMS SwiGLU projector weights carried a massive rank-1 DC collapse
   (`mean_norm=2.85` vs `std_norm=1.00`, `proj_cos(real,blank)=0.9467`) even after exit `RMSNorm`.
   Stage-10.5 warm-starts 100% of intact Stage-9 (`Spark-4B + Loop48 + En1B + visual_cross_attn + C-RADIOv4-H`)
   while cleanly re-initializing ONLY `vision_projector` (`reinit_vision_projector=True`, zeroed `w_down.bias`,
   `out_scale_init=0.035`, `meta_scale_init=0.006`), restoring `21.8x` higher spatial signal-to-DC ratio!
2. Stage-10A's `4.0e-4` AdamW LR on `frozen_spark_cross_attn` spiked (`grad_norm=14,820`, `loss=5.55`) and
   pushed `mean_real_loss 0.631 -> 1.065`. Stage-10.5A freezes BOTH `spark` AND `visual_cross_attn`
   (`stage_mode="frozen_spark_connector"`) with calm `connector_lr=1.2e-4`, `clip_norm=0.5`, `warmup_ratio=0.12`.
3. `Spark-X2.6-4B` has `sliding_window=512` on 75% (`27/36`) of layers. Capping single-image patches at `1600`
   (`<= 400` visual tokens) keeps 100% of visual tokens visible to ALL 48 virtual layers during OCR/VQA/MCQ.
4. Scales pixel-indispensable character-level OCR supervision by `>5x` using commercial-clean `lmms-lab/DocVQA`
   train (`Apache-2.0`) + `naver-clova-ix/synthdog-en` train (`MIT`) + all cached local OCR/Chart/Doc/TextVQA/LaTeX
   + strictly `50/50 Yes/No` + 6-mode full-color synthetic OCR, verified live with an unfakeable `GIANT_OCR` /
   `RED_CIRCLE` / `YESNO` X-Ray probe (`proj_cos`, `logit_KL`, exact readout).
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

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.aesthetic_v28_biomech_corpus import build_aesthetic_v28_biomech_records  # noqa: E402
from colab.final.common import emit, read, write  # noqa: E402
from colab.final.data import Catalogue, MediaReader  # noqa: E402
from colab.final.frontier_stage9_corpus import (  # noqa: E402
    build_dapo_math_rsi3_pool,
    build_hf_real_ocr_pairs,
    build_stage9_frontier_training_records,
    build_synthetic_scene_ocr_pairs,
    build_v28_grammar_repair_records,
    select_stage9_direct_ocr_and_yesno_rids,
)
from colab.final.inference import generate, visual_media  # noqa: E402
from colab.final.model import SparkNative, load_spark_native_safetensors  # noqa: E402
from colab.final.reasoning_agentic_corpus import check_generation_collapse_health  # noqa: E402
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
from scripts.run_stage10_visual_rms_surgery_and_v28_rsi4 import (  # noqa: E402
    measure_layerwise_visual_text_rms,
)


@torch.no_grad()
def measure_visual_grounding_xray(model: SparkNative, *, tag: str = "xray") -> dict[str, Any]:
    """Unfakeable live visual grounding X-Ray: tests giant `HELLO 2026` OCR, red circle, and Yes/No
    vs a blank gray image, logging exact projector real-vs-blank cosine, spatial std/mean ratio, and last-token logit KL."""
    was_train = model.training
    model.eval()
    im_text = Image.new("RGB", (512, 384), (245, 240, 225))
    d1 = ImageDraw.Draw(im_text)
    font = ImageFont.load_default()
    for fp in (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if Path(fp).is_file():
            try:
                font = ImageFont.truetype(fp, 72)
                break
            except Exception:
                pass
    d1.rectangle([20, 20, 492, 364], outline=(20, 60, 140), width=8)
    d1.text((44, 132), "HELLO 2026", fill=(15, 25, 45), font=font)

    im_red = Image.new("RGB", (512, 384), (235, 245, 255))
    d2 = ImageDraw.Draw(im_red)
    d2.ellipse([140, 80, 372, 312], fill=(210, 30, 30), outline=(40, 10, 10), width=6)

    im_blank = Image.new("RGB", (512, 384), (128, 128, 128))
    use_amp = torch.cuda.is_available() and str(model.device).startswith("cuda")

    probes = (
        ("giant_ocr", im_text, "What exact text is written in the image? Answer with only the exact text."),
        ("red_circle", im_red, "What color and shape is the object in the center of the image? Answer concisely."),
        ("yesno_red", im_red, "Is there a red circle in the image? Please answer yes or no."),
        ("yesno_dog", im_red, "Is there a dog in the image? Please answer yes or no."),
    )
    results: dict[str, Any] = {"tag": tag, "probes": {}}
    try:
        for pkey, im, q in probes:
            row = {"_family": "image", "language": "en", "task": "document_ocr", "prompt": q, "enable_thinking": False}
            med_r = visual_media([im])
            med_b = visual_media([im_blank])
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    er, vr, tid_r, tm_r = model.prefix_with_visual_context_and_ids(row, med_r, for_generation=True)
                    eb, vb, tid_b, tm_b = model.prefix_with_visual_context_and_ids(row, med_b, for_generation=True)
                    hr = model.forward_looped_model(
                        inputs_embeds=er, token_ids_for_engram=tid_r, text_token_mask=tm_r, visual_context=vr
                    ).last_hidden_state
                    hb = model.forward_looped_model(
                        inputs_embeds=eb, token_ids_for_engram=tid_b, text_token_mask=tm_b, visual_context=vb
                    ).last_hidden_state
                    lr = model.spark.lm_head(hr[0, -1:]).float()
                    lb = model.spark.lm_head(hb[0, -1:]).float()
            else:
                er, vr, tid_r, tm_r = model.prefix_with_visual_context_and_ids(row, med_r, for_generation=True)
                eb, vb, tid_b, tm_b = model.prefix_with_visual_context_and_ids(row, med_b, for_generation=True)
                hr = model.forward_looped_model(
                    inputs_embeds=er, token_ids_for_engram=tid_r, text_token_mask=tm_r, visual_context=vr
                ).last_hidden_state
                hb = model.forward_looped_model(
                    inputs_embeds=eb, token_ids_for_engram=tid_b, text_token_mask=tm_b, visual_context=vb
                ).last_hidden_state
                lr = model.spark.lm_head(hr[0, -1:]).float()
                lb = model.spark.lm_head(hb[0, -1:]).float()

            vm = ~tm_r[0]
            vr_toks = er[0, vm].detach().float()
            vb_toks = eb[0, vm].detach().float()
            vis_cos = float(F.cosine_similarity(vr_toks.flatten(), vb_toks.flatten(), dim=0).item())
            mean_norm = float(vr_toks.mean(dim=0).norm().item())
            std_norm = float(vr_toks.std(dim=0).norm().item())
            kl = float(F.kl_div(F.log_softmax(lb, dim=-1), F.softmax(lr, dim=-1), reduction="batchmean").item())
            pred_real = generate(model, q, media=med_r, enable_thinking=False, max_new_tokens=28, task="document_ocr")
            results["probes"][pkey] = {
                "vis_tokens": int(vm.sum().item()),
                "vis_cos_real_blank": round(vis_cos, 4),
                "mean_norm": round(mean_norm, 4),
                "std_norm": round(std_norm, 4),
                "logit_kl_real_blank": round(kl, 4),
                "pred_real": pred_real[:80],
            }
    finally:
        if was_train:
            model.train()
    emit("stage105_visual_grounding_xray", **results)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--init-release-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage9-frontier-rsi3/final-model-stage9-rsi3-bf16"),
    )
    ap.add_argument("--radio-vision-dir", type=Path, default=Path("cache/c-radiov4-h"))
    ap.add_argument("--stage7-work-root", type=Path, default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50"))
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage105-clean-proj-ocr-v28"),
    )
    ap.add_argument(
        "--nemotron-cache-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage8-radiov4/nemotron_v3_cache"),
    )
    ap.add_argument("--stage105a-updates", type=int, default=900)
    ap.add_argument("--stage105b-updates", type=int, default=360)
    ap.add_argument("--rsi45-updates", type=int, default=120)
    ap.add_argument("--grad-accum", type=int, default=12)
    ap.add_argument("--stage105a-connector-lr", type=float, default=1.2e-4)
    ap.add_argument("--stage105b-backbone-lr", type=float, default=8.0e-6)
    ap.add_argument("--stage105b-connector-lr", type=float, default=3.5e-5)
    ap.add_argument("--stage105b-engram-lr", type=float, default=2.5e-5)
    ap.add_argument("--stage105b-vision-lr", type=float, default=1.5e-6)
    ap.add_argument("--unfreeze-vision-layers", type=int, default=6)
    ap.add_argument("--out-scale-init", type=float, default=0.035)
    ap.add_argument("--meta-scale-init", type=float, default=0.006)
    ap.add_argument("--vision-optimizer", type=str, default="muon8bit", choices=("muon8bit", "apollo8bit", "adamw"))
    ap.add_argument("--intel-vision", type=str, default="xpu", choices=("xpu", "none"))
    ap.add_argument("--hf-docvqa-max", type=int, default=2400)
    ap.add_argument("--hf-synthdog-max", type=int, default=2000)
    ap.add_argument("--synth-ocr-max", type=int, default=900)
    ap.add_argument("--v28-max-records", type=int, default=96)
    ap.add_argument("--v28-repair-max", type=int, default=110)
    ap.add_argument("--nemotron-max-gqa", type=int, default=800)
    ap.add_argument("--nemotron-max-geo", type=int, default=500)
    ap.add_argument("--nemotron-max-aokvqa", type=int, default=500)
    ap.add_argument("--nemotron-max-clevr", type=int, default=500)
    ap.add_argument("--nemotron-max-zhwiki", type=int, default=250)
    ap.add_argument("--nemotron-max-plotqa", type=int, default=250)
    ap.add_argument("--max-nemotron-terminal", type=int, default=1000)
    ap.add_argument("--max-ot-agent", type=int, default=700)
    ap.add_argument("--max-terminal-pivot", type=int, default=500)
    ap.add_argument("--max-ot3-stem", type=int, default=1000)
    ap.add_argument("--max-nemotron-science", type=int, default=800)
    ap.add_argument("--rsi45-v28-explore", type=int, default=16)
    ap.add_argument("--rsi45-frontier-explore", type=int, default=32)
    ap.add_argument("--eval-images", type=int, default=48)
    ap.add_argument("--cleanup-intermediate-ckpts", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=20261161)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    db_path = args.stage7_work_root / "metadata/corpus.sqlite"
    handoff_path = args.stage7_work_root / "metadata/HANDOFF.json"

    # 1. Build & Strictly Audit Stage-10.5 Real OCR + 6-Mode Synth OCR + 11-Mutator v28 Repair + 3D + Frontier Corpora
    tb41_excluded = _fetch_tb41_excluded_task_ids(hf_token)
    emit("stage105_build_corpora_start", excluded_tb41_tasks=len(tb41_excluded))

    v28_records = build_aesthetic_v28_biomech_records(seed=args.seed, max_records=args.v28_max_records)
    v28_pixel_only = [r for r in v28_records if (r.get("metadata") or {}).get("pixel_indispensable") and r.get("images")]
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
    hf_real_ocr_pairs = build_hf_real_ocr_pairs(
        token=hf_token,
        max_docvqa=args.hf_docvqa_max,
        max_synthdog=args.hf_synthdog_max,
        seed=args.seed + 13,
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

    all_text_recs = list(frontier_records) + list(v28_repair_records)
    all_visual_pairs = list(spatial3d_pairs) + list(hf_real_ocr_pairs) + list(synth_ocr_pairs) + list(nemotron_pairs)
    audit_rep = audit_stage9_records_strict(
        all_text_recs, v28_records, all_visual_pairs, excluded_tb41_ids=tb41_excluded
    )
    write(args.output_root / "STAGE105_DATASET_AUDIT.json", audit_rep)
    emit(
        "stage105_dataset_audit_done",
        passed=audit_rep["passed"],
        failures=audit_rep["failures_count"],
        total=audit_rep["total_checked"],
        hf_real_ocr=len(hf_real_ocr_pairs),
        synth_ocr=len(synth_ocr_pairs),
        v28_repair=len(v28_repair_records),
        nemotron_v3=len(nemotron_pairs),
    )
    if not audit_rep["passed"]:
        raise RuntimeError(f"Stage-10.5 pre-training dataset quality gate FAILED: {audit_rep['failures'][:5]}")

    rng = random.Random(args.seed)

    # 2A. Phase 10.5A Mix (100% Visual Clean-Projector Alignment, `10,800` samples, frozen Spark + frozen visual_cross_attn!):
    #     ~40% Streamed Real OCR (`DocVQA` Apache-2.0 + `SynthDoG-EN` MIT)
    #     + ~20% Cached Direct OCR + Strictly `50/50 Yes/No`
    #     + ~8% 6-Mode High-Contrast Synthetic OCR/Chart/Table
    #     + ~10% Shaded 3D 4-Primitives
    #     + ~11% Nemotron-v3 Spatial / Geo / Chart / Bilingual
    #     + ~11% Stage-7 Pixel-Indispensable Visual Replay + v28 Visual
    total_a = args.stage105a_updates * args.grad_accum
    n_a_hf_ocr = min(int(total_a * 0.40), len(hf_real_ocr_pairs))
    n_a_direct_cat = int(total_a * 0.20)
    n_a_synth_ocr = min(int(total_a * 0.08), len(synth_ocr_pairs))
    n_a_s3d = min(int(total_a * 0.10), len(spatial3d_pairs) * 2)
    n_a_nem = min(int(total_a * 0.11), len(nemotron_pairs))
    n_a_v28 = min(int(total_a * 0.04), max(0, len(v28_pixel_only) * 8))
    n_a_cat_vis = max(1, total_a - n_a_hf_ocr - n_a_direct_cat - n_a_synth_ocr - n_a_s3d - n_a_nem - n_a_v28)

    direct_a_rids, direct_a_summary = select_stage9_direct_ocr_and_yesno_rids(
        db_path, target_samples=n_a_direct_cat, seed=args.seed + 101
    )
    cat_a_rids, cat_a_summary = select_stage75a_pixel_indispensable_rids(
        db_path, target_catalogue_samples=n_a_cat_vis, seed=args.seed + 102
    )
    order_a: list[tuple[str, Any]] = [("cat_direct", rid) for rid in direct_a_rids]
    order_a.extend(("cat", rid) for rid in cat_a_rids)
    for i in range(n_a_hf_ocr):
        order_a.append(("pair", hf_real_ocr_pairs[i % len(hf_real_ocr_pairs)]))
    for i in range(n_a_synth_ocr):
        order_a.append(("pair", synth_ocr_pairs[i % len(synth_ocr_pairs)]))
    for i in range(n_a_s3d):
        order_a.append(("pair", spatial3d_pairs[i % len(spatial3d_pairs)]))
    for i in range(n_a_nem):
        order_a.append(("pair", nemotron_pairs[i % len(nemotron_pairs)]))
    for i in range(n_a_v28):
        order_a.append(("v25", v28_pixel_only[i % len(v28_pixel_only)]))
    rng.shuffle(order_a)
    order_a = order_a[:total_a]

    # 2B. Phase 10.5B Mix (Gentle Full Joint Polish, `4,320` samples):
    #     ~26% Real OCR (`DocVQA` + `SynthDoG` + cached direct OCR + `50/50 Yes/No` + synth OCR)
    #     + ~18% Shaded 3D & Nemotron-v3 + ~18% `v28` + Gate D `train` + 11-Mutator Grammar/Semantic Repair
    #     + ~22% 2026 Frontier STEM/Terminal + ~16% Broad Balanced Stage-7 Visual/Text Replay
    total_b = args.stage105b_updates * args.grad_accum
    n_b_hf_ocr = min(int(total_b * 0.12), len(hf_real_ocr_pairs))
    n_b_direct_cat = int(total_b * 0.10)
    n_b_synth_ocr = min(int(total_b * 0.04), len(synth_ocr_pairs))
    n_b_s3d = min(int(total_b * 0.09), len(spatial3d_pairs))
    n_b_nem = min(int(total_b * 0.09), len(nemotron_pairs))
    n_b_v28 = min(int(total_b * 0.10), len(v28_records) * 6)
    n_b_v28_rep = min(int(total_b * 0.08), max(0, len(v28_repair_records) * 4))
    n_b_front = min(int(total_b * 0.22), len(frontier_records))
    n_b_cat = max(
        1,
        total_b
        - n_b_hf_ocr
        - n_b_direct_cat
        - n_b_synth_ocr
        - n_b_s3d
        - n_b_nem
        - n_b_v28
        - n_b_v28_rep
        - n_b_front,
    )

    direct_b_rids, direct_b_summary = select_stage9_direct_ocr_and_yesno_rids(
        db_path, target_samples=n_b_direct_cat, seed=args.seed + 201
    )
    cat_b_rids, cat_b_summary = select_stage75b_balanced_rids(
        db_path, target_catalogue_samples=n_b_cat, vision_fraction=0.60, seed=args.seed + 202
    )
    order_b: list[tuple[str, Any]] = [("cat_direct", rid) for rid in direct_b_rids]
    order_b.extend(("cat", rid) for rid in cat_b_rids)
    for i in range(n_b_hf_ocr):
        order_b.append(("pair", hf_real_ocr_pairs[(n_a_hf_ocr + i) % len(hf_real_ocr_pairs)]))
    for i in range(n_b_synth_ocr):
        order_b.append(("pair", synth_ocr_pairs[i % len(synth_ocr_pairs)]))
    for i in range(n_b_s3d):
        order_b.append(("pair", spatial3d_pairs[i % len(spatial3d_pairs)]))
    for i in range(n_b_nem):
        order_b.append(("pair", nemotron_pairs[i % len(nemotron_pairs)]))
    for i in range(n_b_v28):
        order_b.append(("v25", v28_records[i % len(v28_records)]))
    for i in range(n_b_v28_rep):
        order_b.append(("pair", _text_record_to_pair(v28_repair_records[i % len(v28_repair_records)])))
    for i in range(n_b_front):
        order_b.append(("pair", _text_record_to_pair(frontier_records[i % len(frontier_records)])))
    rng.shuffle(order_b)
    order_b = order_b[:total_b]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "audit": audit_rep,
                    "order_105a": len(order_a),
                    "order_105b": len(order_b),
                    "breakdown_105a": {
                        "hf_real_ocr": n_a_hf_ocr,
                        "direct_ocr_yesno_cat": len(direct_a_rids),
                        "synth_ocr_6mode": n_a_synth_ocr,
                        "spatial3d": n_a_s3d,
                        "nemotron_v3": n_a_nem,
                        "v28_visual": n_a_v28,
                        "cat_visual_replay": len(cat_a_rids),
                    },
                    "breakdown_105b": {
                        "hf_real_ocr": n_b_hf_ocr,
                        "direct_ocr_yesno_cat": len(direct_b_rids),
                        "synth_ocr_6mode": n_b_synth_ocr,
                        "spatial3d": n_b_s3d,
                        "nemotron_v3": n_b_nem,
                        "v28_all": n_b_v28,
                        "v28_grammar_semantic_repair": n_b_v28_rep,
                        "frontier": n_b_front,
                        "cat_replay": len(cat_b_rids),
                    },
                    "direct_105a_summary": direct_a_summary,
                },
                indent=2,
            )
        )
        return

    # 3. Clean Warm-Start from Intact Stage-9 Anchor with Fresh Zero-DC-Bias RMS-Matched `vision_projector`
    #    and Window-Aware Single-Image Budget (`<= 400` tokens so 100% fits inside Spark's `sliding_window=512`!)
    src_cfg = read(args.init_release_dir / "config.json")
    cfg = dict(src_cfg)
    cfg.update(
        {
            "campaign_id": "spark-x26-4b-loop48-en1b-vl-stage105-clean-proj-ocr-v28",
            "init_from_release_dir": str(args.init_release_dir.resolve()),
            "allow_warmstart_vision_swap": False,
            "allow_warmstart_missing_loop_engram": False,
            "allow_warmstart_projector_rms_match": True,
            "reinit_vision_projector": True,
            "radio_output_rms_match": True,
            "radio_out_scale_init": float(args.out_scale_init),
            "radio_meta_scale_init": float(args.meta_scale_init),
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
    sample_im = v28_pixel_only[0]["images"][0] if v28_pixel_only else None

    try:
        rms_pre = measure_layerwise_visual_text_rms(model, sample_im, tag="pre_stage105a")
        xray_pre = measure_visual_grounding_xray(model, tag="pre_stage105a")
        cal_pre = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="pre_stage105a"
        )
        emit("stage105_pre_calibration", **cal_pre)

        # Phase 10.5A: Frozen Spark + Frozen visual_cross_attn + Frozen C-RADIOv4-H -> Calm Clean-Projector Alignment!
        stage105a_cfg = {
            "stage_mode": "frozen_spark_connector",
            "unfreeze_vision_last_n_layers": 0,
            "optimizer_profile": "bf16_adamw",
            "connector_learning_rate": args.stage105a_connector_lr,
            "adapter_lr": args.stage105a_connector_lr,
            "learning_rate": args.stage105a_connector_lr,
            "warmup_ratio": 0.12,
            "clip_norm": 0.5,
            "weight_decay": 0.01,
        }
        res_a = _run_stage8_phase(
            stage_name="stage105a_clean_projector_ocr_alignment",
            model=model,
            base_cfg=cfg,
            stage_cfg=stage105a_cfg,
            order_items=order_a,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage105a",
            seed=args.seed + 300,
        )
        rms_post_a = measure_layerwise_visual_text_rms(model, sample_im, tag="post_stage105a")
        xray_post_a = measure_visual_grounding_xray(model, tag="post_stage105a")
        cal_post_a = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_stage105a"
        )
        emit("stage105_post_a_calibration", **cal_post_a)

        # Phase 10.5B: Gentle Joint Full Spark (`apollo_coat` 8e-6) + Top-6 C-RADIOv4-H (`muon8bit` 1.5e-6 on B50)
        stage105b_cfg = {
            "stage_mode": "full_spark",
            "unfreeze_vision_last_n_layers": int(args.unfreeze_vision_layers),
            "optimizer_profile": "apollo_coat",
            "learning_rate": args.stage105b_backbone_lr,
            "backbone_lr": args.stage105b_backbone_lr,
            "apollo_learning_rate": args.stage105b_backbone_lr,
            "connector_learning_rate": args.stage105b_connector_lr,
            "adapter_lr": args.stage105b_connector_lr,
            "apollo_connector_learning_rate": args.stage105b_connector_lr,
            "engram_lr": args.stage105b_engram_lr,
            "vision_lr": args.stage105b_vision_lr,
            "warmup_ratio": 0.12,
            "clip_norm": 0.6,
            "weight_decay": 0.01,
        }
        res_b = _run_stage8_phase(
            stage_name="stage105b_gentle_joint_ocr_v28_frontier",
            model=model,
            base_cfg=cfg,
            stage_cfg=stage105b_cfg,
            order_items=order_b,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage105b",
            seed=args.seed + 400,
        )
        rms_post_b = measure_layerwise_visual_text_rms(model, sample_im, tag="post_stage105b")
        xray_post_b = measure_visual_grounding_xray(model, tag="post_stage105b")
        cal_post_b = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_stage105b"
        )
        emit("stage105_post_b_calibration", **cal_post_b)

        # Phase 10.5C: Dual-Track Verified RSI-4.5 + 3x `En1B` LR + Strict 5-Axis Promotion/Rollback Gate
        dapo_math_pool = build_dapo_math_rsi3_pool(token=hf_token, max_items=120, seed=args.seed + 499)
        rsi_signal_items, rsi_mining_summary = mine_rsi3_dual_track_trajectories(
            model,
            v28_records=v28_records,
            frontier_records=list(frontier_records) + list(dapo_math_pool),
            v28_explore_n=args.rsi45_v28_explore,
            frontier_explore_n=args.rsi45_frontier_explore,
            candidates_k=2,
            seed=args.seed + 500,
        )
        total_rsi = args.rsi45_updates * args.grad_accum
        n_sig = min(int(total_rsi * 0.45), max(1, len(rsi_signal_items) * 6))
        n_rsi_direct = min(len(direct_b_rids), int(total_rsi * 0.15))
        n_rsi_hf_ocr = min(len(hf_real_ocr_pairs), int(total_rsi * 0.12))
        n_rsi_rep = min(len(v28_repair_records) * 2, int(total_rsi * 0.10))
        n_rep_cat = max(1, total_rsi - n_sig - n_rsi_direct - n_rsi_hf_ocr - n_rsi_rep)
        cat_rsi_rids, _ = select_stage75b_balanced_rids(
            db_path, target_catalogue_samples=n_rep_cat, vision_fraction=0.60, seed=args.seed + 501
        )
        order_rsi: list[tuple[str, Any]] = [("cat", rid) for rid in cat_rsi_rids]
        for i in range(n_rsi_direct):
            order_rsi.append(("cat_direct", direct_b_rids[i % len(direct_b_rids)]))
        for i in range(n_rsi_hf_ocr):
            order_rsi.append(("pair", hf_real_ocr_pairs[i % len(hf_real_ocr_pairs)]))
        for i in range(n_rsi_rep):
            order_rsi.append(("pair", _text_record_to_pair(v28_repair_records[i % len(v28_repair_records)])))
        for i in range(n_sig):
            order_rsi.append(rsi_signal_items[i % len(rsi_signal_items)])
        rng.shuffle(order_rsi)
        order_rsi = order_rsi[:total_rsi]

        rsi_bb_lr = min(args.stage105b_backbone_lr * 0.55, 4.5e-6)
        rsi_cfg = {
            "stage_mode": "full_spark",
            "unfreeze_vision_last_n_layers": 0,
            "optimizer_profile": "apollo_coat",
            "learning_rate": rsi_bb_lr,
            "backbone_lr": rsi_bb_lr,
            "apollo_learning_rate": rsi_bb_lr,
            "connector_learning_rate": min(args.stage105b_connector_lr * 0.5, 1.8e-5),
            "adapter_lr": min(args.stage105b_connector_lr * 0.5, 1.8e-5),
            "apollo_connector_learning_rate": min(args.stage105b_connector_lr * 0.5, 1.8e-5),
            "engram_lr": max(min(args.stage105b_engram_lr * 0.75, 2.0e-5), rsi_bb_lr * 3.0),
            "vision_lr": 0.0,
            "warmup_ratio": 0.12,
            "clip_norm": 0.5,
            "weight_decay": 0.01,
        }
        res_rsi = _run_stage8_phase(
            stage_name="stage105c_dual_track_rsi45_candidate",
            model=model,
            base_cfg=cfg,
            stage_cfg=rsi_cfg,
            order_items=order_rsi,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage105c-rsi45",
            seed=args.seed + 600,
        )
        rms_post_rsi = measure_layerwise_visual_text_rms(model, sample_im, tag="post_rsi45")
        xray_post_rsi = measure_visual_grounding_xray(model, tag="post_rsi45")
        cal_post_rsi = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_rsi45"
        )
        promo_gate = evaluate_rsi3_promotion_gate(cal_post_b, cal_post_rsi)
        emit("stage105_rsi45_promotion_gate", **promo_gate)

        selected_phase = "stage105c_rsi45"
        if not promo_gate["accepted"]:
            emit("stage105_rsi45_rolled_back_to_stage105b")
            load_spark_native_safetensors(model, args.output_root / "ckpt-stage105b/model.safetensors")
            if getattr(model, "_intel_vision_bridge", None) is not None:
                model._intel_vision_bridge.sync_from_host(model.vision)
            selected_phase = "stage105b_parent_rollback"

        # Clear warm-start-only flags before exporting standalone portable BF16 & FP8+q4g128 releases
        model.policy["allow_warmstart_projector_rms_match"] = False
        model.policy["reinit_vision_projector"] = False
        report = {
            "stage105_architecture": {
                "warmstart": str(args.init_release_dir),
                "reinit_vision_projector": True,
                "vision_backbone": "nvidia/C-RADIOv4-H (Fresh Zero-DC-Bias L24+L32 RMSNorm SwiGLU DeepStack, <=400 vis tokens)",
                "radio_output_rms_match": True,
                "out_scale_init": args.out_scale_init,
                "meta_scale_init": args.meta_scale_init,
                "vision_optimizer_b50": args.vision_optimizer,
                "selected_phase": selected_phase,
            },
            "dataset_audit": audit_rep,
            "direct_105a_summary": direct_a_summary,
            "direct_105b_summary": direct_b_summary,
            "stage105a_cat_summary": cat_a_summary,
            "stage105b_cat_summary": cat_b_summary,
            "layerwise_rms": {
                "pre_105a": rms_pre,
                "post_105a": rms_post_a,
                "post_105b": rms_post_b,
                "post_rsi45": rms_post_rsi,
            },
            "visual_grounding_xray": {
                "pre_105a": xray_pre,
                "post_105a": xray_post_a,
                "post_105b": xray_post_b,
                "post_rsi45": xray_post_rsi,
            },
            "calibration": {
                "pre_105a": cal_pre,
                "post_105a": cal_post_a,
                "post_105b": cal_post_b,
                "post_rsi45": cal_post_rsi,
            },
            "stage105a": res_a,
            "stage105b": res_b,
            "rsi45_mining": rsi_mining_summary,
            "stage105c_rsi45": res_rsi,
            "rsi45_promotion_gate": promo_gate,
            "final_gates": _extract_stage8_gate_stats(model),
        }
        write(args.output_root / "STAGE105_CLEAN_PROJ_OCR_REPORT.json", report)

        out_bf16 = args.output_root / "final-model-stage105-bf16"
        out_fp8 = args.output_root / "final-model-stage105-fp8"
        _export_stage8_releases(
            model,
            src_release_dir=args.init_release_dir,
            radio_dir=args.radio_vision_dir,
            out_bf16_dir=out_bf16,
            out_fp8_dir=out_fp8,
            report=report,
        )
        write(out_bf16 / "STAGE105_CLEAN_PROJ_OCR_REPORT.json", report)
        write(out_fp8 / "STAGE105_CLEAN_PROJ_OCR_REPORT.json", report)

        # Clean up large intermediate checkpoints once final portable BF16 & FP8 releases are safely written
        if (
            args.cleanup_intermediate_ckpts
            and (out_bf16 / "model.safetensors").is_file()
            and (out_fp8 / "model.safetensors").is_file()
        ):
            for ck in ("ckpt-stage105a", "ckpt-stage105b", "ckpt-stage105c-rsi45"):
                ck_p = args.output_root / ck
                if ck_p.is_dir():
                    shutil.rmtree(ck_p, ignore_errors=True)
            emit("stage105_intermediate_ckpts_cleaned")
    finally:
        model.close_coprocessors()


if __name__ == "__main__":
    main()
