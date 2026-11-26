#!/usr/bin/env python3
"""Stage-13 24-Hour Thanksgiving High-Throughput Multimodal Refinement + Online `VC-GRPO` RLVR (`Spark-X2.6-4B-Loop48-En1B-VL`).

Warm-starts directly from Stage-12 BF16 (`reinit_vision_projector=False`, preserving the trained FP32
`RadioMultiLayerGELUProjector` + 2D `\\n` row-break newlines + RMS ceiling) and executes 3 wall-clock-budgeted
phases tailored for 1x NVIDIA RTX PRO 4000 Blackwell 24 GB + 1x Intel Arc Pro B50 16 GB (~70M-95M new tokens in 24h):

  - **Phase 13A (~9.0h, `B=8`, ~65K-75K samples / ~55M-65M tokens)**:
    High-throughput chunked-RADIO (`100`-step chunks -> <= 2.5 GiB host RAM) Early-12L `self_attn` + FP32 GELU
    Projector training across strictly 0%-overlap high-entropy visual grounding pools:
      `TassP0lar/PRISM-Omni-Dataset-v1` (balanced `scenes3d`/`charts`/`paths`/`mazes`/`gvsr`/`pixmo`/downsampled `shapes`)
      + `COCO train2017` All-80-Class Balanced Existence (35% uniform random + 45% co-occurring adv + 20% popular)
      + `COCO` Spatial Layout Descriptions + `DOCCI` dense paragraphs + `ChartQA` + `ScienceQA` + `ViRL39K` + `RefCOCOg`
      + Counterfactual Multi-Color x Multi-Shape x Spatial Grid (breaks `"red circle"` prior!)
      + `DocVQA`/`TextVQA`/`SynthDoG`/`LaTeX`/`Word-OCR` + `Nemotron-v3` (incl. scaled `zhwiki` for `MMBench-CN`).
  - **Phase 13B (~8.5h, Full-36L `APOLLO+COAT` + Top-8 `C-RADIOv4-H` `Muon8bit` on B50, ~11.4K samples)**:
    Joint end-to-end consolidation with `visual_margin_every_n=3`, `visual_margin_target=0.35`, `visual_margin_boost=1.6`,
    blending `PRISM-Omni` (`<think>` XML CoT + direct), dense real photos, `COCO-all80`, `SyncCraft v38.8/v28` + repairs,
    and Frontier STEM/Terminal.
  - **Phase 13C (~5.5h, Genuine On-Policy Multimodal `VC-GRPO` RLVR + User-Feedback DPO)**:
    Live `K=4` rollouts with deterministic ground-truth verifiers + Real-vs-Blank Visual-Causality gating (`delta_logp`),
    interleaved with optional `user_feedback.jsonl` DPO updates, followed by an automatic Counterfactual Promotion Gate
    (auto-rollback to `ckpt-stage13b` if counterfactual visual lift or calibration regresses) and standalone BF16/FP8 export.
"""
from __future__ import annotations

import argparse
import gc
import io
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

from PIL import Image, ImageDraw
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.aesthetic_v28_biomech_corpus import build_aesthetic_v28_biomech_records  # noqa: E402
from colab.final.common import emit, read, write  # noqa: E402
from colab.final.data import Catalogue, MediaReader  # noqa: E402
from colab.final.engine import _save_model_safetensors  # noqa: E402
from colab.final.frontier_stage9_corpus import (  # noqa: E402
    build_3d_spatial_mcq_pairs,
    build_dapo_math_rsi3_pool,
    build_hf_real_ocr_pairs,
    build_realistic_scene_word_ocr_pairs,
    build_stage12_real_vl_alignment_pairs,
    build_stage9_frontier_training_records,
    build_v28_grammar_repair_records,
    load_extra_user_dataset_records,
    select_stage9_direct_ocr_and_yesno_rids,
)
from colab.final.inference import generate as spark_generate, visual_media  # noqa: E402
from colab.final.model import SparkNative, load_spark_native_safetensors  # noqa: E402
from colab.final.multimodal_rlvr import load_user_feedback_jsonl, run_multimodal_vc_grpo_rlvr  # noqa: E402
from colab.final.reasoning_agentic_corpus import check_generation_collapse_health  # noqa: E402
from colab.final.spatial_3d_primitives_corpus import (  # noqa: E402
    build_nemotron_v3_spatial_records,
    build_shaded_3d_spatial_primitives_records,
)
from colab.final.stage13_prism_rlvr_corpus import (  # noqa: E402
    build_cached_real_dense_and_reasoning_pairs,
    build_coco_all80_balanced_and_spatial_pairs,
    build_counterfactual_color_shape_spatial_pairs,
    build_prism_omni_visual_pairs,
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
)
from scripts.run_stage10_visual_rms_surgery_and_v28_rsi4 import (  # noqa: E402
    measure_layerwise_visual_text_rms,
)
from scripts.run_stage105_clean_projector_ocr_v28_rsi import (  # noqa: E402
    measure_visual_grounding_xray,
)
from scripts.run_stage12_frontier_vl_turning_point import (  # noqa: E402
    _batched_precomputed_step,
    _precompute_radio_features_fast,
)


def _run_stage13_counterfactual_vl_probe(model: SparkNative, *, tag: str) -> dict[str, Any]:
    """3-minute counterfactual (Real vs Blank-Gray) visual probe across:
    OCRBench (40), POPE-Random (40), POPE-Adversarial (40), RealWorldQA (40), and Color x Shape (8).
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    model.eval()
    report: dict[str, Any] = {"tag": tag}

    def _pil(val: Any) -> Image.Image:
        if isinstance(val, Image.Image):
            return val.convert("RGB")
        if isinstance(val, dict) and val.get("bytes"):
            return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
        if isinstance(val, (bytes, bytearray)):
            return Image.open(io.BytesIO(val)).convert("RGB")
        return Image.open(val).convert("RGB")

    def _blank(im: Image.Image) -> Image.Image:
        return Image.new("RGB", im.size, (128, 128, 128))

    # 1. Real OCRBench (40 stratified: 15 Reg + 10 Irreg + 8 DocVQA + 7 Scene Text)
    try:
        p_ocr = hf_hub_download(
            "echo840/OCRBench",
            "data/test-00000-of-00001.parquet",
            repo_type="dataset",
            revision="92a54bd1384387c178d5a07140a2d85e0a3d12e1",
            local_files_only=True,
        )
        tbl = pq.read_table(p_ocr).to_pylist()
        by_t: dict[str, list[dict[str, Any]]] = {}
        for r in tbl:
            by_t.setdefault(str(r.get("question_type", "")), []).append(r)
        eval_rows = (
            by_t.get("Regular Text Recognition", [])[:15]
            + by_t.get("Irregular Text Recognition", [])[:10]
            + by_t.get("DocVQA", [])[:8]
            + by_t.get("Scene Text-centric VQA", [])[:7]
        )
        hits_r = hits_b = 0
        by_cat: dict[str, list[int]] = {}
        for r in eval_rows:
            im = _pil(r["image"])
            ans_l = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
            pr = spark_generate(model, str(r["question"]), media=visual_media([im]), enable_thinking=False, max_new_tokens=32, do_sample=False).strip().lower()
            pb = spark_generate(model, str(r["question"]), media=visual_media([_blank(im)]), enable_thinking=False, max_new_tokens=32, do_sample=False).strip().lower()
            ok_r = any(str(a).lower().strip() in pr for a in ans_l if str(a).strip())
            ok_b = any(str(a).lower().strip() in pb for a in ans_l if str(a).strip())
            hits_r += int(ok_r)
            hits_b += int(ok_b)
            qt = str(r.get("question_type", ""))
            by_cat.setdefault(qt, [0, 0])
            by_cat[qt][0] += int(ok_r)
            by_cat[qt][1] += 1
        acc_r = hits_r / max(1, len(eval_rows))
        acc_b = hits_b / max(1, len(eval_rows))
        report["ocrbench_probe"] = {
            "n": len(eval_rows),
            "acc": round(acc_r, 4),
            "blank_acc": round(acc_b, 4),
            "lift": round(acc_r - acc_b, 4),
            "by_type": {k: round(v[0] / max(1, v[1]), 4) for k, v in by_cat.items()},
        }
    except Exception as exc:
        report["ocrbench_probe_error"] = str(exc)[:180]

    # 2. POPE-Random (40 balanced) & POPE-Adversarial (40 balanced), Real vs Blank
    for split_name, pq_rel in (
        ("pope_random_probe", "Full/random-00000-of-00001.parquet"),
        ("pope_adv_probe", "Full/adversarial-00000-of-00001.parquet"),
    ):
        try:
            p_pope = hf_hub_download(
                "lmms-lab/POPE",
                pq_rel,
                repo_type="dataset",
                revision="4db1276663dfa5eb8ad16a52d24c31a09e470896",
                local_files_only=True,
            )
            tbl_p = pq.read_table(p_pope).to_pylist()
            p_rows = (
                [r for r in tbl_p if str(r["answer"]).lower().strip() == "yes"][:20]
                + [r for r in tbl_p if str(r["answer"]).lower().strip() == "no"][:20]
            )
            stats_rb = {"real": [0, 0, 0, 0], "blank": [0, 0, 0, 0]}  # tp, fp, tn, fn
            for r in p_rows:
                im = _pil(r["image"])
                gt_yes = str(r["answer"]).lower().strip() == "yes"
                for mode, img_in in (("real", im), ("blank", _blank(im))):
                    pred = spark_generate(
                        model,
                        f"{r['question']}\nPlease answer yes or no.",
                        media=visual_media([img_in]),
                        enable_thinking=False,
                        max_new_tokens=8,
                        do_sample=False,
                    ).strip().lower()
                    words_p = pred.split()[:3]
                    is_yes = pred.startswith("yes") or ("yes" in words_p and "no" not in words_p)
                    if gt_yes and is_yes:
                        stats_rb[mode][0] += 1
                    elif not gt_yes and is_yes:
                        stats_rb[mode][1] += 1
                    elif not gt_yes and not is_yes:
                        stats_rb[mode][2] += 1
                    else:
                        stats_rb[mode][3] += 1
            tp, fp, tn, fn = stats_rb["real"]
            tpb, fpb, tnb, fnb = stats_rb["blank"]
            acc_r = (tp + tn) / max(1, len(p_rows))
            acc_b = (tpb + tnb) / max(1, len(p_rows))
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1_r = 2.0 * prec * rec / max(1e-9, prec + rec)
            report[split_name] = {
                "n": len(p_rows),
                "acc": round(acc_r, 4),
                "blank_acc": round(acc_b, 4),
                "lift": round(acc_r - acc_b, 4),
                "f1": round(f1_r, 4),
                "yes_ratio": round((tp + fp) / max(1, len(p_rows)), 4),
            }
        except Exception as exc:
            report[f"{split_name}_error"] = str(exc)[:180]

    # 3. RealWorldQA (40 items, Real vs Blank)
    try:
        p_rw = hf_hub_download(
            "xai-org/RealWorldQA",
            "data/test-00000-of-00002.parquet",
            repo_type="dataset",
            revision="17e7f75e092e47169732462ea3cdfebe911105dd",
            local_files_only=True,
        )
        tbl_rw = pq.read_table(p_rw).to_pylist()[:40]
        ok_r = ok_b = 0
        for r in tbl_rw:
            im = _pil(r["image"])
            gt = str(r["answer"]).strip()
            for is_real, img_in in ((True, im), (False, _blank(im))):
                pred = spark_generate(model, str(r["question"]), media=visual_media([img_in]), enable_thinking=False, max_new_tokens=16, do_sample=False).strip().rstrip(".")
                ok = (
                    pred.lower() == gt.lower()
                    or (len(gt) == 1 and pred[:1].upper() == gt.upper())
                    or (len(gt) >= 2 and gt.lower() in pred.lower())
                )
                if is_real:
                    ok_r += int(ok)
                else:
                    ok_b += int(ok)
        acc_r = ok_r / max(1, len(tbl_rw))
        acc_b = ok_b / max(1, len(tbl_rw))
        report["realworldqa_probe"] = {
            "n": len(tbl_rw),
            "acc": round(acc_r, 4),
            "blank_acc": round(acc_b, 4),
            "lift": round(acc_r - acc_b, 4),
        }
    except Exception as exc:
        report["realworldqa_probe_error"] = str(exc)[:180]

    # 4. Counterfactual Color x Shape (8 pairs, Real vs Blank — tests elimination of `"red circle"` prior!)
    cs_specs = [
        ("blue", (35, 80, 220), "square"),
        ("green", (35, 165, 55), "triangle"),
        ("yellow", (235, 200, 25), "circle"),
        ("purple", (145, 50, 195), "diamond"),
        ("red", (215, 35, 35), "square"),
        ("orange", (235, 125, 25), "circle"),
        ("cyan", (30, 185, 200), "triangle"),
        ("black", (28, 28, 32), "rectangle"),
    ]
    cs_r = cs_b = 0
    cs_samples = []
    for cname, rgb, sname in cs_specs:
        im = Image.new("RGB", (512, 384), (242, 242, 242))
        dr = ImageDraw.Draw(im)
        cx, cy, r = 256, 192, 82
        if sname == "circle":
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb, outline=(15, 15, 15), width=4)
        elif sname == "square":
            dr.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb, outline=(15, 15, 15), width=4)
        elif sname == "rectangle":
            dr.rectangle([cx - 115, cy - 60, cx + 115, cy + 60], fill=rgb, outline=(15, 15, 15), width=4)
        elif sname == "triangle":
            dr.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb, outline=(15, 15, 15))
        else:
            dr.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=rgb, outline=(15, 15, 15))
        q = "What color and shape is the object in the center of the image? Answer concisely."
        pr = spark_generate(model, q, media=visual_media([im]), enable_thinking=False, max_new_tokens=16, do_sample=False).strip().lower()
        pb = spark_generate(model, q, media=visual_media([_blank(im)]), enable_thinking=False, max_new_tokens=16, do_sample=False).strip().lower()
        ok_r = (cname in pr) and (sname in pr)
        ok_b = (cname in pb) and (sname in pb)
        cs_r += int(ok_r)
        cs_b += int(ok_b)
        cs_samples.append({"gt": f"{cname} {sname}", "real": pr, "blank": pb, "hit": bool(ok_r)})
    report["color_shape_probe"] = {
        "n": len(cs_specs),
        "acc": round(cs_r / len(cs_specs), 4),
        "blank_acc": round(cs_b / len(cs_specs), 4),
        "lift": round((cs_r - cs_b) / len(cs_specs), 4),
        "samples": cs_samples,
    }

    emit("stage13_counterfactual_vl_probe", **report)
    return report


def _run_stage13a_high_throughput_grounding(
    model: SparkNative,
    *,
    raw_pools: dict[str, list[tuple[dict[str, Any], dict[str, bytes]]]],
    max_updates: int = 7500,
    batch_size: int = 8,
    max_wall_hours: float = 9.0,
    early_attn_layers: int = 12,
    rms_cap: float = 0.09,
    rms_weight: float = 5.0,
    output_root: Path,
    checkpoint_dir: Path,
    seed: int = 20261310,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Phase 13A: High-throughput wall-clock-budgeted chunked-RADIO (`B=8`) visual grounding (~8,885 samples/hr).

    Unfreezes Early-12L Spark `self_attn` + FP32 `RadioMultiLayerGELUProjector` (keeping Stage-12 warm-start
    weights intact!) across 8 strictly balanced multimodal slots per step, targeting ~65K-75K diverse samples
    (~55M-65M tokens) in 9 hours with <= 2.5 GiB host RAM per 100-step chunk!
    """
    rng = random.Random(seed)
    device = model.device
    emb_layer = model.get_input_embeddings()
    nl_id = int(model._row_break_nl_id)
    with torch.no_grad():
        nl_emb = emb_layer(torch.tensor([nl_id], device=device, dtype=torch.long)).to(model.embed_dtype).view(1, 1, 1, -1)

    if getattr(model, "spark", None) is not None and hasattr(model.spark, "model"):
        model.spark.model.gradient_checkpointing = True

    saved_pol = {
        k: model.policy.get(k)
        for k in ("radio_default_max_patches", "radio_highres_max_patches", "radio_default_max_side", "radio_highres_max_side")
    }
    model.policy["radio_default_max_patches"] = 1024
    model.policy["radio_highres_max_patches"] = 1024
    model.policy["radio_default_max_side"] = 512
    model.policy["radio_highres_max_side"] = 512

    vp = model.vision_projector
    if hasattr(vp, "meta_scale"):
        vp.meta_scale.requires_grad_(False)
    proj_fc1_params = [vp.pre_norm.weight, vp.fc1.weight, vp.fc1.bias] + [ln.weight for ln in vp.layer_norms]
    proj_fc2_params = [vp.fc2.weight, vp.fc2.bias]
    proj_all_params = proj_fc1_params + proj_fc2_params

    for p in model.parameters():
        p.requires_grad_(False)
    for p in proj_all_params:
        p.requires_grad_(True)
    attn_params: list[torch.nn.Parameter] = []
    n_early = min(int(early_attn_layers), len(model.spark.model.layers))
    for li in range(n_early):
        for p in model.spark.model.layers[li].self_attn.parameters():
            p.requires_grad_(True)
            attn_params.append(p)
    all_params = proj_all_params + attn_params

    non_empty_fallback = [k for k, v in raw_pools.items() if v]
    if not non_empty_fallback:
        raise RuntimeError("Stage-13A received empty raw_pools!")

    # Strictly balanced 8-slot per-step micro-composition addressing EVERY Stage-12 counterfactual gap:
    slot_recipe = (
        "prism_omni",
        "coco_yes",
        "coco_no_all80",
        "dense_spatial",
        "science_chart_virl",
        "ocr_doc",
        "cf_color_shape_zh",
        "prism_hard_mix",
    )

    opt = torch.optim.AdamW(
        [
            {"params": proj_fc1_params, "lr": 4.5e-5, "weight_decay": 0.02},
            {"params": proj_fc2_params, "lr": 2.5e-5, "weight_decay": 0.02},
            {"params": attn_params, "lr": 1.6e-5, "weight_decay": 0.01},
        ]
    )
    warmup = min(80, max(5, max_updates // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda st: (st + 1) / float(warmup)
        if st < warmup
        else 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, (st - warmup) / float(max(1, max_updates - warmup))))),
    )

    t0 = time.monotonic()
    deadline = t0 + max(0.02, float(max_wall_hours)) * 3600.0
    ema_ce: float | None = None
    step = 0
    samples_seen = 0
    approx_tokens = 0
    chunk_steps = min(100, max(1, max_updates))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    emit(
        "stage13a_start",
        max_updates=max_updates,
        batch_size=batch_size,
        max_wall_hours=max_wall_hours,
        early_attn_layers=n_early,
        pool_counts={k: len(v) for k, v in raw_pools.items()},
    )

    try:
        while step < max_updates and time.monotonic() < deadline:
            cur_n = min(chunk_steps, max_updates - step)
            flat_chunk_pairs: list[tuple[dict[str, Any], dict[str, bytes]]] = []
            for _ in range(cur_n):
                for bi in range(batch_size):
                    bname = slot_recipe[bi % len(slot_recipe)]
                    pool = raw_pools.get(bname) or raw_pools[rng.choice(non_empty_fallback)]
                    flat_chunk_pairs.append(pool[rng.randrange(len(pool))])

            t_cf = time.monotonic()
            c_items, c_fmap, c_hmap = _precompute_radio_features_fast(model, flat_chunk_pairs, bsz=24)
            emit(
                "stage13a_chunk_features_ready",
                step_start=step + 1,
                step_end=step + cur_n,
                items=len(c_items),
                unique_images=len(c_fmap),
                elapsed_s=round(time.monotonic() - t_cf, 1),
            )

            idx_cursor = 0
            for _ in range(cur_n):
                if time.monotonic() >= deadline:
                    break
                batch = c_items[idx_cursor : idx_cursor + batch_size]
                idx_cursor += batch_size
                if not batch:
                    continue
                step += 1
                model.train()
                for mod in model.modules():
                    if isinstance(mod, nn.Dropout):
                        mod.eval()
                loss, ce_v, rms_v = _batched_precomputed_step(
                    model,
                    batch_items=batch,
                    feat_map=c_fmap,
                    hw_map=c_hmap,
                    nl_emb=nl_emb,
                    rms_cap=rms_cap,
                    rms_weight=rms_weight,
                    max_ans_tokens=160,
                )
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    emit("stage13a_nonfinite_skip", step=step)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(all_params, 1.0))
                opt.step()
                sched.step()
                ema_ce = ce_v if ema_ce is None else (0.94 * ema_ce + 0.06 * ce_v)
                samples_seen += len(batch)
                # Accurate token accounting: visual patches ((hh//2)*(ww//2+1)) + prompt + answer chars/3.8
                for row_b, key_b in batch:
                    hh, ww = c_hmap[key_b]
                    vis_t = ((hh + 1) // 2) * (((ww + 1) // 2) + 1)
                    txt_t = max(16, int((len(str(row_b.get("prompt", ""))) + len(str(row_b.get("answer", "")))) / 3.8))
                    approx_tokens += vis_t + txt_t

                if step == 1 or step % 100 == 0 or step == max_updates:
                    el_hr = max(1e-4, (time.monotonic() - t0) / 3600.0)
                    emit(
                        "stage13a_step",
                        step=step,
                        max_updates=max_updates,
                        samples=samples_seen,
                        tokens_m=round(approx_tokens / 1e6, 2),
                        samples_per_hr=round(samples_seen / el_hr, 1),
                        ce=round(ce_v, 4),
                        ema_ce=round(float(ema_ce), 4),
                        vrms=round(rms_v, 4),
                        grad_norm=round(gn, 3),
                        lr_attn=round(float(sched.get_last_lr()[2]), 7),
                        elapsed_min=round(el_hr * 60.0, 1),
                    )
                if step % 1500 == 0:
                    _save_model_safetensors(model, checkpoint_dir / "model.safetensors")

            del c_items, c_fmap, c_hmap, flat_chunk_pairs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for k, v in saved_pol.items():
            if v is not None:
                model.policy[k] = v
        del opt, sched
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save_model_safetensors(model, checkpoint_dir / "model.safetensors")
    probe_13a = _run_stage13_counterfactual_vl_probe(model, tag="post_stage13a")
    write(output_root / "STAGE13A_PROBE.json", probe_13a)
    res_13a = {
        "updates": step,
        "batch_size": batch_size,
        "samples_seen": samples_seen,
        "approx_tokens": approx_tokens,
        "final_ema_ce": round(float(ema_ce), 4) if ema_ce is not None else None,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "checkpoint": str(checkpoint_dir / "model.safetensors"),
    }
    emit("stage13a_complete", **res_13a)
    return res_13a, probe_13a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--init-release-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage12-frontier-vl/final-model-stage12-bf16"),
    )
    ap.add_argument("--radio-vision-dir", type=Path, default=Path("cache/c-radiov4-h"))
    ap.add_argument("--stage7-work-root", type=Path, default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50"))
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage13-prism-rlvr"),
    )
    ap.add_argument(
        "--nemotron-cache-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage8-radiov4/nemotron_v3_cache"),
    )
    ap.add_argument("--user-feedback-jsonl", type=Path, default=Path("work/user_feedback_vl.jsonl"))
    ap.add_argument("--release-tag", type=str, default="stage13")
    # 24-hour wall-clock & update budgets
    ap.add_argument("--phase13a-updates", type=int, default=7500)
    ap.add_argument("--phase13a-batch-size", type=int, default=8)
    ap.add_argument("--phase13a-max-hours", type=float, default=9.0)
    ap.add_argument("--phase13b-updates", type=int, default=950)
    ap.add_argument("--grad-accum", type=int, default=12)
    ap.add_argument("--phase13c-max-prompts", type=int, default=750)
    ap.add_argument("--phase13c-group-size", type=int, default=4)
    ap.add_argument("--phase13c-grad-accum-prompts", type=int, default=4)
    ap.add_argument("--phase13c-max-hours", type=float, default=5.5)
    ap.add_argument("--resume-stage13a-ckpt", type=Path, default=None)
    ap.add_argument("--resume-stage13b-ckpt", type=Path, default=None)
    # Optimizer & vision knobs
    ap.add_argument("--early-spark-layers", type=int, default=12)
    ap.add_argument("--early-spark-lr", type=float, default=1.2e-5)
    ap.add_argument("--backbone-lr", type=float, default=5.0e-6)
    ap.add_argument("--connector-lr", type=float, default=2.2e-5)
    ap.add_argument("--engram-lr", type=float, default=1.8e-5)
    ap.add_argument("--vision-lr", type=float, default=2.5e-6)
    ap.add_argument("--unfreeze-vision-layers", type=int, default=8)
    ap.add_argument("--visual-margin-every-n", type=int, default=3)
    ap.add_argument("--visual-margin-target", type=float, default=0.35)
    ap.add_argument("--visual-margin-boost", type=float, default=1.6)
    ap.add_argument("--rms-cap", type=float, default=0.09)
    ap.add_argument("--rms-weight", type=float, default=5.0)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--vision-optimizer", type=str, default="muon8bit", choices=("muon8bit", "apollo8bit", "adamw"))
    ap.add_argument("--intel-vision", type=str, default="xpu", choices=("xpu", "none"))
    # Dataset caps (all cached locally on workstation!)
    ap.add_argument("--prism-shapes-max", type=int, default=3600)
    ap.add_argument("--prism-scenes3d-max", type=int, default=1610)
    ap.add_argument("--prism-charts-max", type=int, default=1568)
    ap.add_argument("--prism-paths-max", type=int, default=398)
    ap.add_argument("--prism-mazes-max", type=int, default=395)
    ap.add_argument("--prism-real-max", type=int, default=700)
    ap.add_argument("--coco-yesno-max", type=int, default=8000)
    ap.add_argument("--coco-spatial-max", type=int, default=3200)
    ap.add_argument("--coco-count-max", type=int, default=1800)
    ap.add_argument("--coco-shards", type=int, default=4)
    ap.add_argument("--cf-colorshape-max", type=int, default=2400)
    ap.add_argument("--docci-max", type=int, default=1800)
    ap.add_argument("--chartqa-max", type=int, default=2400)
    ap.add_argument("--scienceqa-max", type=int, default=2000)
    ap.add_argument("--virl39k-max", type=int, default=2400)
    ap.add_argument("--refcocog-max", type=int, default=1600)
    ap.add_argument("--word-ocr-max", type=int, default=4800)
    ap.add_argument("--hf-docvqa-max", type=int, default=2400)
    ap.add_argument("--hf-synthdog-max", type=int, default=1200)
    ap.add_argument("--hf-textvqa-max", type=int, default=2400)
    ap.add_argument("--hf-aokvqa-max", type=int, default=2600)
    ap.add_argument("--hf-latex-max", type=int, default=600)
    ap.add_argument("--hf-gqa-open-max", type=int, default=2000)
    ap.add_argument("--nemotron-max-gqa", type=int, default=800)
    ap.add_argument("--nemotron-max-geo", type=int, default=600)
    ap.add_argument("--nemotron-max-aokvqa", type=int, default=600)
    ap.add_argument("--nemotron-max-clevr", type=int, default=600)
    ap.add_argument("--nemotron-max-zhwiki", type=int, default=1200)
    ap.add_argument("--nemotron-max-plotqa", type=int, default=400)
    ap.add_argument("--s3d-mcq-max", type=int, default=480)
    ap.add_argument("--v38-visual-renders", type=int, default=16)
    ap.add_argument("--v38-repair-max", type=int, default=120)
    ap.add_argument("--v28-max-records", type=int, default=60)
    ap.add_argument("--v28-repair-max", type=int, default=100)
    ap.add_argument("--max-nemotron-terminal", type=int, default=600)
    ap.add_argument("--max-ot-agent", type=int, default=400)
    ap.add_argument("--max-terminal-pivot", type=int, default=300)
    ap.add_argument("--max-ot3-stem", type=int, default=600)
    ap.add_argument("--max-nemotron-science", type=int, default=500)
    ap.add_argument("--extra-jsonl", action="append", type=Path, default=[])
    ap.add_argument("--extra-dataset-dir", action="append", type=Path, default=[])
    ap.add_argument("--eval-images", type=int, default=48)
    ap.add_argument("--cleanup-intermediate-ckpts", action="store_true", default=True)
    ap.add_argument("--skip-export", action="store_true", help="Skip BF16/FP8 export (used for rapid smoke tests).")
    ap.add_argument("--skip-quick-probes", action="store_true", help="Skip intermediate quick probes (smoke test).")
    ap.add_argument("--seed", type=int, default=20261300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    db_path = args.stage7_work_root / "metadata/corpus.sqlite"
    handoff_path = args.stage7_work_root / "metadata/HANDOFF.json"
    t_campaign = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Build & Strictly Audit ALL Stage-13 Corpora (0% Benchmark Overlap!)
    # ------------------------------------------------------------------
    tb41_excluded = _fetch_tb41_excluded_task_ids(hf_token)
    emit("stage13_build_corpora_start", excluded_tb41_tasks=len(tb41_excluded))

    prism_pairs = build_prism_omni_visual_pairs(
        token=hf_token,
        max_shapes=args.prism_shapes_max,
        max_scenes3d=args.prism_scenes3d_max,
        max_charts=args.prism_charts_max,
        max_paths=args.prism_paths_max,
        max_mazes=args.prism_mazes_max,
        max_real_prism=args.prism_real_max,
        cot_fraction=0.32,
        seed=args.seed + 1,
    )
    coco_all80_pairs = build_coco_all80_balanced_and_spatial_pairs(
        token=hf_token,
        max_yesno=args.coco_yesno_max,
        max_spatial_desc=args.coco_spatial_max,
        max_counting=args.coco_count_max,
        max_shards=args.coco_shards,
        seed=args.seed + 7,
    )
    cf_colorshape_pairs = build_counterfactual_color_shape_spatial_pairs(
        max_items=args.cf_colorshape_max,
        seed=args.seed + 11,
    )
    cached_real_dense_pairs = build_cached_real_dense_and_reasoning_pairs(
        stage7_work_root=args.stage7_work_root,
        token=hf_token,
        max_docci=args.docci_max,
        max_chartqa=args.chartqa_max,
        max_scienceqa=args.scienceqa_max,
        max_virl39k=args.virl39k_max,
        max_refcocog=args.refcocog_max,
        seed=args.seed + 19,
    )
    word_ocr_pairs = build_realistic_scene_word_ocr_pairs(max_items=args.word_ocr_max, seed=args.seed + 13)
    hf_real_ocr_pairs = build_hf_real_ocr_pairs(
        token=hf_token,
        max_docvqa=args.hf_docvqa_max,
        max_synthdog=args.hf_synthdog_max,
        seed=args.seed + 14,
    )
    real_vl_pairs = build_stage12_real_vl_alignment_pairs(
        token=hf_token,
        max_textvqa=args.hf_textvqa_max,
        max_aokvqa=args.hf_aokvqa_max,
        max_latex=args.hf_latex_max,
        max_gqa_open=args.hf_gqa_open_max,
        seed=args.seed + 15,
    )
    nemotron_vl_pairs = build_nemotron_v3_spatial_records(
        cache_dir=args.nemotron_cache_dir,
        hf_token=hf_token,
        max_gqa=args.nemotron_max_gqa,
        max_geometry3k=args.nemotron_max_geo,
        max_aokvqa=args.nemotron_max_aokvqa,
        max_clevr=args.nemotron_max_clevr,
        max_zhwiki=args.nemotron_max_zhwiki,
        max_plotqa=args.nemotron_max_plotqa,
        seed=args.seed + 21,
    )
    v38_text_recs, v38_vis_pairs, v38_repair_recs = build_v38_64_archetype_corpus(
        seed=args.seed + 25,
        max_visual_renders=args.v38_visual_renders,
        max_repairs=args.v38_repair_max,
    )
    v28_records = build_aesthetic_v28_biomech_records(seed=args.seed + 27, max_records=args.v28_max_records)
    v28_repair_records = build_v28_grammar_repair_records(v28_records, seed=args.seed + 29, max_items=args.v28_repair_max)
    raw_spatial3d_pairs = build_shaded_3d_spatial_primitives_records(seed=args.seed + 31, panel_size=(384, 384), angles_per_asset=6)
    spatial3d_pairs = [
        p
        for p in raw_spatial3d_pairs
        if check_generation_collapse_health(
            str(p[0].get("answer") or ""),
            expect_thinking=bool(p[0].get("enable_thinking", True)),
            max_rep_4gram=0.140,
            max_think_chars=4800,
        )["healthy"]
    ]
    s3d_mcq_pairs = build_3d_spatial_mcq_pairs(spatial3d_pairs, max_items=args.s3d_mcq_max, seed=args.seed + 33)
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
    dapo_math_pool = build_dapo_math_rsi3_pool(token=hf_token, max_items=160, seed=args.seed + 41)
    extra_text_recs, extra_vis_pairs = load_extra_user_dataset_records(
        jsonl_paths=args.extra_jsonl,
        dataset_dirs=args.extra_dataset_dir,
    )
    user_feedback_items = load_user_feedback_jsonl(args.user_feedback_jsonl)

    all_vis_pairs_for_audit = (
        list(prism_pairs)
        + list(coco_all80_pairs)
        + list(cf_colorshape_pairs)
        + list(cached_real_dense_pairs)
        + list(word_ocr_pairs)
        + list(hf_real_ocr_pairs)
        + list(real_vl_pairs)
        + list(nemotron_vl_pairs)
        + list(s3d_mcq_pairs)
        + list(spatial3d_pairs)
        + list(v38_vis_pairs)
        + list(extra_vis_pairs)
    )
    all_text_recs_for_audit = (
        list(frontier_records)
        + list(v38_text_recs)
        + list(v38_repair_recs)
        + list(v28_repair_records)
        + list(extra_text_recs)
    )
    audit_rep = audit_stage9_records_strict(
        all_text_recs_for_audit,
        v28_records,
        all_vis_pairs_for_audit,
        excluded_tb41_ids=tb41_excluded,
    )
    audit_rep["stage13_counts"] = {
        "prism_omni_pairs": len(prism_pairs),
        "coco_all80_pairs": len(coco_all80_pairs),
        "cf_colorshape_pairs": len(cf_colorshape_pairs),
        "cached_real_dense_pairs": len(cached_real_dense_pairs),
        "word_ocr_pairs": len(word_ocr_pairs),
        "hf_real_ocr_pairs": len(hf_real_ocr_pairs),
        "real_vl_pairs": len(real_vl_pairs),
        "nemotron_vl_pairs": len(nemotron_vl_pairs),
        "s3d_mcq_pairs": len(s3d_mcq_pairs),
        "spatial3d_pairs": len(spatial3d_pairs),
        "v38_text_recs": len(v38_text_recs),
        "v38_vis_pairs": len(v38_vis_pairs),
        "v28_records": len(v28_records),
        "frontier_records": len(frontier_records),
        "dapo_math_pool": len(dapo_math_pool),
        "user_feedback_items": len(user_feedback_items),
        "total_visual_pairs": len(all_vis_pairs_for_audit),
    }
    write(args.output_root / "STAGE13_DATASET_AUDIT.json", audit_rep)
    emit("stage13_dataset_audit_complete", **audit_rep)
    if not audit_rep["passed"]:
        raise RuntimeError(f"Stage-13 strict dataset audit FAILED: {audit_rep}")

    # ------------------------------------------------------------------
    # 2. Partition Phase 13A High-Throughput Pools, Phase 13B Full-36L Order, & Phase 13C RLVR Pool
    # ------------------------------------------------------------------
    rng = random.Random(args.seed + 100)
    prism_direct = [p for p in prism_pairs if not p[0].get("enable_thinking", False)]
    prism_non_shape = [p for p in prism_direct if (p[0].get("metadata") or {}).get("prism_family") != "shapes"]
    nem_zh = [p for p in nemotron_vl_pairs if str(p[0].get("language") or "") == "zh" or "zhwiki" in str(p[0].get("task") or "")]
    nem_en = [p for p in nemotron_vl_pairs if p not in nem_zh]

    phase13a_pools: dict[str, list[tuple[dict[str, Any], dict[str, bytes]]]] = {
        "prism_omni": prism_direct or list(prism_pairs),
        "coco_yes": [p for p in coco_all80_pairs if p[0].get("task") == "stage13_coco_pope_yes"],
        "coco_no_all80": [p for p in coco_all80_pairs if str(p[0].get("task") or "").startswith("stage13_coco_pope_no_")],
        "dense_spatial": (
            [p for p in cached_real_dense_pairs if p[0].get("source_group") in ("stage13_docci_dense", "stage13_refcocog")]
            + [p for p in coco_all80_pairs if p[0].get("task") == "stage13_coco_spatial_layout"]
        ),
        "science_chart_virl": (
            [p for p in cached_real_dense_pairs if p[0].get("source_group") in ("stage13_chartqa", "stage13_scienceqa", "stage13_virl39k")]
            + [p for p in real_vl_pairs if str(p[0].get("task") or "").startswith("stage12_aokvqa")]
            + list(s3d_mcq_pairs)
        ),
        "ocr_doc": (
            list(hf_real_ocr_pairs)
            + [p for p in real_vl_pairs if p[0].get("task") in ("stage12_direct_textvqa_real", "stage12_direct_latex_ocr")]
            + list(word_ocr_pairs)
        ),
        "cf_color_shape_zh": (
            list(cf_colorshape_pairs)
            + list(nem_zh)
            + [p for p in real_vl_pairs if p[0].get("task") == "stage12_gqa_open_semantic"]
        ),
        "prism_hard_mix": (
            list(prism_non_shape)
            + [p for p in coco_all80_pairs if p[0].get("task") == "stage13_coco_count"]
            + list(nem_en)
            + list(extra_vis_pairs)
        ),
    }

    # Phase 13B Full-36L + B50 Joint Consolidation Order
    total_13b = args.phase13b_updates * args.grad_accum
    n_b_prism = min(len(prism_pairs), int(total_13b * 0.22))
    n_b_coco = min(len(coco_all80_pairs), int(total_13b * 0.16))
    n_b_dense = min(len(cached_real_dense_pairs), int(total_13b * 0.16))
    n_b_cf = min(len(cf_colorshape_pairs), int(total_13b * 0.05))
    n_b_ocr = min(len(hf_real_ocr_pairs) + len(word_ocr_pairs), int(total_13b * 0.10))
    n_b_real_vl = min(len(real_vl_pairs), int(total_13b * 0.10))
    n_b_nem = min(len(nemotron_vl_pairs), int(total_13b * 0.07))
    n_b_s3d = min(len(s3d_mcq_pairs) + len(spatial3d_pairs), int(total_13b * 0.03))
    n_b_sync = min(len(v38_text_recs) + len(v38_vis_pairs) + len(v38_repair_recs) + len(v28_records) + len(v28_repair_records), int(total_13b * 0.04))
    n_b_front = min(len(frontier_records), int(total_13b * 0.04))
    direct_b_rids, _ = select_stage9_direct_ocr_and_yesno_rids(db_path, target_samples=max(80, int(total_13b * 0.02)), seed=args.seed + 105)
    used_13b = n_b_prism + n_b_coco + n_b_dense + n_b_cf + n_b_ocr + n_b_real_vl + n_b_nem + n_b_s3d + n_b_sync + n_b_front + len(direct_b_rids)
    n_b_cat = max(1, total_13b - used_13b)
    cat_b_rids, _ = select_stage75b_balanced_rids(db_path, target_catalogue_samples=n_b_cat, vision_fraction=0.72, seed=args.seed + 109)

    ocr_combined = list(hf_real_ocr_pairs) + list(word_ocr_pairs)
    rng.shuffle(ocr_combined)
    s3d_combined = list(s3d_mcq_pairs) + list(spatial3d_pairs)
    rng.shuffle(s3d_combined)
    sync_combined: list[tuple[str, Any]] = (
        [("pair", _text_record_to_pair(r)) for r in v38_text_recs]
        + [("pair", p) for p in v38_vis_pairs]
        + [("pair", _text_record_to_pair(r)) for r in v38_repair_recs]
        + [("v25", r) for r in v28_records]
        + [("pair", _text_record_to_pair(r)) for r in v28_repair_records]
    )
    rng.shuffle(sync_combined)

    order_13b: list[tuple[str, Any]] = [("cat", rid) for rid in cat_b_rids] + [("cat_direct", rid) for rid in direct_b_rids]
    for i in range(n_b_prism):
        order_13b.append(("pair", prism_pairs[i % len(prism_pairs)]))
    for i in range(n_b_coco):
        order_13b.append(("pair", coco_all80_pairs[i % len(coco_all80_pairs)]))
    for i in range(n_b_dense):
        order_13b.append(("pair", cached_real_dense_pairs[i % len(cached_real_dense_pairs)]))
    for i in range(n_b_cf):
        order_13b.append(("pair", cf_colorshape_pairs[i % len(cf_colorshape_pairs)]))
    for i in range(n_b_ocr):
        order_13b.append(("pair", ocr_combined[i % len(ocr_combined)]))
    for i in range(n_b_real_vl):
        order_13b.append(("pair", real_vl_pairs[i % len(real_vl_pairs)]))
    for i in range(n_b_nem):
        order_13b.append(("pair", nemotron_vl_pairs[i % len(nemotron_vl_pairs)]))
    for i in range(n_b_s3d):
        order_13b.append(("pair", s3d_combined[i % len(s3d_combined)]))
    for i in range(n_b_sync):
        order_13b.append(sync_combined[i % len(sync_combined)])
    for i in range(n_b_front):
        order_13b.append(("pair", _text_record_to_pair(frontier_records[i % len(frontier_records)])))
    rng.shuffle(order_13b)
    order_13b = order_13b[:total_13b]

    # Phase 13C Verifiable Multimodal RLVR Pool (`VC-GRPO` + optional User Feedback)
    rlvr_verifiable_pool: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    for p in prism_pairs:
        if (p[0].get("metadata") or {}).get("verifier"):
            rlvr_verifiable_pool.append(p)
    for p in coco_all80_pairs:
        if (p[0].get("metadata") or {}).get("verifier"):
            rlvr_verifiable_pool.append(p)
    for p in cf_colorshape_pairs:
        if (p[0].get("metadata") or {}).get("verifier"):
            rlvr_verifiable_pool.append(p)
    for p in cached_real_dense_pairs:
        if (p[0].get("metadata") or {}).get("verifier"):
            rlvr_verifiable_pool.append(p)
    # Add verified OCR crops & DAPO-Math into RLVR pool
    for rec, blobs in word_ocr_pairs[:400]:
        r2 = dict(rec)
        m2 = dict(rec.get("metadata") or {})
        m2["verifier"] = {"kind": "ocr_cer", "gold": str(rec.get("answer") or "")}
        r2["metadata"] = m2
        rlvr_verifiable_pool.append((r2, blobs))
    for rec in dapo_math_pool[:80]:
        r2 = dict(rec)
        m2 = dict(rec.get("metadata") or {})
        gold_m = str((rec.get("target") or {}).get("text") or rec.get("answer") or "")
        m2["verifier"] = {"kind": "dapo_math", "gold": gold_m}
        r2["metadata"] = m2
        rlvr_verifiable_pool.append(_text_record_to_pair(r2))
    rng.shuffle(rlvr_verifiable_pool)

    emit(
        "stage13_plan_ready",
        phase13a={"updates": args.phase13a_updates, "batch_size": args.phase13a_batch_size, "max_hours": args.phase13a_max_hours, "pools": {k: len(v) for k, v in phase13a_pools.items()}},
        phase13b={"updates": args.phase13b_updates, "grad_accum": args.grad_accum, "samples": len(order_13b)},
        phase13c={"max_prompts": args.phase13c_max_prompts, "group_size": args.phase13c_group_size, "max_hours": args.phase13c_max_hours, "rlvr_pool": len(rlvr_verifiable_pool), "user_feedback": len(user_feedback_items)},
    )
    if args.dry_run:
        return

    # ------------------------------------------------------------------
    # 3. Warm-Start SparkNative Directly from Stage-12 BF16 (`reinit_vision_projector=False`!)
    # ------------------------------------------------------------------
    cfg = dict(read(args.init_release_dir / "config.json"))
    cfg.update(
        {
            "campaign_id": str(args.output_root.name),
            "init_from_release_dir": str(args.init_release_dir.resolve()),
            "allow_warmstart_vision_swap": False,
            "allow_warmstart_missing_loop_engram": False,
            "allow_warmstart_projector_rms_match": False,
            "reinit_vision_projector": False,
            "radio_projector_type": "gelu",
            "radio_projector_intermediate_size": 4096,
            "radio_row_break_newline": True,
            "radio_output_rms_match": False,
            "visual_decorr_loss_weight": 0.0,
            "visual_rms_ceiling_weight": float(args.rms_weight),
            "visual_rms_ceiling_cap": 0.10,
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
    sample_im = next((r["images"][0] for r in v28_records if r.get("images")), None) if v28_records else None

    try:
        rms_pre = measure_layerwise_visual_text_rms(model, sample_im, tag="pre_stage13")
        probe_pre = {} if args.skip_quick_probes else _run_stage13_counterfactual_vl_probe(model, tag="pre_stage13")
        if probe_pre:
            write(args.output_root / "STAGE13_PRE_PROBE.json", probe_pre)

        res_13a: dict[str, Any] = {}
        probe_13a: dict[str, Any] = {}
        res_13b: dict[str, Any] = {}
        probe_13b: dict[str, Any] = {}
        cal_13b: dict[str, Any] = {}

        if args.resume_stage13b_ckpt is not None and args.resume_stage13b_ckpt.is_file():
            load_spark_native_safetensors(model, args.resume_stage13b_ckpt)
            if getattr(model, "_intel_vision_bridge", None) is not None:
                model._intel_vision_bridge.sync_from_host(model.vision)
            emit("stage13_resumed_from_stage13b_ckpt", ckpt=str(args.resume_stage13b_ckpt))
            res_13a = {"skipped": True}
            res_13b = {"resumed_from": str(args.resume_stage13b_ckpt)}
        else:
            # Phase 13A: High-Throughput Batched Early-12L + Projector Visual Grounding (~9.0h)
            if args.resume_stage13a_ckpt is not None and args.resume_stage13a_ckpt.is_file():
                load_spark_native_safetensors(model, args.resume_stage13a_ckpt)
                if getattr(model, "_intel_vision_bridge", None) is not None:
                    model._intel_vision_bridge.sync_from_host(model.vision)
                emit("stage13_resumed_from_stage13a_ckpt", ckpt=str(args.resume_stage13a_ckpt))
                res_13a = {"resumed_from": str(args.resume_stage13a_ckpt)}
            elif args.phase13a_updates > 0:
                res_13a, probe_13a = _run_stage13a_high_throughput_grounding(
                    model,
                    raw_pools=phase13a_pools,
                    max_updates=int(args.phase13a_updates),
                    batch_size=int(args.phase13a_batch_size),
                    max_wall_hours=float(args.phase13a_max_hours),
                    early_attn_layers=int(args.early_spark_layers),
                    rms_cap=float(args.rms_cap),
                    rms_weight=float(args.rms_weight),
                    output_root=args.output_root,
                    checkpoint_dir=args.output_root / "ckpt-stage13a",
                    seed=args.seed + 200,
                )

            # Phase 13B: Full-36L `APOLLO+COAT` + Top-8 `C-RADIOv4-H` `Muon8bit` on B50 (~8.5h)
            if args.phase13b_updates > 0 and order_13b:
                stage13b_cfg = {
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
                    "visual_decorr_loss_weight": 0.0,
                    "visual_rms_ceiling_weight": float(args.rms_weight),
                    "visual_rms_ceiling_cap": 0.10,
                    "visual_margin_every_n": int(args.visual_margin_every_n),
                    "visual_margin_target": float(args.visual_margin_target),
                    "visual_margin_boost": float(args.visual_margin_boost),
                    "warmup_ratio": 0.06,
                    "clip_norm": float(args.clip_norm),
                    "weight_decay": 0.01,
                }
                res_13b = _run_stage8_phase(
                    stage_name="stage13b_full36_prism_dense_b50",
                    model=model,
                    base_cfg=cfg,
                    stage_cfg=stage13b_cfg,
                    order_items=order_13b,
                    catalogue=catalogue,
                    reader=reader,
                    grad_accum=args.grad_accum,
                    checkpoint_dir=args.output_root / "ckpt-stage13b",
                    seed=args.seed + 300,
                )

        rms_13b = measure_layerwise_visual_text_rms(model, sample_im, tag="post_stage13b")
        xray_13b = measure_visual_grounding_xray(model, tag="post_stage13b")
        cal_13b = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_stage13b"
        )
        probe_13b = {} if args.skip_quick_probes else _run_stage13_counterfactual_vl_probe(model, tag="post_stage13b")
        if probe_13b:
            write(args.output_root / "STAGE13B_PROBE.json", probe_13b)

        # ------------------------------------------------------------------
        # Phase 13C: Genuine On-Policy Multimodal `VC-GRPO` RLVR + User-Feedback DPO (~5.5h)
        # ------------------------------------------------------------------
        res_13c: dict[str, Any] = {}
        probe_13c: dict[str, Any] = {}
        cal_13c: dict[str, Any] = {}
        promo_gate: dict[str, Any] = {"accepted": True}
        selected_phase = "stage13c_vc_grpo"

        if args.phase13c_max_prompts > 0 and rlvr_verifiable_pool:
            res_13c = run_multimodal_vc_grpo_rlvr(
                model,
                base_cfg=cfg,
                rlvr_items=rlvr_verifiable_pool,
                user_feedback_items=user_feedback_items,
                max_prompts=int(args.phase13c_max_prompts),
                group_size=int(args.phase13c_group_size),
                grad_accum_prompts=int(args.phase13c_grad_accum_prompts),
                max_wall_hours=float(args.phase13c_max_hours),
                lr=min(float(args.backbone_lr) * 0.55, 3.0e-6),
                connector_lr=min(float(args.connector_lr) * 0.50, 1.2e-5),
                kl_beta=0.04,
                clip_eps=0.20,
                rms_cap=0.10,
                rms_weight=float(args.rms_weight),
                checkpoint_dir=args.output_root / "ckpt-stage13c-rlvr",
                seed=args.seed + 500,
            )
            cal_13c = measure_visual_alignment_diagnostics(
                model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_stage13c"
            )
            probe_13c = {} if args.skip_quick_probes else _run_stage13_counterfactual_vl_probe(model, tag="post_stage13c")
            if probe_13c:
                write(args.output_root / "STAGE13C_PROBE.json", probe_13c)

            promo_gate = evaluate_rsi3_promotion_gate(cal_13b, cal_13c)
            if probe_13b and probe_13c:
                # Counterfactual Visual Lift non-regression check across all 5 probe tracks!
                checks = {
                    "ocr": float(probe_13c.get("ocrbench_probe", {}).get("acc", 0.0)) >= float(probe_13b.get("ocrbench_probe", {}).get("acc", 0.0)) - 0.04,
                    "pope_rand": float(probe_13c.get("pope_random_probe", {}).get("acc", 0.0)) >= float(probe_13b.get("pope_random_probe", {}).get("acc", 0.0)) - 0.05,
                    "pope_adv": float(probe_13c.get("pope_adv_probe", {}).get("acc", 0.0)) >= float(probe_13b.get("pope_adv_probe", {}).get("acc", 0.0)) - 0.05,
                    "realworldqa": float(probe_13c.get("realworldqa_probe", {}).get("acc", 0.0)) >= float(probe_13b.get("realworldqa_probe", {}).get("acc", 0.0)) - 0.05,
                    "color_shape": float(probe_13c.get("color_shape_probe", {}).get("acc", 0.0)) >= float(probe_13b.get("color_shape_probe", {}).get("acc", 0.0)) - 0.13,
                }
                promo_gate["counterfactual_probe_checks"] = checks
                promo_gate["accepted"] = bool(promo_gate["accepted"]) and all(checks.values())
            emit("stage13_vc_grpo_promotion_gate", **promo_gate)

            ckpt_13b_sf = args.output_root / "ckpt-stage13b/model.safetensors"
            if not promo_gate["accepted"] and ckpt_13b_sf.is_file():
                emit("stage13c_rolled_back_to_stage13b")
                load_spark_native_safetensors(model, ckpt_13b_sf)
                if getattr(model, "_intel_vision_bridge", None) is not None:
                    model._intel_vision_bridge.sync_from_host(model.vision)
                selected_phase = "stage13b_parent_rollback"
        else:
            selected_phase = "stage13b"

        # Clean standalone inference policy before export
        model.policy["reinit_vision_projector"] = False
        model.policy["radio_projector_type"] = "gelu"
        model.policy["radio_projector_intermediate_size"] = 4096
        model.policy["radio_row_break_newline"] = True
        model.policy["radio_output_rms_match"] = False
        for k in (
            "visual_decorr_loss_weight",
            "visual_decorr_target_dc_ratio",
            "visual_decorr_target_cos",
            "visual_rms_ceiling_weight",
            "visual_rms_ceiling_cap",
        ):
            model.policy.pop(k, None)

        est_13b_tokens = len(order_13b) * 850
        est_13c_tokens = int(res_13c.get("tokens_scored", 0))
        total_new_tokens = int(res_13a.get("approx_tokens", 0)) + est_13b_tokens + est_13c_tokens

        report = {
            "stage13_architecture": {
                "release_tag": str(args.release_tag),
                "warmstart": str(args.init_release_dir),
                "selected_phase": selected_phase,
                "total_campaign_hours": round((time.monotonic() - t_campaign) / 3600.0, 2),
                "estimated_new_tokens_total": total_new_tokens,
                "estimated_new_tokens_millions": round(total_new_tokens / 1e6, 2),
                "phase13a_tokens": int(res_13a.get("approx_tokens", 0)),
                "phase13b_est_tokens": est_13b_tokens,
                "phase13c_rlvr_tokens": est_13c_tokens,
            },
            "dataset_audit": audit_rep,
            "phase13a": res_13a,
            "phase13b": res_13b,
            "phase13c_vc_grpo": res_13c,
            "promotion_gate": promo_gate,
            "counterfactual_vl_probes": {
                "pre_13": probe_pre,
                "post_13a": probe_13a,
                "post_13b": probe_13b,
                "post_13c": probe_13c,
            },
            "layerwise_rms": {"pre_13": rms_pre, "post_13b": rms_13b},
            "visual_grounding_xray": {"post_13b": xray_13b},
            "calibration": {"post_13b": cal_13b, "post_13c": cal_13c},
            "final_gates": _extract_stage8_gate_stats(model),
        }
        write(args.output_root / "STAGE13_MULTIMODAL_RLVR_REPORT.json", report)

        if not args.skip_export:
            tag = str(args.release_tag).strip() or "stage13"
            out_bf16 = args.output_root / f"final-model-{tag}-bf16"
            out_fp8 = args.output_root / f"final-model-{tag}-fp8"
            _export_stage8_releases(
                model,
                src_release_dir=args.init_release_dir,
                radio_dir=args.radio_vision_dir,
                out_bf16_dir=out_bf16,
                out_fp8_dir=out_fp8,
                report=report,
            )
            write(out_bf16 / "STAGE13_MULTIMODAL_RLVR_REPORT.json", report)
            write(out_fp8 / "STAGE13_MULTIMODAL_RLVR_REPORT.json", report)

            if (
                args.cleanup_intermediate_ckpts
                and (out_bf16 / "model.safetensors").is_file()
                and (out_fp8 / "model.safetensors").is_file()
            ):
                for ck in ("ckpt-stage13a", "ckpt-stage13b", "ckpt-stage13c-rlvr"):
                    ck_p = args.output_root / ck
                    if ck_p.is_dir():
                        shutil.rmtree(ck_p, ignore_errors=True)
                emit("stage13_intermediate_ckpts_cleaned")
        emit("stage13_complete", selected_phase=selected_phase, total_new_tokens_m=round(total_new_tokens / 1e6, 2))
    finally:
        model.close_coprocessors()


if __name__ == "__main__":
    main()
