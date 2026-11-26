#!/usr/bin/env python3
"""Independent Visual-Causal Re-Verification & Reload-Parity Audit for Spark-X2.6-4B-Loop48-En1B-VL.

Scientifically verifies whether the exported standalone checkpoint genuinely conditions on
visual inputs (vs. language priors or template memorization) by executing controlled
Real-Image vs. Blank-Gray vs. Shuffled-Image counterfactual ablations across:
  1. Synthetic Counterfactual Color x Shape x Novel-Text Grid (unfakeable 0-prior check)
  2. Real OCRBench reload-parity & prompt-format ablation + character-level similarity (1 - CER)
  3. Real POPE (Random / Popular / Adversarial) counterfactual Real vs. Blank vs. Shuffled + error audit
  4. Real RealWorldQA counterfactual Real vs. Blank + first-token KL divergence
  5. Real AI2D counterfactual Real vs. Blank (visual lift delta_vis)
  6. Real BLINK 6-Spatial counterfactual Real vs. Blank (spatial visual lift delta_vis)
  7. Exact parameter & architecture partition audit + optional BF16 vs. FP8+q4g128 consistency check.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.common import write  # noqa: E402
from colab.final.inference import generate, load, visual_media  # noqa: E402
from scripts.eval_standard_multimodal_benchmarks import (  # noqa: E402
    BENCHMARK_PINS,
    build_frozen_benchmark_suites,
    decode_parquet_image,
    extract_mcq_choice,
    extract_yes_no,
    parse_safe_list,
    score_ocrbench_answer,
    score_realworldqa_answer,
)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _char_similarity(pred: str, golds: list[str]) -> float:
    p = " ".join(str(pred or "").lower().split()).strip()
    if not p:
        return 0.0
    best = 0.0
    for g in golds:
        gn = " ".join(str(g or "").lower().split()).strip()
        if not gn:
            continue
        if gn in p:
            return 1.0
        d = _levenshtein(p[: max(len(gn) * 2, 32)], gn)
        best = max(best, max(0.0, 1.0 - d / max(1, len(gn))))
    return round(best, 4)


def _blank_like(im: Image.Image) -> Image.Image:
    return Image.new("RGB", im.size, (128, 128, 128))


def _load_font(size: int = 64) -> ImageFont.ImageFont:
    for fp in (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if Path(fp).is_file():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


@torch.no_grad()
def _first_token_kl_real_vs_blank(model: Any, prompt: str, im: Image.Image, *, task: str = "general_vqa") -> dict[str, float]:
    row = {"_family": "image", "language": "en", "task": task, "prompt": prompt, "enable_thinking": False}
    med_r = visual_media([im])
    med_b = visual_media([_blank_like(im)])
    use_amp = torch.cuda.is_available() and str(model.device).startswith("cuda")
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
    vr_t = er[0, vm].detach().float()
    vb_t = eb[0, vm].detach().float()
    cos_rb = float(F.cosine_similarity(vr_t.flatten(), vb_t.flatten(), dim=0).item())
    kl_rb = float(F.kl_div(F.log_softmax(lb, dim=-1), F.softmax(lr, dim=-1), reduction="batchmean").item())
    kl_sym = 0.5 * (
        kl_rb + float(F.kl_div(F.log_softmax(lr, dim=-1), F.softmax(lb, dim=-1), reduction="batchmean").item())
    )
    return {"vis_cos_rb": round(cos_rb, 4), "kl_rb": round(kl_rb, 4), "kl_sym": round(kl_sym, 4), "vis_tokens": int(vm.sum().item())}


def audit_parameter_breakdown(model: Any) -> dict[str, Any]:
    buckets: dict[str, int] = defaultdict(int)
    for name, p in model.named_parameters():
        n = int(p.numel())
        if name.startswith("spark."):
            buckets["spark_4b_backbone"] += n
        elif name.startswith("vision."):
            buckets["c_radiov4_h_vision"] += n
        elif name.startswith("engram_anchors."):
            buckets["engram_gpu_hot_and_routers"] += n
        elif name.startswith("loop_controllers."):
            buckets["loop48_dual_mhc"] += n
        elif name.startswith("vision_projector."):
            buckets["radio_gelu_projector"] += n
        elif name.startswith("visual_cross_attn."):
            buckets["gated_visual_cross_attn"] += n
        else:
            buckets["aux_spatial_modal_connectors"] += n
    # Include CPU-pinned Tier-1B cold Engram buffers
    tier1b = 0
    for name, b in model.named_buffers():
        if name.endswith("tier1b_cold_weight"):
            tier1b += int(b.numel())
    buckets["engram_cpu_tier1b_cold_buffers"] = tier1b
    total_all = sum(buckets.values())
    return {
        "by_module": dict(sorted(buckets.items())),
        "total_parameters_incl_tier1b": total_all,
        "total_billion": round(total_all / 1e9, 4),
    }


def run_synthetic_counterfactual_grid(model: Any) -> dict[str, Any]:
    """Tests 4 colors x 2 shapes (8 images) + 8 novel scene words against blank gray controls."""
    colors = {
        "red": (215, 30, 30),
        "blue": (30, 75, 220),
        "green": (30, 170, 50),
        "yellow": (235, 200, 20),
    }
    shapes = ("circle", "square")
    color_hits_real = color_hits_blank = 0
    shape_hits_real = shape_hits_blank = 0
    yn_real_tp = yn_real_tn = yn_real_fp = yn_real_fn = 0
    yn_blank_tp = yn_blank_tn = yn_blank_fp = yn_blank_fn = 0
    cs_samples = []

    for cname, rgb in colors.items():
        for sname in shapes:
            im = Image.new("RGB", (512, 384), (242, 244, 248))
            dr = ImageDraw.Draw(im)
            if sname == "circle":
                dr.ellipse([146, 82, 366, 302], fill=rgb, outline=(25, 25, 25), width=6)
            else:
                dr.rectangle([146, 82, 366, 302], fill=rgb, outline=(25, 25, 25), width=6)
            im_b = _blank_like(im)

            q_open = "What color and shape is the object in the center of the image? Answer concisely."
            p_r = generate(model, q_open, media=visual_media([im]), enable_thinking=False, max_new_tokens=16).strip()
            p_b = generate(model, q_open, media=visual_media([im_b]), enable_thinking=False, max_new_tokens=16).strip()
            kl_info = _first_token_kl_real_vs_blank(model, q_open, im)
            color_hits_real += int(cname in p_r.lower())
            color_hits_blank += int(cname in p_b.lower())
            shape_hits_real += int(sname in p_r.lower())
            shape_hits_blank += int(sname in p_b.lower())
            cs_samples.append({
                "gt": f"{cname} {sname}",
                "pred_real": p_r,
                "pred_blank": p_b,
                "kl_sym": kl_info["kl_sym"],
            })

            # Counterfactual Yes/No: 1 positive (exact color+shape), 1 hard negative (wrong color, same shape)
            wrong_color = [c for c in colors if c != cname][0]
            for query_cs, gt_yes in ((f"{cname} {sname}", True), (f"{wrong_color} {sname}", False)):
                q_yn = f"Is there a {query_cs} in the image? Please answer yes or no."
                yr = extract_yes_no(generate(model, q_yn, media=visual_media([im]), enable_thinking=False, max_new_tokens=8)) or "no"
                yb = extract_yes_no(generate(model, q_yn, media=visual_media([im_b]), enable_thinking=False, max_new_tokens=8)) or "no"
                if gt_yes and yr == "yes":
                    yn_real_tp += 1
                elif gt_yes and yr == "no":
                    yn_real_fn += 1
                elif (not gt_yes) and yr == "yes":
                    yn_real_fp += 1
                else:
                    yn_real_tn += 1
                if gt_yes and yb == "yes":
                    yn_blank_tp += 1
                elif gt_yes and yb == "no":
                    yn_blank_fn += 1
                elif (not gt_yes) and yb == "yes":
                    yn_blank_fp += 1
                else:
                    yn_blank_tn += 1

    # Novel OCR words
    font = _load_font(68)
    words = ["BRIDGE", "WATER", "TIGER", "CLOCK", "METAL", "NORTH", "SPARK", "ALPHA", "CLOUD", "TRAIN", "PLANT", "STONE"]
    ocr_exact_r = ocr_exact_b = 0
    ocr_sim_r: list[float] = []
    ocr_sim_b: list[float] = []
    ocr_kls: list[float] = []
    ocr_samples = []
    for idx, w in enumerate(words):
        bg = (245, 242, 232) if idx % 2 == 0 else (30, 40, 55)
        fg = (20, 25, 35) if idx % 2 == 0 else (245, 245, 240)
        im = Image.new("RGB", (512, 256), bg)
        dr = ImageDraw.Draw(im)
        dr.text((70, 90), w, fill=fg, font=font)
        im_b = _blank_like(im)
        q = "What text is written in this image?"
        pr = generate(model, q, media=visual_media([im]), enable_thinking=False, max_new_tokens=16).strip()
        pb = generate(model, q, media=visual_media([im_b]), enable_thinking=False, max_new_tokens=16).strip()
        kl_info = _first_token_kl_real_vs_blank(model, q, im)
        ok_r = w.lower() in pr.lower()
        ok_b = w.lower() in pb.lower()
        ocr_exact_r += int(ok_r)
        ocr_exact_b += int(ok_b)
        sr = _char_similarity(pr, [w])
        sb = _char_similarity(pb, [w])
        ocr_sim_r.append(sr)
        ocr_sim_b.append(sb)
        ocr_kls.append(kl_info["kl_sym"])
        ocr_samples.append({"gt": w, "pred_real": pr, "pred_blank": pb, "sim_real": sr, "kl_sym": kl_info["kl_sym"]})

    n_cs = len(cs_samples)
    n_yn = yn_real_tp + yn_real_tn + yn_real_fp + yn_real_fn
    return {
        "color_shape_open": {
            "n": n_cs,
            "color_acc_real": round(color_hits_real / n_cs, 4),
            "color_acc_blank": round(color_hits_blank / n_cs, 4),
            "shape_acc_real": round(shape_hits_real / n_cs, 4),
            "shape_acc_blank": round(shape_hits_blank / n_cs, 4),
            "samples": cs_samples,
        },
        "color_shape_yesno": {
            "n": n_yn,
            "acc_real": round((yn_real_tp + yn_real_tn) / max(1, n_yn), 4),
            "acc_blank": round((yn_blank_tp + yn_blank_tn) / max(1, n_yn), 4),
            "real_cm": {"tp": yn_real_tp, "fp": yn_real_fp, "tn": yn_real_tn, "fn": yn_real_fn},
            "blank_cm": {"tp": yn_blank_tp, "fp": yn_blank_fp, "tn": yn_blank_tn, "fn": yn_blank_fn},
        },
        "novel_word_ocr": {
            "n": len(words),
            "exact_acc_real": round(ocr_exact_r / len(words), 4),
            "exact_acc_blank": round(ocr_exact_b / len(words), 4),
            "mean_char_sim_real": round(sum(ocr_sim_r) / len(words), 4),
            "mean_char_sim_blank": round(sum(ocr_sim_b) / len(words), 4),
            "mean_kl_sym": round(sum(ocr_kls) / len(words), 4),
            "samples": ocr_samples,
        },
    }


def run_ocrbench_reload_and_counterfactual(model: Any, suites: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Tests:
      (A) Exact reload parity on the Stage-12 quick probe (raw prompt vs eval suffix prompt, Real vs Blank)
      (B) Full 5-group OCRBench (`n=120`) raw prompt vs eval suffix prompt, Real vs Blank, plus char similarity!
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    p_ocr = hf_hub_download(
        "echo840/OCRBench",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=BENCHMARK_PINS["ocrbench"]["revision"],
        local_files_only=True,
    )
    tbl = pq.read_table(p_ocr).to_pylist()
    by_t: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in tbl:
        by_t[str(r.get("question_type", ""))].append(r)

    # (A) Exact quick probe rows (same 32 items as _run_stage12_quick_vl_probe)
    probe_rows = (
        by_t["Regular Text Recognition"][:15]
        + by_t["Irregular Text Recognition"][:10]
        + by_t["Scene Text-centric VQA"][:7]
    )
    probe_modes = {}
    for mode_name, use_suffix, use_blank in (
        ("real_raw_prompt", False, False),
        ("real_eval_suffix_prompt", True, False),
        ("blank_raw_prompt", False, True),
    ):
        hits = 0
        sims: list[float] = []
        by_cat: dict[str, list[int]] = defaultdict(list)
        samples = []
        for r in probe_rows:
            im = decode_parquet_image(r["image"])
            if im is None:
                continue
            im_in = _blank_like(im) if use_blank else im
            q = str(r["question"]).strip()
            prompt = f"{q}\nRead the text/formula carefully from the image and answer directly and concisely." if use_suffix else q
            pred = generate(
                model,
                prompt,
                media=visual_media([im_in]),
                enable_thinking=False,
                max_new_tokens=32,
                task=("document_ocr" if use_suffix else "general_vqa"),
            ).strip()
            golds = parse_safe_list(r["answer"])
            ok = score_ocrbench_answer(pred, golds, str(r["question_type"]))
            hits += int(ok)
            sim = _char_similarity(pred, golds)
            sims.append(sim)
            qt = str(r["question_type"])
            by_cat[qt].append(int(ok))
            if len(samples) < 10:
                samples.append({"type": qt, "gt": golds[0] if golds else "", "pred": pred[:50], "hit": ok, "sim": sim})
        probe_modes[mode_name] = {
            "n": len(probe_rows),
            "acc": round(hits / max(1, len(probe_rows)), 4),
            "mean_char_sim": round(sum(sims) / max(1, len(sims)), 4),
            "by_type": {k: round(sum(v) / max(1, len(v)), 4) for k, v in sorted(by_cat.items())},
            "samples": samples,
        }

    # (B) Stratified 5-group OCRBench (n=120 from eval_standard_multimodal_benchmarks): raw prompt vs blank
    s_rows = suites["ocrbench"]
    full_raw_by_grp: dict[str, list[int]] = defaultdict(list)
    full_raw_by_sub: dict[str, list[int]] = defaultdict(list)
    full_blank_hits = 0
    full_raw_sims: list[float] = []
    full_blank_sims: list[float] = []
    for idx, r in enumerate(s_rows):
        im = decode_parquet_image(r["image"])
        if im is None:
            continue
        q = str(r["question"]).strip()
        golds = parse_safe_list(r["answer"])
        pred_r = generate(model, q, media=visual_media([im]), enable_thinking=False, max_new_tokens=64).strip()
        ok_r = int(score_ocrbench_answer(pred_r, golds, str(r["question_type"])))
        full_raw_by_grp[str(r["_group"])].append(ok_r)
        full_raw_by_sub[str(r["question_type"])].append(ok_r)
        full_raw_sims.append(_char_similarity(pred_r, golds))
        if idx < 40:
            pred_b = generate(model, q, media=visual_media([_blank_like(im)]), enable_thinking=False, max_new_tokens=64).strip()
            full_blank_hits += int(score_ocrbench_answer(pred_b, golds, str(r["question_type"])))
            full_blank_sims.append(_char_similarity(pred_b, golds))

    total_ok = sum(sum(v) for v in full_raw_by_grp.values())
    return {
        "quick_probe_32_parity": probe_modes,
        "stratified_5group_120_raw_prompt": {
            "n": len(s_rows),
            "correct": total_ok,
            "accuracy": round(total_ok / max(1, len(s_rows)), 4),
            "mean_char_sim_real": round(sum(full_raw_sims) / max(1, len(full_raw_sims)), 4),
            "blank_acc_first40": round(full_blank_hits / 40.0, 4),
            "mean_char_sim_blank_first40": round(sum(full_blank_sims) / max(1, len(full_blank_sims)), 4),
            "by_task_group": {k: round(sum(v) / max(1, len(v)), 4) for k, v in sorted(full_raw_by_grp.items())},
            "by_question_type": {k: {"n": len(v), "acc": round(sum(v) / max(1, len(v)), 4)} for k, v in sorted(full_raw_by_sub.items())},
        },
    }


def run_pope_counterfactual_and_random_audit(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluates POPE (n=180) under Real, Blank-Gray, and Shuffled-Image conditions + audits POPE-Random errors."""
    rng = random.Random(20261125)
    imgs = [decode_parquet_image(r["image"]) for r in rows]
    shuffled_idx = list(range(len(rows)))
    for i in range(len(rows)):
        j = (i + 37) % len(rows)
        shuffled_idx[i] = j

    def _eval_mode(mode: str) -> dict[str, Any]:
        stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0})
        random_errors = []
        kls: list[float] = []
        for i, r in enumerate(rows):
            im = imgs[i]
            if im is None:
                continue
            if mode == "real":
                im_in = im
            elif mode == "blank":
                im_in = _blank_like(im)
            else:
                im_in = imgs[shuffled_idx[i]] or _blank_like(im)
            prompt = f"{str(r['question']).strip()} Please answer yes or no."
            if mode == "real" and i % 6 == 0:
                kls.append(_first_token_kl_real_vs_blank(model, prompt, im)["kl_sym"])
            pred = generate(model, prompt, media=visual_media([im_in]), enable_thinking=False, max_new_tokens=16)
            yn = extract_yes_no(pred) or "no"
            gold = str(r["answer"]).strip().lower()
            sp = str(r["_split_only"])
            for b in (sp, "overall"):
                st = stats[b]
                st["n"] += 1
                if gold == "yes" and yn == "yes":
                    st["tp"] += 1
                elif gold == "no" and yn == "yes":
                    st["fp"] += 1
                elif gold == "no" and yn == "no":
                    st["tn"] += 1
                else:
                    st["fn"] += 1
            if mode == "real" and sp == "random" and yn != gold and len(random_errors) < 20:
                random_errors.append({
                    "image_source": str(r.get("image_source", "")),
                    "question": str(r["question"]),
                    "gold": gold,
                    "pred": yn,
                    "raw_pred": pred.strip()[:40],
                })

        def _m(st: dict[str, int]) -> dict[str, Any]:
            tp, fp, tn, fn, n = st["tp"], st["fp"], st["tn"], st["fn"], max(1, st["n"])
            acc = (tp + tn) / n
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0
            return {
                "n": st["n"],
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "yes_ratio": round((tp + fp) / n, 4),
                "cm": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            }

        out = {
            "overall": _m(stats["overall"]),
            "by_split": {k: _m(v) for k, v in sorted(stats.items()) if k != "overall"},
        }
        if mode == "real":
            out["mean_kl_sym_sampled"] = round(sum(kls) / max(1, len(kls)), 4)
            out["random_split_error_samples"] = random_errors
        return out

    res_real = _eval_mode("real")
    res_blank = _eval_mode("blank")
    res_shuf = _eval_mode("shuffled")
    return {
        "real": res_real,
        "blank": res_blank,
        "shuffled": res_shuf,
        "visual_lift_acc_vs_blank": round(res_real["overall"]["accuracy"] - res_blank["overall"]["accuracy"], 4),
        "visual_lift_acc_vs_shuffled": round(res_real["overall"]["accuracy"] - res_shuf["overall"]["accuracy"], 4),
        "visual_lift_f1_vs_blank": round(res_real["overall"]["f1"] - res_blank["overall"]["f1"], 4),
        "popular_lift_vs_blank": round(
            res_real["by_split"]["popular"]["accuracy"] - res_blank["by_split"]["popular"]["accuracy"], 4
        ),
        "adversarial_lift_vs_blank": round(
            res_real["by_split"]["adversarial"]["accuracy"] - res_blank["by_split"]["adversarial"]["accuracy"], 4
        ),
    }


def run_vqa_counterfactual(
    model: Any,
    suites: dict[str, list[dict[str, Any]]],
    *,
    n_rwqa: int = 80,
    n_ai2d: int = 80,
    n_blink: int = 72,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # 1. RealWorldQA Real vs Blank
    rw_rows = suites["realworldqa"][:n_rwqa]
    rw_r = rw_b = rw_agree = 0
    rw_kls: list[float] = []
    for i, r in enumerate(rw_rows):
        im = decode_parquet_image(r["image"])
        if im is None:
            continue
        q = str(r["question"]).strip()
        if i % 4 == 0:
            rw_kls.append(_first_token_kl_real_vs_blank(model, q, im)["kl_sym"])
        pr = generate(model, q, media=visual_media([im]), enable_thinking=False, max_new_tokens=32)
        pb = generate(model, q, media=visual_media([_blank_like(im)]), enable_thinking=False, max_new_tokens=32)
        rw_r += int(score_realworldqa_answer(pr, q, str(r["answer"])))
        rw_b += int(score_realworldqa_answer(pb, q, str(r["answer"])))
        rw_agree += int(pr.strip().lower() == pb.strip().lower())
    out["realworldqa"] = {
        "n": len(rw_rows),
        "acc_real": round(rw_r / max(1, len(rw_rows)), 4),
        "acc_blank": round(rw_b / max(1, len(rw_rows)), 4),
        "visual_lift": round((rw_r - rw_b) / max(1, len(rw_rows)), 4),
        "pred_change_rate": round(1.0 - rw_agree / max(1, len(rw_rows)), 4),
        "mean_kl_sym_sampled": round(sum(rw_kls) / max(1, len(rw_kls)), 4),
    }

    # 2. AI2D Real vs Blank
    ai_rows = suites["ai2d_test"][:n_ai2d]
    ai_r = ai_b = ai_agree = 0
    ai_kls: list[float] = []
    for i, r in enumerate(ai_rows):
        im = decode_parquet_image(r["image"])
        if im is None:
            continue
        opts = parse_safe_list(r["options"])
        letters = [chr(ord("A") + k) for k in range(len(opts))]
        gold_ans = str(r["answer"]).strip()
        gold_ltr = chr(ord("A") + int(gold_ans)) if gold_ans.isdigit() else gold_ans.upper()
        opt_block = "\n".join(f"({letters[k]}) {opts[k]}" for k in range(len(opts)))
        prompt = (
            f"{str(r['question']).strip()}\nOptions:\n{opt_block}\n"
            f"Answer with only the option letter ({', '.join(letters)}) in the format: Answer: <letter>"
        )
        if i % 4 == 0:
            ai_kls.append(_first_token_kl_real_vs_blank(model, prompt, im)["kl_sym"])
        pr = generate(model, prompt, media=visual_media([im]), enable_thinking=False, max_new_tokens=32)
        pb = generate(model, prompt, media=visual_media([_blank_like(im)]), enable_thinking=False, max_new_tokens=32)
        cr = extract_mcq_choice(pr, letters, opts, min_option_text_len=3)
        cb = extract_mcq_choice(pb, letters, opts, min_option_text_len=3)
        ai_r += int(cr == gold_ltr)
        ai_b += int(cb == gold_ltr)
        ai_agree += int(cr == cb)
    out["ai2d_test"] = {
        "n": len(ai_rows),
        "acc_real": round(ai_r / max(1, len(ai_rows)), 4),
        "acc_blank": round(ai_b / max(1, len(ai_rows)), 4),
        "visual_lift": round((ai_r - ai_b) / max(1, len(ai_rows)), 4),
        "choice_change_rate": round(1.0 - ai_agree / max(1, len(ai_rows)), 4),
        "mean_kl_sym_sampled": round(sum(ai_kls) / max(1, len(ai_kls)), 4),
    }

    # 3. BLINK 6-Spatial Real vs Blank
    bl_rows = suites["blink_val"][:n_blink]
    bl_r = bl_b = bl_agree = 0
    for r in bl_rows:
        imgs = [decode_parquet_image(r.get(f"image_{k}")) for k in range(1, 5)]
        imgs = [im for im in imgs if im is not None]
        if not imgs:
            continue
        blanks = [_blank_like(im) for im in imgs]
        vkind = "state_pair" if len(imgs) == 2 else ("multiview_render" if len(imgs) > 2 else "independent_image")
        v_units = (
            [{"visual_kind": "state_pair", "view_id": "before"}, {"visual_kind": "state_pair", "view_id": "after"}]
            if len(imgs) == 2
            else None
        )
        choices = parse_safe_list(r.get("choices", ""))
        letters = [chr(ord("A") + k) for k in range(max(2, len(choices)))]
        prompt = f"{str(r['prompt']).strip()}\nAnswer with only the option letter ({', '.join(letters)}) in the format: Answer: <letter>"
        gold_ltr = str(r["answer"]).strip().strip("()").upper()
        pr = generate(model, prompt, media=visual_media(imgs, visual_kind=vkind, visual_units=v_units), enable_thinking=False, max_new_tokens=32)
        pb = generate(model, prompt, media=visual_media(blanks, visual_kind=vkind, visual_units=v_units), enable_thinking=False, max_new_tokens=32)
        cr = extract_mcq_choice(pr, letters, choices, min_option_text_len=2)
        cb = extract_mcq_choice(pb, letters, choices, min_option_text_len=2)
        bl_r += int(cr == gold_ltr)
        bl_b += int(cb == gold_ltr)
        bl_agree += int(cr == cb)
    out["blink_6spatial"] = {
        "n": len(bl_rows),
        "acc_real": round(bl_r / max(1, len(bl_rows)), 4),
        "acc_blank": round(bl_b / max(1, len(bl_rows)), 4),
        "visual_lift": round((bl_r - bl_b) / max(1, len(bl_rows)), 4),
        "choice_change_rate": round(1.0 - bl_agree / max(1, len(bl_rows)), 4),
    }

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage12-frontier-vl/final-model-stage12-bf16"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage12-frontier-vl/STAGE12_VISUAL_CAUSAL_REVERIFY.json"),
    )
    ap.add_argument("--scale", type=float, default=0.6)
    args = ap.parse_args()

    t0 = time.time()
    token = os.environ.get("HF_TOKEN") or None
    suites = build_frozen_benchmark_suites(token, scale=float(args.scale))
    model = load(args.model_dir, device="cuda:0" if torch.cuda.is_available() else "cpu")

    print("[1/5] Parameter breakdown audit...", flush=True)
    param_audit = audit_parameter_breakdown(model)
    print(json.dumps(param_audit, indent=2), flush=True)

    print("[2/5] Synthetic counterfactual Color x Shape x Novel-Text grid...", flush=True)
    synth_res = run_synthetic_counterfactual_grid(model)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in synth_res.items()}, indent=2), flush=True)

    print("[3/5] OCRBench reload-parity, suffix ablation & Real-vs-Blank audit...", flush=True)
    ocr_res = run_ocrbench_reload_and_counterfactual(model, suites)
    print(json.dumps({
        "quick_probe_32_parity": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in ocr_res["quick_probe_32_parity"].items()},
        "stratified_5group_120_raw_prompt": ocr_res["stratified_5group_120_raw_prompt"],
    }, indent=2), flush=True)

    print("[4/5] POPE Real vs Blank vs Shuffled counterfactual + Random-split error audit...", flush=True)
    pope_res = run_pope_counterfactual_and_random_audit(model, suites["pope"])
    print(json.dumps({
        "real_overall": pope_res["real"]["overall"],
        "real_by_split": pope_res["real"]["by_split"],
        "blank_overall": pope_res["blank"]["overall"],
        "shuffled_overall": pope_res["shuffled"]["overall"],
        "visual_lift_acc_vs_blank": pope_res["visual_lift_acc_vs_blank"],
        "visual_lift_acc_vs_shuffled": pope_res["visual_lift_acc_vs_shuffled"],
    }, indent=2), flush=True)

    print("[5/5] RealWorldQA / AI2D / BLINK Real-vs-Blank visual lift...", flush=True)
    vqa_res = run_vqa_counterfactual(model, suites)
    print(json.dumps(vqa_res, indent=2), flush=True)

    report = {
        "schema_version": 1,
        "model_dir": str(args.model_dir),
        "elapsed_s": round(time.time() - t0, 2),
        "parameter_audit": param_audit,
        "synthetic_counterfactual": synth_res,
        "ocrbench_parity_and_counterfactual": ocr_res,
        "pope_counterfactual": pope_res,
        "vqa_counterfactual": vqa_res,
    }
    write(args.out_json, report)
    print(f"COMPLETED_VISUAL_CAUSAL_REVERIFY_OK -> {args.out_json} ({report['elapsed_s']}s)", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
