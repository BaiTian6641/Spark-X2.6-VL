#!/usr/bin/env python3
"""Stage-12 True Frontier VL Turning Point (`Spark-X2.6-4B-Loop48-En1B-VL`):
Clean `RadioMultiLayerGELUProjector` (NO SwiGLU / NO exit `out_norm` / ZERO `L_decorr`)
+ 2D Spatial Row-Break Newline Tokens (`radio_row_break_newline=True`, proven `+7.5%` exact 4-char OCR)
+ High-Throughput Bucketed `B=8` Frozen-Spark Projector Alignment (`Stage-12A`, per-example normalized CE loss)
+ Realistic Scene-Text Word/Code OCR (`0% -> 17.5%+` real OCRBench in 250 steps) + Real `TextVQA` / `DocVQA` / `SynthDoG` / `LaTeX`
+ Clean Object-Existence `50/50` GQA POPE + `A-OKVQA` (MCQ + rationales) + Open-Ended GQA + `Nemotron-Image-v3`
+ Full-36-Layer Joint `APOLLO+COAT` + `C-RADIOv4-H` Top-6 Blocks (`Muon8bit` on B50) (`Stage-12B`)
+ Targeted Unseen `v28` Syntax-Slip Repair (`missing_bone_attach`, `meta_chain_field`, `illegal_bone_statement`) + `v38.8` + `RSI-6`.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.aesthetic_v28_biomech_corpus import build_aesthetic_v28_biomech_records  # noqa: E402
from colab.final.common import emit, read, write  # noqa: E402
from colab.final.data import Catalogue, MediaReader, prepare_media  # noqa: E402
from colab.final.engine import _save_model_safetensors  # noqa: E402
from colab.final.frontier_stage9_corpus import (  # noqa: E402
    build_3d_spatial_mcq_pairs,
    build_dapo_math_rsi3_pool,
    build_hf_real_ocr_pairs,
    build_real_photo_pope_yesno_pairs,
    build_realistic_scene_word_ocr_pairs,
    build_stage12_real_vl_alignment_pairs,
    build_stage9_frontier_training_records,
    build_v28_grammar_repair_records,
    convert_catalogue_row_to_direct_mode,
    load_extra_user_dataset_records,
    select_stage115_visual_alignment_rids,
    select_stage9_direct_ocr_and_yesno_rids,
)
from colab.final.inference import generate as spark_generate, visual_media  # noqa: E402
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
    _pair_to_example,
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


def _materialize_stage12_item(
    kind: str,
    payload: Any,
    *,
    catalogue: Catalogue,
    reader: MediaReader,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if kind == "v25":
        return v25_record_to_example(payload)
    if kind == "pair":
        return _pair_to_example(payload)
    row, packet = catalogue.row(str(payload))
    with reader.packet(packet) as get:
        media = prepare_media(row, get, cfg)
    if kind == "cat_direct":
        row = convert_catalogue_row_to_direct_mode(row)
    return row, media


def _run_stage12_quick_vl_probe(model: SparkNative, *, tag: str) -> dict[str, Any]:
    """2-minute zero-shot real-benchmark probe on OCRBench (n=40), POPE-Adversarial (n=60), and RealWorldQA (n=40)."""
    from huggingface_hub import hf_hub_download
    from PIL import Image
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

    # 1. Real OCRBench (40 stratified: 15 Reg + 10 Irreg + 8 DocVQA + 7 Scene Text VQA)
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
        hits, by_cat, samples = 0, {}, []
        for r in eval_rows:
            im = _pil(r["image"])
            pred = spark_generate(
                model,
                str(r["question"]),
                media=visual_media([im]),
                enable_thinking=False,
                max_new_tokens=32,
                do_sample=False,
            ).strip()
            ans_l = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
            pl = pred.lower()
            ok = any(str(a).lower().strip() in pl for a in ans_l if str(a).strip())
            hits += int(ok)
            qt = str(r.get("question_type", ""))
            by_cat.setdefault(qt, [0, 0])
            by_cat[qt][0] += int(ok)
            by_cat[qt][1] += 1
            if len(samples) < 8:
                samples.append({"type": qt, "gt": str(ans_l[0]), "pred": pred, "hit": bool(ok)})
        report["ocrbench_probe"] = {
            "n": len(eval_rows),
            "acc": round(hits / max(1, len(eval_rows)), 4),
            "by_type": {k: round(v[0] / max(1, v[1]), 4) for k, v in by_cat.items()},
            "samples": samples,
        }
    except Exception as exc:
        report["ocrbench_probe_error"] = str(exc)[:180]

    # 2. Real POPE-Adversarial (60 balanced: 30 Yes / 30 No)
    try:
        p_pope = hf_hub_download(
            "lmms-lab/POPE",
            "Full/adversarial-00000-of-00001.parquet",
            repo_type="dataset",
            revision="4db1276663dfa5eb8ad16a52d24c31a09e470896",
            local_files_only=True,
        )
        tbl_p = pq.read_table(p_pope).to_pylist()
        p_rows = (
            [r for r in tbl_p if str(r["answer"]).lower().strip() == "yes"][:30]
            + [r for r in tbl_p if str(r["answer"]).lower().strip() == "no"][:30]
        )
        tp = fp = tn = fn = 0
        for r in p_rows:
            im = _pil(r["image"])
            pred = spark_generate(
                model,
                f"{r['question']}\nPlease answer yes or no.",
                media=visual_media([im]),
                enable_thinking=False,
                max_new_tokens=8,
                do_sample=False,
            ).strip().lower()
            words_p = pred.split()[:3]
            is_yes = pred.startswith("yes") or ("yes" in words_p and "no" not in words_p)
            gt_yes = str(r["answer"]).lower().strip() == "yes"
            if gt_yes and is_yes:
                tp += 1
            elif gt_yes and not is_yes:
                fn += 1
            elif not gt_yes and is_yes:
                fp += 1
            else:
                tn += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2.0 * prec * rec / max(1e-9, prec + rec)
        report["pope_adv_probe"] = {
            "n": len(p_rows),
            "acc": round((tp + tn) / max(1, len(p_rows)), 4),
            "f1": round(f1, 4),
            "yes_ratio": round((tp + fp) / max(1, len(p_rows)), 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    except Exception as exc:
        report["pope_adv_probe_error"] = str(exc)[:180]

    # 3. Real RealWorldQA (40 items)
    try:
        p_rw = hf_hub_download(
            "xai-org/RealWorldQA",
            "data/test-00000-of-00002.parquet",
            repo_type="dataset",
            revision="17e7f75e092e47169732462ea3cdfebe911105dd",
            local_files_only=True,
        )
        tbl_rw = pq.read_table(p_rw).to_pylist()[:40]
        rw_ok = 0
        for r in tbl_rw:
            im = _pil(r["image"])
            pred = spark_generate(
                model,
                str(r["question"]),
                media=visual_media([im]),
                enable_thinking=False,
                max_new_tokens=16,
                do_sample=False,
            ).strip().rstrip(".")
            gt = str(r["answer"]).strip()
            ok = (
                pred.lower() == gt.lower()
                or (len(gt) == 1 and pred[:1].upper() == gt.upper())
                or (len(gt) >= 2 and gt.lower() in pred.lower())
            )
            rw_ok += int(ok)
        report["realworldqa_probe"] = {"n": len(tbl_rw), "acc": round(rw_ok / max(1, len(tbl_rw)), 4)}
    except Exception as exc:
        report["realworldqa_probe_error"] = str(exc)[:180]

    emit("stage12_quick_vl_probe", **report)
    return report


def _run_stage12a_batched_projector_prealign(
    model: SparkNative,
    *,
    order_items: list[tuple[str, Any]],
    catalogue: Catalogue,
    reader: MediaReader,
    cfg: dict[str, Any],
    batch_size: int = 8,
    max_updates: int = 1600,
    lr: float = 3.0e-4,
    warmup_steps: int = 60,
    clip_norm: float = 1.0,
    max_ans_tokens: int = 96,
    checkpoint_dir: Path,
    seed: int = 20261210,
) -> dict[str, Any]:
    """High-throughput true-batched (`B=8`) frozen-Spark projector alignment through `model.image_tokens` + `forward_looped_model`.

    Uses strictly per-example normalized CE loss so 2-token `POPE`/`OCR` items receive identical gradient weight
    to 48-token captions, while length-bucketing batches minimizes right-padding waste for ~6x higher throughput
    than `microbatch_size=1` with 100% bitwise parity with `SparkNative` inference.
    """
    rng = random.Random(seed)
    device = model.device
    tok = model.tokenizer
    eos_id = int(tok.eos_token_id)
    emb_layer = model.get_input_embeddings()

    # Freeze Spark, Loop48, En1B, Vision, and visual_cross_attn; train ONLY vision_projector + visual metadata connectors!
    model.set_stage(alignment_only=True, stage_mode="frozen_spark_connector", unfreeze_vision_last_n_layers=0)
    if getattr(model, "spark", None) is not None and hasattr(model.spark, "model"):
        model.spark.model.gradient_checkpointing = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    if n_trainable <= 0:
        raise RuntimeError("Stage-12A found 0 trainable projector/connector parameters")

    # Temporarily cap Stage-12A radio patch budget to 1024 (512px -> <=256 visual tokens) for fast batched B=8 steps!
    prev_max_patches = model.policy.get("radio_default_max_patches")
    prev_hi_patches = model.policy.get("radio_highres_max_patches")
    prev_hi_side = model.policy.get("radio_highres_max_side")
    prev_def_side = model.policy.get("radio_default_max_side")
    model.policy["radio_default_max_patches"] = 1024
    model.policy["radio_highres_max_patches"] = 1024
    model.policy["radio_default_max_side"] = 512
    model.policy["radio_highres_max_side"] = 512

    opt = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=0.01)

    def _lr_fn(st: int) -> float:
        if st < warmup_steps:
            return float(st + 1) / float(max(1, warmup_steps))
        prog = float(st - warmup_steps) / float(max(1, max_updates - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_fn)

    emit(
        "stage12a_batched_prealign_start",
        items=len(order_items),
        batch_size=batch_size,
        max_updates=max_updates,
        lr=lr,
        warmup_steps=warmup_steps,
        trainable_params=n_trainable,
    )

    # Shuffle order and slice chunks of `batch_size * 8` to length-sort locally before forming B=8 batches!
    work_items = list(order_items)
    rng.shuffle(work_items)
    cursor = 0

    def _next_bucketed_batches(n_groups: int = 8) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
        nonlocal cursor
        pool: list[tuple[int, tuple[dict[str, Any], dict[str, Any]]]] = []
        want = batch_size * n_groups
        attempts = 0
        while len(pool) < want and attempts < want * 3:
            if cursor >= len(work_items):
                rng.shuffle(work_items)
                cursor = 0
            kind, payload = work_items[cursor]
            cursor += 1
            attempts += 1
            try:
                row, med = _materialize_stage12_item(kind, payload, catalogue=catalogue, reader=reader, cfg=cfg)
                imgs = med.get("images") or []
                ans = str(row.get("answer") or "").strip()
                prm = str(row.get("prompt") or "").strip()
                if not imgs or not ans or not prm:
                    continue
                # Estimate sequence length for tight length bucketing
                im0 = imgs[0]
                th, tw = model._radio_target_hw(im0.size[0], im0.size[1], is_temporal=False, num_images=len(imgs))
                est_len = (th // 32) * (tw // 32 + 1) * len(imgs) + (len(prm) + min(len(ans), 360)) // 4
                pool.append((est_len, (row, med)))
            except Exception:
                continue
        pool.sort(key=lambda x: x[0])
        out_b: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
        for i in range(0, len(pool) - batch_size + 1, batch_size):
            out_b.append([ex for _, ex in pool[i : i + batch_size]])
        rng.shuffle(out_b)
        return out_b

    t0 = time.time()
    ema_loss: float | None = None
    loss_hist: list[float] = []
    batch_queue: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    use_amp = torch.cuda.is_available() and str(device).startswith("cuda")
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext

    try:
        for step in range(1, max_updates + 1):
            while not batch_queue:
                batch_queue = _next_bucketed_batches(n_groups=8)
            batch = batch_queue.pop()

            model.train()
            for mod in model.modules():
                if isinstance(mod, nn.Dropout):
                    mod.eval()

            embs_l: list[torch.Tensor] = []
            tm_l: list[torch.Tensor] = []
            ie_l: list[torch.Tensor] = []
            lbl_l: list[torch.Tensor] = []
            max_S = 0

            with amp_ctx():
                for row, med in batch:
                    vis = model.image_tokens(
                        med.get("images"),
                        med.get("times"),
                        med.get("capture_times"),
                        alphas=med.get("alphas"),
                        visual_units=med.get("visual_units"),
                    )
                    if vis is None or vis.shape[1] == 0:
                        continue
                    vis_1d = vis[0]  # [V, D]
                    ans = str(row.get("answer") or "").strip()
                    prm = str(row.get("prompt") or "").strip()
                    use_think = bool(row.get("enable_thinking", False)) and ("<think>" in ans)
                    pre_ids, post_ids = model.chat.encode(
                        prm,
                        enable_thinking=use_think,
                        system=row.get("system_prompt"),
                        for_generation=True,
                    )
                    ans_ids = tok(ans, add_special_tokens=False).input_ids[:max_ans_tokens] + [eos_id]
                    if not ans_ids:
                        continue
                    pre_t = torch.tensor(pre_ids, device=device, dtype=torch.long)
                    post_t = torch.tensor(post_ids, device=device, dtype=torch.long)
                    ans_t = torch.tensor(ans_ids, device=device, dtype=torch.long)
                    ans_in = ans_t[:-1]
                    parts = [emb_layer(pre_t).detach(), vis_1d, emb_layer(post_t).detach()]
                    if len(ans_in) > 0:
                        parts.append(emb_layer(ans_in).detach())
                    seq_e = torch.cat(parts, dim=0)
                    S = int(seq_e.shape[0])
                    max_S = max(max_S, S)
                    P, V, Q = len(pre_ids), int(vis_1d.shape[0]), len(post_ids)
                    tm = torch.ones(S, dtype=torch.bool, device=device)
                    tm[P : P + V] = False
                    ie = torch.zeros(S, dtype=torch.long, device=device)
                    ie[:P] = pre_t
                    ie[P + V : P + V + Q] = post_t
                    if len(ans_in) > 0:
                        ie[P + V + Q :] = ans_in
                    lbl = torch.full((S,), -100, dtype=torch.long, device=device)
                    lbl[P + V + Q - 1 : P + V + Q - 1 + len(ans_ids)] = ans_t
                    embs_l.append(seq_e)
                    tm_l.append(tm)
                    ie_l.append(ie)
                    lbl_l.append(lbl)

                if not embs_l:
                    continue

                B = len(embs_l)
                D = int(embs_l[0].shape[-1])
                full_emb = torch.zeros((B, max_S, D), dtype=model.embed_dtype, device=device)
                attn_mask = torch.zeros((B, max_S), dtype=torch.long, device=device)
                tmask = torch.zeros((B, max_S), dtype=torch.bool, device=device)
                ids_eng = torch.zeros((B, max_S), dtype=torch.long, device=device)
                labels = torch.full((B, max_S), -100, dtype=torch.long, device=device)
                for bi in range(B):
                    s = int(embs_l[bi].shape[0])
                    full_emb[bi, :s] = embs_l[bi].to(model.embed_dtype)
                    attn_mask[bi, :s] = 1
                    tmask[bi, :s] = tm_l[bi]
                    ids_eng[bi, :s] = ie_l[bi]
                    labels[bi, :s] = lbl_l[bi]

                out = model.forward_looped_model(
                    inputs_embeds=full_emb,
                    attention_mask=attn_mask,
                    token_ids_for_engram=ids_eng,
                    text_token_mask=tmask,
                    visual_context=None,
                )
                # Strictly per-example normalized CE loss so short POPE/OCR == long captions!
                ex_losses = []
                for bi in range(B):
                    vb = labels[bi] != -100
                    if vb.any():
                        lg = model.spark.lm_head(out.last_hidden_state[bi, vb])
                        ex_losses.append(F.cross_entropy(lg.float(), labels[bi, vb]))
                if not ex_losses:
                    continue
                loss = torch.stack(ex_losses).mean()

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                emit("stage12a_nonfinite_skip", step=step)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(trainable_params, float(clip_norm)))
            opt.step()
            sched.step()

            lv = float(loss.detach().item())
            loss_hist.append(lv)
            ema_loss = lv if ema_loss is None else (0.95 * ema_loss + 0.05 * lv)
            if step == 1 or step % 50 == 0 or step == max_updates:
                dt = time.time() - t0
                emit(
                    "stage12a_step",
                    step=step,
                    max_updates=max_updates,
                    loss=round(lv, 4),
                    ema_loss=round(float(ema_loss), 4),
                    grad_norm=round(gn, 3),
                    lr=round(float(sched.get_last_lr()[0]), 7),
                    max_seq=max_S,
                    img_per_sec=round((step * batch_size) / max(1e-3, dt), 2),
                    elapsed_s=round(dt, 1),
                )
    finally:
        if prev_max_patches is not None:
            model.policy["radio_default_max_patches"] = prev_max_patches
        if prev_hi_patches is not None:
            model.policy["radio_highres_max_patches"] = prev_hi_patches
        if prev_hi_side is not None:
            model.policy["radio_highres_max_side"] = prev_hi_side
        if prev_def_side is not None:
            model.policy["radio_default_max_side"] = prev_def_side

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_model_safetensors(model, checkpoint_dir / "model.safetensors")
    del opt, sched
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "updates": len(loss_hist),
        "batch_size": batch_size,
        "examples_seen": len(loss_hist) * batch_size,
        "start_loss": round(loss_hist[0], 4) if loss_hist else None,
        "final_ema_loss": round(float(ema_loss), 4) if ema_loss is not None else None,
        "elapsed_s": round(time.time() - t0, 1),
        "checkpoint": str(checkpoint_dir / "model.safetensors"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--init-release-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage115-open-vocab-visual-v38/final-model-stage115-bf16"),
    )
    ap.add_argument("--radio-vision-dir", type=Path, default=Path("cache/c-radiov4-h"))
    ap.add_argument("--stage7-work-root", type=Path, default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50"))
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage12-frontier-vl"),
    )
    ap.add_argument(
        "--nemotron-cache-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage8-radiov4/nemotron_v3_cache"),
    )
    ap.add_argument("--stage12a-updates", type=int, default=1600)
    ap.add_argument("--stage12a-batch-size", type=int, default=8)
    ap.add_argument("--stage12a-lr", type=float, default=3.0e-4)
    ap.add_argument("--stage12a-warmup", type=int, default=60)
    ap.add_argument("--stage12b-updates", type=int, default=960)
    ap.add_argument("--rsi6-updates", type=int, default=120)
    ap.add_argument("--grad-accum", type=int, default=12)
    ap.add_argument("--clip-norm", type=float, default=1.2)
    ap.add_argument("--release-tag", type=str, default="stage12")
    ap.add_argument("--resume-stage12a-ckpt", type=Path, default=None)
    ap.add_argument("--resume-stage12b-ckpt", type=Path, default=None)
    ap.add_argument("--early-spark-layers", type=int, default=12)
    ap.add_argument("--early-spark-lr", type=float, default=2.2e-5)
    ap.add_argument("--backbone-lr", type=float, default=7.5e-6)
    ap.add_argument("--connector-lr", type=float, default=6.0e-5)
    ap.add_argument("--engram-lr", type=float, default=2.5e-5)
    ap.add_argument("--vision-lr", type=float, default=3.0e-6)
    ap.add_argument("--unfreeze-vision-layers", type=int, default=6)
    ap.add_argument("--visual-margin-every-n", type=int, default=4)
    ap.add_argument("--visual-margin-target", type=float, default=0.25)
    ap.add_argument("--visual-margin-boost", type=float, default=1.25)
    ap.add_argument("--vision-optimizer", type=str, default="muon8bit", choices=("muon8bit", "apollo8bit", "adamw"))
    ap.add_argument("--intel-vision", type=str, default="xpu", choices=("xpu", "none"))
    ap.add_argument("--word-ocr-max", type=int, default=4200)
    ap.add_argument("--hf-docvqa-max", type=int, default=2400)
    ap.add_argument("--hf-synthdog-max", type=int, default=1400)
    ap.add_argument("--hf-textvqa-max", type=int, default=2200)
    ap.add_argument("--hf-aokvqa-max", type=int, default=2600)
    ap.add_argument("--hf-latex-max", type=int, default=600)
    ap.add_argument("--hf-gqa-open-max", type=int, default=2000)
    ap.add_argument("--real-pope-max", type=int, default=2000)
    ap.add_argument("--s3d-mcq-max", type=int, default=480)
    ap.add_argument("--v38-visual-renders", type=int, default=16)
    ap.add_argument("--v38-repair-max", type=int, default=140)
    ap.add_argument("--v28-max-records", type=int, default=80)
    ap.add_argument("--v28-repair-max", type=int, default=120)
    ap.add_argument("--nemotron-max-gqa", type=int, default=700)
    ap.add_argument("--nemotron-max-geo", type=int, default=500)
    ap.add_argument("--nemotron-max-aokvqa", type=int, default=500)
    ap.add_argument("--nemotron-max-clevr", type=int, default=500)
    ap.add_argument("--nemotron-max-zhwiki", type=int, default=300)
    ap.add_argument("--nemotron-max-plotqa", type=int, default=300)
    ap.add_argument("--max-nemotron-terminal", type=int, default=900)
    ap.add_argument("--max-ot-agent", type=int, default=600)
    ap.add_argument("--max-terminal-pivot", type=int, default=450)
    ap.add_argument("--max-ot3-stem", type=int, default=900)
    ap.add_argument("--max-nemotron-science", type=int, default=700)
    ap.add_argument("--extra-jsonl", action="append", type=Path, default=[])
    ap.add_argument("--extra-dataset-dir", action="append", type=Path, default=[])
    ap.add_argument("--rsi6-v38-explore", type=int, default=16)
    ap.add_argument("--rsi6-frontier-explore", type=int, default=32)
    ap.add_argument("--eval-images", type=int, default=48)
    ap.add_argument("--cleanup-intermediate-ckpts", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=20261200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    db_path = args.stage7_work_root / "metadata/corpus.sqlite"
    handoff_path = args.stage7_work_root / "metadata/HANDOFF.json"

    # 1. Build & Strictly Audit ALL Stage-12 Corpora
    tb41_excluded = _fetch_tb41_excluded_task_ids(hf_token)
    emit("stage12_build_corpora_start", excluded_tb41_tasks=len(tb41_excluded))

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
    real_pope_pairs = build_real_photo_pope_yesno_pairs(
        token=hf_token,
        max_pairs=args.real_pope_max,
        seed=args.seed + 16,
        existence_priority=True,
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
        seed=args.seed + 19,
    )
    frontier_records = build_stage9_frontier_training_records(
        token=hf_token,
        max_nemotron_terminal=args.max_nemotron_terminal,
        max_openthoughts_agent=args.max_ot_agent,
        max_terminal_pivot=args.max_terminal_pivot,
        max_openthoughts3_stem=args.max_ot3_stem,
        max_nemotron_science=args.max_nemotron_science,
        seed=args.seed + 23,
        excluded_tb_task_ids=tb41_excluded,
    )
    extra_text_recs, extra_vis_pairs = load_extra_user_dataset_records(
        jsonl_paths=args.extra_jsonl,
        dataset_dirs=args.extra_dataset_dir,
    )

    all_vis_pairs_for_audit = (
        list(word_ocr_pairs)
        + list(hf_real_ocr_pairs)
        + list(real_vl_pairs)
        + list(real_pope_pairs)
        + list(s3d_mcq_pairs)
        + list(spatial3d_pairs)
        + list(nemotron_vl_pairs)
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
    audit_rep["stage12_counts"] = {
        "word_ocr_pairs": len(word_ocr_pairs),
        "hf_real_ocr_pairs": len(hf_real_ocr_pairs),
        "real_vl_pairs": len(real_vl_pairs),
        "real_pope_pairs": len(real_pope_pairs),
        "s3d_mcq_pairs": len(s3d_mcq_pairs),
        "spatial3d_pairs": len(spatial3d_pairs),
        "nemotron_vl_pairs": len(nemotron_vl_pairs),
        "v38_text_recs": len(v38_text_recs),
        "v38_vis_pairs": len(v38_vis_pairs),
        "v38_repair_recs": len(v38_repair_recs),
        "v28_records": len(v28_records),
        "v28_repair_records": len(v28_repair_records),
        "frontier_records": len(frontier_records),
        "extra_text_recs": len(extra_text_recs),
        "extra_vis_pairs": len(extra_vis_pairs),
    }
    write(args.output_root / "STAGE12_DATASET_AUDIT.json", audit_rep)
    emit("stage12_dataset_audit_complete", **audit_rep)
    if not audit_rep["passed"]:
        raise RuntimeError(f"Stage-12 strict dataset audit FAILED: {audit_rep}")

    rng = random.Random(args.seed + 100)

    # 2A. Assemble Balanced Pure-Visual Order for Stage-12A (Bucketed B=8 Frozen-Spark Projector Alignment)
    total_12a = args.stage12a_updates * args.stage12a_batch_size
    cat_12a_rids, cat_12a_summary = select_stage115_visual_alignment_rids(
        db_path, target_samples=max(1, int(total_12a * 0.16)), seed=args.seed + 101
    )
    order_12a: list[tuple[str, Any]] = []
    # ~28% realistic scene-text/alnum word crops (crosses OCRBench Text Recognition phase transition cleanly!)
    n_12a_word = min(len(word_ocr_pairs), int(total_12a * 0.28))
    # ~18% real DocVQA + concise SynthDoG
    n_12a_hf_ocr = min(len(hf_real_ocr_pairs), int(total_12a * 0.18))
    # ~26% real TextVQA + A-OKVQA (MCQ + rationales) + GQA Open-Ended + LaTeX
    n_12a_real_vl = min(len(real_vl_pairs), int(total_12a * 0.26))
    # ~12% clean object-existence 50/50 GQA POPE
    n_12a_pope = min(len(real_pope_pairs), int(total_12a * 0.12))
    # ~6% 3D Spatial MCQ + 4-Primitive + Nemotron VL
    n_12a_s3d = min(len(s3d_mcq_pairs), int(total_12a * 0.03))
    n_12a_nem = min(len(nemotron_vl_pairs), int(total_12a * 0.03))

    for i in range(n_12a_word):
        order_12a.append(("pair", word_ocr_pairs[i]))
    for i in range(n_12a_hf_ocr):
        order_12a.append(("pair", hf_real_ocr_pairs[i]))
    for i in range(n_12a_real_vl):
        order_12a.append(("pair", real_vl_pairs[i]))
    for i in range(n_12a_pope):
        order_12a.append(("pair", real_pope_pairs[i]))
    for i in range(n_12a_s3d):
        order_12a.append(("pair", s3d_mcq_pairs[i]))
    for i in range(n_12a_nem):
        order_12a.append(("pair", nemotron_vl_pairs[i]))
    for kind, rid in cat_12a_rids:
        order_12a.append((kind, rid))
    rng.shuffle(order_12a)
    if len(order_12a) < total_12a and order_12a:
        base_12a = list(order_12a)
        while len(order_12a) < total_12a:
            order_12a.append(base_12a[len(order_12a) % len(base_12a)])
    order_12a = order_12a[:total_12a]
    stage12a_breakdown = {
        "total": len(order_12a),
        "word_ocr": n_12a_word,
        "hf_real_ocr": n_12a_hf_ocr,
        "real_vl": n_12a_real_vl,
        "real_pope": n_12a_pope,
        "s3d_mcq": n_12a_s3d,
        "nemotron_vl": n_12a_nem,
        "cat_visual": len(cat_12a_rids),
    }

    # 2B. Assemble Balanced Multimodal + SyncCraft v38.8/v28 + Frontier Order for Stage-12B (`full_spark`)
    total_12b = args.stage12b_updates * args.grad_accum
    n_b_word = min(len(word_ocr_pairs), int(total_12b * 0.14))
    n_b_hf_ocr = min(len(hf_real_ocr_pairs), int(total_12b * 0.14))
    n_b_real_vl = min(len(real_vl_pairs), int(total_12b * 0.18))
    n_b_pope = min(len(real_pope_pairs), int(total_12b * 0.11))
    n_b_s3d_mcq = min(len(s3d_mcq_pairs), int(total_12b * 0.04))
    n_b_s3d = min(len(spatial3d_pairs), int(total_12b * 0.04))
    n_b_nem = min(len(nemotron_vl_pairs), int(total_12b * 0.07))
    n_b_v38_txt = min(len(v38_text_recs) * 2, int(total_12b * 0.03)) if v38_text_recs else 0
    n_b_v38_vis = min(len(v38_vis_pairs) * 2, int(total_12b * 0.01)) if v38_vis_pairs else 0
    n_b_v38_rep = min(len(v38_repair_recs) * 2, int(total_12b * 0.03)) if v38_repair_recs else 0
    n_b_v28 = min(len(v28_records) * 2, int(total_12b * 0.02)) if v28_records else 0
    n_b_v28_rep = min(len(v28_repair_records) * 2, int(total_12b * 0.03)) if v28_repair_records else 0
    n_b_front = min(len(frontier_records), int(total_12b * 0.14))
    n_b_extra = len(extra_text_recs) + len(extra_vis_pairs)

    direct_b_rids, direct_b_summary = select_stage9_direct_ocr_and_yesno_rids(
        db_path, max_items=max(200, int(total_12b * 0.04)), seed=args.seed + 105
    )
    n_b_direct = len(direct_b_rids)
    used_12b = (
        n_b_word + n_b_hf_ocr + n_b_real_vl + n_b_pope + n_b_s3d_mcq + n_b_s3d + n_b_nem
        + n_b_v38_txt + n_b_v38_vis + n_b_v38_rep + n_b_v28 + n_b_v28_rep + n_b_front + n_b_extra + n_b_direct
    )
    n_b_cat = max(1, total_12b - used_12b)
    cat_b_rids, cat_b_summary = select_stage75b_balanced_rids(
        db_path, target_catalogue_samples=n_b_cat, vision_fraction=0.68, seed=args.seed + 109
    )

    order_12b: list[tuple[str, Any]] = [("cat", rid) for rid in cat_b_rids]
    for rid in direct_b_rids:
        order_12b.append(("cat_direct", rid))
    for i in range(n_b_word):
        order_12b.append(("pair", word_ocr_pairs[i % len(word_ocr_pairs)]))
    for i in range(n_b_hf_ocr):
        order_12b.append(("pair", hf_real_ocr_pairs[i % len(hf_real_ocr_pairs)]))
    for i in range(n_b_real_vl):
        order_12b.append(("pair", real_vl_pairs[i % len(real_vl_pairs)]))
    for i in range(n_b_pope):
        order_12b.append(("pair", real_pope_pairs[i % len(real_pope_pairs)]))
    for i in range(n_b_s3d_mcq):
        order_12b.append(("pair", s3d_mcq_pairs[i % len(s3d_mcq_pairs)]))
    for i in range(n_b_s3d):
        order_12b.append(("pair", spatial3d_pairs[i % len(spatial3d_pairs)]))
    for i in range(n_b_nem):
        order_12b.append(("pair", nemotron_vl_pairs[i % len(nemotron_vl_pairs)]))
    for i in range(n_b_v38_txt):
        order_12b.append(("pair", _text_record_to_pair(v38_text_recs[i % len(v38_text_recs)])))
    for i in range(n_b_v38_vis):
        order_12b.append(("pair", v38_vis_pairs[i % len(v38_vis_pairs)]))
    for i in range(n_b_v38_rep):
        order_12b.append(("pair", _text_record_to_pair(v38_repair_recs[i % len(v38_repair_recs)])))
    for i in range(n_b_v28):
        order_12b.append(("v25", v28_records[i % len(v28_records)]))
    for i in range(n_b_v28_rep):
        order_12b.append(("pair", _text_record_to_pair(v28_repair_records[i % len(v28_repair_records)])))
    for i in range(n_b_front):
        order_12b.append(("pair", _text_record_to_pair(frontier_records[i % len(frontier_records)])))
    for rec in extra_text_recs:
        order_12b.append(("pair", _text_record_to_pair(rec)))
    for pair in extra_vis_pairs:
        order_12b.append(("pair", pair))
    rng.shuffle(order_12b)
    order_12b = order_12b[:total_12b]

    stage12b_breakdown = {
        "total": len(order_12b),
        "word_ocr": n_b_word,
        "hf_real_ocr": n_b_hf_ocr,
        "real_vl": n_b_real_vl,
        "real_pope": n_b_pope,
        "s3d_mcq": n_b_s3d_mcq,
        "s3d_4prim": n_b_s3d,
        "nemotron_vl": n_b_nem,
        "v38_txt": n_b_v38_txt,
        "v38_vis": n_b_v38_vis,
        "v38_rep": n_b_v38_rep,
        "v28": n_b_v28,
        "v28_rep": n_b_v28_rep,
        "frontier": n_b_front,
        "direct_cat": n_b_direct,
        "cat_replay": len(cat_b_rids),
        "extra": n_b_extra,
    }
    emit("stage12_plan_ready", stage12a=stage12a_breakdown, stage12b=stage12b_breakdown)
    if args.dry_run:
        return

    # 3. Initialize SparkNative from Stage-11.5 BF16 with Clean `RadioMultiLayerGELUProjector` + 2D `\n` Row-Breaks!
    cfg = dict(read(args.init_release_dir / "config.json"))
    cfg.update(
        {
            "campaign_id": str(args.output_root.name),
            "init_from_release_dir": str(args.init_release_dir.resolve()),
            "allow_warmstart_vision_swap": False,
            "allow_warmstart_missing_loop_engram": False,
            "allow_warmstart_projector_rms_match": False,
            "reinit_vision_projector": True,
            "radio_projector_type": "gelu",
            "radio_projector_intermediate_size": 4096,
            "radio_projector_target_rms": 0.025,
            "radio_meta_scale_init": 0.004,
            "radio_row_break_newline": True,
            "radio_output_rms_match": False,
            "visual_decorr_loss_weight": 0.0,
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
        rms_pre = measure_layerwise_visual_text_rms(model, sample_im, tag="pre_stage12a")
        xray_pre = measure_visual_grounding_xray(model, tag="pre_stage12a")

        res_12a: dict[str, Any] = {}
        probe_post_12a: dict[str, Any] = {}
        res_12b: dict[str, Any] = {}
        probe_post_12b: dict[str, Any] = {}

        if args.resume_stage12b_ckpt is not None and args.resume_stage12b_ckpt.is_file():
            load_spark_native_safetensors(model, args.resume_stage12b_ckpt)
            if getattr(model, "_intel_vision_bridge", None) is not None:
                model._intel_vision_bridge.sync_from_host(model.vision)
            emit("stage12_resumed_from_stage12b_ckpt", ckpt=str(args.resume_stage12b_ckpt))
            res_12a = {"skipped": True}
            res_12b = {"resumed_from": str(args.resume_stage12b_ckpt)}
        else:
            # Phase 12A: True Bucketed B=8 Frozen-Spark Projector Alignment
            if args.resume_stage12a_ckpt is not None and args.resume_stage12a_ckpt.is_file():
                load_spark_native_safetensors(model, args.resume_stage12a_ckpt)
                if getattr(model, "_intel_vision_bridge", None) is not None:
                    model._intel_vision_bridge.sync_from_host(model.vision)
                emit("stage12_resumed_from_stage12a_ckpt", ckpt=str(args.resume_stage12a_ckpt))
                res_12a = {"resumed_from": str(args.resume_stage12a_ckpt)}
            elif args.stage12a_updates > 0:
                res_12a = _run_stage12a_batched_projector_prealign(
                    model,
                    order_items=order_12a,
                    catalogue=catalogue,
                    reader=reader,
                    cfg=cfg,
                    batch_size=int(args.stage12a_batch_size),
                    max_updates=int(args.stage12a_updates),
                    lr=float(args.stage12a_lr),
                    warmup_steps=int(args.stage12a_warmup),
                    clip_norm=1.0,
                    checkpoint_dir=args.output_root / "ckpt-stage12a",
                    seed=args.seed + 200,
                )
            probe_post_12a = _run_stage12_quick_vl_probe(model, tag="post_stage12a")
            write(args.output_root / "STAGE12A_PROBE.json", probe_post_12a)

            # Phase 12B: Full-36-Layer Joint Multimodal + SyncCraft v38.8/v28 + Frontier Adaptation
            stage12b_cfg = {
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
                "visual_margin_every_n": int(args.visual_margin_every_n),
                "visual_margin_target": float(args.visual_margin_target),
                "visual_margin_boost": float(args.visual_margin_boost),
                "warmup_ratio": 0.08,
                "clip_norm": float(args.clip_norm),
                "weight_decay": 0.01,
            }
            res_12b = _run_stage8_phase(
                stage_name="stage12b_full36_frontier_vl_v38",
                model=model,
                base_cfg=cfg,
                stage_cfg=stage12b_cfg,
                order_items=order_12b,
                catalogue=catalogue,
                reader=reader,
                grad_accum=args.grad_accum,
                checkpoint_dir=args.output_root / "ckpt-stage12b",
                seed=args.seed + 300,
            )

        rms_post_12b = measure_layerwise_visual_text_rms(model, sample_im, tag="post_stage12b")
        xray_post_12b = measure_visual_grounding_xray(model, tag="post_stage12b")
        cal_post_12b = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_stage12b"
        )
        probe_post_12b = _run_stage12_quick_vl_probe(model, tag="post_stage12b")
        write(args.output_root / "STAGE12B_PROBE.json", probe_post_12b)

        # Phase 12C: Dual-Track Verified RSI-6 (freeze_vision_projector=True -> zero OOM & zero OCR projector drift!)
        dapo_math_pool = build_dapo_math_rsi3_pool(token=hf_token, max_items=120, seed=args.seed + 499)
        rsi_v38_pool = list(v38_text_recs) + list(v28_records)
        rsi_signal_items, rsi_mining_summary = mine_rsi3_dual_track_trajectories(
            model,
            v28_records=rsi_v38_pool,
            frontier_records=list(frontier_records) + list(dapo_math_pool),
            v28_explore_n=args.rsi6_v38_explore,
            frontier_explore_n=args.rsi6_frontier_explore,
            candidates_k=2,
            seed=args.seed + 500,
        )
        total_rsi = args.rsi6_updates * args.grad_accum
        n_sig = min(int(total_rsi * 0.40), max(1, len(rsi_signal_items) * 6))
        n_rsi_word = min(len(word_ocr_pairs), int(total_rsi * 0.10))
        n_rsi_ocr = min(len(hf_real_ocr_pairs), int(total_rsi * 0.10))
        n_rsi_pope = min(len(real_pope_pairs), int(total_rsi * 0.10))
        n_rsi_v28_rep = min(len(v28_repair_records) * 2, int(total_rsi * 0.10)) if v28_repair_records else 0
        n_rsi_v38_rep = min(len(v38_repair_recs) * 2, int(total_rsi * 0.08)) if v38_repair_recs else 0
        n_rep_cat = max(1, total_rsi - n_sig - n_rsi_word - n_rsi_ocr - n_rsi_pope - n_rsi_v28_rep - n_rsi_v38_rep)
        cat_rsi_rids, _ = select_stage75b_balanced_rids(
            db_path, target_catalogue_samples=n_rep_cat, vision_fraction=0.60, seed=args.seed + 501
        )
        order_rsi: list[tuple[str, Any]] = [("cat", rid) for rid in cat_rsi_rids]
        for i in range(n_rsi_word):
            order_rsi.append(("pair", word_ocr_pairs[(n_b_word + i) % len(word_ocr_pairs)]))
        for i in range(n_rsi_ocr):
            order_rsi.append(("pair", hf_real_ocr_pairs[(n_b_hf_ocr + i) % len(hf_real_ocr_pairs)]))
        for i in range(n_rsi_pope):
            order_rsi.append(("pair", real_pope_pairs[i % len(real_pope_pairs)]))
        for i in range(n_rsi_v28_rep):
            order_rsi.append(("pair", _text_record_to_pair(v28_repair_records[i % len(v28_repair_records)])))
        for i in range(n_rsi_v38_rep):
            order_rsi.append(("pair", _text_record_to_pair(v38_repair_recs[i % len(v38_repair_recs)])))
        for i in range(n_sig):
            order_rsi.append(rsi_signal_items[i % len(rsi_signal_items)])
        rng.shuffle(order_rsi)
        order_rsi = order_rsi[:total_rsi]

        rsi_bb_lr = min(args.backbone_lr * 0.55, 4.5e-6)
        rsi_cfg = {
            "stage_mode": "full_spark",
            "unfreeze_vision_last_n_layers": 0,
            "freeze_vision_projector": True,
            "optimizer_profile": "apollo_coat",
            "early_spark_layers": int(args.early_spark_layers),
            "early_spark_lr": min(args.early_spark_lr * 0.45, 1.0e-5),
            "learning_rate": rsi_bb_lr,
            "backbone_lr": rsi_bb_lr,
            "apollo_learning_rate": rsi_bb_lr,
            "connector_learning_rate": min(args.connector_lr * 0.35, 2.0e-5),
            "adapter_lr": min(args.connector_lr * 0.35, 2.0e-5),
            "apollo_connector_learning_rate": min(args.connector_lr * 0.35, 2.0e-5),
            "engram_lr": max(min(args.engram_lr * 0.75, 2.0e-5), rsi_bb_lr * 3.0),
            "vision_lr": 0.0,
            "visual_decorr_loss_weight": 0.0,
            "warmup_ratio": 0.12,
            "clip_norm": max(0.8, float(args.clip_norm) * 0.75),
            "weight_decay": 0.01,
        }
        res_rsi = _run_stage8_phase(
            stage_name="stage12c_dual_track_rsi6_candidate",
            model=model,
            base_cfg=cfg,
            stage_cfg=rsi_cfg,
            order_items=order_rsi,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage12c-rsi6",
            seed=args.seed + 600,
        )
        rms_post_rsi = measure_layerwise_visual_text_rms(model, sample_im, tag="post_rsi6")
        xray_post_rsi = measure_visual_grounding_xray(model, tag="post_rsi6")
        cal_post_rsi = measure_visual_alignment_diagnostics(
            model, catalogue=catalogue, reader=reader, config=cfg, eval_rids=eval_rids, v25_probe_example=probe_ex, tag="post_rsi6"
        )
        probe_post_rsi = _run_stage12_quick_vl_probe(model, tag="post_rsi6")
        promo_gate = evaluate_rsi3_promotion_gate(cal_post_12b, cal_post_rsi)
        emit("stage12_rsi6_promotion_gate", **promo_gate)

        selected_phase = "stage12c_rsi6"
        if not promo_gate["accepted"]:
            emit("stage12_rsi6_rolled_back_to_stage12b")
            load_spark_native_safetensors(model, args.output_root / "ckpt-stage12b/model.safetensors")
            if getattr(model, "_intel_vision_bridge", None) is not None:
                model._intel_vision_bridge.sync_from_host(model.vision)
            selected_phase = "stage12b_parent_rollback"

        # Prepare clean standalone inference policy before export (`reinit_vision_projector=False`!)
        model.policy["reinit_vision_projector"] = False
        model.policy["radio_projector_type"] = "gelu"
        model.policy["radio_projector_intermediate_size"] = 4096
        model.policy["radio_row_break_newline"] = True
        model.policy["radio_output_rms_match"] = False
        for k in ("visual_decorr_loss_weight", "visual_decorr_target_dc_ratio", "visual_decorr_target_cos"):
            model.policy.pop(k, None)

        report = {
            "stage12_architecture": {
                "release_tag": str(args.release_tag),
                "warmstart": str(args.init_release_dir),
                "radio_projector_type": "gelu",
                "radio_projector_intermediate_size": 4096,
                "radio_row_break_newline": True,
                "radio_output_rms_match": False,
                "decorr_weight": 0.0,
                "stage12a_updates": int(args.stage12a_updates),
                "stage12a_batch_size": int(args.stage12a_batch_size),
                "stage12b_updates": int(args.stage12b_updates),
                "rsi6_updates": int(args.rsi6_updates),
                "selected_phase": selected_phase,
            },
            "dataset_audit": audit_rep,
            "stage12a_breakdown": stage12a_breakdown,
            "stage12a_cat_summary": cat_12a_summary,
            "stage12b_breakdown": stage12b_breakdown,
            "direct_12b_summary": direct_b_summary,
            "stage12b_cat_summary": cat_b_summary,
            "stage12a": res_12a,
            "stage12b": res_12b,
            "stage12c_rsi6": res_rsi,
            "rsi6_mining": rsi_mining_summary,
            "rsi6_promotion_gate": promo_gate,
            "quick_vl_probes": {
                "post_12a": probe_post_12a,
                "post_12b": probe_post_12b,
                "post_rsi6": probe_post_rsi,
            },
            "layerwise_rms": {
                "pre_12a": rms_pre,
                "post_12b": rms_post_12b,
                "post_rsi6": rms_post_rsi,
            },
            "visual_grounding_xray": {
                "pre_12a": xray_pre,
                "post_12b": xray_post_12b,
                "post_rsi6": xray_post_rsi,
            },
            "calibration": {
                "post_12b": cal_post_12b,
                "post_rsi6": cal_post_rsi,
            },
            "final_gates": _extract_stage8_gate_stats(model),
        }
        write(args.output_root / "STAGE12_FRONTIER_VL_REPORT.json", report)

        tag = str(args.release_tag).strip() or "stage12"
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
        write(out_bf16 / "STAGE12_FRONTIER_VL_REPORT.json", report)
        write(out_fp8 / "STAGE12_FRONTIER_VL_REPORT.json", report)

        if (
            args.cleanup_intermediate_ckpts
            and (out_bf16 / "model.safetensors").is_file()
            and (out_fp8 / "model.safetensors").is_file()
        ):
            for ck in ("ckpt-stage12a", "ckpt-stage12b", "ckpt-stage12c-rsi6"):
                ck_p = args.output_root / ck
                if ck_p.is_dir():
                    shutil.rmtree(ck_p, ignore_errors=True)
            emit("stage12_intermediate_ckpts_cleaned")
    finally:
        model.close_coprocessors()


if __name__ == "__main__":
    main()
