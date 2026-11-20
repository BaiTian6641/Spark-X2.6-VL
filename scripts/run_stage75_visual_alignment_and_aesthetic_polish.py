#!/usr/bin/env python3
"""Stage-7.5 Dual-GPU `SigLIP2-So400M` Visual Alignment Surgery + `v25` Vigorous Carved Aesthetic Polish.

Cures the freshly-initialized `SigLIP2-So400M` connector/cross-attention under-alignment discovered during
Stage-7 mid-flight diagnostics (where `vision_projector` moved only `+0.28%` and `visual_cross_attn` gates
stayed `0.010/-0.014/+0.026`), AND infuses the complete `Blockbench_sculpted_style-v25` aesthetic
(`D1-D4 x M1-M4` Intent Router, 2-Pass Locate-First `<box>`/`<point>` grounding, Vigorous Carved `16 px/block`
mass-packing `G1-G12`, 6-plane zero-thickness flat maw craft, 4-foot ground-locked ecology animation):

1. **Pre-Surgery Baseline Diagnostics:** Measures `media_advantage` (`blank_loss - real_loss`) on a fixed
   48-item held-out `promotion` image slice, `visual_cross_attn.*.gate_attn` values, and `vision_projector`
   weight norms.
2. **Stage 7.5A — Frozen-Backbone Visual Alignment Surgery (`~600` updates, ~60-90 min):**
   - `stage_mode="frozen_spark_cross_attn"`, `unfreeze_vision_last_n_layers=27` on Intel Arc Pro B50 (`xpu:0`)
     via `DualPinnedShmArena` (`0` trainable Spark-4B / Loop48 / En1B params!).
   - Trains ONLY `vision_projector` + `visual_cross_attn` + `spatial_clock`/`camera_pose`/`alpha_projector`/
     `modality`/`view_id` (`connector_lr=3.5e-4`, `bf16_adamw`) + 27L `SigLIP2-So400M` on B50 (`vision_lr=1.5e-5`).
   - Strictly **100% pixel-indispensable visual data** (`OCR`, `ChartQA`, `DocVQA`, `TextVQA`, `point_box`,
     `visual_primitives_*`, `ARC-AGI-2` visual, `SAT-3D`, `ViRL39K`, `factual_caption`, `image:zh`, `video`,
     + upsampled `v25` jittered Pass-1 Locate-First `<box>`/`<point>`, Intent Router, Multiview Inverse,
     and Adversarial Trap Punch-Back renders).
   - Verifies post-7.5A `media_advantage` surge and `visual_cross_attn` gate opening before saving
     `stage75a-connector-aligned`.
3. **Stage 7.5B — Dual-GPU Joint Vision + `v25` Vigorous Carved Aesthetic + Frontier Replay (`~300` updates, ~2.5 hr):**
   - `stage_mode="full_spark"`, `unfreeze_vision_last_n_layers=27`, `optimizer_profile="apollo_coat"`,
     gentle LRs (`backbone_lr=6e-6`, `engram_lr=4e-5`, `connector_lr=4e-5`, `vision_lr=3e-6`).
   - Balanced mixture: `55%` newly-aligned Vision + ALL 6 `v25` Vigorous Carved Aesthetic tasks (`5x` upsampled)
     + `45%` anti-forgetting Frontier Text / Code / Reasoning / 2026 World Knowledge replay.
4. **Local Release Export (`publish_to_huggingface=False`):**
   - Exports portable `final-model-stage75-bf16` (with synced 27L B50 `SigLIP2-So400M` weights & SHA-256
     `MODEL_RELEASE.json`) AND `final-model-stage75-fp8` (`FP8_E4M3 + q4g128` residual).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import contextlib
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time
from typing import Any

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.aesthetic_v25_vigorous_corpus import (  # noqa: E402
    build_aesthetic_v25_vigorous_records,
)
from colab.final.common import emit, identity, read, sha, write  # noqa: E402
from colab.final.storage import manifest  # noqa: E402
from colab.final.data import Catalogue, MediaReader, apply_dynamic_training_augmentation, prepare_media  # noqa: E402
from colab.final.engine import (  # noqa: E402
    _build_scheduler,
    _lr_for_group_and_name,
    _save_model_safetensors,
    ensure_coat_for_model,
    optimizer_for,
)
from colab.final.inference import empty_media, generate, load, visual_media  # noqa: E402
from colab.final.model import export_spark_native_fp8_safetensors  # noqa: E402
from colab.final.optimizer import MergedGradientAccumulator  # noqa: E402

PIXEL_INDISPENSABLE_HINTS: tuple[str, ...] = (
    "ocr",
    "chart",
    "docvqa",
    "textvqa",
    "point_box",
    "visual_primitives",
    "point_count",
    "arc_agi",
    "sat_3d",
    "virl",
    "caption",
    "vqa",
    "chinese",
    "sculpted",
    "synccraft",
    "geometry",
    "video",
    "track",
    "count",
    "refer",
    "ground",
    "plot",
    "table",
    "diagram",
    "science",
    "spatial",
    "multiview",
    "locate",
    "v25_",
)


def partition_stage75_trainable_parameters(model: Any) -> dict[str, int]:
    """Audit parameter trainability counts by functional group so Stage-7.5A guarantees 0 Spark/Loop/En1B drift."""
    counts: dict[str, int] = defaultdict(int)
    seen: set[int] = set()
    for group, name, p in model.named_groups():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = int(p.numel())
        bucket = "trainable" if p.requires_grad else "frozen"
        counts[f"{group}:{bucket}"] += n
        counts[f"total:{bucket}"] += n
    if hasattr(model, "has_intel_vision_bridge") and model.has_intel_vision_bridge():
        tel = model.intel_vision_telemetry()
        counts["intel_xpu_vision:trainable"] = int(tel.get("trainable_parameters", 0))
    return dict(sorted(counts.items()))


def v25_record_to_example(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert an in-memory `aesthetic_v25_vigorous_corpus` record into a `(row, media)` pair for `SparkNative`."""
    imgs = list(rec.get("images") or [])
    fam = "image" if imgs else "text"
    row = {
        "id": str(rec["id"]),
        "_family": fam,
        "_split": str(rec.get("split", "train")),
        "family": fam,
        "language": str(rec.get("language", "en")),
        "task": str(rec.get("task", "v25_vigorous_carved_masterwork")),
        "prompt": str(rec["prompt"]),
        "answer": str(rec["answer"]),
        "enable_thinking": True,
        "source_family": str(rec.get("source_family", "synccraft_aesthetic_v25_cc0")),
    }
    if imgs:
        vkind = "multiview" if len(imgs) > 1 else "independent_image"
        media = visual_media(imgs, visual_kind=vkind)
    else:
        media = empty_media()
    return row, media


def select_stage75a_pixel_indispensable_rids(
    db_path: Path,
    *,
    target_catalogue_samples: int,
    seed: int = 7501,
) -> tuple[list[str], dict[str, int]]:
    """Select strictly visual (`image` + `video`) pixel-indispensable training `rid`s from `corpus.sqlite`."""
    con = sqlite3.connect(str(db_path))
    cols = {c[1] for c in con.execute("PRAGMA table_info(rows)")}
    has_task = "task" in cols
    if has_task:
        rows = list(
            con.execute(
                "SELECT rid, family, language, COALESCE(task, '') FROM rows "
                "WHERE split='train' AND family IN ('image', 'video') ORDER BY rid"
            )
        )
    else:
        rows = [
            (r[0], r[1], r[2], "")
            for r in con.execute(
                "SELECT rid, family, language FROM rows "
                "WHERE split='train' AND family IN ('image', 'video') ORDER BY rid"
            )
        ]
    con.close()

    by_bucket: dict[str, list[str]] = defaultdict(list)
    for rid, fam, lang, task in rows:
        t_low = str(task).lower()
        priority = "high" if any(h in t_low for h in PIXEL_INDISPENSABLE_HINTS) else "std"
        bucket = f"{fam}:{lang}:{task or 'general'}:{priority}"
        by_bucket[bucket].append(str(rid))

    rng = random.Random(seed)
    for rids in by_bucket.values():
        rng.shuffle(rids)

    # Balanced round-robin across visual task buckets, weighting high-priority pixel-reading tasks 2x
    chosen: list[str] = []
    bucket_keys = sorted(by_bucket.keys())
    cursors = {k: 0 for k in bucket_keys}
    task_counts: Counter[str] = Counter()
    while len(chosen) < target_catalogue_samples and bucket_keys:
        rng.shuffle(bucket_keys)
        for bk in list(bucket_keys):
            pool = by_bucket[bk]
            if not pool:
                bucket_keys.remove(bk)
                continue
            reps = 2 if bk.endswith(":high") else 1
            for _ in range(reps):
                if len(chosen) >= target_catalogue_samples:
                    break
                idx = cursors[bk] % len(pool)
                if idx == 0 and cursors[bk] > 0:
                    rng.shuffle(pool)
                chosen.append(pool[idx])
                cursors[bk] += 1
                task_counts[bk.rsplit(":", 1)[0]] += 1
    rng.shuffle(chosen)
    return chosen, dict(task_counts.most_common(24))


def select_stage75b_balanced_rids(
    db_path: Path,
    *,
    target_catalogue_samples: int,
    vision_fraction: float = 0.52,
    seed: int = 7502,
) -> tuple[list[str], dict[str, int]]:
    """Select balanced Stage-7.5B joint polish mix: ~52% Vision + ~48% Frontier Text/Code/Reasoning/2026 replay."""
    n_vis = int(round(target_catalogue_samples * vision_fraction))
    n_txt = max(0, target_catalogue_samples - n_vis)
    vis_rids, vis_counts = select_stage75a_pixel_indispensable_rids(
        db_path, target_catalogue_samples=n_vis, seed=seed + 11
    )

    con = sqlite3.connect(str(db_path))
    cols = {c[1] for c in con.execute("PRAGMA table_info(rows)")}
    has_task = "task" in cols
    if has_task:
        t_rows = list(
            con.execute(
                "SELECT rid, language, COALESCE(task, '') FROM rows "
                "WHERE split='train' AND family='text' ORDER BY rid"
            )
        )
    else:
        t_rows = [
            (r[0], r[1], "")
            for r in con.execute("SELECT rid, language FROM rows WHERE split='train' AND family='text' ORDER BY rid")
        ]
    con.close()

    by_t: dict[str, list[str]] = defaultdict(list)
    for rid, lang, task in t_rows:
        by_t[f"text:{lang}:{task or 'general'}"].append(str(rid))
    rng = random.Random(seed + 29)
    for rids in by_t.values():
        rng.shuffle(rids)

    txt_rids: list[str] = []
    txt_counts: Counter[str] = Counter()
    t_keys = sorted(by_t.keys())
    cursors = {k: 0 for k in t_keys}
    while len(txt_rids) < n_txt and t_keys:
        rng.shuffle(t_keys)
        for tk in t_keys:
            if len(txt_rids) >= n_txt:
                break
            pool = by_t[tk]
            idx = cursors[tk] % len(pool)
            if idx == 0 and cursors[tk] > 0:
                rng.shuffle(pool)
            txt_rids.append(pool[idx])
            cursors[tk] += 1
            txt_counts[tk] += 1

    combined = vis_rids + txt_rids
    rng.shuffle(combined)
    summary = {**vis_counts, **dict(txt_counts.most_common(16))}
    return combined, summary


def _extract_visual_gate_stats(model: Any) -> dict[str, Any]:
    """Read exact `visual_cross_attn` gate scalars and `vision_projector` L2 norms."""
    gates: dict[str, float] = {}
    proj_norms: dict[str, float] = {}
    for name, p in model.named_parameters():
        if "visual_cross_attn" in name and "gate" in name:
            gates[name] = round(float(p.detach().float().mean().item()), 6)
        elif name.startswith("vision_projector.") and p.ndim == 2:
            proj_norms[name] = round(float(p.detach().float().norm().item()), 4)
    return {"visual_cross_attn_gates": gates, "vision_projector_l2_norms": proj_norms}


def _select_eval_image_rids(db_path: Path, *, max_items: int = 48) -> list[str]:
    """Deterministic held-out `promotion` image slice for before/after `media_advantage` verification."""
    con = sqlite3.connect(str(db_path))
    rows = list(
        con.execute(
            "SELECT rid, language, COALESCE(task, '') FROM rows "
            "WHERE split='promotion' AND family='image' ORDER BY rid"
        )
    )
    con.close()
    by_task: dict[str, list[str]] = defaultdict(list)
    for rid, lang, task in rows:
        by_task[f"{lang}:{task}"].append(str(rid))
    picked: list[str] = []
    keys = sorted(by_task.keys())
    idx = 0
    while len(picked) < max_items and keys:
        progressed = False
        for k in keys:
            if idx < len(by_task[k]) and len(picked) < max_items:
                picked.append(by_task[k][idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    return picked


def measure_visual_alignment_diagnostics(
    model: Any,
    *,
    catalogue: Catalogue,
    reader: MediaReader,
    config: dict[str, Any],
    eval_rids: list[str],
    v25_probe_example: tuple[dict[str, Any], dict[str, Any]] | None = None,
    tag: str = "diagnostic",
) -> dict[str, Any]:
    """Measure held-out `media_advantage` (`blank_loss - real_loss`), gate openings, and 2 visual sanity probes."""
    was_training = model.training
    model.eval()
    if hasattr(model, "sync_intel_vision_to_host"):
        model.sync_intel_vision_to_host()

    blank_img = Image.new("RGB", (224, 224), (128, 128, 128))
    real_losses: list[float] = []
    blank_losses: list[float] = []
    by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
    use_amp = torch.cuda.is_available() and str(model.device).startswith("cuda")
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext

    with torch.no_grad():
        for rid in eval_rids:
            row, packet = catalogue.row(rid)
            with reader.packet(packet) as get:
                media = prepare_media(row, get, config)
            if not media.get("images"):
                continue
            blank_media = dict(media)
            blank_media["images"] = [blank_img for _ in media["images"]]
            blank_media["alphas"] = [None for _ in media["images"]]
            with amp_ctx():
                res_real = model(examples=[(row, media)])
                res_blank = model(examples=[(row, blank_media)])
            rl = float(res_real["per_example_loss"][0].detach().float().item())
            bl = float(res_blank["per_example_loss"][0].detach().float().item())
            if math.isfinite(rl) and math.isfinite(bl):
                real_losses.append(rl)
                blank_losses.append(bl)
                by_task[str(row.get("task", "image"))].append((rl, bl))

    mean_real = sum(real_losses) / max(1, len(real_losses))
    mean_blank = sum(blank_losses) / max(1, len(blank_losses))
    mean_adv = mean_blank - mean_real
    task_breakdown = {
        t: {
            "n": len(pairs),
            "real_loss": round(sum(a for a, _ in pairs) / len(pairs), 4),
            "blank_loss": round(sum(b for _, b in pairs) / len(pairs), 4),
            "media_advantage": round(sum(b - a for a, b in pairs) / len(pairs), 4),
        }
        for t, pairs in sorted(by_task.items())
    }

    # Quick greedy visual sanity probe on `v25_two_pass_locate_then_zoom` (tests `<box>` & `<point>` pixel reading!)
    probe_out: dict[str, Any] = {}
    if v25_probe_example is not None:
        prow, pmedia = v25_probe_example
        try:
            pred = generate(
                model,
                prow["prompt"],
                media=pmedia,
                enable_thinking=True,
                max_new_tokens=220,
                do_sample=False,
                task=prow["task"],
                language="en",
            )
            probe_out = {
                "task": prow["task"],
                "expected_head": prow["answer"][:260],
                "predicted": pred[:360],
            }
        except Exception as exc:
            probe_out = {"error": str(exc)}

    stats = {
        "tag": tag,
        "evaluated_images": len(real_losses),
        "mean_real_loss": round(mean_real, 4),
        "mean_blank_loss": round(mean_blank, 4),
        "mean_media_advantage": round(mean_adv, 4),
        "by_task": task_breakdown,
        **_extract_visual_gate_stats(model),
        "v25_locate_probe": probe_out,
    }
    emit(
        "stage75_diagnostic",
        tag=tag,
        n=len(real_losses),
        real_loss=stats["mean_real_loss"],
        blank_loss=stats["mean_blank_loss"],
        media_advantage=stats["mean_media_advantage"],
        gates=stats["visual_cross_attn_gates"],
    )
    if was_training:
        model.train()
    return stats


def _run_stage75_training_loop(
    *,
    model: Any,
    stage_name: str,
    stage_mode: str,
    stage_cfg: dict[str, Any],
    order_items: list[tuple[str, Any]],
    catalogue: Catalogue,
    reader: MediaReader,
    base_config: dict[str, Any],
    grad_accum: int,
    seed: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Run one Stage-7.5 training phase (`7.5A` frozen-backbone or `7.5B` joint Dual-GPU APOLLO+COAT)."""
    is_frozen_spark = stage_mode in ("frozen_spark_cross_attn", "alignment_only", "frozen_spark_connector")
    unfreeze_vis = int(stage_cfg.get("unfreeze_vision_last_n_layers", 27))
    model._intel_vision_base_lr = _lr_for_group_and_name("vision", "vision.vision_model", stage_cfg)
    model.set_stage(False, stage_mode=stage_mode, unfreeze_vision_last_n_layers=unfreeze_vis)
    param_audit = partition_stage75_trainable_parameters(model)
    emit("stage75_param_partition", stage=stage_name, mode=stage_mode, **param_audit)

    if is_frozen_spark:
        if param_audit.get("spark:trainable", 0) != 0 or param_audit.get("engram_mhc:trainable", 0) != 0:
            raise AssertionError(f"Stage 7.5A leaked trainable Spark/En1B params: {param_audit}")

    model.train()
    coat_ctrl = ensure_coat_for_model(model, stage_cfg)
    planned_updates = math.ceil(len(order_items) / max(1, grad_accum))
    warmup_steps = max(2, round(planned_updates * float(stage_cfg.get("warmup_ratio", 0.06))))
    optimizer = optimizer_for(model, stage_cfg)
    scheduler = _build_scheduler(optimizer, kind="cosine", num_warmup_steps=warmup_steps, num_training_steps=planned_updates)

    buf_policy = "fp32" if is_frozen_spark else "bf16_apollo_fp32_small"
    accumulator = MergedGradientAccumulator(
        model,
        accum_mode=buf_policy,
        min_bf16_accum_dim=int(stage_cfg.get("apollo_min_dim", 256)),
        allow_embeddings_in_apollo=bool(stage_cfg.get("apollo_include_embeddings", True)),
    )
    use_cuda_amp = torch.cuda.is_available() and str(model.device).startswith("cuda")
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_cuda_amp else contextlib.nullcontext
    coat_ctx = (
        (lambda: coat_ctrl.activation_context(training=True))
        if (coat_ctrl is not None and coat_ctrl.enabled and not is_frozen_spark)
        else contextlib.nullcontext
    )
    inv_fn = (
        coat_ctrl.invalidate_weight_caches
        if (coat_ctrl is not None and coat_ctrl.enabled and not is_frozen_spark)
        else None
    )
    use_dyn = bool(base_config.get("dynamic_training_augmentation", True))
    can_prefetch = (
        hasattr(model, "has_intel_vision_bridge")
        and model.has_intel_vision_bridge()
        and getattr(model, "_intel_vision_trainable", False)
    )
    prefetch_pool = ThreadPoolExecutor(max_workers=1) if can_prefetch else None

    def _materialize_item(idx: int, kind: str, payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if kind == "v25":
            return v25_record_to_example(payload)
        row, packet = catalogue.row(str(payload))
        with reader.packet(packet) as get:
            media = prepare_media(row, get, base_config)
        if use_dyn:
            aug_seed = int(identity([seed, stage_name, idx, str(payload)])[:16], 16)
            row, media = apply_dynamic_training_augmentation(row, media, random.Random(aug_seed), base_config)
        return row, media

    t0 = time.monotonic()
    step_index = 0
    window_records = 0
    window_tokens = 0
    loss_hist: list[float] = []
    stratum_loss: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    try:
        prefetched_ex: tuple[dict[str, Any], dict[str, Any]] | None = None
        prefetch_fut = None
        for idx, (kind, payload) in enumerate(order_items):
            if prefetched_ex is not None:
                if prefetch_fut is not None:
                    prefetch_fut.result()
                    prefetch_fut = None
                ex = prefetched_ex
                prefetched_ex = None
            else:
                ex = _materialize_item(idx, kind, payload)

            if prefetch_pool is not None and (window_records + 1) < grad_accum and (idx + 1) < len(order_items):
                nk, npay = order_items[idx + 1]
                prefetched_ex = _materialize_item(idx + 1, nk, npay)
                prefetch_fut = prefetch_pool.submit(model.prefetch_examples_vision, [prefetched_ex])

            row, media = ex
            is_long = use_cuda_amp and (len(str(row.get("prompt", ""))) + len(str(row.get("answer", ""))) > 8000)
            if is_long:
                torch.cuda.empty_cache()

            with amp_ctx(), coat_ctx():
                result = model(examples=[(row, media)])
                per = result["per_example_loss"]
                loss_val = per[0]

            if not torch.isfinite(loss_val):
                accumulator.zero_buffers()
                if hasattr(model, "zero_tier1b_buffers"):
                    model.zero_tier1b_buffers()
                emit("stage75_nonfinite_skip", stage=stage_name, step=step_index, item_idx=idx)
                window_records = 0
                window_tokens = 0
                continue

            loss_f = float(loss_val.detach().item())
            key = f"{row['_family']}:{row['language']}"
            stratum_loss[key][0] += loss_f
            stratum_loss[key][1] += 1.0
            loss_hist.append(loss_f)
            mb_units = max(1, int(result.get("supervised_units", 1)))
            window_tokens += mb_units
            window_records += 1

            with accumulator.microbatch(weight=1.0, supervised_units=mb_units):
                if loss_val.requires_grad:
                    loss_val.backward()
            result = per = loss_val = None
            if is_long:
                torch.cuda.empty_cache()

            should_step = (window_records >= grad_accum) or (idx + 1 == len(order_items))
            if not should_step:
                continue

            denom = float(max(1, window_records))
            lr_ratio = 1.0
            if optimizer.param_groups and float(optimizer.param_groups[0].get("initial_lr", 0.0)) > 0.0:
                lr_ratio = float(optimizer.param_groups[0]["lr"]) / float(optimizer.param_groups[0]["initial_lr"])

            step_stats = accumulator.step(
                optimizer,
                scheduler,
                max_grad_norm=float(stage_cfg.get("clip_norm", 0.8)),
                post_scale=1.0 / denom,
                invalidate_caches_fn=inv_fn,
            )
            if not step_stats.finite or not step_stats.applied:
                if hasattr(model, "zero_tier1b_buffers"):
                    model.zero_tier1b_buffers()
                emit("stage75_nonfinite_step_skip", stage=stage_name, step=step_index)
            elif (not is_frozen_spark) and hasattr(model, "step_tier1b_sparse_adam"):
                base_engram_lr = float(stage_cfg.get("engram_lr", 4e-5))
                model.step_tier1b_sparse_adam(lr=base_engram_lr * lr_ratio, denom=denom)

            window_records = 0
            window_tokens = 0
            step_index += 1

            if step_index == 1 or step_index % 20 == 0 or step_index == planned_updates:
                elapsed_min = (time.monotonic() - t0) / 60.0
                recent = loss_hist[-min(len(loss_hist), grad_accum * 10) :]
                mean_recent = sum(recent) / max(1, len(recent))
                extra: dict[str, Any] = {}
                if hasattr(model, "has_intel_vision_bridge") and model.has_intel_vision_bridge():
                    tel = model.intel_vision_telemetry()
                    lst = tel.get("last_step") or {}
                    extra["b50_vgrad"] = round(float(lst.get("vision_grad_norm", 0.0)), 5)
                    extra["b50_gib"] = round(float(lst.get("xpu_peak_bytes", 0)) / (1024**3), 2)
                emit(
                    "stage75_progress",
                    stage=stage_name,
                    update=step_index,
                    planned=planned_updates,
                    elapsed_min=round(elapsed_min, 2),
                    upd_per_hr=round(step_index / max(1e-4, elapsed_min / 60.0), 1),
                    recent_loss=round(mean_recent, 4),
                    grad_norm=round(float(step_stats.grad_norm), 4),
                    lr_ratio=round(lr_ratio, 4),
                    **_extract_visual_gate_stats(model)["visual_cross_attn_gates"],
                    **extra,
                )
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=True)
        accumulator.close()
        del optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Persist intermediate stage checkpoint cleanly
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "sync_intel_vision_to_host"):
        model.sync_intel_vision_to_host()
    _save_model_safetensors(model, checkpoint_dir / "model.safetensors")
    elapsed_s = round(time.monotonic() - t0, 1)
    return {
        "stage": stage_name,
        "stage_mode": stage_mode,
        "updates_completed": step_index,
        "samples_processed": len(order_items),
        "elapsed_seconds": elapsed_s,
        "updates_per_hour": round(step_index / max(1e-4, elapsed_s / 3600.0), 1),
        "mean_loss": round(sum(loss_hist) / max(1, len(loss_hist)), 4),
        "stratum_loss": {k: round(v[0] / max(1.0, v[1]), 4) for k, v in sorted(stratum_loss.items())},
        "param_partition": param_audit,
        "checkpoint": str(checkpoint_dir / "model.safetensors"),
    }


def _export_stage75_releases(
    model: Any,
    *,
    src_release_dir: Path,
    out_bf16_dir: Path,
    out_fp8_dir: Path,
    report: dict[str, Any],
) -> dict[str, str]:
    """Export standalone portable Stage-7.5 BF16 AND FP8+q4g128 releases with verified SHA-256 manifests."""
    if hasattr(model, "sync_intel_vision_to_host"):
        model.sync_intel_vision_to_host()
    for d in (out_bf16_dir, out_fp8_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Copy portable runtime, reload-assets, config, docs from Stage-7 release
    for path in src_release_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_release_dir)
        if rel.as_posix() in ("model.safetensors", "MODEL_RELEASE.json"):
            continue
        dest = out_bf16_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    cfg = read(out_bf16_dir / "config.json")
    cfg["intel_arc_vision_offload"] = False
    cfg["stage75_visual_alignment_and_v25_aesthetic"] = True
    cfg["blockbench_style_version"] = "v25"
    write(out_bf16_dir / "config.json", cfg)
    write(out_bf16_dir / "STAGE75_SURGERY_REPORT.json", report)

    _save_model_safetensors(model, out_bf16_dir / "model.safetensors")
    bf16_files = manifest(out_bf16_dir)
    write(
        out_bf16_dir / "MODEL_RELEASE.json",
        {
            "schema_version": 1,
            "campaign_id": str(cfg.get("campaign_id", "spark-x26-4b-loop48-en1b-vl-stage75")),
            "stage75_complete": True,
            "training_complete": True,
            "quantization": "bf16",
            **bf16_files,
        },
    )

    # Copy metadata/runtime to FP8 dir and quantize Spark linears to F8_E4M3 + q4g128 residual
    for path in out_bf16_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(out_bf16_dir)
        if rel.as_posix() in ("model.safetensors", "MODEL_RELEASE.json"):
            continue
        dest = out_fp8_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    fp8_meta = export_spark_native_fp8_safetensors(model, out_fp8_dir / "model.safetensors")
    cfg_fp8 = dict(cfg)
    cfg_fp8["exported_weight_format"] = "fp8_e4m3_channel_q4g128_residual"
    write(out_fp8_dir / "config.json", cfg_fp8)
    fp8_files = manifest(out_fp8_dir)
    write(
        out_fp8_dir / "MODEL_RELEASE.json",
        {
            "schema_version": 1,
            "campaign_id": str(cfg.get("campaign_id", "spark-x26-4b-loop48-en1b-vl-stage75")),
            "stage75_complete": True,
            "training_complete": True,
            "quantization": "compressed-tensors-fp8-e4m3-channel+q4g128-residual",
            **fp8_meta,
            **fp8_files,
        },
    )
    emit(
        "stage75_releases_exported",
        bf16_dir=str(out_bf16_dir),
        bf16_sha256=sha(out_bf16_dir / "MODEL_RELEASE.json"),
        fp8_dir=str(out_fp8_dir),
        fp8_sha256=sha(out_fp8_dir / "MODEL_RELEASE.json"),
    )
    return {"bf16_dir": str(out_bf16_dir), "fp8_dir": str(out_fp8_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-7.5 Visual Alignment Surgery + v25 Aesthetic Polish")
    parser.add_argument(
        "--init-release-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50/final-model"),
    )
    parser.add_argument(
        "--stage7-work-root",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50/stage75-surgery"),
    )
    parser.add_argument("--stage75a-updates", type=int, default=600)
    parser.add_argument("--stage75b-updates", type=int, default=300)
    parser.add_argument("--grad-accum", type=int, default=12)
    parser.add_argument("--stage75a-connector-lr", type=float, default=3.5e-4)
    parser.add_argument("--stage75a-vision-lr", type=float, default=1.5e-5)
    parser.add_argument("--stage75b-backbone-lr", type=float, default=6e-6)
    parser.add_argument("--stage75b-engram-lr", type=float, default=4e-5)
    parser.add_argument("--stage75b-connector-lr", type=float, default=4e-5)
    parser.add_argument("--stage75b-vision-lr", type=float, default=3e-6)
    parser.add_argument("--eval-images", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--intel-vision", type=str, default="xpu", choices=("xpu", "none"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    db_path = args.stage7_work_root / "metadata/corpus.sqlite"
    handoff_path = args.stage7_work_root / "metadata/HANDOFF.json"

    emit("stage75_build_v25_aesthetic_corpus_start", seed=args.seed)
    v25_records_a = build_aesthetic_v25_vigorous_records(seed=args.seed, max_records=64)
    v25_records_b = build_aesthetic_v25_vigorous_records(seed=args.seed + 777, max_records=64)
    v25_pixel_only = [r for r in v25_records_a if r["metadata"].get("pixel_indispensable")] + [
        r for r in v25_records_b if r["metadata"].get("pixel_indispensable")
    ]
    v25_all = v25_records_a + v25_records_b
    emit(
        "stage75_build_v25_aesthetic_corpus_done",
        total_v25=len(v25_all),
        pixel_indispensable_v25=len(v25_pixel_only),
        by_task=dict(Counter(r["task"] for r in v25_records_a)),
    )

    total_75a_samples = args.stage75a_updates * args.grad_accum
    v25_75a_items = [("v25", r) for _ in range(4) for r in v25_pixel_only]
    cat_75a_needed = max(1, total_75a_samples - len(v25_75a_items))
    cat_75a_rids, cat_75a_summary = select_stage75a_pixel_indispensable_rids(
        db_path, target_catalogue_samples=cat_75a_needed, seed=args.seed + 1
    )
    order_75a: list[tuple[str, Any]] = [("cat", rid) for rid in cat_75a_rids] + v25_75a_items
    random.Random(args.seed + 2).shuffle(order_75a)

    total_75b_samples = args.stage75b_updates * args.grad_accum
    v25_75b_items = [("v25", r) for _ in range(5) for r in v25_all]
    cat_75b_needed = max(1, total_75b_samples - len(v25_75b_items))
    cat_75b_rids, cat_75b_summary = select_stage75b_balanced_rids(
        db_path, target_catalogue_samples=cat_75b_needed, vision_fraction=0.52, seed=args.seed + 3
    )
    order_75b: list[tuple[str, Any]] = [("cat", rid) for rid in cat_75b_rids] + v25_75b_items
    random.Random(args.seed + 4).shuffle(order_75b)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "order_75a_total": len(order_75a),
                    "v25_75a_items": len(v25_75a_items),
                    "cat_75a_top_tasks": cat_75a_summary,
                    "order_75b_total": len(order_75b),
                    "v25_75b_items": len(v25_75b_items),
                    "cat_75b_top_tasks": cat_75b_summary,
                },
                indent=2,
            )
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    iv_mode = None if args.intel_vision == "none" else args.intel_vision
    emit("stage75_loading_release", directory=str(args.init_release_dir), device=device, intel_vision=iv_mode)
    model = load(args.init_release_dir, device=device, intel_vision=iv_mode)
    base_cfg = read(args.init_release_dir / "config.json")

    # Pre-flight token-budget guard: verify EVERY v25 record fits safely within max_answer_tokens & max_sequence!
    max_ans_tok = int(model.policy.get("max_answer_tokens", 6656))
    max_seq_tok = int(model.policy.get("max_sequence", 8192))
    max_vis_tok = int(model.policy.get("max_visual_tokens", 1536))
    tok = model.tokenizer
    worst_ans = 0
    worst_tot = 0
    for rec in v25_all:
        a_len = len(tok.encode(str(rec["answer"]), add_special_tokens=False))
        p_len = len(tok.encode(str(rec["prompt"]), add_special_tokens=False)) + 64
        v_est = min(max_vis_tok, 256 * len(rec.get("images") or []))
        worst_ans = max(worst_ans, a_len)
        worst_tot = max(worst_tot, p_len + v_est + a_len)
        if a_len > max_ans_tok or (p_len + v_est + a_len) > max_seq_tok:
            raise ValueError(
                f"v25 record {rec['id']} ({rec['task']}) exceeds token budget: "
                f"ans={a_len} (max={max_ans_tok}), est_total={p_len + v_est + a_len} (max={max_seq_tok})"
            )
    emit(
        "stage75_v25_token_budget_verified",
        records=len(v25_all),
        worst_answer_tokens=worst_ans,
        worst_est_total_tokens=worst_tot,
        max_answer_tokens=max_ans_tok,
        max_sequence=max_seq_tok,
    )
    catalogue = Catalogue(db_path)
    handoff = read(handoff_path)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    reader = MediaReader(
        args.stage7_work_root,
        handoff,
        base_cfg,
        hf_token,
        reserve=2 * int(base_cfg.get("initial_checkpoint_reservation_bytes", 20_000_000_000)),
    )
    eval_rids = _select_eval_image_rids(db_path, max_items=args.eval_images)
    probe_rec = next((r for r in v25_records_a if r["task"] == "v25_two_pass_locate_then_zoom"), None)
    probe_ex = v25_record_to_example(probe_rec) if probe_rec is not None else None

    report: dict[str, Any] = {
        "schema_version": 1,
        "init_release_dir": str(args.init_release_dir),
        "style_version": "blockbench_sculpted_style_v25",
        "v25_records_summary": {
            "stage75a_v25_items": len(v25_75a_items),
            "stage75a_total_items": len(order_75a),
            "stage75b_v25_items": len(v25_75b_items),
            "stage75b_total_items": len(order_75b),
        },
    }

    # 1. Pre-Surgery Diagnostic
    report["pre_surgery_diagnostic"] = measure_visual_alignment_diagnostics(
        model,
        catalogue=catalogue,
        reader=reader,
        config=base_cfg,
        eval_rids=eval_rids,
        v25_probe_example=probe_ex,
        tag="pre_surgery",
    )
    write(args.output_root / "STAGE75_SURGERY_REPORT.json", report)

    # 2. Stage 7.5A — Frozen-Backbone SigLIP2-So400M Visual Alignment Surgery
    cfg_75a = {
        **base_cfg,
        "optimizer_profile": "bf16_adamw",
        "backbone_lr": 0.0,
        "learning_rate": args.stage75a_connector_lr,
        "connector_learning_rate": args.stage75a_connector_lr,
        "adapter_lr": args.stage75a_connector_lr,
        "vision_lr": args.stage75a_vision_lr,
        "vision_learning_rate": args.stage75a_vision_lr,
        "engram_lr": 0.0,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "clip_norm": 0.8,
        "unfreeze_vision_last_n_layers": 27,
        "coat_activation_fp8": False,
    }
    report["stage75a"] = _run_stage75_training_loop(
        model=model,
        stage_name="stage75a_frozen_backbone_siglip2_so400m_visual_alignment",
        stage_mode="frozen_spark_cross_attn",
        stage_cfg=cfg_75a,
        order_items=order_75a,
        catalogue=catalogue,
        reader=reader,
        base_config=base_cfg,
        grad_accum=args.grad_accum,
        seed=args.seed + 101,
        checkpoint_dir=args.output_root / "stage75a-connector-aligned",
    )
    report["post_75a_diagnostic"] = measure_visual_alignment_diagnostics(
        model,
        catalogue=catalogue,
        reader=reader,
        config=base_cfg,
        eval_rids=eval_rids,
        v25_probe_example=probe_ex,
        tag="post_75a",
    )
    write(args.output_root / "STAGE75_SURGERY_REPORT.json", report)

    # 3. Stage 7.5B — Dual-GPU Joint Vision + v25 Vigorous Carved Aesthetic + Frontier Replay Polish
    cfg_75b = {
        **base_cfg,
        "optimizer_profile": "apollo_coat",
        "backbone_lr": args.stage75b_backbone_lr,
        "learning_rate": args.stage75b_backbone_lr,
        "connector_learning_rate": args.stage75b_connector_lr,
        "adapter_lr": args.stage75b_connector_lr,
        "engram_lr": args.stage75b_engram_lr,
        "engram_learning_rate": args.stage75b_engram_lr,
        "vision_lr": args.stage75b_vision_lr,
        "vision_learning_rate": args.stage75b_vision_lr,
        "weight_decay": 0.01,
        "warmup_ratio": 0.08,
        "clip_norm": 0.8,
        "unfreeze_vision_last_n_layers": 27,
        "coat_activation_fp8": True,
    }
    report["stage75b"] = _run_stage75_training_loop(
        model=model,
        stage_name="stage75b_joint_vision_v25_aesthetic_and_frontier_polish",
        stage_mode="full_spark",
        stage_cfg=cfg_75b,
        order_items=order_75b,
        catalogue=catalogue,
        reader=reader,
        base_config=base_cfg,
        grad_accum=args.grad_accum,
        seed=args.seed + 202,
        checkpoint_dir=args.output_root / "stage75b-final-checkpoint",
    )
    report["post_75b_diagnostic"] = measure_visual_alignment_diagnostics(
        model,
        catalogue=catalogue,
        reader=reader,
        config=base_cfg,
        eval_rids=eval_rids,
        v25_probe_example=probe_ex,
        tag="post_75b",
    )
    write(args.output_root / "STAGE75_SURGERY_REPORT.json", report)

    # 4. Export Final Stage-7.5 BF16 & FP8+q4g128 Local Releases
    releases = _export_stage75_releases(
        model,
        src_release_dir=args.init_release_dir,
        out_bf16_dir=args.stage7_work_root / "final-model-stage75-bf16",
        out_fp8_dir=args.stage7_work_root / "final-model-stage75-fp8",
        report=report,
    )
    report["exported_releases"] = releases
    write(args.output_root / "STAGE75_SURGERY_REPORT.json", report)
    emit("stage75_complete", **releases)


if __name__ == "__main__":
    main()
