#!/usr/bin/env python3
"""Stage-8 Unified `nvidia/C-RADIOv4-H` (651.6M) Super-Encoder + Multi-Layer DeepStack Fusion
(`RMSNorm(L24) || RMSNorm(L32) -> 2x2 Pixel Shuffle -> SwiGLU MLP` + Early `L24` Cross-Attn Injection)
+ Adaptive Native-Aspect `256px-768px` (High-Res Single-Image Tier `1024px-1152px`, small crops `>= 384px`)
+ Shaded 3D 4-Primitive Visual Primitives (`<point>`, `<box>`, `<dash>`, `<circle>` + Front/Back + Left/Right)
+ `nvidia/Nemotron-Image-Training-v3` (`CC-BY-4.0`) Grounded Spatial & Geometric Reasoning.

Strictly enforces BOTH `C-RADIOv4-H` architectural safeguards:
1. **DINOv3-7B + SAM3 Prior Protection**:
   - Stage-8A (`frozen_spark_cross_attn`, `unfreeze_vision_last_n_layers=0`): `C-RADIOv4-H` is 100% FROZEN on
     Intel Arc Pro B50 (`xpu:0`), while `RadioMultiLayerSwiGLUProjector` + `visual_cross_attn` (with `wd=0.0`
     on all gates/RMSNorms!) align cleanly to `Spark-4B + Loop48 + En1B` warm-started from `final-model-stage75-bf16`.
   - Stage-8B (`full_spark`, `unfreeze_vision_last_n_layers=6`): ONLY the top 6 blocks (`blocks[26..31]` + `norm`,
     `118.06M` params) are unfrozen at ~10x lower LR (`vision_lr=2.5e-6`) using 8-bit `muon8bit` / `apollo8bit`
     on Intel Arc Pro B50 (`xpu:0`); lower 26 blocks (`0..25`) remain 100% frozen.
2. **Left/Right Orientation & Real 3D Model Understanding (NEVER black-and-white cutouts)**:
   - Combines full-color shaded 3D renders with `<point>`, `<box>`, `<dash>` (bone chains, facing arrows,
     `front_to_back` depth rays, 2-frame SE(3) animation arcs), `<circle>` joint pivots, balanced 50/50 Yes/No,
     and EN/ZH bilingual spatial QA + `nvidia/Nemotron-Image-Training-v3` spatial/geometric CoT
     (`clevr_1`, `zhwiki_1`, `plotqa_1`, `gqa_1`, `gqa_2`, `geometry3k`, `aokvqa_2`) + Stage-7 Visual Primitives
     + `v25` Vigorous Carved Masterworks + Frontier replay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import contextlib
import gc
import io
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

from colab.final.aesthetic_v25_vigorous_corpus import build_aesthetic_v25_vigorous_records  # noqa: E402
from colab.final.common import emit, read, sha, write  # noqa: E402
from colab.final.data import Catalogue, MediaReader, prepare_media  # noqa: E402
from colab.final.engine import (  # noqa: E402
    _build_scheduler,
    _lr_for_group_and_name,
    _save_model_safetensors,
    ensure_coat_for_model,
    optimizer_for,
)
from colab.final.inference import empty_media, visual_media  # noqa: E402
from colab.final.model import (  # noqa: E402
    RadioMultiLayerSwiGLUProjector,
    SparkNative,
    export_spark_native_fp8_safetensors,
)
from colab.final.optimizer import MergedGradientAccumulator  # noqa: E402
from colab.final.spatial_3d_primitives_corpus import (  # noqa: E402
    build_nemotron_v3_spatial_records,
    build_shaded_3d_spatial_primitives_records,
)
from colab.final.storage import manifest  # noqa: E402
from scripts.run_stage75_visual_alignment_and_aesthetic_polish import (  # noqa: E402
    _select_eval_image_rids,
    measure_visual_alignment_diagnostics,
    partition_stage75_trainable_parameters,
    select_stage75a_pixel_indispensable_rids,
    select_stage75b_balanced_rids,
    v25_record_to_example,
)


def _pair_to_example(pair: tuple[dict[str, Any], dict[str, bytes]]) -> tuple[dict[str, Any], dict[str, Any]]:
    row_raw, blobs = pair
    frames = list(row_raw.get("frames") or [])
    imgs: list[Image.Image] = []
    for mem in frames:
        b = blobs.get(mem)
        if b is not None:
            with Image.open(io.BytesIO(b)) as im:
                imgs.append(im.convert("RGB"))
    fam = "image" if imgs else "text"
    row = {
        "id": str(row_raw["id"]),
        "_family": fam,
        "_split": "train",
        "family": fam,
        "language": str(row_raw.get("language", "en")),
        "task": str(row_raw.get("task", "spatial_3d")),
        "prompt": str(row_raw["prompt"]),
        "answer": str(row_raw["answer"]),
        "enable_thinking": bool(row_raw.get("enable_thinking", True)),
        "source_family": str((row_raw.get("metadata") or {}).get("source", "stage8_spatial3d")),
    }
    if imgs:
        vkind = "multiview_render" if len(imgs) > 1 else "independent_image"
        media = visual_media(imgs, visual_kind=vkind)
        if row_raw.get("visual_units"):
            media["visual_units"] = row_raw["visual_units"]
    else:
        media = empty_media()
    return row, media


def _extract_stage8_gate_stats(model: Any) -> dict[str, Any]:
    gates: dict[str, float] = {}
    proj_norms: dict[str, float] = {}
    for name, p in model.named_parameters():
        if any(
            k in name
            for k in ("gate_attn", "gate_ffn", "early_mid_gate", "out_scale", "early_out_scale", "meta_scale")
        ):
            gates[name] = round(float(p.detach().float().mean().item()), 6)
        elif name.startswith("vision_projector.") and p.ndim == 2:
            proj_norms[name] = round(float(p.detach().float().norm().item()), 4)
    return {"gates": gates, "projector_l2_norms": proj_norms}


def _run_stage8_phase(
    *,
    stage_name: str,
    model: SparkNative,
    base_cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    order_items: list[tuple[str, Any]],
    catalogue: Catalogue,
    reader: MediaReader,
    grad_accum: int,
    checkpoint_dir: Path,
    seed: int,
) -> dict[str, Any]:
    t0 = time.monotonic()
    stage_mode = str(stage_cfg["stage_mode"])
    unfreeze_vis = int(stage_cfg.get("unfreeze_vision_last_n_layers", 0))
    is_frozen_spark = stage_mode in (
        "alignment_only",
        "frozen_spark_connector",
        "frozen_spark_cross_attn",
        "vision_late_unfreeze",
        "defrost_loop_engram",
    )
    eff_cfg = {**base_cfg, **stage_cfg}
    _vis_reg_keys = (
        "visual_decorr_loss_weight",
        "visual_decorr_target_dc_ratio",
        "visual_decorr_target_cos",
        "visual_rms_ceiling_weight",
        "visual_rms_ceiling_cap",
    )
    prev_decorr_keys = {k: model.policy.get(k) for k in _vis_reg_keys if k in model.policy}
    for k in _vis_reg_keys:
        if k in stage_cfg:
            model.policy[k] = stage_cfg[k]
    vis_margin_every_n = int(stage_cfg.get("visual_margin_every_n", 0) or 0)
    vis_margin_target = float(stage_cfg.get("visual_margin_target", 0.25))
    vis_margin_boost = float(stage_cfg.get("visual_margin_boost", 1.25))
    model._intel_vision_base_lr = _lr_for_group_and_name("vision", "vision.radio_model", eff_cfg)
    model.set_stage(
        alignment_only=is_frozen_spark,
        stage_mode=stage_mode,
        unfreeze_vision_last_n_layers=unfreeze_vis,
    )
    if bool(stage_cfg.get("freeze_vision_projector", False)) and getattr(model, "vision_projector", None) is not None:
        model.vision_projector.requires_grad_(False)
    model.train()

    param_audit = partition_stage75_trainable_parameters(model)
    if is_frozen_spark and param_audit.get("spark:trainable", 0) != 0:
        raise AssertionError(f"Expected frozen Spark in {stage_name}, got {param_audit}")
    if (not is_frozen_spark) and param_audit.get("spark:trainable", 0) == 0:
        raise AssertionError(f"Expected trainable Spark in {stage_name}, got {param_audit}")

    coat_ctrl = ensure_coat_for_model(model, eff_cfg)
    planned_updates = max(1, math.ceil(len(order_items) / max(1, grad_accum)))
    warmup_steps = max(1, round(planned_updates * float(stage_cfg.get("warmup_ratio", 0.08))))

    optimizer = optimizer_for(model, eff_cfg)
    scheduler = _build_scheduler(
        optimizer,
        kind="cosine",
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_updates,
    )
    buf_policy = "fp32" if is_frozen_spark else "bf16_apollo_fp32_small"
    accumulator = MergedGradientAccumulator(
        model,
        accum_mode=buf_policy,
        min_bf16_accum_dim=int(eff_cfg.get("apollo_min_dim", 128)),
        allow_embeddings_in_apollo=bool(eff_cfg.get("apollo_include_embeddings", False)),
    )

    use_cuda_amp = torch.cuda.is_available() and str(model.device).startswith("cuda")
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_cuda_amp else contextlib.nullcontext
    coat_ctx = (
        (lambda: coat_ctrl.activation_context(training=True))
        if (coat_ctrl is not None and coat_ctrl.enabled and not is_frozen_spark)
        else contextlib.nullcontext
    )
    inv_fn = coat_ctrl.invalidate_weight_caches if (coat_ctrl is not None and coat_ctrl.enabled) else None

    use_vis_prefetch = (
        hasattr(model, "has_intel_vision_bridge")
        and model.has_intel_vision_bridge()
        and getattr(model, "_intel_vision_trainable", False)
        and hasattr(model, "prefetch_examples_vision")
    )
    prefetch_pool = ThreadPoolExecutor(max_workers=1) if use_vis_prefetch else None

    def _materialize(item_idx: int, kind: str, payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if kind == "v25":
            return v25_record_to_example(payload)
        if kind == "pair":
            return _pair_to_example(payload)
        row, packet = catalogue.row(str(payload))
        with reader.packet(packet) as get:
            media = prepare_media(row, get, base_cfg)
        if kind == "cat_direct":
            from colab.final.frontier_stage9_corpus import convert_catalogue_row_to_direct_mode

            row = convert_catalogue_row_to_direct_mode(row)
        return row, media

    emit(
        "stage8_phase_start",
        stage=stage_name,
        stage_mode=stage_mode,
        unfreeze_vision_last_n_layers=unfreeze_vis,
        samples=len(order_items),
        planned_updates=planned_updates,
        grad_accum=grad_accum,
        param_audit=param_audit,
        vision_optimizer=eff_cfg.get("intel_arc_vision_optimizer"),
    )

    window_records = 0
    step_index = 0
    loss_hist: list[float] = []
    stratum_loss: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    prefetched_ex: tuple[dict[str, Any], dict[str, Any]] | None = None
    prefetch_fut = None
    vis_items_seen = 0
    margin_adv_sum = 0.0
    margin_hard_cnt = 0
    margin_probe_cnt = 0

    try:
        for idx, (kind, payload) in enumerate(order_items):
            if prefetched_ex is not None:
                if prefetch_fut is not None:
                    prefetch_fut.result()
                    prefetch_fut = None
                ex = prefetched_ex
                prefetched_ex = None
            else:
                ex = _materialize(idx, kind, payload)

            row, media = ex
            has_vis_imgs = row.get("_family") == "image" and bool(media.get("images"))
            do_blank_margin = False
            if has_vis_imgs:
                vis_items_seen += 1
                if vis_margin_every_n > 0 and (vis_items_seen % vis_margin_every_n == 0):
                    do_blank_margin = True

            if (
                prefetch_pool is not None
                and not do_blank_margin
                and (window_records + 1) < grad_accum
                and (idx + 1) < len(order_items)
            ):
                nk, npay = order_items[idx + 1]
                prefetched_ex = _materialize(idx + 1, nk, npay)
                prefetch_fut = prefetch_pool.submit(model.prefetch_examples_vision, [prefetched_ex])

            is_long = use_cuda_amp and (len(str(row.get("prompt", ""))) + len(str(row.get("answer", ""))) > 4000)
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
                emit("stage8_nonfinite_skip", stage=stage_name, step=step_index, item_idx=idx)
                window_records = 0
                continue

            loss_f = float(loss_val.detach().item())
            mb_weight = 1.0
            if do_blank_margin:
                blank_imgs = [Image.new("RGB", im.size, (128, 128, 128)) for im in media["images"]]
                blank_media = visual_media(
                    blank_imgs,
                    visual_kind=str(media.get("visual_kind") or "independent_image"),
                    visual_units=media.get("visual_units"),
                )
                with torch.no_grad(), amp_ctx():
                    b_res = model(examples=[(row, blank_media)])
                    loss_blank_f = float(b_res["per_example_loss"][0].detach().item())
                if math.isfinite(loss_blank_f):
                    adv = loss_blank_f - loss_f
                    margin_adv_sum += adv
                    margin_probe_cnt += 1
                    if adv < vis_margin_target:
                        margin_hard_cnt += 1
                        deficit = min(1.0, max(0.0, (vis_margin_target - adv) / max(1e-4, vis_margin_target)))
                        mb_weight = 1.0 + vis_margin_boost * deficit
                b_res = None

            key = f"{row['_family']}:{row['language']}"
            stratum_loss[key][0] += loss_f
            stratum_loss[key][1] += 1.0
            loss_hist.append(loss_f)
            mb_units = max(1, int(result.get("supervised_units", 1)))
            window_records += 1

            with accumulator.microbatch(weight=mb_weight, supervised_units=mb_units):
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
                emit("stage8_nonfinite_step_skip", stage=stage_name, step=step_index)
            elif (not is_frozen_spark) and hasattr(model, "step_tier1b_sparse_adam"):
                base_engram_lr = float(stage_cfg.get("engram_lr", 5e-5))
                model.step_tier1b_sparse_adam(lr=base_engram_lr * lr_ratio, denom=denom)

            window_records = 0
            step_index += 1

            if step_index == 1 or step_index % 20 == 0 or step_index == planned_updates:
                elapsed_min = (time.monotonic() - t0) / 60.0
                recent = loss_hist[-min(len(loss_hist), grad_accum * 10) :]
                mean_recent = sum(recent) / max(1, len(recent))
                extra: dict[str, Any] = {}
                if margin_probe_cnt > 0:
                    extra["vis_adv_mean"] = round(margin_adv_sum / margin_probe_cnt, 4)
                    extra["vis_hard_frac"] = round(margin_hard_cnt / margin_probe_cnt, 4)
                if hasattr(model, "has_intel_vision_bridge") and model.has_intel_vision_bridge():
                    tel = model.intel_vision_telemetry()
                    lst = tel.get("last_step") or {}
                    extra["b50_vgrad"] = round(float(lst.get("vision_grad_norm", 0.0)), 5)
                    extra["b50_gib"] = round(float(lst.get("xpu_peak_bytes", 0)) / (1024**3), 2)
                    extra["b50_opt"] = str(tel.get("vision_optimizer", ""))
                    extra["b50_fwd"] = int(tel.get("fwd_eval_count", 0)) + int(tel.get("fwd_train_count", 0))
                    extra["b50_pid"] = tel.get("xpu_pid")
                emit(
                    "stage8_progress",
                    stage=stage_name,
                    update=step_index,
                    planned=planned_updates,
                    elapsed_min=round(elapsed_min, 2),
                    upd_per_hr=round(step_index / max(1e-4, elapsed_min / 60.0), 1),
                    recent_loss=round(mean_recent, 4),
                    grad_norm=round(float(step_stats.grad_norm), 4),
                    lr_ratio=round(lr_ratio, 4),
                    **_extract_stage8_gate_stats(model)["gates"],
                    **extra,
                )
    finally:
        for k in _vis_reg_keys:
            if k in prev_decorr_keys:
                model.policy[k] = prev_decorr_keys[k]
            else:
                model.policy.pop(k, None)
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=True)
        accumulator.close()
        del optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        "visual_margin_summary": {
            "probes": margin_probe_cnt,
            "mean_adv": round(margin_adv_sum / max(1, margin_probe_cnt), 4),
            "hard_frac": round(margin_hard_cnt / max(1, margin_probe_cnt), 4),
        }
        if margin_probe_cnt > 0
        else None,
        "param_partition": param_audit,
        "checkpoint": str(checkpoint_dir / "model.safetensors"),
    }


def _export_stage8_releases(
    model: SparkNative,
    *,
    src_release_dir: Path,
    radio_dir: Path,
    out_bf16_dir: Path,
    out_fp8_dir: Path,
    report: dict[str, Any],
) -> dict[str, str]:
    """Export standalone portable Stage-8 C-RADIOv4-H BF16 AND FP8+q4g128 releases with updated runtime & vision assets."""
    if hasattr(model, "sync_intel_vision_to_host"):
        model.sync_intel_vision_to_host()
    for d in (out_bf16_dir, out_fp8_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy base reload-assets/spark & docs from Stage-7.5 release (skip old SigLIP2 reload-assets/vision & old runtime)
    for path in src_release_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_release_dir)
        posix = rel.as_posix()
        if posix in ("model.safetensors", "MODEL_RELEASE.json") or posix.startswith(("reload-assets/vision/", "runtime/")):
            continue
        dest = out_bf16_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    # 2. Copy C-RADIOv4-H code + config + preprocessor into reload-assets/vision/ (excluding heavy weight blobs)
    vis_dest = out_bf16_dir / "reload-assets/vision"
    vis_dest.mkdir(parents=True, exist_ok=True)
    for p in radio_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in (".safetensors", ".bin", ".pt", ".tar"):
            continue
        rel = p.relative_to(radio_dir)
        dst = vis_dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

    # 3. Copy fresh repository runtime (`colab/final/*.py`) so offline reload supports C-RADIOv4-H + SwiGLU DeepStack!
    rt_root = out_bf16_dir / "runtime"
    for p in (REPO_ROOT / "colab").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        dst = rt_root / p.relative_to(REPO_ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

    cfg = dict(model.policy)
    cfg["intel_arc_vision_offload"] = False
    cfg["intel_arc_openvino_eval"] = False
    cfg["vision_backbone_type"] = "c_radio_v4"
    cfg["radio_layer_indices"] = list(model._radio_layer_indices)
    cfg["stage8_cradiov4_deepstack_3d_spatial"] = True
    write(out_bf16_dir / "config.json", cfg)
    write(out_bf16_dir / "STAGE8_RADIOV4_REPORT.json", report)

    _save_model_safetensors(model, out_bf16_dir / "model.safetensors")
    bf16_files = manifest(out_bf16_dir)
    write(
        out_bf16_dir / "MODEL_RELEASE.json",
        {
            "schema_version": 1,
            "campaign_id": str(cfg.get("campaign_id", "spark-x26-4b-loop48-en1b-vl-stage8-radiov4")),
            "stage8_complete": True,
            "training_complete": True,
            "vision_backbone": "nvidia/C-RADIOv4-H",
            "quantization": "bf16",
            **bf16_files,
        },
    )

    # 4. FP8 + q4g128 residual export (~9.8 GB total!)
    for path in out_bf16_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(out_bf16_dir)
        if rel.as_posix() in ("model.safetensors", "MODEL_RELEASE.json"):
            continue
        dst = out_fp8_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)

    fp8_meta = export_spark_native_fp8_safetensors(model, out_fp8_dir / "model.safetensors")
    cfg_fp8 = dict(cfg)
    cfg_fp8["exported_weight_format"] = "fp8_e4m3_channel_q4g128_residual"
    write(out_fp8_dir / "config.json", cfg_fp8)
    fp8_files = manifest(out_fp8_dir)
    write(
        out_fp8_dir / "MODEL_RELEASE.json",
        {
            "schema_version": 1,
            "campaign_id": str(cfg.get("campaign_id", "spark-x26-4b-loop48-en1b-vl-stage8-radiov4")),
            "stage8_complete": True,
            "training_complete": True,
            "vision_backbone": "nvidia/C-RADIOv4-H",
            "quantization": "compressed-tensors-fp8-e4m3-channel+q4g128-residual",
            **fp8_meta,
            **fp8_files,
        },
    )
    emit(
        "stage8_releases_exported",
        bf16_dir=str(out_bf16_dir),
        bf16_sha256=sha(out_bf16_dir / "MODEL_RELEASE.json"),
        fp8_dir=str(out_fp8_dir),
        fp8_sha256=sha(out_fp8_dir / "MODEL_RELEASE.json"),
    )
    return {"bf16_dir": str(out_bf16_dir), "fp8_dir": str(out_fp8_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--init-release-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage7-b50/final-model-stage75-bf16"),
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
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage8-radiov4"),
    )
    ap.add_argument("--stage8a-updates", type=int, default=800)
    ap.add_argument("--stage8b-updates", type=int, default=400)
    ap.add_argument("--grad-accum", type=int, default=12)
    ap.add_argument("--stage8a-connector-lr", type=float, default=4.0e-4)
    ap.add_argument("--stage8b-backbone-lr", type=float, default=2.5e-5)
    ap.add_argument("--stage8b-engram-lr", type=float, default=6.0e-5)
    ap.add_argument("--stage8b-connector-lr", type=float, default=1.2e-4)
    ap.add_argument("--stage8b-vision-lr", type=float, default=2.5e-6)
    ap.add_argument("--stage8b-unfreeze-layers", type=int, default=6)
    ap.add_argument("--vision-optimizer", type=str, default="muon8bit", choices=("muon8bit", "apollo8bit", "adamw"))
    ap.add_argument("--nemotron-max-gqa", type=int, default=1200)
    ap.add_argument("--nemotron-max-geo", type=int, default=900)
    ap.add_argument("--nemotron-max-aokvqa", type=int, default=900)
    ap.add_argument("--nemotron-max-clevr", type=int, default=800)
    ap.add_argument("--nemotron-max-zhwiki", type=int, default=400)
    ap.add_argument("--nemotron-max-plotqa", type=int, default=300)
    ap.add_argument("--eval-images", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--intel-vision", type=str, default="xpu", choices=("xpu", "none"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    db_path = args.stage7_work_root / "metadata/corpus.sqlite"
    handoff_path = args.stage7_work_root / "metadata/HANDOFF.json"
    hf_token = os.environ.get("HF_TOKEN")

    # 1. Build Stage-8 Shaded 3D 4-Primitive Corpus (<point>, <box>, <dash>, <circle> + Front/Back + Left/Right)
    emit("stage8_build_spatial_3d_primitives_start", seed=args.seed)
    spatial3d_pairs = build_shaded_3d_spatial_primitives_records(
        seed=args.seed,
        panel_size=(384, 384),
        angles_per_asset=6,
    )
    v25_records = build_aesthetic_v25_vigorous_records(seed=args.seed + 101, max_records=64)
    v25_pixel_only = [r for r in v25_records if r["metadata"].get("pixel_indispensable")]
    emit("stage8_build_nemotron_v3_start")
    nemotron_pairs = build_nemotron_v3_spatial_records(
        cache_dir=args.output_root / "nemotron_v3_cache",
        hf_token=hf_token,
        max_gqa=args.nemotron_max_gqa,
        max_geometry3k=args.nemotron_max_geo,
        max_aokvqa=args.nemotron_max_aokvqa,
        max_clevr=args.nemotron_max_clevr,
        max_zhwiki=args.nemotron_max_zhwiki,
        max_plotqa=args.nemotron_max_plotqa,
        seed=args.seed + 202,
    )
    emit(
        "stage8_corpora_ready",
        spatial3d_records=len(spatial3d_pairs),
        nemotron_v3_records=len(nemotron_pairs),
        v25_records=len(v25_records),
    )

    # 2. Construct Stage-8A (100% visual alignment) and Stage-8B (spatial 3D + Nemotron + Visual Primitives + v25 + Frontier)
    rng = random.Random(args.seed)
    total_8a = args.stage8a_updates * args.grad_accum
    total_8b = args.stage8b_updates * args.grad_accum

    # Stage-8A mix: ~25% Shaded 3D 4-Primitives + ~30% Nemotron-v3 Spatial/Geometry + ~10% v25 Visual + ~35% Stage-7 Visual Primitives/Main Dishes
    n8a_s3d = min(int(total_8a * 0.25), max(1, len(spatial3d_pairs) * 4))
    n8a_nem = min(int(total_8a * 0.30), max(0, len(nemotron_pairs) * 2))
    n8a_v25 = min(int(total_8a * 0.10), max(0, len(v25_pixel_only) * 4))
    n8a_cat = max(1, total_8a - n8a_s3d - n8a_nem - n8a_v25)

    cat_8a_rids, cat_8a_summary = select_stage75a_pixel_indispensable_rids(
        db_path, target_catalogue_samples=n8a_cat, seed=args.seed + 1
    )
    order_8a: list[tuple[str, Any]] = [("cat", rid) for rid in cat_8a_rids]
    for i in range(n8a_s3d):
        order_8a.append(("pair", spatial3d_pairs[i % len(spatial3d_pairs)]))
    for i in range(n8a_nem):
        order_8a.append(("pair", nemotron_pairs[i % len(nemotron_pairs)]))
    for i in range(n8a_v25):
        order_8a.append(("v25", v25_pixel_only[i % len(v25_pixel_only)]))
    rng.shuffle(order_8a)
    order_8a = order_8a[:total_8a]

    # Stage-8B mix: ~20% Shaded 3D 4-Primitives + ~22% Nemotron-v3 Spatial/Geo + ~8% v25 All + ~50% Balanced Stage-7 Vision/Frontier Text
    n8b_s3d = min(int(total_8b * 0.20), max(1, len(spatial3d_pairs) * 3))
    n8b_nem = min(int(total_8b * 0.22), max(0, len(nemotron_pairs)))
    n8b_v25 = min(int(total_8b * 0.08), max(0, len(v25_records) * 3))
    n8b_cat = max(1, total_8b - n8b_s3d - n8b_nem - n8b_v25)

    cat_8b_rids, cat_8b_summary = select_stage75b_balanced_rids(
        db_path, target_catalogue_samples=n8b_cat, vision_fraction=0.55, seed=args.seed + 2
    )
    order_8b: list[tuple[str, Any]] = [("cat", rid) for rid in cat_8b_rids]
    for i in range(n8b_s3d):
        order_8b.append(("pair", spatial3d_pairs[i % len(spatial3d_pairs)]))
    for i in range(n8b_nem):
        order_8b.append(("pair", nemotron_pairs[i % len(nemotron_pairs)]))
    for i in range(n8b_v25):
        order_8b.append(("v25", v25_records[i % len(v25_records)]))
    rng.shuffle(order_8b)
    order_8b = order_8b[:total_8b]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "order_8a": len(order_8a),
                    "order_8b": len(order_8b),
                    "spatial3d_pairs": len(spatial3d_pairs),
                    "nemotron_pairs": len(nemotron_pairs),
                    "v25_records": len(v25_records),
                },
                indent=2,
            )
        )
        return

    # 3. Load Spark-4B + Loop48 + En1B from Stage-7.5 AND swap vision encoder cleanly to `nvidia/C-RADIOv4-H`!
    src_cfg = read(args.init_release_dir / "config.json")
    cfg = dict(src_cfg)
    cfg.update(
        {
            "campaign_id": "spark-x26-4b-loop48-en1b-vl-stage8-radiov4",
            "init_from_release_dir": str(args.init_release_dir.resolve()),
            "allow_warmstart_vision_swap": True,
            "allow_warmstart_missing_loop_engram": False,
            "vision_backbone_type": "c_radio_v4",
            "radio_layer_indices": [23, 31],  # L24 + L32 (exact ViTDet global-attention boundary layers)
            "radio_min_side": 256,
            "radio_default_max_side": 768,
            "radio_small_crop_min_side": 384,
            "radio_highres_tier_enabled": True,
            "radio_highres_trigger_px": 900,
            "radio_highres_max_side": 1152,
            "radio_default_max_patches": 1536,
            "radio_highres_max_patches": 4096,
            "radio_default_unfreeze_layers": int(args.stage8b_unfreeze_layers),
            "max_visual_tokens": 1024,
            "intel_arc_vision_offload": (args.intel_vision == "xpu"),
            "intel_arc_openvino_eval": False,
            "intel_arc_vision_optimizer": str(args.vision_optimizer),
            "intel_arc_vision_full_defrost": True,
            "vision_dtype": "bfloat16",
            "spark_dtype": "bfloat16",
        }
    )

    paths = {
        "spark": args.init_release_dir / "reload-assets/spark",
        "vision": args.radio_vision_dir.resolve(),
    }
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    emit("stage8_loading_model_start", init_dir=str(args.init_release_dir), radio_dir=str(args.radio_vision_dir))
    model = SparkNative(paths, cfg, device=device, pretrained=True)
    model.gradient_checkpointing_enable()
    emit(
        "stage8_model_loaded",
        vision_backbone_type=model.vision_backbone_type,
        radio_layers=list(model._radio_layer_indices),
        intel_vision_telemetry=model.intel_vision_telemetry(),
        initial_gates=_extract_stage8_gate_stats(model),
    )

    catalogue = Catalogue(db_path)
    handoff = read(handoff_path)
    reader = MediaReader(args.stage7_work_root, handoff, cfg, token=hf_token, reserve=0)

    eval_rids = _select_eval_image_rids(db_path, max_items=args.eval_images)
    probe_ex = _pair_to_example(spatial3d_pairs[0]) if spatial3d_pairs else None
    try:
        # Baseline visual calibration right after C-RADIOv4-H swap (before Stage-8A)
        cal_pre = measure_visual_alignment_diagnostics(
            model,
            catalogue=catalogue,
            reader=reader,
            config=cfg,
            eval_rids=eval_rids,
            v25_probe_example=probe_ex,
            tag="pre_stage8a",
        )
        emit("stage8_pre_calibration", **cal_pre)

        # Phase 8A: Frozen C-RADIOv4-H + Frozen Spark -> High-Speed SwiGLU DeepStack Projector & Cross-Attn Alignment
        stage8a_cfg = {
            "stage_mode": "frozen_spark_cross_attn",
            "unfreeze_vision_last_n_layers": 0,
            "optimizer_profile": "bf16_adamw",
            "connector_learning_rate": args.stage8a_connector_lr,
            "adapter_lr": args.stage8a_connector_lr,
            "learning_rate": args.stage8a_connector_lr,
            "warmup_ratio": 0.08,
            "clip_norm": 1.0,
            "weight_decay": 0.01,
        }
        res_8a = _run_stage8_phase(
            stage_name="stage8a_radio_deepstack_alignment",
            model=model,
            base_cfg=cfg,
            stage_cfg=stage8a_cfg,
            order_items=order_8a,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage8a",
            seed=args.seed + 10,
        )
        cal_post_8a = measure_visual_alignment_diagnostics(
            model,
            catalogue=catalogue,
            reader=reader,
            config=cfg,
            eval_rids=eval_rids,
            v25_probe_example=probe_ex,
            tag="post_stage8a",
        )
        emit("stage8_post_8a_calibration", **cal_post_8a)

        # Phase 8B: Full Spark-4B + Loop48 + En1B + Top-6 ONLY C-RADIOv4-H blocks (10x lower LR, muon8bit/apollo8bit!)
        stage8b_cfg = {
            "stage_mode": "full_spark",
            "unfreeze_vision_last_n_layers": int(args.stage8b_unfreeze_layers),
            "optimizer_profile": "apollo_coat",
            "learning_rate": args.stage8b_backbone_lr,
            "backbone_lr": args.stage8b_backbone_lr,
            "apollo_learning_rate": args.stage8b_backbone_lr,
            "connector_learning_rate": args.stage8b_connector_lr,
            "adapter_lr": args.stage8b_connector_lr,
            "apollo_connector_learning_rate": args.stage8b_connector_lr,
            "engram_lr": args.stage8b_engram_lr,
            "vision_lr": args.stage8b_vision_lr,
            "warmup_ratio": 0.10,
            "clip_norm": 0.8,
            "weight_decay": 0.01,
        }
        res_8b = _run_stage8_phase(
            stage_name="stage8b_full_spark_radio_top6_3d_spatial",
            model=model,
            base_cfg=cfg,
            stage_cfg=stage8b_cfg,
            order_items=order_8b,
            catalogue=catalogue,
            reader=reader,
            grad_accum=args.grad_accum,
            checkpoint_dir=args.output_root / "ckpt-stage8b",
            seed=args.seed + 20,
        )
        cal_post_8b = measure_visual_alignment_diagnostics(
            model,
            catalogue=catalogue,
            reader=reader,
            config=cfg,
            eval_rids=eval_rids,
            v25_probe_example=probe_ex,
            tag="post_stage8b",
        )
        emit("stage8_post_8b_calibration", **cal_post_8b)

        report = {
            "stage8_architecture": {
                "vision_backbone": "nvidia/C-RADIOv4-H (651.6M, 32L ViT-H/16, SigLIP2-g + DINOv3-7B + SAM3)",
                "multi_layer_fusion": "RMSNorm(L24) || RMSNorm(L32) -> 2x2 pixel shuffle -> SwiGLU MLP + Early L24 Cross-Attn",
                "resolution_tiers": "default 256-768px adaptive native-aspect; small crops >= 384px; high-res single-image tier 1024-1152px",
                "vision_optimizer_b50": args.vision_optimizer,
                "stage8b_unfreeze_layers": int(args.stage8b_unfreeze_layers),
                "visual_primitives": ["point", "box", "dash", "circle"],
            },
            "corpora": {
                "spatial3d_shaded_records": len(spatial3d_pairs),
                "nemotron_v3_spatial_records": len(nemotron_pairs),
                "v25_vigorous_records": len(v25_records),
                "stage8a_cat_summary": cat_8a_summary,
                "stage8b_cat_summary": cat_8b_summary,
            },
            "calibration": {
                "pre_8a": cal_pre,
                "post_8a": cal_post_8a,
                "post_8b": cal_post_8b,
            },
            "stage8a": res_8a,
            "stage8b": res_8b,
            "final_gates": _extract_stage8_gate_stats(model),
        }
        write(args.output_root / "STAGE8_RADIOV4_REPORT.json", report)

        _export_stage8_releases(
            model,
            src_release_dir=args.init_release_dir,
            radio_dir=args.radio_vision_dir,
            out_bf16_dir=args.output_root / "final-model-stage8-radiov4-bf16",
            out_fp8_dir=args.output_root / "final-model-stage8-radiov4-fp8",
            report=report,
        )
    finally:
        model.close_coprocessors()


if __name__ == "__main__":
    main()
