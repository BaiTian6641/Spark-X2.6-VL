#!/usr/bin/env python3
"""Live Hardware Micro-Benchmark on Intel Arc Pro B50 (`xpu:0`, 16 GB) for Stage-8 Unified `nvidia/C-RADIOv4-H` (651.6M).

Measures exact real hardware numbers through `IntelArcXpuVisionBridge` + `DualPinnedShmArena`:
1. `C-RADIOv4-H` (`L24 + L32`, `indices=(23, 31)`) Frozen Stage-8A Forward Latency across:
   - `512x384` (192 tok), `768x512` (384 tok), High-Res Single-Image Tier `1024x768` (768 tok) & `1152x768` (864 tok).
2. `C-RADIOv4-H` Top-6 Blocks Unfrozen (`118.06M` trainable params, blocks `26..31` + `norm`) Stage-8B Head-to-Head:
   - **`muon8bit`** vs **`apollo8bit`** vs **`adamw`**:
     - Exact B50 Peak VRAM (`GiB / 16 GiB`)
     - Optimizer State Bytes & Grad Accum Bytes
     - Forward + Async Backward + Optimizer Step Latency (`ms`)
     - Numerical Gradient Norm Health (`finite`, non-zero)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.intel_vision_coproc import IntelArcXpuVisionBridge, load_cradiov4_vision_model  # noqa: E402


def bench_mode(
    model_dir: Path,
    *,
    optimizer_mode: str,
    unfreeze_layers: int,
    resolutions: list[tuple[int, int]],
    steps: int,
    xpu_python: str | None,
    device_spec: str,
) -> dict[str, Any]:
    out_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    bridge = IntelArcXpuVisionBridge(
        model_dir,
        vision_optimizer=optimizer_mode,
        radio_layer_indices=(23, 31),
        xpu_python=xpu_python,
        device_spec=device_spec,
        vision_dtype="bfloat16",
        weight_decay=0.01,
        grad_checkpoint_token_threshold=2304,
    )
    try:
        host_vis = load_cradiov4_vision_model(model_dir, dtype=torch.bfloat16, pretrained=True)
        bridge.sync_from_host(host_vis)
        del host_vis

        tr_resp = bridge.configure_trainability(
            trainable=(unfreeze_layers > 0),
            unfreeze_last_n_layers=int(unfreeze_layers),
        )

        per_res: dict[str, Any] = {}
        for h_px, w_px in resolutions:
            fwd_bwd_ms: list[float] = []
            step_ms_list: list[float] = []
            last_step_res: dict[str, Any] = {}
            out_shape: list[int] = []
            for it in range(steps + 1):
                pv = torch.rand(1, 3, h_px, w_px, dtype=torch.bfloat16)
                t0 = time.perf_counter()
                if unfreeze_layers > 0:
                    feats = bridge.forward(
                        {"pixel_values": pv},
                        training_mode=True,
                        out_device=out_dev,
                        out_dtype=torch.bfloat16,
                    )
                    out_shape = list(feats.shape)
                    # Backprop through L24+L32 stacked features -> async DMA grad to B50!
                    loss = feats.float().mean()
                    loss.backward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_fb = (time.perf_counter() - t0) * 1000.0
                    t1 = time.perf_counter()
                    sres = bridge.step(lr=3e-5, max_grad_norm=1.0, post_scale=1.0)
                    t_st = (time.perf_counter() - t1) * 1000.0
                    last_step_res = sres
                    if it > 0:
                        fwd_bwd_ms.append(t_fb)
                        step_ms_list.append(t_st)
                else:
                    with torch.no_grad():
                        feats = bridge.forward(
                            {"pixel_values": pv},
                            training_mode=False,
                            out_device=out_dev,
                            out_dtype=torch.bfloat16,
                        )
                    out_shape = list(feats.shape)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_fb = (time.perf_counter() - t0) * 1000.0
                    if it > 0:
                        fwd_bwd_ms.append(t_fb)
            key = f"{h_px}x{w_px}"
            per_res[key] = {
                "out_shape": out_shape,
                "tokens_2x2": (h_px // 32) * (w_px // 32),
                "mean_fwd_bwd_ms": round(sum(fwd_bwd_ms) / max(1, len(fwd_bwd_ms)), 2),
                "mean_step_ms": round(sum(step_ms_list) / max(1, len(step_ms_list)), 2) if step_ms_list else 0.0,
                "xpu_peak_gib": round(float(last_step_res.get("xpu_peak_bytes", 0)) / (1024**3), 3),
                "optim_state_mib": round(float(last_step_res.get("optimizer_state_bytes", 0)) / (1024**2), 2),
                "grad_accum_mib": round(float(last_step_res.get("grad_accum_bytes", 0)) / (1024**2), 2),
                "vision_grad_norm": round(float(last_step_res.get("vision_grad_norm", 0.0)), 6),
                "finite": bool(last_step_res.get("finite", True)),
            }

        return {
            "optimizer_mode": optimizer_mode,
            "unfreeze_layers": unfreeze_layers,
            "device_name": bridge.device_info.get("device_name"),
            "xpu_ze_shm_pinned": bridge.device_info.get("xpu_ze_shm_pinned"),
            "cuda_shm_pinned": bridge.arena.cuda_registered,
            "total_parameters": tr_resp.get("total_parameters"),
            "trainable_parameters": tr_resp.get("trainable_parameters"),
            "resolutions": per_res,
        }
    finally:
        bridge.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="cache/c-radiov4-h")
    ap.add_argument("--xpu-python", default=None)
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--output-json", default="work/b50_cradiov4_muon_vs_apollo_bench.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    resolutions = [(512, 384), (768, 512), (1024, 768), (1152, 768)]
    results: dict[str, Any] = {"model_dir": str(model_dir), "runs": {}}

    # 1. Stage-8A Frozen C-RADIOv4-H Forward across default + High-Res Single-Image Tier!
    print("=== Benchmarking Stage-8A Frozen C-RADIOv4-H (L24+L32) ===")
    results["runs"]["frozen_stage8a"] = bench_mode(
        model_dir,
        optimizer_mode="muon8bit",
        unfreeze_layers=0,
        resolutions=resolutions,
        steps=args.steps,
        xpu_python=args.xpu_python,
        device_spec=args.device,
    )
    print(json.dumps(results["runs"]["frozen_stage8a"], indent=2))

    # 2. Stage-8B Top-6 Blocks Unfrozen (118.06M params): muon8bit vs apollo8bit vs adamw!
    for opt_mode in ("muon8bit", "apollo8bit", "adamw"):
        print(f"\n=== Benchmarking Stage-8B Top-6 Unfrozen C-RADIOv4-H [{opt_mode}] ===")
        results["runs"][f"top6_{opt_mode}"] = bench_mode(
            model_dir,
            optimizer_mode=opt_mode,
            unfreeze_layers=6,
            resolutions=resolutions,
            steps=args.steps,
            xpu_python=args.xpu_python,
            device_spec=args.device,
        )
        print(json.dumps(results["runs"][f"top6_{opt_mode}"], indent=2))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved live B50 benchmark report to {out_path}")


if __name__ == "__main__":
    main()
