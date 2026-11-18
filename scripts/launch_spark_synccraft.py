#!/usr/bin/env python3
"""Pinned launcher for Spark-X2.6-VL + SyncCraft v3.2 with APOLLO + COAT + merged gradients.

Target workstation: NVIDIA RTX PRO 4000 Blackwell SFF (24 GB GDDR7, SM120).
Secrets (`HF_TOKEN`, `OPENCODE_API_KEY`) are read from environment variables and
delivered to `colab.final.run` strictly through stdin JSON, never CLI argv or logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.common import HERE, identity, read, sha, validate_config, write

BASELINE_GIT_COMMIT = "22ff1605d75143870743d26fa5d9445a32d6e627"
DEFAULT_CONFIG_PATH = HERE / "campaign_rtx_pro_4000_24gb.json"

PINNED_RUNTIME_PATHS = (
    "colab/final/common.py",
    "colab/final/loop_engram.py",
    "colab/final/intel_vision_coproc.py",
    "colab/final/optimizer.py",
    "colab/final/coat_spark.py",
    "colab/final/synccraft_bridge.py",
    "colab/final/sculpted_style_corpus.py",
    "colab/final/sculpted_v8_v41_corpus.py",
    "colab/final/visual_primitives_corpus.py",
    "colab/final/frontier_stage7_corpus.py",
    "colab/final/world_pipeline.py",
    "colab/final/reasoning_agentic_corpus.py",
    "colab/final/vision_rsi.py",
    "colab/final/hybrid_corpus.py",
    "colab/final/data.py",
    "colab/final/model.py",
    "colab/final/engine.py",
    "colab/final/evaluate.py",
    "colab/final/qualify.py",
    "colab/final/storage.py",
    "colab/final/assets.py",
    "colab/final/inference.py",
    "colab/final/publish.py",
    "colab/final/teacher.py",
    "colab/final/run.py",
    "colab/final/components.lock.json",
)


def compute_source_manifest(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    files: dict[str, str] = {}
    for rel in PINNED_RUNTIME_PATHS:
        p = repo_root / rel
        if not p.is_file():
            raise FileNotFoundError(f"Missing pinned runtime file: {rel}")
        files[rel] = sha(p)
    git_head = None
    baseline_reachable = False
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_head = out.stdout.strip()
        anc = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", BASELINE_GIT_COMMIT, "HEAD"],
            check=False,
            capture_output=True,
        )
        baseline_reachable = anc.returncode == 0
    except Exception:
        pass
    source_sha = hashlib.sha256(
        json.dumps({"baseline": BASELINE_GIT_COMMIT, "files": files}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "baseline_git_commit": BASELINE_GIT_COMMIT,
        "git_head": git_head,
        "baseline_is_ancestor": baseline_reachable,
        "source_sha256": source_sha,
        "files": files,
    }


def analytical_vram_budget_24gb() -> dict[str, Any]:
    """Analytical tensor memory breakdown (bytes & GiB) for Spark-X2.6-4B + SigLIP2 on 24 GB SM120."""
    # Spark-X2.6-4B: ~4.12B params (~4.06B in 2D matrices + ~0.06B in norms/gates/biases)
    # SigLIP2-SO400M: ~0.43B params (frozen in BF16 during full_spark stage)
    # Multimodal connectors + alpha/pose + 3 gated visual x-attn blocks: ~0.045B FP32 params
    gib = float(1024**3)
    spark_2d_params = 4_060_000_000
    spark_1d_params = 60_000_000
    siglip_params = 430_000_000
    connector_params = 45_000_000

    weights_bf16 = (spark_2d_params + siglip_params) * 2 + (spark_1d_params + connector_params) * 4
    merged_grad_bytes = spark_2d_params * 2 + (spark_1d_params + connector_params) * 4

    def _entry(opt_state_bytes: int, act_note: str) -> dict[str, Any]:
        static_total = weights_bf16 + merged_grad_bytes + opt_state_bytes
        headroom_24gb = (24 * (1024**3)) - static_total
        return {
            "weights_gib": round(weights_bf16 / gib, 2),
            "merged_grad_buffer_gib": round(merged_grad_bytes / gib, 2),
            "optimizer_state_gib": round(opt_state_bytes / gib, 2),
            "total_static_gib": round(static_total / gib, 2),
            "free_for_activations_on_24gib": round(headroom_24gb / gib, 2),
            "fits_24gb_rtx_pro_4000": headroom_24gb >= 3.5 * gib,
            "activation_strategy": act_note,
        }

    # Standard AdamW FP32 master + m + v = 12 bytes/param on 4.165B trainable params = ~50.0 GB -> OOM on 24 GB!
    # Pure BF16 AdamW m + v (8 bytes/param FP32 moments) = ~33.3 GB -> OOM on 24 GB!
    adamw_opt_bytes = (spark_2d_params + spark_1d_params + connector_params) * 8
    # APOLLO rank-64 on ~252 large 2D matrices (avg shape ~4096x2560): 252 * (64*2560*2 * 4) + FP32 master on small params
    apollo_r64_bytes = 252 * (64 * 2560 * 8) + (spark_1d_params + connector_params) * 12
    apollo_r32_bytes = 252 * (32 * 2560 * 8) + (spark_1d_params + connector_params) * 12
    apollo_fp8_r64_bytes = 252 * (64 * 2560 * 3) + (spark_1d_params + connector_params) * 12

    return {
        "target_gpu": "NVIDIA RTX PRO 4000 Blackwell SFF (24 GB GDDR7, SM120)",
        "note": "Analytical byte accounting on development host; run `ablate` on SM120 GPU for live max_memory_allocated.",
        "profiles": {
            "bf16_adamw": _entry(adamw_opt_bytes, "Standard BF16 AdamW moments exceed 24 GB on full 4B unfreeze"),
            "apollo_merged_r64": _entry(apollo_r64_bytes, "Gradient checkpointing (BF16 saved activations)"),
            "apollo_merged_r32": _entry(apollo_r32_bytes, "Gradient checkpointing (BF16 saved activations)"),
            "apollo_coat_r64": _entry(apollo_r64_bytes, "Gradient checkpointing + COAT FP8 E4M3 saved activations (~5.7 GiB headroom for 8192 ctx)"),
            "apollo_coat_fp8_moments_r64": _entry(apollo_fp8_r64_bytes, "Gradient checkpointing + COAT FP8 activations + FP8 low-rank moments"),
        },
    }


def _invoke_runner(mode: str, config_path: Path, source_sha: str, extra_args: list[str] | None = None) -> int:
    secrets = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "OPENCODE_API_KEY": os.environ.get("OPENCODE_API_KEY", ""),
    }
    cmd = [
        sys.executable,
        "-m",
        "colab.final.run",
        mode,
        "--config",
        str(config_path),
        "--source-sha",
        source_sha,
        *(extra_args or []),
    ]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        cmd,
        input=(json.dumps(secrets) + "\n").encode("utf-8"),
        env=env,
        check=False,
    )
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "manifest",
            "budget",
            "prepare-synccraft",
            "prepare-assets",
            "doctor",
            "ablate",
            "pilot",
            "qualify",
            "recover",
            "train",
            "run",
            "direct",
            "resume",
            "discover",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Campaign JSON config path")
    parser.add_argument("--work-root", "--root", dest="work_root", type=Path, default=None, help="Optional override for config work_root")
    parser.add_argument("--dataset-mode", choices=("local_synccraft", "hybrid_world_v32", "hf_full"), default=None, help="Dataset source override")
    parser.add_argument("--steps", type=int, default=None, help="Number of optimizer updates per profile in ablate mode")
    parser.add_argument("--resume-receipt", type=str, default=None, help="Receipt JSON path for resume mode")
    parser.add_argument("--run-phase", choices=("full", "train", "rsi", "audit"), default=None, help="Optional phase override (train -> teacher subagent -> rsi -> audit)")
    parser.add_argument("--no-animation", action="store_true", help="Exclude Stage-B animation clips during static pilot prep")
    args = parser.parse_args(argv)

    manifest = compute_source_manifest()
    if args.command == "manifest":
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "budget":
        print(json.dumps(analytical_vram_budget_24gb(), indent=2))
        return 0

    raw_cfg = read(args.config)
    if args.work_root is not None:
        raw_cfg["work_root"] = str(args.work_root.resolve())
    if args.dataset_mode is not None:
        if args.dataset_mode == "hf_full":
            raw_cfg.pop("dataset_mode", None)
        else:
            raw_cfg["dataset_mode"] = args.dataset_mode
    if args.steps is not None:
        raw_cfg["ablation_updates"] = max(1, int(args.steps))
    cfg = validate_config(raw_cfg)
    work_root = Path(cfg["work_root"])
    work_root.mkdir(parents=True, exist_ok=True)

    # Write an effective pinned config inside work_root so overrides remain auditable.
    effective_cfg_path = work_root / "effective_campaign_config.json"
    if args.command == "pilot":
        pilot_cfg = dict(cfg)
        pilot_cfg["stages"] = [cfg["stages"][0]]
        pilot_cfg = validate_config(pilot_cfg)
        write(effective_cfg_path, pilot_cfg)
    else:
        write(effective_cfg_path, cfg)

    if args.command == "prepare-synccraft":
        from colab.final.assets import prepare_data, prepare_local_synccraft_data

        if cfg.get("dataset_mode") == "hybrid_world_v32":
            report = prepare_data(work_root, os.environ.get("HF_TOKEN") or None, cfg)
        else:
            report = prepare_local_synccraft_data(
                work_root,
                cfg,
                include_animation=not args.no_animation,
            )
        print(json.dumps({"status": "ready", "work_root": str(work_root), "report": report}, indent=2))
        return 0

    if args.command == "prepare-assets":
        from colab.final.assets import prepare_assets

        paths = prepare_assets(
            work_root,
            os.environ.get("HF_TOKEN") or None,
            cfg,
            download_weights=True,
        )
        print(json.dumps({"status": "assets_ready", "work_root": str(work_root), "paths": {k: str(v) for k, v in paths.items()}}, indent=2))
        return 0

    run_mode = "train" if args.command in ("pilot", "run") else args.command
    extra: list[str] = []
    if args.resume_receipt:
        extra.extend(["--resume-receipt", args.resume_receipt])
    if args.run_phase:
        extra.extend(["--run-phase", args.run_phase])
    return _invoke_runner(run_mode, effective_cfg_path, manifest["source_sha256"], extra)


if __name__ == "__main__":
    raise SystemExit(main())
