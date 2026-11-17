#!/usr/bin/env python3
"""Evaluate Terminal-Bench 4.0 (`v4.0.0`) disjoint holdout tasks across Stage-4/5/RSI-2 and verify 3D pipeline fixes."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.common import read, write
from colab.final.inference import empty_media, generate, load
from colab.final.reasoning_agentic_corpus import (
    TERMINAL_BENCH3_REPO_ID,
    TERMINAL_BENCH3_REVISION,
    TERMINAL_BENCH4_REPO_ID,
    TERMINAL_BENCH4_REVISION,
    check_generation_collapse_health,
)
from colab.final.synccraft_bridge import ensure_synccraft_importable, render_world_views_rgba, strip_think_block
from colab.final.world_pipeline import (
    SyncCraftWorldPipeline,
    build_default_world_state_for_program,
    evaluate_3d_asset_aesthetics,
)


def _hf_file(repo_id: str, filename: str, *, revision: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision, token=token))


def _collect_disjoint_tb4_holdout_tasks(token: str | None, limit: int = 12) -> list[str]:
    """Select Terminal-Bench 4.0 tasks that were NEVER in the Terminal-Bench 3.0 training split."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info3 = api.dataset_info(TERMINAL_BENCH3_REPO_ID, revision=TERMINAL_BENCH3_REVISION)
    tb3_train_tids: set[str] = set()
    for sib in info3.siblings or ():
        parts = sib.rfilename.split("/")
        if len(parts) >= 3 and parts[0] == "tasks":
            tid = parts[1]
            if int(hashlib.sha256(f"tb3_holdout_v1:{tid}".encode("utf-8")).hexdigest()[:8], 16) % 3 != 0:
                tb3_train_tids.add(tid)

    info4 = api.dataset_info(TERMINAL_BENCH4_REPO_ID, revision=TERMINAL_BENCH4_REVISION)
    tb4_by_task: dict[str, set[str]] = {}
    for sib in info4.siblings or ():
        parts = sib.rfilename.split("/")
        if len(parts) >= 3 and parts[0] == "tasks":
            tb4_by_task.setdefault(parts[1], set()).add(sib.rfilename)

    # Prioritize newly added v4.0.0 tasks absent from TB3, plus held-out `bucket == 0` TB4 tasks
    new_v4 = [
        tid
        for tid in sorted(tb4_by_task)
        if tid not in tb3_train_tids and f"tasks/{tid}/instruction.md" in tb4_by_task[tid]
    ]
    return new_v4[:limit]


def _validate_terminal_response(body: str, tid: str) -> tuple[bool, str]:
    """Validate either fenced ```bash/sh/python``` scripts OR structured JSON `tool_calls` shell/code payloads."""
    code_re = re.compile(r"```(bash|sh|python|json)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    blocks = code_re.findall(body)
    if blocks:
        valid_any = False
        for lang, snippet in blocks:
            lt = lang.lower()
            if lt == "python":
                try:
                    compile(snippet, f"<{tid}>", "exec")
                    valid_any = True
                except SyntaxError:
                    return False, "invalid_python_syntax"
            elif lt in ("bash", "sh"):
                p = subprocess.run(["bash", "-n"], input=snippet.encode("utf-8"), capture_output=True, timeout=3, check=False)
                if p.returncode != 0:
                    return False, "invalid_bash_syntax"
                valid_any = True
            elif lt == "json":
                try:
                    obj = json.loads(snippet)
                    if isinstance(obj, dict) and ("tool_calls" in obj or "command" in obj or "name" in obj):
                        valid_any = True
                except Exception:
                    return False, "invalid_tool_json"
        if valid_any:
            return True, "fenced_script_or_tool_call_valid"

    # Also check unfenced shebang script
    if body.lstrip().startswith("#!/"):
        if "python" in body.splitlines()[0]:
            try:
                compile(body, f"<{tid}>", "exec")
                return True, "shebang_python_valid"
            except SyntaxError:
                return False, "invalid_shebang_python"
        p = subprocess.run(["bash", "-n"], input=body.encode("utf-8"), capture_output=True, timeout=3, check=False)
        if p.returncode == 0:
            return True, "shebang_bash_valid"
    return False, "no_valid_script_or_tool_block"


def eval_tb4_for_model(model: Any, holdout_tids: list[str], token: str | None) -> dict[str, Any]:
    valid_cnt = 0
    healthy_cnt = 0
    think_closed_cnt = 0
    per_task: list[dict[str, Any]] = []

    for tid in holdout_tids:
        inst = _hf_file(
            TERMINAL_BENCH4_REPO_ID,
            f"tasks/{tid}/instruction.md",
            revision=TERMINAL_BENCH4_REVISION,
            token=token,
        ).read_text(encoding="utf-8", errors="replace").strip()
        out = generate(
            model,
            (
                f"[Terminal-Bench 4.0 Task: {tid}]\n{inst[:4500]}\n\n"
                "Author a complete, non-interactive solution inside a ```bash ... ``` or ```python ... ``` code block."
            ),
            media=empty_media(),
            enable_thinking=True,
            max_new_tokens=1664,
            max_think_tokens=576,
            task="terminal_bench_agent",
        )
        h = check_generation_collapse_health(out, expect_thinking=True, max_rep_4gram=0.16, max_think_chars=5200)
        if h["healthy"]:
            healthy_cnt += 1
        if h.get("think_closed"):
            think_closed_cnt += 1
        _, body = strip_think_block(out)
        ok, kind = _validate_terminal_response(body, tid)
        if ok:
            valid_cnt += 1
        per_task.append({"task_id": tid, "valid": ok, "kind": kind, "healthy": h["healthy"], "think_closed": h.get("think_closed")})

    n = max(1, len(holdout_tids))
    return {
        "benchmark": "Terminal-Bench 4.0 (v4.0.0 Disjoint Holdout)",
        "revision": TERMINAL_BENCH4_REVISION,
        "evaluated": len(holdout_tids),
        "valid_executable_or_tool_actions": valid_cnt,
        "valid_rate": round(valid_cnt / n, 4),
        "healthy_generation_rate": round(healthy_cnt / n, 4),
        "think_close_rate": round(think_closed_cnt / n, 4),
        "tasks": per_task,
    }


def eval_fixed_3d_pipeline(model: Any, cfg: dict[str, Any], out_dir: Path, token: str | None) -> dict[str, Any]:
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SyncCraftWorldPipeline(model, config={**cfg, "min_aesthetic_score": 0.72}, default_max_new_tokens=3072)

    hound_res = pipeline.generate_3d_asset(
        "Design an artist-grade SyncCraft Block-DSL v3.2 armored sentinel hound quadruped (`bedrock_entity`, `texel_density: 16`) "
        "with distinct ribcage, tapered pelvis, chest armor (`inflate`), articulated neck, skull, muzzle, nose, lower jaw, "
        "bilateral `mirror_x` ears, 2-segment tail, 4 three-part legs (`thigh -> shin -> paw`), 5 OKLCH `_mat` ramps, and a `walk_cycle` IK gait.",
        asset_name="sentinel_hound_rsi2_fixed",
        output_dir=out_dir / "sentinel_hound_rsi2_fixed",
        export_formats=("bbmodel", "bedrock", "geckolib", "gltf_pbr", "labpbr"),
        min_aesthetic_score=0.74,
    )

    val_p = _hf_file(
        "TassP0lar/synccraft-v3-human-artist-dataset",
        "data/permissive_cc/val.jsonl",
        revision="46b826aa7928372b9609c65ce4766274272632ba",
        token=token,
    )
    val_rows = [json.loads(ln) for ln in val_p.open("r", encoding="utf-8") if ln.strip()]
    artic_val = [
        r
        for r in val_rows
        if 14 <= int(r.get("box_count") or 0) <= 45
        and int(r.get("bone_count") or 0) >= 4
        and not r.get("ai_assisted")
        and r.get("commercial_weights_eligible")
    ]
    target_human = artic_val[0] if artic_val else val_rows[0]
    h_prog = parse_program(target_human["canonical_dsl"])
    h_ws = build_default_world_state_for_program(h_prog, seed=99)
    h_pngs, _, _, _ = render_world_views_rgba(h_ws, panel_size=(192, 192))
    ref_pil = [Image.open(io.BytesIO(h_pngs[v])) for v in ("south", "right", "top", "iso")]
    h_aud = str(getattr(h_prog.meta, "audit_profile", "authored"))
    h_prof = str(getattr(h_prog.meta, "target_profile", "bbmodel_5"))

    inv_res = pipeline.generate_3d_asset(
        f"Reconstruct a collision-free SyncCraft Block-DSL v3.2 program matching these 4 orthographic views "
        f"(south, right, top, iso) of `{target_human.get('title', target_human['asset_id'])}` with "
        f"target_profile={h_prof!r}, texel_density={h_prog.meta.texel_density}, and audit_profile={h_aud!r}.",
        images=ref_pil,
        asset_name="human_artist_multiview_rsi2_fixed",
        output_dir=out_dir / "human_artist_multiview_rsi2_fixed",
        export_formats=("bbmodel", "bedrock", "gltf_pbr"),
        min_aesthetic_score=0.68,
    )

    stage4_toy_dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: strict }
@palette {
  pelt_mat: ramp(base_lch=[0.64, 0.11, 42], steps=6)
  hoof_mat: ramp(base_lch=[0.4, 0.06, 132], steps=6)
}
@rig {
  root bone body_root {
    pivot_local: [0, 10, 0]
    box torso { size: [6, 5, 9], material: pelt_mat }
    bone fl_thigh attach(parent_box=torso, anchor=[-1, -0.6, -0.6], align=[1, 1, 0]) mirror_x(fr_thigh) {
      box fl_thigh_seg { size: [2, 4, 2], material: pelt_mat }
      bone fl_shin attach(parent_box=fl_thigh_seg, anchor=[0, -1, 0], align=[0, 1, 0]) {
        box fl_shin_seg { size: [2, 4, 2], material: hoof_mat }
      }
    }
    bone hl_thigh attach(parent_box=torso, anchor=[-1, -0.6, 0.6], align=[1, 1, 0]) mirror_x(hr_thigh) {
      box hl_thigh_seg { size: [2, 4, 2], material: pelt_mat }
      bone hl_shin attach(parent_box=hl_thigh_seg, anchor=[0, -1, 0], align=[0, 1, 0]) {
        box hl_shin_seg { size: [2, 4, 2], material: hoof_mat }
      }
    }
  }
}
"""
    up_res = pipeline.generate_3d_asset(
        "Upgrade this simplistic 9-box headless toy quadruped blockout into an artist-grade 20–32 box "
        "articulated armored hound (`bedrock_entity`) with neck, skull, muzzle, jaw, bilateral `mirror_x` ears, "
        "pelvis taper, chest armor (`inflate`), 2-segment tail, paws, >=4 OKLCH `_mat` ramps, and `walk_cycle` IK gait:\n"
        f"```blockdsl\n{stage4_toy_dsl.strip()}\n```",
        asset_name="upgraded_hound_rsi2_fixed",
        output_dir=out_dir / "upgraded_hound_rsi2_fixed",
        export_formats=("bbmodel", "bedrock", "gltf_pbr"),
        min_aesthetic_score=0.72,
    )

    return {
        "sentinel_hound_rsi2_fixed": {
            "passed": hound_res["passed"],
            "boxes": hound_res["boxes"],
            "bones": hound_res["bones"],
            "reaudits_used": hound_res["reaudits_used"],
            "failure_reason": hound_res["failure_reason"],
            "aesthetic_evaluation": hound_res["aesthetic_evaluation"],
            "preview_views": (hound_res["bundle"] or {}).get("preview_views", {}),
        },
        "human_artist_multiview_rsi2_fixed": {
            "reference_asset_id": target_human["asset_id"],
            "passed": inv_res["passed"],
            "boxes": inv_res["boxes"],
            "bones": inv_res["bones"],
            "reaudits_used": inv_res["reaudits_used"],
            "failure_reason": inv_res["failure_reason"],
            "aesthetic_evaluation": inv_res["aesthetic_evaluation"],
            "preview_views": (inv_res["bundle"] or {}).get("preview_views", {}),
        },
        "upgraded_hound_rsi2_fixed": {
            "passed": up_res["passed"],
            "boxes": up_res["boxes"],
            "bones": up_res["bones"],
            "reaudits_used": up_res["reaudits_used"],
            "failure_reason": up_res["failure_reason"],
            "aesthetic_evaluation": up_res["aesthetic_evaluation"],
            "preview_views": (up_res["bundle"] or {}).get("preview_views", {}),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="colab/final/campaign_rtx_pro_4000_stage5_rsi.json")
    ap.add_argument("--tb4-limit", type=int, default=12)
    args = ap.parse_args()

    cfg = read(args.config)
    root = Path(cfg["work_root"]).resolve()
    token = os.environ.get("HF_TOKEN") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    holdout_tids = _collect_disjoint_tb4_holdout_tasks(token, limit=args.tb4_limit)
    checkpoints = [
        ("stage4_baseline", Path("/home/baitian6641/Spark-X2.6-VL/work/spark-world-v32-24gb/final-model")),
        ("stage5_parent", root / "final-model"),
        ("stage5_rsi2_promoted", root / "final-model-rsi2"),
    ]

    tb4_results: dict[str, Any] = {"disjoint_holdout_task_ids": holdout_tids, "by_checkpoint": {}}
    fixed_3d_results: dict[str, Any] = {}

    for label, ckpt in checkpoints:
        if not (ckpt / "MODEL_RELEASE.json").is_file():
            continue
        model = load(ckpt, device=device)
        tb4_results["by_checkpoint"][label] = eval_tb4_for_model(model, holdout_tids, token)
        if label == "stage5_rsi2_promoted":
            fixed_3d_results = eval_fixed_3d_pipeline(model, cfg, root / "stage5_rsi2_fixed_3d_artifacts", token)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "terminal_bench_4_v4_0_0": tb4_results,
        "fixed_3d_pipeline_rsi2": fixed_3d_results,
    }
    write(root / "TERMINAL_BENCH4_AND_3D_FIXES_REPORT.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
