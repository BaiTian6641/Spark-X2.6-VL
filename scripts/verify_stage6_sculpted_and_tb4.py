#!/usr/bin/env python3
"""Verify Stage-6 Sculpted Blockbench Style (`v3.2` & `v4.0`), Human-Artist Studio Views, Terminal-Bench 4.0, and General VL/Science."""
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
from colab.final.inference import empty_media, generate, load, visual_media
from colab.final.reasoning_agentic_corpus import (
    GPQA_REPO_ID,
    GPQA_REVISION,
    TERMINAL_BENCH4_REPO_ID,
    TERMINAL_BENCH4_REVISION,
    check_generation_collapse_health,
)
from colab.final.sculpted_style_corpus import (
    build_sculpted_crimson_western_dragon_dsl,
    evaluate_sculpted_blockbench_style,
    evaluate_sculpted_task_completion,
    render_sculpted_turnaround_views,
)
from colab.final.synccraft_bridge import _camera_unit_meta, ensure_synccraft_importable, strip_think_block
from colab.final.world_pipeline import SyncCraftWorldPipeline


def _hf_file(repo_id: str, filename: str, *, revision: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision, token=token))


def _collect_tb4_holdout_bucket0(token: str | None, limit: int = 16) -> list[str]:
    """Return strictly held-out `bucket == 0` Terminal-Bench 4.0 (`v4.0.0`) task IDs (`0%` overlap with train)."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info4 = api.dataset_info(TERMINAL_BENCH4_REPO_ID, revision=TERMINAL_BENCH4_REVISION)
    tb4_by_task: dict[str, set[str]] = {}
    for sib in info4.siblings or ():
        parts = sib.rfilename.split("/")
        if len(parts) >= 3 and parts[0] == "tasks":
            tb4_by_task.setdefault(parts[1], set()).add(sib.rfilename)

    holdout = [
        tid
        for tid in sorted(tb4_by_task)
        if int(hashlib.sha256(f"tb3_holdout_v1:{tid}".encode("utf-8")).hexdigest()[:8], 16) % 3 == 0
        and f"tasks/{tid}/instruction.md" in tb4_by_task[tid]
    ]
    return holdout[:limit]


def _validate_terminal_response(body: str, tid: str) -> tuple[bool, str]:
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


def eval_terminal_bench4_holdout(model: Any, holdout_tids: list[str], token: str | None) -> dict[str, Any]:
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
        "benchmark": "Terminal-Bench 4.0 (v4.0.0 Disjoint Holdout Bucket-0)",
        "revision": TERMINAL_BENCH4_REVISION,
        "evaluated": len(holdout_tids),
        "valid_executable_or_tool_actions": valid_cnt,
        "valid_rate": round(valid_cnt / n, 4),
        "healthy_generation_rate": round(healthy_cnt / n, 4),
        "think_close_rate": round(think_closed_cnt / n, 4),
        "tasks": per_task,
    }


def eval_gpqa_diamond_and_live_python(model: Any, token: str | None, n_gpqa: int = 24) -> dict[str, Any]:
    import csv

    gpqa_csv = _hf_file(GPQA_REPO_ID, "gpqa_diamond.csv", revision=GPQA_REVISION, token=token)
    with gpqa_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:n_gpqa]

    correct = 0
    closed = 0
    letter_re = re.compile(r"(?:Answer|Final Answer)\s*:\s*\(?([A-D])\)?", re.IGNORECASE)
    for idx, r in enumerate(rows):
        q = str(r["Question"]).strip()
        opts = [
            str(r["Correct Answer"]).strip(),
            str(r["Incorrect Answer 1"]).strip(),
            str(r["Incorrect Answer 2"]).strip(),
            str(r["Incorrect Answer 3"]).strip(),
        ]
        # Deterministic rotation by question index
        rot = idx % 4
        ordered = opts[rot:] + opts[:rot]
        gold_letter = "ABCD"[(4 - rot) % 4]
        prompt = (
            f"{q}\n\n"
            f"A) {ordered[0]}\nB) {ordered[1]}\nC) {ordered[2]}\nD) {ordered[3]}\n\n"
            "Think step-by-step inside `<think>...</think>`, then end with `Answer: X` (where X is A, B, C, or D)."
        )
        out = generate(
            model,
            prompt,
            media=empty_media(),
            enable_thinking=True,
            max_new_tokens=1536,
            max_think_tokens=896,
            task="gpqa_extended_science",
        )
        if "</think>" in out:
            closed += 1
        _, body = strip_think_block(out)
        m = letter_re.search(body)
        pred_l = m.group(1).upper() if m else ""
        if pred_l == gold_letter:
            correct += 1

    # Live Python BFS execution (`thinking=False` and `thinking=True`)
    bfs_results: dict[str, bool] = {}
    for th_mode in (False, True):
        out_py = generate(
            model,
            (
                "Write a self-contained Python 3 function `shortest_grid_path(grid)` that returns the integer shortest path "
                "length from `(0, 0)` to `(R-1, C-1)` moving 4-directionally over `0` cells (`1` = obstacle), or `-1` if unreachable. "
                "Return the code inside ```python ... ```."
            ),
            media=empty_media(),
            enable_thinking=th_mode,
            max_new_tokens=1280,
            max_think_tokens=512 if th_mode else None,
        )
        _, b_py = strip_think_block(out_py)
        m_py = re.search(r"```python\s*\n(.*?)```", b_py, re.DOTALL | re.IGNORECASE)
        code_s = m_py.group(1) if m_py else b_py
        test_harness = (
            code_s
            + "\nassert shortest_grid_path([[0,0,0],[1,1,0],[0,0,0]]) == 4\n"
            + "assert shortest_grid_path([[0,1],[1,0]]) == -1\nprint('BFS_OK')\n"
        )
        proc = subprocess.run([sys.executable, "-c", test_harness], capture_output=True, text=True, timeout=5, check=False)
        bfs_results[f"thinking_{th_mode}"] = proc.returncode == 0 and "BFS_OK" in proc.stdout

    return {
        "gpqa_diamond_evaluated": len(rows),
        "gpqa_diamond_accuracy": round(correct / max(1, len(rows)), 4),
        "gpqa_think_close_rate": round(closed / max(1, len(rows)), 4),
        "live_python_bfs": bfs_results,
    }


def eval_sculpted_3d_and_v4_suite(model: Any, cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program
    from synccraft.world.assembly_world_tasks import (
        build_modular_assembly_supervision_suite,
        evaluate_modular_assembly_world_tasks,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SyncCraftWorldPipeline(
        model,
        config={**cfg, "style_mode": "sculpted", "min_aesthetic_score": 0.74},
        default_max_new_tokens=3584,
    )

    # 1. Sculpted Blockbench Dragon Generation
    dragon_res = pipeline.generate_3d_asset(
        "Design a vibrant, deliberately sculpted 3D Blockbench asset for: crimson and warm-cream western dragon "
        "(Standard), using Sculpted Block-DSL (`audit_profile: sculpted`, `size_step: 0.5`, `hue_shift` ramps, "
        "seated joints, S-curved neck, partitioned wing membrane `plane`s, curved horns, and glowing focal eyes). "
        "Avoid rigid grid-aligned boxes or flat unpainted faces.",
        asset_name="sculpted_crimson_dragon_stage6",
        output_dir=out_dir / "sculpted_crimson_dragon_stage6",
        export_formats=("bbmodel", "bedrock", "geckolib", "gltf_pbr", "labpbr"),
    )

    # 2. Sculpted Hard-Surface Expedition Vehicle / Automaton Generation
    crawler_res = pipeline.generate_3d_asset(
        "Design a vibrant, deliberately sculpted 3D Blockbench asset for: six-wheeled polar expedition crawler truck "
        "(Standard Hard-Surface Vehicle), using Sculpted Block-DSL (`audit_profile: sculpted`, `size_step: 0.5`, "
        "`hue_shift` ramps, snow-capped roof slabs, safety-orange rack/ladder/antenna greebles, 45-degree octagonal wheel rims, and glowing headlamps).",
        asset_name="sculpted_polar_crawler_stage6",
        output_dir=out_dir / "sculpted_polar_crawler_stage6",
        export_formats=("bbmodel", "bedrock", "gltf_pbr", "labpbr"),
    )

    # 3. Sculpted Cosy Diorama Scene Generation
    diorama_res = pipeline.generate_3d_asset(
        "Design a vibrant, deliberately sculpted 3D Blockbench asset for: cosy alchemist workbench alcove with warm "
        "glowing lantern focal point, skewed leaning books, mortar & pestle, and potted leaf `plane`s (Diorama Scene), "
        "using Sculpted Block-DSL (`audit_profile: sculpted`, `size_step: 0.5`, `hue_shift` ramps).",
        asset_name="sculpted_alchemist_diorama_stage6",
        output_dir=out_dir / "sculpted_alchemist_diorama_stage6",
        export_formats=("bbmodel", "gltf_pbr", "labpbr"),
    )

    # 4. Stage-C Stiff Blockout -> Stage-E Sculpted Upgrade
    _, stage_c_dragon, _ = build_sculpted_crimson_western_dragon_dsl(99)
    prog_c = parse_program(stage_c_dragon)
    pngs_c, _ = render_sculpted_turnaround_views(prog_c, views=("south", "right", "top", "iso"), panel_size=(192, 192))
    pil_c = [Image.open(io.BytesIO(pngs_c[v])) for v in ("south", "right", "top", "iso")]
    upgrade_res = pipeline.generate_3d_asset(
        "Critique this stiff Stage-C axis-aligned blockout from its 4 views (`south`, `right`, `top`, `iso`) "
        "and upgrade it into a vibrant Stage-E sculpted Blockbench western dragon (`audit_profile: sculpted`, `size_step: 0.5`):\n"
        f"```blockdsl\n{stage_c_dragon.strip()}\n```",
        images=pil_c,
        asset_name="upgraded_stage_c_to_sculpted_e",
        output_dir=out_dir / "upgraded_stage_c_to_sculpted_e",
        export_formats=("bbmodel", "bedrock", "gltf_pbr"),
    )

    # 5. Visual Sculpted Self-Critique Accuracy (Sculpted E vs Stiff C)
    se_99, _, _ = build_sculpted_crimson_western_dragon_dsl(99)
    prog_e = parse_program(se_99)
    pngs_e, _ = render_sculpted_turnaround_views(prog_e, panel_size=(192, 192), include_silhouette_sheet=True)
    pngs_c_sil, _ = render_sculpted_turnaround_views(prog_c, panel_size=(192, 192), include_silhouette_sheet=True)
    critique_checks: list[dict[str, Any]] = []
    for lbl, iso_b, sil_b, p_ref in (
        ("sculpted_e_holdout", pngs_e["iso"], pngs_e["silhouette_sheet"], prog_e),
        ("stiff_c_holdout", pngs_c["iso"], pngs_c_sil["silhouette_sheet"], prog_c),
    ):
        ev_ref = evaluate_sculpted_blockbench_style(p_ref)
        oracle_crit = {
            "grade": ev_ref["grade"],
            "is_stiff_or_voxel_like": ev_ref["is_stiff_or_voxel_like"],
            "has_spine_gesture_curve": ev_ref["has_spine_gesture_curve"],
            "has_cutout_or_thin_planes": ev_ref["has_cutout_or_thin_planes"],
            "hue_shifted_palettes": ev_ref["hue_shifted_ramp_count"] >= 2,
            "audit_passed": ev_ref["audit_passed"],
        }
        med = visual_media(
            [Image.open(io.BytesIO(iso_b)), Image.open(io.BytesIO(sil_b))],
            visual_kind="state_pair",
            visual_units=[{"visual_kind": "multiview_render", "view_id": "iso"}, {"visual_kind": "multiview_render", "view_id": "top"}],
        )
        pred_crit = generate(
            model,
            (
                "Inspect the 3/4 studio render (image 1) and 4-view black silhouette check sheet (image 2). "
                "Perform a Sculpted Blockbench Style critique and output JSON with keys "
                "`audit_passed`, `grade` (`sculpted_master_tier` | `sculpted_standard` | `stiff_blockout`), "
                "`has_cutout_or_thin_planes`, `has_spine_gesture_curve`, `hue_shifted_palettes`, and `is_stiff_or_voxel_like`."
            ),
            media=med,
            enable_thinking=True,
            max_new_tokens=640,
            max_think_tokens=256,
            task="synccraft_sculpted_self_critique",
        )
        scored_c = evaluate_sculpted_task_completion({"task": "synccraft_sculpted_self_critique", "metadata": {"oracle_json": oracle_crit}}, pred_crit)
        critique_checks.append({"case": lbl, **scored_c})

    # 6. Block-DSL v4.0 Modular Assembly Suite (`@layout`, `@component`, `@assembly`)
    v4_suite = build_modular_assembly_supervision_suite()
    v4_completions: dict[str, str] = {}
    from colab.final.synccraft_bridge import _extract_dsl_candidate, _extract_json_candidate

    for item in v4_suite:
        tid = str(item["task_id"])
        raw_v4 = generate(
            model,
            str(item["prompt"]),
            media=empty_media(),
            enable_thinking=True,
            max_new_tokens=1536,
            max_think_tokens=384,
            task="synccraft_v4_modular_assembly",
        )
        _, b_v4 = strip_think_block(raw_v4)
        if item["task_type"] == "propagate_layout_and_instance_controls":
            try:
                v4_completions[tid] = json.dumps(_extract_json_candidate(b_v4))
            except Exception:
                v4_completions[tid] = b_v4
        else:
            v4_completions[tid] = _extract_dsl_candidate(b_v4)
    v4_eval = evaluate_modular_assembly_world_tasks(v4_suite, v4_completions)

    def _summarize_gen(r: dict[str, Any]) -> dict[str, Any]:
        sc_ev = r.get("sculpted_evaluation") or {}
        aes_ev = r.get("aesthetic_evaluation") or {}
        bnd = r.get("bundle") or {}
        return {
            "passed": r["passed"],
            "initial_passed": r["initial_passed"],
            "reaudits_used": r["reaudits_used"],
            "boxes": r["boxes"],
            "bones": r["bones"],
            "audit_profile": bnd.get("audit_profile"),
            "size_step": bnd.get("size_step"),
            "sculpted_score": sc_ev.get("sculpted_score"),
            "sculpted_grade": sc_ev.get("grade"),
            "plane_count": sc_ev.get("plane_count"),
            "rotated_fraction": sc_ev.get("rotated_element_fraction"),
            "aesthetic_score": aes_ev.get("aesthetic_score"),
            "studio_views": bnd.get("studio_views"),
            "failure_reason": r["failure_reason"],
        }

    return {
        "sculpted_dragon": _summarize_gen(dragon_res),
        "sculpted_polar_crawler": _summarize_gen(crawler_res),
        "sculpted_alchemist_diorama": _summarize_gen(diorama_res),
        "stage_c_to_sculpted_e_upgrade": _summarize_gen(upgrade_res),
        "visual_sculpted_self_critique": critique_checks,
        "blockdsl_v4_modular_suite": v4_eval,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True, help="Promoted Stage-6 BF16 release directory")
    ap.add_argument("--fp8-model-dir", default="", help="Optional Stage-6 FP8 release directory for spot-check")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    cfg = read(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb4_holdout = _collect_tb4_holdout_bucket0(token, limit=16)

    model = load(args.model_dir)
    sculpted_rep = eval_sculpted_3d_and_v4_suite(model, cfg, out_dir / "bf16_assets")
    tb4_rep = eval_terminal_bench4_holdout(model, tb4_holdout, token)
    sci_rep = eval_gpqa_diamond_and_live_python(model, token, n_gpqa=24)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fp8_spot: dict[str, Any] | None = None
    if args.fp8_model_dir and Path(args.fp8_model_dir).is_dir():
        fp8_m = load(args.fp8_model_dir)
        pipe_fp8 = SyncCraftWorldPipeline(
            fp8_m,
            config={**cfg, "style_mode": "sculpted", "min_aesthetic_score": 0.72},
            default_max_new_tokens=3584,
        )
        dr_fp8 = pipe_fp8.generate_3d_asset(
            "Design a vibrant, deliberately sculpted 3D Blockbench asset for: chunky expressive ice drake with oversized "
            "fanged jaw and dorsal cyan crystal cluster (Standard), using Sculpted Block-DSL (`audit_profile: sculpted`, `size_step: 0.5`).",
            asset_name="sculpted_glacial_drake_fp8",
            output_dir=out_dir / "fp8_assets" / "sculpted_glacial_drake_fp8",
            export_formats=("bbmodel", "bedrock", "gltf_pbr"),
        )
        sc_fp8 = dr_fp8.get("sculpted_evaluation") or {}
        fp8_spot = {
            "passed": dr_fp8["passed"],
            "boxes": dr_fp8["boxes"],
            "bones": dr_fp8["bones"],
            "sculpted_score": sc_fp8.get("sculpted_score"),
            "sculpted_grade": sc_fp8.get("grade"),
            "studio_views": (dr_fp8.get("bundle") or {}).get("studio_views"),
        }
        del fp8_m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_doc = {
        "model_dir": str(args.model_dir),
        "fp8_model_dir": str(args.fp8_model_dir) if args.fp8_model_dir else None,
        "sculpted_3d_and_v4": sculpted_rep,
        "terminal_bench_4_holdout": tb4_rep,
        "science_and_live_python": sci_rep,
        "fp8_sculpted_spot_check": fp8_spot,
    }
    write(out_dir / "STAGE6_SCULPTED_AND_TB4_VERIFICATION.json", final_doc)
    print(json.dumps(final_doc, indent=2))


if __name__ == "__main__":
    main()
