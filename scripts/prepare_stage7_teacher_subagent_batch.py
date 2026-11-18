#!/usr/bin/env python3
"""Prepare a student exploration + rendered `v8` Region Sheet batch for the Stage-7 Teacher Subagent.

Runs on the workstation right before a Phase C Teacher-Subagent Instructed RSI round:
1. Restores the latest committed `Spark-X2.6-4B-Loop48-En1B-VL` checkpoint from `work_root`.
2. Samples greedy + diverse `<think>` drafts on stratified `train`-only records across:
   - `synccraft_v41_sculpted_v8` / `synccraft_sculpted_stage_progression` / `synccraft_v41_proposer`,
   - `visual_primitives_point_count` / `visual_primitives_cot_grounding`,
   - `arc_agi_2_visual_primitives`,
   - `swebench_pro_patch` / `terminal_bench_agent`.
3. Renders 4-view Studio Turnarounds + 2x2 `SKILL.md` v8 Region Sheets for student Block-DSL drafts
   (and exports input images for visual tasks) into `<work_root>/teacher_subagent_round_<N>/`
   so the orchestrator can pull them locally and invoke a dedicated Teacher/Reviewer Subagent
   equipped with `skills/blockbench-sculpted-style/SKILL.md` (v8).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.common import read, validate_config, write  # noqa: E402
from colab.final.engine import Campaign  # noqa: E402
from colab.final.sculpted_style_corpus import render_sculpted_turnaround_views  # noqa: E402
from colab.final.sculpted_v8_v41_corpus import (  # noqa: E402
    render_v8_zoomed_region_sheet,
    run_v8_construct_check,
)
from colab.final.synccraft_bridge import _extract_dsl_candidate, strip_think_block  # noqa: E402
from colab.final.vision_rsi import _select_stratified_rsi_pool, score_rsi_candidate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=18)
    args = parser.parse_args()

    cfg = validate_config(read(args.config))
    root = Path(cfg["work_root"])
    receipt = read(root / "asset-receipt.json")
    paths = {k: Path(v) for k, v in receipt["paths"].items()}

    campaign = Campaign(cfg, paths, args.source_sha, token=None, key=None)
    state = read(root / "CAMPAIGN.json")
    opt, sched, _ = campaign.restore(state["last_checkpoint"])
    del opt, sched

    campaign.model.eval()
    explore_rids, _ = _select_stratified_rsi_pool(
        campaign.catalogue,
        seed=int(cfg["seed"]),
        round_no=int(args.round),
        total_records=int(cfg.get("vision_rsi_records_per_round", 64)),
    )

    out_dir = root / f"teacher_subagent_round_{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items_out: list[dict[str, Any]] = []

    for rid in explore_rids[: args.max_items]:
        row, media = campaign.media(rid)
        task = str(row.get("task") or "")
        gen_max = 1400 if task.startswith("synccraft_") else 768
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = campaign.model.answer(
                row,
                media,
                max_new_tokens=gen_max,
                enable_thinking=True,
                do_sample=False,
            )
        diag = score_rsi_candidate(row, pred, min_aesthetic_score=float(cfg.get("vision_rsi_min_aesthetic_score", 0.75)))
        entry: dict[str, Any] = {
            "rid": rid,
            "task": task,
            "prompt": str(row.get("prompt") or ""),
            "student_prediction": pred,
            "student_verified": bool(diag["verified"]),
            "student_score": float(diag["score"]),
            "student_reason": str(diag.get("reason") or ""),
            "oracle_reference_answer": str(row.get("answer") or ""),
            "rendered_files": [],
        }

        safe_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in rid)[:48]
        # Save any input images
        for im_i, im_obj in enumerate((media.get("images") or ())[:2]):
            if isinstance(im_obj, Image.Image):
                in_name = f"{safe_id}_input_{im_i}.png"
                im_obj.save(out_dir / in_name, format="PNG")
                entry["rendered_files"].append(in_name)

        # If task produces Block-DSL, compile & render student's own Region Sheet + Iso hero!
        if task in ("synccraft_v41_sculpted_v8", "synccraft_sculpted_style_creation", "synccraft_sculpted_stage_progression", "synccraft_sculpted_multiview_inverse"):
            _, body = strip_think_block(pred)
            dsl_cand = _extract_dsl_candidate(body)
            try:
                from synccraft.dsl.parser import parse_program

                prog = parse_program(dsl_cand)
                v8_rep = run_v8_construct_check(dsl_cand)
                entry["student_v8_check"] = v8_rep
                reg_im, reg_boxes = render_v8_zoomed_region_sheet(prog, panel_size=(256, 256))
                reg_name = f"{safe_id}_student_region_sheet.png"
                reg_im.save(out_dir / reg_name, format="PNG")
                entry["rendered_files"].append(reg_name)
                entry["region_boxes_1000"] = reg_boxes

                pngs, _ = render_sculpted_turnaround_views(prog, panel_size=(256, 256), include_silhouette_sheet=True)
                iso_name = f"{safe_id}_student_iso.png"
                (out_dir / iso_name).write_bytes(pngs["iso"])
                entry["rendered_files"].append(iso_name)
            except Exception as exc:
                entry["student_compile_error"] = str(exc)[:400]

        items_out.append(entry)

    write(
        out_dir / "STUDENT_BATCH_MANIFEST.json",
        {
            "round": int(args.round),
            "campaign_id": cfg["campaign_id"],
            "checkpoint": state.get("last_checkpoint"),
            "count": len(items_out),
            "items": items_out,
        },
    )
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "items": len(items_out)}))


if __name__ == "__main__":
    main()
