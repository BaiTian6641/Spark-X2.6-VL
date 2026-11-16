#!/usr/bin/env python3
"""Comprehensive General + 3D Spatial + SyncCraft v3.2 World-Model & Asset-Pipeline Verifier.

Evaluates a trained `Spark-X2.6-VL` checkpoint (`final-model` and optional `final-model-fp8`)
across five real capability pillars on held-out split data and live 3D asset generation:

1. General Bilingual Vision-Language & Compositional RGBA Grounding (`SparkX2.6-MM` + `compositional_vqa`)
2. 3D Spatial Reasoning (`array/SAT` held-out `audit`/`promotion` split across 5 spatial categories)
3. Exact 4-Way RGBA/UV Seam Diagnosis (`a0_o0`, `a1_o0`, `a0_o1`, `a1_o1`) + Counterfactual Alpha Flips
4. SyncCraft v3.2 World Capability Ladder (L1-L7), `ADR 0007` `sculpted` Rigs & `spin(...)`/`foot_cycle` Gait
5. End-to-End Working-Pipeline 3D Asset Generation (`SyncCraftWorldPipeline` -> `.bbmodel`, `.geo.json`,
   `.animation.json`, `.gltf` PBR, and LabPBR 1.3 with `serialized_vertex_max_error <= 1e-6` and `CostLedger`)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.common import local, read, validate_config, write
from colab.final.data import Catalogue, MediaReader, prepare_media
from colab.final.evaluate import error_rate
from colab.final.inference import generate, load, visual_media
from colab.final.synccraft_bridge import (
    build_parametric_creature_world,
    build_parametric_walking_citadel_world,
    build_uv_seam_alpha_diagnosis_record,
    evaluate_synccraft_completion,
    render_world_views_rgba,
)
from colab.final.world_pipeline import SyncCraftWorldPipeline


def run_verification(
    config_path: Path,
    *,
    checkpoint_dir: Path | None = None,
    compare_fp8: bool = True,
    max_per_task: int = 4,
) -> dict[str, Any]:
    import io
    from PIL import Image

    cfg = validate_config(read(config_path))
    root = local(cfg["work_root"], execution_target=cfg.get("execution_target", "local_workstation"))
    ckpt = checkpoint_dir or (root / "final-model")
    model = load(ckpt, device="cuda")

    handoff = read(root / "metadata" / "HANDOFF.json")
    catalogue = Catalogue(root / "metadata" / "corpus.sqlite", modalities=cfg.get("modalities", "vision_text"))
    reader = MediaReader(root, handoff, cfg, token=os.environ.get("HF_TOKEN"))

    # 1. Held-out task-stratified evaluation on `audit` split
    audit_items = catalogue.select("audit", ("image", "video", "text"))
    by_task: dict[str, list[str]] = defaultdict(list)
    for rid, _fam, _lang, _comp, _shard in audit_items:
        row, _ = catalogue.row(rid)
        by_task[str(row["task"])].append(rid)

    heldout_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_name in sorted(by_task):
        for rid in by_task[task_name][:max_per_task]:
            row, pkt = catalogue.row(rid)
            with reader.packet(pkt) as get:
                media = prepare_media(row, get, cfg)
            max_toks = 6400 if task_name.startswith("synccraft_") else int(cfg.get("evaluation_max_new_tokens", 384))
            pred = generate(
                model,
                row["prompt"],
                media=media,
                enable_thinking=bool(row.get("enable_thinking", False)),
                max_new_tokens=max_toks,
                task=row["task"],
                language=row["language"],
            )
            entry: dict[str, Any] = {
                "id": rid,
                "language": row["language"],
                "prompt_preview": row["prompt"][:140],
                "prediction_preview": pred[:220],
                "reference_preview": str(row["answer"])[:220],
            }
            if task_name.startswith("synccraft_"):
                sc = evaluate_synccraft_completion(row, pred)
                entry["passed"] = bool(sc.get("passed"))
                entry["synccraft_eval"] = sc
            elif task_name.startswith("sat_3d_"):
                norm_p = pred.strip().lower()
                norm_r = str(row["answer"]).strip().lower()
                entry["passed"] = (norm_r in norm_p) or (norm_p == norm_r)
            else:
                er = error_rate(str(row["answer"]), pred, row["language"])
                entry["edit_error_rate"] = round(float(er["rate"]), 4)
            heldout_results[task_name].append(entry)

    # 2. Unseen 4-way RGBA/UV diagnosis (`variant=91` unseen creature rig)
    unseen_ws = build_parametric_creature_world(variant=91, subdivisions_per_unit=2)
    uv_4way: list[dict[str, Any]] = []
    for cut_flag in (False, True):
        for ov_flag in (False, True):
            rec, blobs = build_uv_seam_alpha_diagnosis_record(
                unseen_ws,
                asset_id=f"unseen_v91_c{int(cut_flag)}_o{int(ov_flag)}",
                family_id="unseen_probe_bio",
                inject_alpha_cutout=cut_flag,
                inject_uv_overlap=ov_flag,
                vary_target_faces=True,
                panel_size=(160, 160),
            )
            imgs = [Image.open(io.BytesIO(blobs[fp])) for fp in rec["frames"]]
            med = visual_media(imgs, visual_kind="uv_atlas", visual_units=rec["visual_units"])
            pred = generate(
                model,
                rec["prompt"],
                media=med,
                enable_thinking=True,
                max_new_tokens=768,
                task=rec["task"],
            )
            ev = evaluate_synccraft_completion(rec, pred)
            uv_4way.append(
                {
                    "case": f"a{int(cut_flag)}_o{int(ov_flag)}",
                    "passed": bool(ev["passed"]),
                    "field_accuracy": ev.get("field_accuracy"),
                    "prediction": pred[-240:],
                }
            )

    # 3. End-to-End Working Pipeline 3D Asset Generation & Multi-Engine Export
    out_assets_dir = root / "generated-3d-assets"
    pipeline = SyncCraftWorldPipeline(model, config=cfg, default_max_new_tokens=2560, enable_thinking=True)

    # 3a. Text-to-3D asset generation with bounded SAT/PGD repair & multi-engine export
    text_to_3d = pipeline.generate_3d_asset(
        (
            "Design a complete SyncCraft Block-DSL v3.2 3D asset for an articulated quadruped sentinel hound "
            "(archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: strict, "
            "at least 5 boxes across 3+ bones, including @kinematics two-bone leg IK and @behavior walk_cycle "
            "with foot_cycle(...) stance slip <= 0.02u) that passes 15-axis SAT + P1-02 audit."
        ),
        prompt_spec={"min_boxes": 5},
        asset_name="sentinel_hound_v32",
        output_dir=out_assets_dir / "sentinel_hound_v32",
        export_formats=("bbmodel", "bedrock", "geckolib", "gltf_pbr", "labpbr"),
        max_reaudits=3,
        seed=901,
    )

    # 3b. Multiview-Image-to-3D asset reconstruction on an unseen parametric creature (`variant=93`)
    unseen_mv_ws = build_parametric_creature_world(variant=93, subdivisions_per_unit=2)
    png_views, _, _, _ = render_world_views_rgba(unseen_mv_ws, views=("south", "right", "top", "iso"), panel_size=(160, 160))
    ref_pil_views = [Image.open(io.BytesIO(png_views[v])) for v in ("south", "right", "top", "iso")]
    image_to_3d = pipeline.generate_3d_asset(
        (
            "Reconstruct a collision-free SyncCraft Block-DSL v3.2 program matching these 4 orthographic views "
            "(south, right, top, iso) with target_profile='bedrock_entity', texel_density=16."
        ),
        images=ref_pil_views,
        prompt_spec={"min_boxes": 4},
        asset_name="reconstructed_creature_v93",
        output_dir=out_assets_dir / "reconstructed_creature_v93",
        export_formats=("bbmodel", "bedrock", "gltf_pbr"),
        max_reaudits=3,
        seed=903,
    )

    # 3c. Flagship ADR 0007 Sculpted Colossal Citadel multi-format export smoke verification
    citadel_ws = build_parametric_walking_citadel_world(variant=2, subdivisions_per_unit=1)
    from colab.final.world_pipeline import export_program_bundle

    citadel_export = export_program_bundle(
        citadel_ws.program,
        asset_name="colossal_walking_citadel_v32",
        output_dir=out_assets_dir / "colossal_walking_citadel_v32",
        formats=("bbmodel", "bedrock", "gltf_pbr", "labpbr"),
        world_state=citadel_ws,
        seed=905,
    )

    fp8_parity: dict[str, Any] | None = None
    fp8_dir = root / "final-model-fp8"
    if compare_fp8 and fp8_dir.is_dir() and (fp8_dir / "MODEL_RELEASE.json").is_file():
        del model
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        model_fp8 = load(fp8_dir, device="cuda")
        pipe_fp8 = SyncCraftWorldPipeline(model_fp8, config=cfg, default_max_new_tokens=2560, enable_thinking=True)
        fp8_gen = pipe_fp8.generate_3d_asset(
            (
                "Design a complete SyncCraft Block-DSL v3.2 3D asset for an articulated quadruped sentinel hound "
                "(archetype: entity, target_profile: bedrock_entity, texel_density: 16, audit_profile: strict, "
                "at least 5 boxes across 3+ bones, including @kinematics and @behavior) that passes 15-axis SAT + P1-02 audit."
            ),
            prompt_spec={"min_boxes": 5},
            asset_name="sentinel_hound_v32_fp8",
            output_dir=out_assets_dir / "sentinel_hound_v32_fp8",
            export_formats=("bbmodel", "bedrock", "gltf_pbr"),
            max_reaudits=3,
            seed=901,
        )
        fp8_parity = {
            "passed": fp8_gen["passed"],
            "reaudits_used": fp8_gen["reaudits_used"],
            "boxes": fp8_gen["boxes"],
            "bones": fp8_gen["bones"],
            "cost_ledger": fp8_gen["cost_ledger"],
            "written_files": (fp8_gen["bundle"] or {}).get("written_files", {}),
        }

    task_pass_rates = {}
    for tname, items in heldout_results.items():
        scored = [x for x in items if "passed" in x]
        if scored:
            task_pass_rates[tname] = {
                "n": len(scored),
                "pass_rate": round(sum(1 for x in scored if x["passed"]) / len(scored), 4),
            }
        else:
            errs = [x["edit_error_rate"] for x in items if "edit_error_rate" in x]
            task_pass_rates[tname] = {
                "n": len(items),
                "mean_edit_error_rate": round(sum(errs) / max(1, len(errs)), 4),
            }

    summary = {
        "campaign_id": cfg["campaign_id"],
        "checkpoint": str(ckpt),
        "heldout_task_summary": task_pass_rates,
        "unseen_4way_uv_diagnosis": {
            "pass_count": sum(1 for x in uv_4way if x["passed"]),
            "total": len(uv_4way),
            "cases": uv_4way,
        },
        "working_pipeline_3d_generation": {
            "text_to_3d_sentinel_hound": {
                "passed": text_to_3d["passed"],
                "reaudits_used": text_to_3d["reaudits_used"],
                "boxes": text_to_3d["boxes"],
                "bones": text_to_3d["bones"],
                "cost_ledger": text_to_3d["cost_ledger"],
                "bedrock_vertex_max_error": ((text_to_3d["bundle"] or {}).get("exports", {}).get("bedrock", {}) or {}).get("serialized_vertex_max_error"),
                "written_files": (text_to_3d["bundle"] or {}).get("written_files", {}),
            },
            "multiview_to_3d_v93": {
                "passed": image_to_3d["passed"],
                "reaudits_used": image_to_3d["reaudits_used"],
                "boxes": image_to_3d["boxes"],
                "bones": image_to_3d["bones"],
                "cost_ledger": image_to_3d["cost_ledger"],
                "written_files": (image_to_3d["bundle"] or {}).get("written_files", {}),
            },
            "sculpted_colossal_citadel_export": {
                "audit_profile": citadel_export["audit_profile"],
                "boxes": citadel_export["boxes"],
                "bones": citadel_export["bones"],
                "sculpted_overlap_count": citadel_export["scene_audit"]["sculpted_overlap_count"],
                "behavior_audits": citadel_export["behavior_audits"],
                "bedrock_vertex_max_error": citadel_export["exports"]["bedrock"]["serialized_vertex_max_error"],
                "written_files": citadel_export["written_files"],
            },
            "fp8_q4g128_pipeline_generation": fp8_parity,
        },
        "heldout_details": dict(heldout_results),
    }
    report_path = root / "evaluations" / "WORLD_V32_CAPABILITY_VERIFICATION.json"
    write(report_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "colab/final/campaign_rtx_pro_4000_world_v32.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-fp8", action="store_true")
    parser.add_argument("--max-per-task", type=int, default=4)
    args = parser.parse_args()
    rep = run_verification(
        args.config,
        checkpoint_dir=args.checkpoint,
        compare_fp8=not args.no_fp8,
        max_per_task=args.max_per_task,
    )
    print(
        json.dumps(
            {
                "heldout_task_summary": rep["heldout_task_summary"],
                "unseen_4way_uv_diagnosis": rep["unseen_4way_uv_diagnosis"],
                "working_pipeline_3d_generation": rep["working_pipeline_3d_generation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
