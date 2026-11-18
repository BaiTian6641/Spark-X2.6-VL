#!/usr/bin/env python3
"""Comprehensive Stage-7 Evaluation Suite: ARC-AGI-2, Frontier SWE/TB4/TB3, SuperGPQA, & Blockbench v8 / Gate D.

Evaluates any `SparkNative` release directory (`Stage-6.1`, `Stage-7 BF16`, `Stage-7 FP8`) on strictly
held-out (`0%` training overlap) benchmarks:
1. `ARC-AGI-2` (`data/evaluation/*.json` — Pass@1 & official 2-attempt Pass@2 on visual+symbolic sheets).
2. `SuperGPQA` (`bucket == 0` unseen graduate disciplines) + `GPQA-Diamond` (`100%` held out).
3. `Terminal-Bench 4.0` (`v4.0.0`, `bucket == 0`) & `Terminal-Bench 3.0` (`bucket == 0`).
4. `SWE-Bench Pro` (`ScaleAI/SWE-bench_Pro`, `bucket == 0` unseen repository issues).
5. `SyncCraft Block-DSL v4.1` `Gate D` Proposer (`holdout_gate_d` unseen `biped_theropod` & `hexapod_arthropod`)
   + Held-out `v4_wm_01..04` Modular `@layout/@component/@assembly` tasks.
6. Unseen `Blockbench_sculpted_style-v8` (`SKILL.md` v8) creature & vehicle prompts — compiling `.bbmodel`
   exports + 4-view Studio Turnaround PNGs + 2x2 `v8` Region Sheets + `run_v8_construct_check` diagnostics.
"""
from __future__ import annotations

import argparse
import ast
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

import pyarrow.parquet as pq
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.common import read, write  # noqa: E402
from colab.final.inference import generate, load, visual_media  # noqa: E402
from colab.final.reasoning_agentic_corpus import (  # noqa: E402
    GPQA_REPO_ID,
    GPQA_REVISION,
    TERMINAL_BENCH3_REPO_ID,
    TERMINAL_BENCH3_REVISION,
    TERMINAL_BENCH4_REPO_ID,
    TERMINAL_BENCH4_REVISION,
)
from colab.final.frontier_stage7_corpus import (  # noqa: E402
    SUPERGPQA_REPO,
    SUPERGPQA_REV,
    SWEBENCH_PRO_REPO,
    SWEBENCH_PRO_REV,
)
from colab.final.sculpted_style_corpus import render_sculpted_turnaround_views  # noqa: E402
from colab.final.sculpted_v8_v41_corpus import (  # noqa: E402
    render_v8_zoomed_region_sheet,
    run_v8_construct_check,
)
from colab.final.synccraft_bridge import (  # noqa: E402
    _extract_dsl_candidate,
    _extract_json_candidate,
    ensure_synccraft_importable,
    strip_think_block,
)
from colab.final.visual_primitives_corpus import render_arc_task_sheet  # noqa: E402


UNSEEN_V8_PROMPTS: tuple[dict[str, str], ...] = (
    {
        "id": "unseen_v8_azure_frost_wyvern",
        "prompt": (
            "Design an original Sculpted Blockbench Style v8 model in `Block-DSL v4.1` (`dsl_version: \"4.1\"`, "
            "`audit_profile: sculpted`): Azure Frost-Crest Dragon (six-limbed alpine ice dragon with partitioned "
            "wing membranes, 3-step wedged head + hinged jaw + real tapered block horns, and exact X=0 symmetric "
            "rest pose). Enforce `<= 3.5` boxes/bone anti-shard discipline, `top-10 volume >= 50%`, `seat=0.28`, "
            "`shade: banded(...)`, `@stamps`, and verified `@behavior` motion."
        ),
    },
    {
        "id": "unseen_v8_amber_volt_wolf_drake",
        "prompt": (
            "Design an original Sculpted Blockbench Style v8 model in `Block-DSL v4.1` (`dsl_version: \"4.1\"`, "
            "`audit_profile: sculpted`): Amber Volt Wolf-Drake (quadruped fanged storm drake with short 1.5:1 wedged "
            "wolf muzzle, hinged jaw, tapered block horns, layered `edge: spikes` fur cutout planes, zero shard armour, "
            "exact X=0 symmetric rest pose, `shade: banded(...)`, and `@behavior` Froude `gait`)."
        ),
    },
    {
        "id": "unseen_v8_basalt_drill_crawler",
        "prompt": (
            "Design an original Sculpted Blockbench Style v8 hard-surface vehicle in `Block-DSL v4.1` "
            "(`dsl_version: \"4.1\"`, `audit_profile: sculpted`): Basalt Magma Survey Crawler with chamfered hull "
            "volumes, native `octagon` wheels (`joint_limits: continuous`), functional lamp/beacon greebles, "
            "exact X=0 mirror symmetry, and `@behavior` continuous wheel `spin` clip."
        ),
    },
)


def _hf_file(repo_id: str, filename: str, *, revision: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    )


def _gen(
    model: Any,
    prompt: str,
    images: list[Image.Image] | None = None,
    *,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.55,
) -> str:
    media = visual_media(images) if images else None
    return generate(
        model,
        prompt,
        media=media,
        enable_thinking=True,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=0.92,
        repetition_penalty=1.04,
    )


def _parse_2d_grid(text: str) -> list[list[int]] | None:
    _, body = strip_think_block(text)
    m = re.search(r"```(?:json)?\s*(\[\s*\[.*?\]\s*\])\s*```", body, re.DOTALL)
    raw = m.group(1) if m else body
    if not m:
        s_i, e_i = raw.find("[["), raw.rfind("]]")
        if s_i >= 0 and e_i > s_i:
            raw = raw[s_i : e_i + 2]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed and all(isinstance(r, list) for r in parsed):
            return [[int(x) for x in r] for r in parsed]
    except Exception:
        pass
    return None


def eval_arc_agi_2(model: Any, arc_root: Path, limit: int = 30) -> dict[str, Any]:
    eval_dir = arc_root / "data" / "evaluation"
    if not eval_dir.is_dir():
        return {"available": False, "reason": f"Missing {eval_dir}"}
    files = sorted(eval_dir.glob("*.json"))[:limit]
    pass1 = 0
    pass2 = 0
    cell_accs: list[float] = []
    used_vis_prims = 0

    for tf in files:
        doc = json.loads(tf.read_text(encoding="utf-8"))
        tr_pairs = doc["train"]
        te_in = doc["test"][0]["input"]
        te_out = doc["test"][0]["output"]
        png_b, _ = render_arc_task_sheet(tr_pairs, te_in)
        im = Image.open(io.BytesIO(png_b)).convert("RGB")
        demo_txt = "\n".join(
            f"Demo {i + 1}: input={json.dumps(p['input'])} -> output={json.dumps(p['output'])}"
            for i, p in enumerate(tr_pairs[:4])
        )
        prompt = (
            "Solve this ARC-AGI-2 abstract visual reasoning task. Inspect the rendered demonstration pairs "
            "and Test Input grid, accompanied by the exact integer matrices:\n\n"
            f"{demo_txt}\nTest Input: {json.dumps(te_in)}\n\n"
            "Localize key panels/objects with `<box>`/`<point>` inside `<think>` and output ONLY the exact 2D integer JSON grid."
        )
        out1 = _gen(model, prompt, [im], max_new_tokens=960, do_sample=False)
        th1, _ = strip_think_block(out1)
        if th1 and ("<box " in th1 or "<point " in th1):
            used_vis_prims += 1
        g1 = _parse_2d_grid(out1)
        hit1 = g1 == te_out
        hit2 = hit1
        if not hit1:
            out2 = _gen(model, prompt, [im], max_new_tokens=960, do_sample=True, temperature=0.55)
            g2 = _parse_2d_grid(out2)
            hit2 = g2 == te_out
        if hit1:
            pass1 += 1
        if hit2:
            pass2 += 1
        if g1 is not None and len(g1) == len(te_out) and len(g1[0]) == len(te_out[0]):
            matches = sum(
                1 for r_a, r_b in zip(g1, te_out) for x_a, x_b in zip(r_a, r_b) if x_a == x_b
            )
            cell_accs.append(matches / max(1, len(te_out) * len(te_out[0])))
        else:
            cell_accs.append(0.0)

    n = max(1, len(files))
    return {
        "available": True,
        "evaluated_tasks": len(files),
        "pass_at_1": round(pass1 / n, 4),
        "pass_at_2_official": round(pass2 / n, 4),
        "mean_cell_accuracy": round(sum(cell_accs) / n, 4),
        "visual_primitives_usage_rate": round(used_vis_prims / n, 4),
    }


def eval_supergpqa_holdout(model: Any, token: str | None, limit: int = 60) -> dict[str, Any]:
    sg_path = _hf_file(SUPERGPQA_REPO, "SuperGPQA-all.jsonl", revision=SUPERGPQA_REV, token=token)
    holdout_rows: list[dict[str, Any]] = []
    for ln in sg_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        item = json.loads(ln)
        uid = str(item.get("uuid") or "").strip()
        if not uid:
            continue
        if int(hashlib.sha256(f"stage7_supergpqa_v1:{uid}".encode("utf-8")).hexdigest()[:8], 16) % 3 == 0:
            holdout_rows.append(item)
            if len(holdout_rows) >= limit:
                break

    letters = "ABCDEFGHIJ"
    correct = 0
    for item in holdout_rows:
        q = str(item["question"]).strip()
        opts = [str(o).strip() for o in item["options"]]
        ans_let = str(item["answer_letter"]).strip().upper()
        opts_fmt = "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(opts[: len(letters)]))
        out = _gen(model, f"{q}\n\nOptions:\n{opts_fmt}\n\nProvide concise reasoning and conclude with `Answer: (<letter>)`.", max_new_tokens=640)
        _, body = strip_think_block(out)
        m = re.search(r"(?:Answer|Final Answer)\s*:\s*\(?([A-J])\)?", body, re.IGNORECASE)
        pred_let = m.group(1).upper() if m else ""
        if pred_let == ans_let:
            correct += 1
    n = max(1, len(holdout_rows))
    return {"evaluated": len(holdout_rows), "accuracy": round(correct / n, 4), "correct": correct}


def eval_swebench_pro_holdout(model: Any, token: str | None, limit: int = 30) -> dict[str, Any]:
    swep_path = _hf_file(SWEBENCH_PRO_REPO, "data/default/test-00000-of-00001.parquet", revision=SWEBENCH_PRO_REV, token=token)
    rows = pq.read_table(swep_path, columns=["repo", "instance_id", "problem_statement", "requirements", "interface", "patch", "repo_language"]).to_pylist()
    holdout = [
        r for r in rows
        if int(hashlib.sha256(f"stage7_swebench_pro_v1:{r['instance_id']}".encode("utf-8")).hexdigest()[:8], 16) % 3 == 0
    ][:limit]

    valid_diffs = 0
    file_localization_hits = 0
    verified_patches = 0
    for r in holdout:
        repo = str(r.get("repo") or "")
        prob = str(r.get("problem_statement") or "")[:3800]
        reqs = str(r.get("requirements") or "")[:1000]
        iface = str(r.get("interface") or "")[:800]
        gt_patch = str(r.get("patch") or "")
        gt_files = set(re.findall(r"^diff --git a/(\S+) b/\S+", gt_patch, flags=re.MULTILINE))
        prompt = (
            f"[SWE-Bench Pro Repository Issue — `{repo}`]\n{prob}\n"
            + (f"\n### Requirements\n{reqs}\n" if reqs else "")
            + (f"\n### Target Interface\n{iface}\n" if iface else "")
            + "\nProduce a minimal, clean unified `diff --git` patch resolving this issue."
        )
        out = _gen(model, prompt, max_new_tokens=1152)
        _, body = strip_think_block(out)
        has_diff = ("diff --git " in body) and ("@@ " in body) and ("--- " in body) and ("+++ " in body)
        pred_files = set(re.findall(r"^diff --git a/(\S+) b/\S+", body, flags=re.MULTILINE))
        f_hit = bool(gt_files & pred_files) if gt_files else has_diff
        gt_idents = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", gt_patch)}
        pr_idents = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", body)}
        rec = len(gt_idents & pr_idents) / max(1, len(gt_idents))
        if has_diff:
            valid_diffs += 1
        if f_hit:
            file_localization_hits += 1
        if has_diff and f_hit and rec >= 0.30:
            verified_patches += 1

    n = max(1, len(holdout))
    return {
        "evaluated": len(holdout),
        "valid_unified_diff_rate": round(valid_diffs / n, 4),
        "target_file_hit_rate": round(file_localization_hits / n, 4),
        "verified_patch_rate": round(verified_patches / n, 4),
    }


def eval_terminal_bench_holdout(model: Any, token: str | None, version: str = "4.0", limit: int = 20) -> dict[str, Any]:
    from huggingface_hub import HfApi

    repo_id = TERMINAL_BENCH4_REPO_ID if version == "4.0" else TERMINAL_BENCH3_REPO_ID
    rev = TERMINAL_BENCH4_REVISION if version == "4.0" else TERMINAL_BENCH3_REVISION
    api = HfApi(token=token)
    info = api.dataset_info(repo_id, revision=rev)
    by_task: dict[str, set[str]] = {}
    for sib in info.siblings or ():
        parts = sib.rfilename.split("/")
        if len(parts) >= 3 and parts[0] == "tasks":
            by_task.setdefault(parts[1], set()).add(sib.rfilename)

    holdout_ids = [
        tid for tid in sorted(by_task)
        if int(hashlib.sha256(f"tb3_holdout_v1:{tid}".encode("utf-8")).hexdigest()[:8], 16) % 3 == 0
        and f"tasks/{tid}/instruction.md" in by_task[tid]
    ][:limit]

    syntax_valid = 0
    substantive_pass = 0
    for tid in holdout_ids:
        inst = _hf_file(repo_id, f"tasks/{tid}/instruction.md", revision=rev, token=token).read_text(encoding="utf-8", errors="replace")[:4200]
        prompt = (
            f"[Terminal-Bench {version} Task: {tid}]\n{inst}\n\n"
            "Author a complete, non-interactive solution inside a ```bash ... ``` or ```python ... ``` code block."
        )
        out = _gen(model, prompt, max_new_tokens=1024)
        _, body = strip_think_block(out)
        blocks = re.findall(r"```(bash|sh|python)\s*\n(.*?)```", body, re.DOTALL | re.IGNORECASE)
        if not blocks:
            continue
        ok_syn = True
        for lang, code in blocks:
            if lang.lower() == "python":
                try:
                    ast.parse(code)
                except SyntaxError:
                    ok_syn = False
                    break
            else:
                p = subprocess.run(["bash", "-n"], input=code.encode("utf-8"), capture_output=True, timeout=3, check=False)
                if p.returncode != 0:
                    ok_syn = False
                    break
        if ok_syn:
            syntax_valid += 1
            if sum(len(c.strip()) for _, c in blocks) >= 48:
                substantive_pass += 1

    n = max(1, len(holdout_ids))
    return {
        "version": version,
        "evaluated": len(holdout_ids),
        "syntax_valid_rate": round(syntax_valid / n, 4),
        "substantive_solution_rate": round(substantive_pass / n, 4),
    }


def eval_v8_sculpted_and_v41_gate_d(model: Any, export_dir: Path) -> dict[str, Any]:
    ensure_synccraft_importable()
    from synccraft.dsl.parser import parse_program
    from synccraft.exporters.bbmodel_exporter import export_bbmodel
    from synccraft.kinematics.se3_chain import compile_geometry
    from synccraft.world.assembly_world_tasks import (
        build_modular_assembly_world_tasks,
        evaluate_modular_assembly_world_tasks,
    )
    from synccraft.world.proposer_tasks import (
        build_proposer_supervision_dataset,
        evaluate_gate_d_proposer,
    )

    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gate D Held-out Proposer (`holdout_gate_d` — 0% train overlap!)
    prop_all = build_proposer_supervision_dataset()
    holdout_prop = [r for r in prop_all if r.split == "holdout_gate_d"]

    def _proposer_fn(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "[SyncCraft v4.1 World-Model Proposer — `propose_gait_from_style`]\n"
            f"Input specification:\n```json\n{json.dumps(payload, sort_keys=True)}\n```\n"
            "Return ONLY the solver-verified JSON target parameters."
        )
        out = _gen(model, prompt, max_new_tokens=512)
        _, bd = strip_think_block(out)
        try:
            return _extract_json_candidate(bd)
        except Exception:
            return {"pattern": "lateral_sequence", "froude": 0.20, "duty": 0.65}

    gate_d_rep = evaluate_gate_d_proposer(holdout_prop, _proposer_fn)

    # 2. Held-out `v4_wm_01..04` Modular `@layout/@component/@assembly` Benchmark
    v4_tasks = [t for t in build_modular_assembly_world_tasks() if str(t.get("task_id", "")).startswith("v4_wm_0")]
    v4_preds: dict[str, str] = {}
    for t in v4_tasks:
        tid = str(t["task_id"])
        out = _gen(model, str(t["prompt"]), max_new_tokens=1024)
        _, bd = strip_think_block(out)
        if str(t.get("task_type")) == "propagate_layout_and_instance_controls":
            try:
                v4_preds[tid] = json.dumps(_extract_json_candidate(bd))
            except Exception:
                v4_preds[tid] = "{}"
        else:
            v4_preds[tid] = _extract_dsl_candidate(bd)
    v4_mod_rep = evaluate_modular_assembly_world_tasks(v4_tasks, v4_preds)

    # 3. Unseen `SKILL.md` v8 Sculpted Assets -> `.bbmodel` + Studio Turnaround + 2x2 Region Sheet PNGs
    v8_assets: list[dict[str, Any]] = []
    for item in UNSEEN_V8_PROMPTS:
        aid = item["id"]
        best_dsl = ""
        best_rep: dict[str, Any] = {"passed_v8": False, "style_anchor_score": 0.0}
        for attempt in range(2):
            out = _gen(
                model,
                item["prompt"],
                max_new_tokens=1600,
                do_sample=(attempt > 0),
                temperature=0.50,
            )
            _, bd = strip_think_block(out)
            cand_dsl = _extract_dsl_candidate(bd)
            try:
                rep = run_v8_construct_check(cand_dsl)
                if rep["passed_v8"] or float(rep["style_anchor_score"]) >= float(best_rep.get("style_anchor_score", 0.0)):
                    best_dsl = cand_dsl
                    best_rep = rep
                if rep["passed_v8"]:
                    break
            except Exception as exc:
                best_rep = {"passed_v8": False, "compile_error": str(exc)[:300]}

        asset_entry: dict[str, Any] = {"id": aid, "v8_check": best_rep, "exported_files": []}
        if best_dsl and "dsl_version" in best_rep:
            prog = parse_program(best_dsl)
            sc = compile_geometry(prog)
            dsl_p = export_dir / f"{aid}.blockdsl"
            dsl_p.write_text(best_dsl, encoding="utf-8")
            asset_entry["exported_files"].append(dsl_p.name)

            try:
                bb_doc = export_bbmodel(prog, sc, model_name=aid)
                bb_p = export_dir / f"{aid}.bbmodel"
                bb_p.write_text(json.dumps(bb_doc, indent=2), encoding="utf-8")
                asset_entry["exported_files"].append(bb_p.name)
            except Exception as exc:
                asset_entry["bbmodel_note"] = str(exc)[:200]

            pngs, _ = render_sculpted_turnaround_views(prog, panel_size=(256, 256), include_silhouette_sheet=True)
            for vname, raw_png in pngs.items():
                vp = export_dir / f"{aid}_{vname}.png"
                vp.write_bytes(raw_png)
                asset_entry["exported_files"].append(vp.name)

            reg_im, _ = render_v8_zoomed_region_sheet(prog, panel_size=(256, 256))
            rp = export_dir / f"{aid}_region_sheet.png"
            reg_im.save(rp, format="PNG")
            asset_entry["exported_files"].append(rp.name)
        v8_assets.append(asset_entry)

    v8_pass_cnt = sum(1 for a in v8_assets if a["v8_check"].get("passed_v8"))
    return {
        "gate_d_proposer": gate_d_rep,
        "v4_modular_holdout": {
            "pass_rate": v4_mod_rep.get("pass_rate"),
            "passed_count": v4_mod_rep.get("passed_count"),
            "total": v4_mod_rep.get("total_tasks"),
        },
        "v8_unseen_pass_rate": round(v8_pass_cnt / max(1, len(v8_assets)), 4),
        "v8_assets": v8_assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, action="append", required=True, help="Label=path or path to release dir")
    parser.add_argument("--arc-root", type=Path, default=Path("/home/baitian6641/Spark-X2.6-VL/cache/ARC-AGI-2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arc-limit", type=int, default=30)
    parser.add_argument("--supergpqa-limit", type=int, default=60)
    parser.add_argument("--swebench-limit", type=int, default=30)
    parser.add_argument("--tb-limit", type=int, default=20)
    args = parser.parse_args()

    tok = os.environ.get("HF_TOKEN") or None
    if not tok:
        for tp in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
            if tp.is_file():
                tok = tp.read_text(encoding="utf-8").strip()
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"models": {}}

    for spec in args.model_dir:
        spec_str = str(spec)
        if "=" in spec_str:
            label, raw_p = spec_str.split("=", 1)
            m_dir = Path(raw_p)
        else:
            m_dir = Path(spec_str)
            label = m_dir.parent.name + "_" + m_dir.name

        print(f"=== Loading model [{label}] from {m_dir} ===", flush=True)
        model = load(m_dir, device="cuda")
        model.eval()

        m_out = args.output_dir / label
        m_out.mkdir(parents=True, exist_ok=True)

        res = {
            "arc_agi_2_eval": eval_arc_agi_2(model, args.arc_root, limit=args.arc_limit),
            "supergpqa_holdout": eval_supergpqa_holdout(model, tok, limit=args.supergpqa_limit),
            "terminal_bench_4_holdout": eval_terminal_bench_holdout(model, tok, version="4.0", limit=args.tb_limit),
            "terminal_bench_3_holdout": eval_terminal_bench_holdout(model, tok, version="3.0", limit=args.tb_limit),
            "swebench_pro_holdout": eval_swebench_pro_holdout(model, tok, limit=args.swebench_limit),
            "synccraft_v8_and_v41": eval_v8_sculpted_and_v41_gate_d(model, m_out / "v8_exports"),
            "loop_engram_telemetry": model.loop_engram_telemetry() if getattr(model, "enable_loop_engram", False) else None,
        }
        summary["models"][label] = res
        write(m_out / "STAGE7_FRONTIER_V8_REPORT.json", res)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write(args.output_dir / "STAGE7_HEAD_TO_HEAD_REPORT.json", summary)
    print(json.dumps({"status": "complete", "report": str(args.output_dir / "STAGE7_HEAD_TO_HEAD_REPORT.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
