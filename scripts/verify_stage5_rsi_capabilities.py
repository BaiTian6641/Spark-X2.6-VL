#!/usr/bin/env python3
"""Comprehensive 6-Pillar Stage-5 + Vision RSI Verification & Anti-Collapse Suite.

Evaluates a trained Stage-5 / Vision-RSI `SparkNative` package across:
1. General EN/ZH Vision-Language, OCR & Video Grounding (`audit` split + blank/shuffled controls)
2. Unseen 3D Spatial Reasoning (`SAT_test.parquet`) & Unseen 4-Way RGBA UV/Alpha Diagnosis (standard + dense ribbon)
3. Held-Out Frontier `<think>` Reasoning, Agentic Coding & Anti-Collapse Health:
   - Strictly held-out `GPQA-Diamond` subset (0% overlap with `gpqa_extended` non-Diamond train)
   - Strictly held-out `HLE` disjoint text + multimodal vision subset (`bucket == 0` holdout)
   - Strictly held-out `Terminal-Bench 3.0` tasks (`bucket == 0` holdout)
   - Dual-mode (`enable_thinking=True` & `False`) live Python algorithmic execution + collapse audit
4. Unseen SyncCraft 7-Level World Ladder (`held_out_compositional_val`) & Visual Aesthetic Judgment (`toy` vs `artist`)
5. Artist-Grade End-to-End 3D Asset Generation & Upgrade (`SyncCraftWorldPipeline` with geometry-neutral grammar,
   non-toy aesthetic critic, 15-axis SAT, `<0.02u` stance slip, `.bbmodel`/`.geo.json`/`.gltf` `<=1e-6` vertex error, and PNGs)
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.common import identity, read, sha, write
from colab.final.data import Catalogue, MediaReader
from colab.final.evaluate import evaluate
from colab.final.inference import empty_media, generate, load, visual_media
from colab.final.model import export_spark_native_fp8_safetensors
from colab.final.reasoning_agentic_corpus import (
    GPQA_REPO_ID,
    GPQA_REVISION,
    HLE_REPO_ID,
    HLE_REVISION,
    TERMINAL_BENCH3_REPO_ID,
    TERMINAL_BENCH3_REVISION,
    _decode_hle_image_bytes,
    check_generation_collapse_health,
)
from colab.final.storage import manifest
from colab.final.synccraft_bridge import (
    build_animated_quadruped_program,
    build_parametric_sentinel_hound_world,
    build_uv_seam_alpha_diagnosis_record,
    build_visual_aesthetic_judgment_record,
    build_world_capability_record_from_ladder_row,
    ensure_synccraft_importable,
    evaluate_synccraft_completion,
    render_world_views_rgba,
    strip_think_block,
)
from colab.final.world_pipeline import (
    SyncCraftWorldPipeline,
    build_default_world_state_for_program,
    evaluate_3d_asset_aesthetics,
)


def export_cold_checkpoint_package(
    cold_dir: Path,
    base_export_dir: Path,
    dest_bf16_dir: Path,
    dest_fp8_dir: Path | None = None,
) -> dict[str, Any]:
    """Package a verified `local-cold-store/checkpoint-*` directory into standalone BF16 (+ optional FP8) release dirs."""
    payload = cold_dir / "payload" if (cold_dir / "payload").is_dir() else cold_dir
    src_sf = payload / "model.safetensors"
    if not src_sf.is_file():
        raise FileNotFoundError(f"Missing model.safetensors in cold checkpoint {cold_dir}")

    if dest_bf16_dir.exists():
        shutil.rmtree(dest_bf16_dir)
    dest_bf16_dir.mkdir(parents=True, exist_ok=True)

    for item in base_export_dir.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(base_export_dir)
        if rel.as_posix() in ("model.safetensors", "MODEL_RELEASE.json"):
            continue
        target = dest_bf16_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)

    for fn in ("model.safetensors", "BINDING.json", "PARAMETERS.json"):
        sp = payload / fn
        if sp.is_file():
            dp = dest_bf16_dir / fn
            if dp.exists():
                dp.unlink()
            os.link(sp, dp)

    base_rel = read(base_export_dir / "MODEL_RELEASE.json")
    bf16_files = manifest(dest_bf16_dir)
    write(
        dest_bf16_dir / "MODEL_RELEASE.json",
        {
            "schema_version": 1,
            "campaign_id": str(base_rel.get("campaign_id", "")) + "-rsi-promoted",
            "binding": base_rel["binding"],
            "training_complete": True,
            "source_cold_checkpoint": cold_dir.name,
            **bf16_files,
        },
    )
    out_info: dict[str, Any] = {
        "bf16_dir": str(dest_bf16_dir),
        "bf16_release_sha256": sha(dest_bf16_dir / "MODEL_RELEASE.json"),
    }

    if dest_fp8_dir is not None:
        if dest_fp8_dir.exists():
            shutil.rmtree(dest_fp8_dir)
        dest_fp8_dir.mkdir(parents=True, exist_ok=True)
        for item in dest_bf16_dir.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(dest_bf16_dir)
            if rel.as_posix() in ("model.safetensors", "MODEL_RELEASE.json"):
                continue
            target = dest_fp8_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        tmp_model = load(dest_bf16_dir, device="cpu")
        fp8_meta = export_spark_native_fp8_safetensors(tmp_model, dest_fp8_dir / "model.safetensors")
        del tmp_model
        gc.collect()
        cfg_fp8 = read(dest_fp8_dir / "config.json")
        cfg_fp8["exported_weight_format"] = "fp8_e4m3_channel_q4g128_residual"
        write(dest_fp8_dir / "config.json", cfg_fp8)
        fp8_files = manifest(dest_fp8_dir)
        write(
            dest_fp8_dir / "MODEL_RELEASE.json",
            {
                "schema_version": 1,
                "campaign_id": str(base_rel.get("campaign_id", "")) + "-rsi-promoted-fp8",
                "binding": base_rel["binding"],
                "training_complete": True,
                "quantization": fp8_meta,
                **fp8_files,
            },
        )
        out_info["fp8_dir"] = str(dest_fp8_dir)
        out_info["fp8_release_sha256"] = sha(dest_fp8_dir / "MODEL_RELEASE.json")

    return out_info


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


def _eval_heldout_gpqa_diamond(model: Any, token: str | None, n_items: int = 24) -> dict[str, Any]:
    dia_path = _hf_file(GPQA_REPO_ID, "gpqa_diamond.csv", revision=GPQA_REVISION, token=token)
    with dia_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: identity(["gpqa_diamond_eval_v1", r.get("Question", "")[:80]]))
    subset = rows[:n_items]

    correct_cnt = 0
    healthy_cnt = 0
    think_closed_cnt = 0
    choice_re = re.compile(r"(?:Answer|Final Answer)\s*:\s*\(?([A-D])\)?", re.IGNORECASE)
    fallback_re = re.compile(r"\(([A-D])\)\s*$|\b([A-D])\s*\.?\s*$")

    for idx, r in enumerate(subset):
        q = str(r.get("Question") or "").strip()
        ans = str(r.get("Correct Answer") or "").strip()
        wrongs = [str(r.get(f"Incorrect Answer {k}") or "").strip() for k in (1, 2, 3)]
        rng = random.Random(9100 + idx)
        opts = [ans, *wrongs]
        rng.shuffle(opts)
        letters = ("A", "B", "C", "D")
        gold_letter = letters[opts.index(ans)]
        opt_str = "\n".join(f"({letters[i]}) {opts[i]}" for i in range(4))
        prompt = (
            f"{q}\n\nChoices:\n{opt_str}\n\n"
            "Analyze the scientific principles concisely and conclude with `Answer: (X)` where X is A, B, C, or D."
        )
        out = generate(
            model,
            prompt,
            media=empty_media(),
            enable_thinking=True,
            max_new_tokens=1536,
            max_think_tokens=640,
            task="gpqa_extended_science",
        )
        health = check_generation_collapse_health(out, expect_thinking=True, max_rep_4gram=0.14, max_think_chars=5200)
        if health["healthy"]:
            healthy_cnt += 1
        if health.get("think_closed"):
            think_closed_cnt += 1
        _, body = strip_think_block(out)
        m = choice_re.search(body)
        if m:
            pred_letter = m.group(1).upper()
        else:
            fm = fallback_re.search(body.strip())
            pred_letter = (fm.group(1) or fm.group(2) or "").upper() if fm else ""
        if pred_letter == gold_letter:
            correct_cnt += 1

    return {
        "benchmark": "GPQA-Diamond (100% Held-Out Split)",
        "evaluated": len(subset),
        "correct": correct_cnt,
        "accuracy": round(correct_cnt / max(1, len(subset)), 4),
        "healthy_generation_rate": round(healthy_cnt / max(1, len(subset)), 4),
        "think_close_rate": round(think_closed_cnt / max(1, len(subset)), 4),
    }


def _eval_heldout_hle(model: Any, token: str | None, n_text: int = 16, n_mm: int = 10) -> dict[str, Any]:
    import pyarrow.parquet as pq

    hle_path = _hf_file(HLE_REPO_ID, "data/test-00000-of-00001.parquet", revision=HLE_REVISION, token=token)
    pf = pq.ParquetFile(hle_path)
    text_holdout: list[dict[str, Any]] = []
    mm_holdout: list[dict[str, Any]] = []
    for batch in pf.iter_batches(batch_size=128, columns=["id", "question", "image", "answer", "answer_type", "category"]):
        for item in batch.to_pylist():
            hid = str(item.get("id") or "").strip()
            if int(hashlib.sha256(f"hle_holdout_v1:{hid}".encode("utf-8")).hexdigest()[:8], 16) % 5 != 0:
                continue
            if str(item.get("answer_type") or "") == "multipleChoice" or len(str(item.get("answer") or "")) <= 32:
                if item.get("image"):
                    mm_holdout.append(item)
                else:
                    text_holdout.append(item)

    text_holdout.sort(key=lambda x: identity(["hle_eval_txt", x["id"]]))
    mm_holdout.sort(key=lambda x: identity(["hle_eval_mm", x["id"]]))
    chosen = [*mm_holdout[:n_mm], *text_holdout[:n_text]]

    ans_re = re.compile(r"Final Answer\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    correct = 0
    mm_correct = 0
    mm_total = 0
    healthy = 0
    think_closed = 0

    for item in chosen:
        q = str(item["question"]).strip()
        gold = str(item["answer"]).strip().casefold()
        media = empty_media()
        is_mm = False
        if item.get("image"):
            dec = _decode_hle_image_bytes(item["image"])
            if dec is not None:
                im = Image.open(io.BytesIO(dec[0]))
                media = visual_media([im], visual_kind="independent_image")
                is_mm = True
                mm_total += 1
        out = generate(
            model,
            f"{q}\n\nProvide a concise derivation and end with `Final Answer: <answer>`.",
            media=media,
            enable_thinking=True,
            max_new_tokens=1536,
            max_think_tokens=640,
            task="hle_multimodal_reasoning" if is_mm else "hle_text_reasoning",
        )
        h = check_generation_collapse_health(out, expect_thinking=True, max_rep_4gram=0.14, max_think_chars=5200)
        if h["healthy"]:
            healthy += 1
        if h.get("think_closed"):
            think_closed += 1
        _, body = strip_think_block(out)
        m = ans_re.search(body)
        pred = (m.group(1) if m else (body.splitlines()[-1] if body else "")).strip().strip(".").casefold()
        if gold and (pred == gold or f"({gold})" in pred or pred.startswith(gold + " ") or pred.strip("()") == gold):
            correct += 1
            if is_mm:
                mm_correct += 1

    return {
        "benchmark": "HLE Disjoint Holdout (Text + Multimodal Vision, Zero Train Overlap)",
        "evaluated_total": len(chosen),
        "evaluated_multimodal": mm_total,
        "correct_total": correct,
        "accuracy_total": round(correct / max(1, len(chosen)), 4),
        "accuracy_multimodal": round(mm_correct / max(1, mm_total), 4),
        "healthy_generation_rate": round(healthy / max(1, len(chosen)), 4),
        "think_close_rate": round(think_closed / max(1, len(chosen)), 4),
    }


def _eval_heldout_terminal_and_agentic_coding(model: Any, token: str | None, n_tb: int = 8) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.dataset_info(TERMINAL_BENCH3_REPO_ID, revision=TERMINAL_BENCH3_REVISION)
    by_task: dict[str, set[str]] = {}
    for sib in info.siblings or ():
        parts = sib.rfilename.split("/")
        if len(parts) >= 3 and parts[0] == "tasks":
            by_task.setdefault(parts[1], set()).add(sib.rfilename)

    holdout_tids = [
        tid
        for tid in sorted(by_task)
        if int(hashlib.sha256(f"tb3_holdout_v1:{tid}".encode("utf-8")).hexdigest()[:8], 16) % 3 == 0
        and f"tasks/{tid}/instruction.md" in by_task[tid]
    ][:n_tb]

    tb_valid_scripts = 0
    tb_healthy = 0
    tb_think_closed = 0
    code_re = re.compile(r"```(bash|sh|python)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

    for tid in holdout_tids:
        inst = _hf_file(
            TERMINAL_BENCH3_REPO_ID,
            f"tasks/{tid}/instruction.md",
            revision=TERMINAL_BENCH3_REVISION,
            token=token,
        ).read_text(encoding="utf-8", errors="replace").strip()
        out = generate(
            model,
            f"[Terminal Agent Task]\n{inst[:4500]}\n\nAuthor a complete, reproducible ```bash ... ``` or ```python ... ``` solution script.",
            media=empty_media(),
            enable_thinking=True,
            max_new_tokens=1664,
            max_think_tokens=576,
            task="terminal_bench_agent",
        )
        h = check_generation_collapse_health(out, expect_thinking=True, max_rep_4gram=0.16, max_think_chars=5200)
        if h["healthy"]:
            tb_healthy += 1
        if h.get("think_closed"):
            tb_think_closed += 1
        _, body = strip_think_block(out)
        blocks = code_re.findall(body)
        if blocks:
            ok_all = True
            for lang, code in blocks:
                if lang.lower() == "python":
                    try:
                        compile(code, f"<{tid}>", "exec")
                    except SyntaxError:
                        ok_all = False
                        break
                else:
                    p = subprocess.run(["bash", "-n"], input=code.encode("utf-8"), capture_output=True, timeout=3, check=False)
                    if p.returncode != 0:
                        ok_all = False
                        break
            if ok_all:
                tb_valid_scripts += 1

    coding_probe = (
        "Write a self-contained Python function `shortest_grid_path(grid: list[list[int]]) -> int` that returns "
        "the minimum number of 4-directional steps from `(0, 0)` to `(R-1, C-1)` avoiding obstacles (`1`), or `-1` "
        "if unreachable (and `-1` if start or goal is `1`). Output ONLY a ```python ... ``` block."
    )
    exec_modes: dict[str, bool] = {}
    for think_mode in (False, True):
        resp = generate(
            model,
            coding_probe,
            media=empty_media(),
            enable_thinking=think_mode,
            max_new_tokens=1408 if think_mode else 768,
            max_think_tokens=512 if think_mode else None,
            task="fable5_swe_agent",
        )
        _, cbody = strip_think_block(resp)
        m = re.search(r"```(?:python)?\s*\n(.*?)```", cbody, re.DOTALL | re.IGNORECASE)
        snippet = m.group(1) if m else cbody
        test_harness = (
            snippet
            + "\nassert shortest_grid_path([[0,0,0],[1,1,0],[0,0,0]]) == 4\n"
            + "assert shortest_grid_path([[0,1],[1,0]]) == -1\n"
            + "assert shortest_grid_path([[0]]) == 0\n"
            + "print('PASS_GRID')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", test_harness],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        exec_modes[f"bfs_exec_thinking_{think_mode}"] = proc.returncode == 0 and "PASS_GRID" in proc.stdout

    return {
        "terminal_bench_3_holdout_evaluated": len(holdout_tids),
        "terminal_bench_3_valid_executable_syntax": tb_valid_scripts,
        "terminal_bench_3_syntax_pass_rate": round(tb_valid_scripts / max(1, len(holdout_tids)), 4),
        "terminal_bench_3_healthy_rate": round(tb_healthy / max(1, len(holdout_tids)), 4),
        "terminal_bench_3_think_close_rate": round(tb_think_closed / max(1, len(holdout_tids)), 4),
        "live_python_execution": exec_modes,
    }


def _eval_sat_test_and_4way_uv_and_aesthetics(model: Any, token: str | None) -> dict[str, Any]:
    import pyarrow.parquet as pq

    ensure_synccraft_importable()
    # 1. Unseen `array/SAT` (`SAT_test.parquet`)
    sat_test = _hf_file("array/SAT", "SAT_test.parquet", revision="bda5dde942d7f7b41bff7935f086ed9a9e348ae3", token=token)
    pf = pq.ParquetFile(sat_test)
    items = next(pf.iter_batches(batch_size=32)).to_pylist()[:16]
    sat_ok = 0
    for it in items:
        imgs = [Image.open(io.BytesIO(bytes(b))) for b in (it.get("image_bytes") or ())]
        q = str(it.get("question") or "").strip()
        gold = str(it.get("correct_answer") or "").strip()
        choices = [str(c).strip() for c in (it.get("answers") or ()) if str(c).strip()]
        if gold not in choices:
            choices.append(gold)
        opt_s = "\n".join(f"- {c}" for c in choices)
        media = visual_media(
            imgs,
            visual_kind="state_pair" if len(imgs) > 1 else "independent_image",
        )
        out = generate(
            model,
            f"[3D Spatial Reasoning]\n{q}\nOptions:\n{opt_s}\nState the exact correct option.",
            media=media,
            enable_thinking=True,
            max_new_tokens=512,
            max_think_tokens=256,
            task="sat_3d_spatial",
        )
        _, body = strip_think_block(out)
        if gold.casefold() in body.strip().casefold():
            sat_ok += 1

    # 2a. Unseen 4-Way RGBA UV/Alpha Diagnosis on standard-aspect unseen quadruped rig (`128x64` atlas)
    ws_std = build_default_world_state_for_program(build_animated_quadruped_program(97, rich_anatomy=False), seed=97)
    uv_std_cells: dict[str, Any] = {}
    for a_flag, o_flag in ((False, False), (True, False), (False, True), (True, True)):
        key = f"a{int(a_flag)}_o{int(o_flag)}"
        row, blobs = build_uv_seam_alpha_diagnosis_record(
            ws_std,
            asset_id=f"unseen_quad_97_{key}",
            family_id="quad_holdout",
            inject_alpha_cutout=a_flag,
            inject_uv_overlap=o_flag,
            vary_target_faces=True,
            enable_thinking=True,
        )
        imgs = [Image.open(io.BytesIO(blobs[p])) for p in row["frames"]]
        media = visual_media(imgs, visual_kind="uv_atlas", visual_units=row["visual_units"])
        pred = generate(
            model,
            row["prompt"],
            media=media,
            enable_thinking=True,
            max_new_tokens=768,
            max_think_tokens=384,
            task="synccraft_uv_alpha_diagnosis",
        )
        ev = evaluate_synccraft_completion(row, pred)
        uv_std_cells[key] = {"passed": ev["passed"], "field_accuracy": ev.get("field_accuracy", 0.0)}

    # 2b. Unseen 4-Way RGBA UV/Alpha Diagnosis on 27-box Sentinel Hound (`512x16` ultra-dense 162-face ribbon)
    ws_hound = build_parametric_sentinel_hound_world(variant=99, subdivisions_per_unit=1)
    uv_dense_cells: dict[str, Any] = {}
    for a_flag, o_flag in ((False, False), (True, False), (False, True), (True, True)):
        key = f"a{int(a_flag)}_o{int(o_flag)}"
        row, blobs = build_uv_seam_alpha_diagnosis_record(
            ws_hound,
            asset_id=f"unseen_hound_99_{key}",
            family_id="sentinel_hound_holdout",
            inject_alpha_cutout=a_flag,
            inject_uv_overlap=o_flag,
            vary_target_faces=True,
            enable_thinking=True,
        )
        imgs = [Image.open(io.BytesIO(blobs[p])) for p in row["frames"]]
        media = visual_media(imgs, visual_kind="uv_atlas", visual_units=row["visual_units"])
        pred = generate(
            model,
            row["prompt"],
            media=media,
            enable_thinking=True,
            max_new_tokens=768,
            max_think_tokens=384,
            task="synccraft_uv_alpha_diagnosis",
        )
        ev = evaluate_synccraft_completion(row, pred)
        uv_dense_cells[key] = {"passed": ev["passed"], "field_accuracy": ev.get("field_accuracy", 0.0)}

    # 3. Visual Aesthetic Judgment (`toy_blockout` vs `artist_grade`) on unseen 27-box rig
    aes_judge: dict[str, Any] = {}
    for toy_flag in (True, False):
        label = "coarsened_toy_blockout" if toy_flag else "full_27box_artist_asset"
        j_row, j_blobs = build_visual_aesthetic_judgment_record(
            ws_hound,
            asset_id="unseen_hound_judge_99",
            family_id="sentinel_hound_holdout",
            inject_toy_coarsening=toy_flag,
            enable_thinking=True,
        )
        j_imgs = [Image.open(io.BytesIO(j_blobs[p])) for p in j_row["frames"]]
        j_media = visual_media(j_imgs, visual_kind="multiview_render", visual_units=j_row["visual_units"])
        j_pred = generate(
            model,
            j_row["prompt"],
            media=j_media,
            enable_thinking=True,
            max_new_tokens=768,
            max_think_tokens=384,
            task="synccraft_visual_aesthetic_judgment",
        )
        j_ev = evaluate_synccraft_completion(j_row, j_pred)
        aes_judge[label] = {
            "passed": j_ev["passed"],
            "field_accuracy": j_ev.get("field_accuracy", 0.0),
            "oracle": j_row["metadata"]["oracle_json"],
        }

    # 4. Held-out SyncCraft 7-Level World Ladder (`held_out_compositional_val`)
    ladder_p = _hf_file(
        "TassP0lar/synccraft-v3-blockdsl-dataset",
        "world_capability_ladder.jsonl",
        revision="705e426536286a3632f163d886756704abaad575",
        token=token,
    )
    ladder_by_level: dict[str, list[dict[str, Any]]] = {}
    with ladder_p.open("r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            lr = json.loads(ln)
            if lr.get("split") in ("held_out_compositional_val", "test"):
                lvl = str(lr["level"])
                if len(ladder_by_level.get(lvl, ())) < 2:
                    ladder_by_level.setdefault(lvl, []).append(lr)

    ladder_scores: dict[str, float] = {}
    for lvl, lrows in sorted(ladder_by_level.items()):
        sc_vals = []
        for lr in lrows:
            rec, blobs = build_world_capability_record_from_ladder_row(lr, enable_thinking=True)
            imgs = [Image.open(io.BytesIO(blobs[p])) for p in rec.get("frames", ())]
            med = (
                visual_media(imgs, visual_kind=rec.get("visual_kind", "multiview_render"), visual_units=rec.get("visual_units"))
                if imgs
                else empty_media()
            )
            pred = generate(
                model,
                rec["prompt"],
                media=med,
                enable_thinking=True,
                max_new_tokens=1536,
                max_think_tokens=512,
                task="synccraft_world_capability",
            )
            ev = evaluate_synccraft_completion(rec, pred)
            sc_vals.append(float(ev.get("ladder_score", 1.0 if ev.get("passed") else 0.0)))
        ladder_scores[lvl] = round(sum(sc_vals) / max(1, len(sc_vals)), 4)

    return {
        "sat_test_unseen_accuracy": round(sat_ok / max(1, len(items)), 4),
        "sat_test_correct": sat_ok,
        "sat_test_total": len(items),
        "unseen_4way_uv_standard_rig": uv_std_cells,
        "unseen_4way_uv_standard_pass_count": sum(1 for v in uv_std_cells.values() if v["passed"]),
        "unseen_4way_uv_dense_27box": uv_dense_cells,
        "unseen_4way_uv_dense_mean_field_acc": round(
            sum(float(v["field_accuracy"]) for v in uv_dense_cells.values()) / max(1, len(uv_dense_cells)), 4
        ),
        "visual_aesthetic_judgment": aes_judge,
        "world_ladder_compositional_holdout_scores": ladder_scores,
        "world_ladder_compositional_holdout_mean": round(
            sum(ladder_scores.values()) / max(1, len(ladder_scores)), 4
        ),
    }


def _eval_artist_grade_3d_pipeline(model: Any, cfg: dict[str, Any], out_dir: Path, token: str | None) -> dict[str, Any]:
    from synccraft.dsl.parser import parse_program

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SyncCraftWorldPipeline(model, config={**cfg, "min_aesthetic_score": 0.72}, default_max_new_tokens=3072)

    # 1. Text-to-3D Sentinel Hound (with geometry-neutral system grammar — ZERO hound exemplar in system prompt!)
    hound_res = pipeline.generate_3d_asset(
        "Design an artist-grade SyncCraft Block-DSL v3.2 armored sentinel hound quadruped (`bedrock_entity`, `texel_density: 16`) "
        "with distinct ribcage, tapered pelvis, chest armor (`inflate`), articulated neck, skull, muzzle, nose, lower jaw, "
        "bilateral `mirror_x` ears, 2-segment tail, 4 three-part legs (`thigh -> shin -> paw`), 5 OKLCH `_mat` ramps, and a `walk_cycle` IK gait.",
        asset_name="sentinel_hound_stage5_artist",
        output_dir=out_dir / "sentinel_hound_stage5_artist",
        export_formats=("bbmodel", "bedrock", "geckolib", "gltf_pbr", "labpbr"),
        min_aesthetic_score=0.74,
    )

    # 2. Multiview Human-Artist Asset Inversion (`permissive_cc/val.jsonl`, articulated >=4-bone asset with `system_prompt=None`)
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
    rich_val = artic_val or [
        r
        for r in val_rows
        if 14 <= int(r.get("box_count") or 0) <= 60
        and not r.get("ai_assisted")
        and r.get("commercial_weights_eligible")
    ]
    target_human = rich_val[0] if rich_val else val_rows[0]
    h_prog = parse_program(target_human["canonical_dsl"])
    h_ws = build_default_world_state_for_program(h_prog, seed=99)
    h_pngs, _, _, _ = render_world_views_rgba(h_ws, panel_size=(192, 192))
    ref_dir = out_dir / "human_artist_val_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_pil = []
    for vname in ("south", "right", "top", "iso"):
        (ref_dir / f"human_ref_{vname}.png").write_bytes(h_pngs[vname])
        ref_pil.append(Image.open(io.BytesIO(h_pngs[vname])))

    h_aud = str(getattr(h_prog.meta, "audit_profile", "authored"))
    h_prof = str(getattr(h_prog.meta, "target_profile", "bbmodel_5"))
    inv_res = pipeline.generate_3d_asset(
        f"Reconstruct a collision-free SyncCraft Block-DSL v3.2 program matching these 4 orthographic views "
        f"(south, right, top, iso) of `{target_human.get('title', target_human['asset_id'])}` with "
        f"target_profile={h_prof!r}, texel_density={h_prog.meta.texel_density}, and audit_profile={h_aud!r}.",
        images=ref_pil,
        asset_name="human_artist_multiview_stage5",
        output_dir=out_dir / "human_artist_multiview_stage5",
        export_formats=("bbmodel", "bedrock", "gltf_pbr"),
        min_aesthetic_score=0.68,
    )

    # 3. Critique & Upgrade of Stage-4's simplistic 9-box headless toy into an artist-grade asset!
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
    toy_before_aes = evaluate_3d_asset_aesthetics(stage4_toy_dsl, prompt="sentinel hound quadruped")
    upgraded_res = pipeline.critique_and_upgrade_asset(
        stage4_toy_dsl,
        format_kind="blockdsl",
        upgrade_goal=(
            "Upgrade this simplistic 9-box headless toy quadruped blockout into an artist-grade 20–32 box "
            "articulated armored hound (`bedrock_entity`) with neck, skull, muzzle, jaw, bilateral `mirror_x` ears, "
            "pelvis taper, chest armor (`inflate`), 2-segment tail, paws, >=4 OKLCH `_mat` ramps, and `walk_cycle` IK gait."
        ),
        use_model_first=True,
        asset_name="upgraded_from_stage4_toy",
        output_dir=out_dir / "upgraded_from_stage4_toy",
        export_formats=("bbmodel", "bedrock", "gltf_pbr"),
    )
    up_inner = upgraded_res["result"]
    up_aes = (
        up_inner.get("aesthetic_evaluation")
        if up_inner.get("aesthetic_evaluation") is not None
        else evaluate_3d_asset_aesthetics(up_inner["final_dsl"], prompt="sentinel hound quadruped")
    )

    return {
        "stage4_toy_baseline_aesthetics": toy_before_aes,
        "sentinel_hound_stage5": {
            "passed": hound_res["passed"],
            "boxes": hound_res["boxes"],
            "bones": hound_res["bones"],
            "reaudits_used": hound_res["reaudits_used"],
            "aesthetic_evaluation": hound_res["aesthetic_evaluation"],
            "preview_views": (hound_res["bundle"] or {}).get("preview_views", {}),
        },
        "human_artist_multiview_stage5": {
            "reference_asset_id": target_human["asset_id"],
            "reference_boxes": target_human.get("box_count"),
            "reference_bones": target_human.get("bone_count"),
            "passed": inv_res["passed"],
            "boxes": inv_res["boxes"],
            "bones": inv_res["bones"],
            "reaudits_used": inv_res["reaudits_used"],
            "aesthetic_evaluation": inv_res["aesthetic_evaluation"],
            "preview_views": (inv_res["bundle"] or {}).get("preview_views", {}),
        },
        "upgraded_from_stage4_toy": {
            "route": upgraded_res["route"],
            "passed": up_inner["passed"],
            "boxes": up_inner["boxes"],
            "bones": up_inner["bones"],
            "reaudits_used": up_inner.get("reaudits_used", 0),
            "aesthetic_evaluation": up_aes,
            "preview_views": (up_inner["bundle"] or {}).get("preview_views", {}),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Stage-5 campaign JSON config")
    ap.add_argument("--work-root", default=None, help="Override work_root")
    ap.add_argument("--checkpoint", default=None, help="Explicit model dir (default: <work_root>/final-model)")
    ap.add_argument("--fp8-checkpoint", default=None, help="Optional explicit FP8 model dir")
    ap.add_argument("--out-dir", default=None, help="Artifacts output dir")
    ap.add_argument("--out-json", default=None, help="Output verification JSON path")
    ap.add_argument("--export-cold-checkpoint", default=None, help="Optional cold-store checkpoint dir to package first")
    ap.add_argument("--export-dest", default=None, help="Destination BF16 release dir when --export-cold-checkpoint is set")
    ap.add_argument("--export-fp8-dest", default=None, help="Destination FP8 release dir when --export-cold-checkpoint is set")
    args = ap.parse_args()

    cfg = read(args.config)
    root = Path(args.work_root) if args.work_root else Path(cfg["work_root"]).resolve()

    export_receipt = None
    if args.export_cold_checkpoint and args.export_dest:
        export_receipt = export_cold_checkpoint_package(
            Path(args.export_cold_checkpoint),
            root / "final-model",
            Path(args.export_dest),
            Path(args.export_fp8_dest) if args.export_fp8_dest else None,
        )

    ckpt = Path(args.checkpoint) if args.checkpoint else (Path(args.export_dest) if args.export_dest else (root / "final-model"))
    token = os.environ.get("HF_TOKEN") or None

    model = load(ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
    handoff = read(root / "metadata" / "HANDOFF.json")
    catalogue = Catalogue(root / "metadata" / "corpus.sqlite", modalities="vision_text")
    reader = MediaReader(root, handoff, cfg, token, reserve=0)

    out_dir = Path(args.out_dir) if args.out_dir else (root / "stage5_verification_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Protected internal audit split (EN/ZH image, video, text + blank/shuffled ablations)
    audit_rep = evaluate(model, catalogue, reader, cfg, "audit", out_dir / "protected_audit_eval", generate=True)
    audit_health = {
        "by_stratum": audit_rep["by_stratum"],
        "synccraft_summary": audit_rep["synccraft_summary"],
    }

    # 2. Held-out external benchmarks (GPQA-Diamond, HLE Holdout, Terminal-Bench 3.0 Holdout, Live Python Exec)
    gpqa_res = _eval_heldout_gpqa_diamond(model, token, n_items=24)
    hle_res = _eval_heldout_hle(model, token, n_text=16, n_mm=10)
    tb_code_res = _eval_heldout_terminal_and_agentic_coding(model, token, n_tb=8)

    # 3. Unseen SAT_test, 4-Way UV/Alpha, Visual Aesthetic Judgment, and World Ladder Holdout
    vision_3d_res = _eval_sat_test_and_4way_uv_and_aesthetics(model, token)

    # 4. Artist-Grade 3D Asset Generation & Upgrade Pipeline (BF16)
    pipeline_bf16 = _eval_artist_grade_3d_pipeline(model, cfg, out_dir / "assets_bf16", token)

    # 5. Also verify exported FP8 (`final-model-fp8` or `--fp8-checkpoint`) if present!
    fp8_summary: dict[str, Any] | None = None
    fp8_dir = (
        Path(args.fp8_checkpoint)
        if args.fp8_checkpoint
        else (Path(args.export_fp8_dest) if args.export_fp8_dest else (root / "final-model-fp8"))
    )
    if (fp8_dir / "MODEL_RELEASE.json").is_file():
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fp8_model = load(fp8_dir, device="cuda" if torch.cuda.is_available() else "cpu")
        fp8_pipe = SyncCraftWorldPipeline(fp8_model, config={**cfg, "min_aesthetic_score": 0.72}, default_max_new_tokens=3072)
        fp8_hound = fp8_pipe.generate_3d_asset(
            "Design an artist-grade SyncCraft Block-DSL v3.2 armored sentinel hound quadruped (`bedrock_entity`, `texel_density: 16`) "
            "with ribcage, pelvis, chest armor, neck, skull, muzzle, jaw, bilateral `mirror_x` ears, tail, 4 three-part legs, 5 `_mat` ramps, and `walk_cycle` IK.",
            asset_name="sentinel_hound_stage5_fp8",
            output_dir=out_dir / "assets_fp8",
            export_formats=("bbmodel", "bedrock", "gltf_pbr"),
            min_aesthetic_score=0.72,
        )
        fp8_summary = {
            "passed": fp8_hound["passed"],
            "boxes": fp8_hound["boxes"],
            "bones": fp8_hound["bones"],
            "reaudits_used": fp8_hound["reaudits_used"],
            "aesthetic_evaluation": fp8_hound["aesthetic_evaluation"],
            "preview_views": (fp8_hound["bundle"] or {}).get("preview_views", {}),
        }

    full_summary = {
        "checkpoint": str(ckpt),
        "export_receipt": export_receipt,
        "protected_audit": audit_health,
        "gpqa_diamond_holdout": gpqa_res,
        "hle_disjoint_holdout": hle_res,
        "terminal_bench3_and_coding": tb_code_res,
        "vision_3d_and_aesthetic_judgment": vision_3d_res,
        "artist_grade_3d_pipeline_bf16": pipeline_bf16,
        "artist_grade_3d_pipeline_fp8": fp8_summary,
    }
    dest = Path(args.out_json) if args.out_json else (root / "STAGE5_RSI_CAPABILITY_VERIFICATION.json")
    write(dest, full_summary)
    print(json.dumps(full_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
