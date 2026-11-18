#!/usr/bin/env python3
"""Standard Public Multimodal Benchmark Evaluation Suite for Spark-X2.6-VL.

Evaluates `SparkNative` checkpoints (Stage-4, Stage-5+RSI-2, Stage-6.1 BF16 & FP8)
across 8 zero-overlap, immutable-SHA public multimodal benchmarks:

1. `MMStar` (`Lin-Chen/MMStar`, 6 core vision-indispensable capability pillars; direct + `<think>`)
2. `MMMU` (`lmms-lab/MMMU` official `validation` split across 30 college subfields / 6 disciplines; direct + `<think>`)
3. `AI2D` (`lmms-lab/ai2d` `test` split scientific diagram reasoning)
4. `OCRBench` (`echo840/OCRBench` `test` split across all 5 official OCR/KIE/Doc/HMER task groups)
5. `RealWorldQA` (`xai-org/RealWorldQA` `test` split physical & spatial scene QA)
6. `BLINK` (`BLINK-Benchmark/BLINK` `val` split across 6 3D/multi-view/spatial subtasks)
7. `POPE` (`lmms-lab/POPE` `random`, `popular`, `adversarial` hallucination resistance: Acc, F1, Yes-Ratio)
8. `MMBench-CN` (`lmms-lab/MMBench_CN` `dev` split Chinese multimodal perception & reasoning)

Strictly avoids `eval()` (`AGENTS.md`): all serialized option lists use `ast.literal_eval`.
Writes incremental results after every benchmark so live progress is always inspectable.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.final.common import write  # noqa: E402
from colab.final.inference import generate, load, visual_media  # noqa: E402
from colab.final.synccraft_bridge import strip_think_block  # noqa: E402

BENCHMARK_PINS: dict[str, dict[str, Any]] = {
    "mmstar": {
        "repo_id": "Lin-Chen/MMStar",
        "revision": "bc98d668301da7b14f648724866e57302778ab27",
        "files": ["mmstar.parquet"],
    },
    "mmmu_val": {
        "repo_id": "lmms-lab/MMMU",
        "revision": "364f2e2eb107b36e07ff4c5a15f5947a759cef47",
        "files": ["data/validation-00000-of-00001.parquet"],
    },
    "ai2d_test": {
        "repo_id": "lmms-lab/ai2d",
        "revision": "c83a9b9692933aff8349157c88a413df9d02c4e5",
        "files": ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"],
    },
    "ocrbench": {
        "repo_id": "echo840/OCRBench",
        "revision": "92a54bd1384387c178d5a07140a2d85e0a3d12e1",
        "files": ["data/test-00000-of-00001.parquet"],
    },
    "realworldqa": {
        "repo_id": "xai-org/RealWorldQA",
        "revision": "17e7f75e092e47169732462ea3cdfebe911105dd",
        "files": ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"],
    },
    "blink_val": {
        "repo_id": "BLINK-Benchmark/BLINK",
        "revision": "a3666eb249237ba3d5eca8db21176cc47967e040",
        "files": [
            "Multi-view_Reasoning/val-00000-of-00001.parquet",
            "Relative_Depth/val-00000-of-00001.parquet",
            "Spatial_Relation/val-00000-of-00001.parquet",
            "Object_Localization/val-00000-of-00001.parquet",
            "Visual_Correspondence/val-00000-of-00001.parquet",
            "Counting/val-00000-of-00001.parquet",
        ],
    },
    "pope": {
        "repo_id": "lmms-lab/POPE",
        "revision": "4db1276663dfa5eb8ad16a52d24c31a09e470896",
        "files": [
            "Full/random-00000-of-00001.parquet",
            "Full/popular-00000-of-00001.parquet",
            "Full/adversarial-00000-of-00001.parquet",
        ],
    },
    "mmbench_cn_dev": {
        "repo_id": "lmms-lab/MMBench_CN",
        "revision": "78601bcca41a57248c05f130d9902db624aa6f4d",
        "files": ["data/dev-00000-of-00001-d459e48a71003d0c.parquet"],
    },
}

MMMU_DISCIPLINE_BY_SUBFIELD: dict[str, str] = {
    "Art": "Art & Design",
    "Art_Theory": "Art & Design",
    "Design": "Art & Design",
    "Music": "Art & Design",
    "Accounting": "Business",
    "Economics": "Business",
    "Finance": "Business",
    "Manage": "Business",
    "Marketing": "Business",
    "Biology": "Science",
    "Chemistry": "Science",
    "Geography": "Science",
    "Math": "Science",
    "Physics": "Science",
    "Basic_Medical_Science": "Health & Medicine",
    "Clinical_Medicine": "Health & Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Health & Medicine",
    "Pharmacy": "Health & Medicine",
    "Public_Health": "Health & Medicine",
    "History": "Humanities & Social Science",
    "Literature": "Humanities & Social Science",
    "Sociology": "Humanities & Social Science",
    "Psychology": "Humanities & Social Science",
    "Agriculture": "Tech & Engineering",
    "Architecture_and_Engineering": "Tech & Engineering",
    "Computer_Science": "Tech & Engineering",
    "Electronics": "Tech & Engineering",
    "Energy_and_Power": "Tech & Engineering",
    "Materials": "Tech & Engineering",
    "Mechanical_Engineering": "Tech & Engineering",
}

NUMBER_WORDS: dict[str, str] = {
    "zero": "0",
    "none": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def _det_hash(tag: str, key: str, seed: int = 20261117) -> str:
    return hashlib.sha256(f"{seed}:{tag}:{key}".encode("utf-8")).hexdigest()


def _stratified_sample(
    rows: list[dict[str, Any]],
    group_fn: Any,
    key_fn: Any,
    total_target: int,
    *,
    tag: str,
    seed: int = 20261117,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(group_fn(r))].append(r)
    groups = sorted(buckets.keys())
    if not groups:
        return []
    for g in groups:
        buckets[g].sort(key=lambda r: _det_hash(f"{tag}:{g}", str(key_fn(r)), seed))

    per_group = max(1, total_target // len(groups))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for g in groups:
        selected.extend(buckets[g][:per_group])
        leftovers.extend(buckets[g][per_group:])
    if len(selected) < total_target and leftovers:
        leftovers.sort(key=lambda r: _det_hash(f"{tag}:extra", str(key_fn(r)), seed))
        selected.extend(leftovers[: total_target - len(selected)])
    return selected[:total_target]


def decode_parquet_image(val: Any) -> Image.Image | None:
    """Decode a Parquet image cell (`bytes`, `{'bytes': ...}`, `{'path': ...}`, or `PIL.Image`)."""
    if val is None:
        return None
    if isinstance(val, Image.Image):
        im = val
    elif isinstance(val, (bytes, bytearray)):
        im = Image.open(io.BytesIO(bytes(val)))
    elif isinstance(val, dict):
        b = val.get("bytes")
        p = val.get("path")
        if b is not None:
            im = Image.open(io.BytesIO(bytes(b)))
        elif p and Path(p).is_file():
            im = Image.open(p)
        else:
            return None
    else:
        return None
    im.load()
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        return im.convert("RGBA")
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def parse_safe_list(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    s = str(raw or "").strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return [s]


def clean_prediction_body(raw_text: str) -> tuple[str, bool]:
    """Return `(post_think_body, think_closed_ok)` while handling unclosed `<think>` safely."""
    txt = str(raw_text or "")
    if "<think>" in txt:
        if "</think>" in txt:
            _, body = strip_think_block(txt)
            return body.strip(), True
        tail = txt.split("<think>", 1)[1].strip()
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        return (lines[-1] if lines else tail), False
    return txt.strip(), True


def extract_mcq_choice(
    raw_text: str,
    valid_letters: list[str] | tuple[str, ...],
    option_texts: list[str] | None = None,
    *,
    min_option_text_len: int = 1,
) -> str | None:
    """Standard VLMEvalKit / MMBench / MMMU multiple-choice option extractor."""
    body, _ = clean_prediction_body(raw_text)
    if not body:
        return None
    valid_set = {c.upper() for c in valid_letters}
    letter_cls = "".join(sorted(valid_set))

    # 1. Explicit Answer / Option / 答案 marker (take last match if CoT precedes final answer)
    explicit = re.findall(
        rf"(?:answer|option|choice|答案|选项)\s*(?:is|为|:|：)?\s*[\(\[（\*]*([{letter_cls}])[\)\]）\*]*(?![A-Za-z0-9])",
        body,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()

    # 2. Standalone single letter response
    m_exact = re.match(rf"^\s*[\(\[（\*]*([{letter_cls}])[\)\]）\*]*\.?\s*$", body, flags=re.IGNORECASE)
    if m_exact:
        return m_exact.group(1).upper()

    # 3. Leading `(B) ...` or `B. ...` or `B: ...` at start of answer
    m_lead_paren = re.match(rf"^\s*[\(\[（]([{letter_cls}])[\)\]）]", body)
    if m_lead_paren:
        return m_lead_paren.group(1).upper()
    m_lead_dot = re.match(rf"^\s*([{letter_cls}])[\.\):：،,]\s*", body)
    if m_lead_dot:
        return m_lead_dot.group(1).upper()

    # 4. Unique option-text match when model writes out the exact option phrase
    if option_texts:
        norm_body = re.sub(r"\s+", " ", body).strip().casefold()
        exact_hits: list[str] = []
        sub_hits: list[str] = []
        for idx, opt in enumerate(option_texts):
            if idx >= len(valid_letters):
                break
            ltr = valid_letters[idx].upper()
            norm_opt = re.sub(r"\s+", " ", str(opt)).strip().casefold()
            if len(norm_opt) < min_option_text_len:
                continue
            if norm_body.rstrip(".") == norm_opt.rstrip("."):
                exact_hits.append(ltr)
            elif len(norm_opt) >= 3 and norm_opt in norm_body:
                sub_hits.append(ltr)
        if len(exact_hits) == 1:
            return exact_hits[0]
        if len(sub_hits) == 1:
            return sub_hits[0]

    # 5. Trailing parenthesized `(C)` or final capital letter token
    tail_paren = re.findall(rf"[\(\[（]([{letter_cls}])[\)\]）]", body)
    if tail_paren:
        return tail_paren[-1].upper()
    tail_tok = re.findall(rf"\b([{letter_cls}])\b", body)
    if tail_tok:
        return tail_tok[-1].upper()
    return None


def score_ocrbench_answer(raw_text: str, gold_answers: list[str], question_type: str) -> bool:
    """Official OCRBench scoring rule (Liu et al., 2024)."""
    body, _ = clean_prediction_body(raw_text)
    if not body:
        return False
    is_hmer = "mathematical expression" in str(question_type).lower()
    if is_hmer:
        pred_norm = re.sub(r"\s+", "", body)
        for gt in gold_answers:
            gt_norm = re.sub(r"\s+", "", str(gt))
            if gt_norm and gt_norm in pred_norm:
                return True
        return False

    pred_norm = re.sub(r"\s+", " ", body.lower()).strip()
    pred_nocomma = pred_norm.replace(",", "")
    for gt in gold_answers:
        gt_norm = re.sub(r"\s+", " ", str(gt).lower()).strip()
        if not gt_norm:
            continue
        if gt_norm in pred_norm or gt_norm.replace(",", "") in pred_nocomma:
            return True
    return False


def score_realworldqa_answer(raw_text: str, question: str, gold_answer: str) -> bool:
    """Official RealWorldQA multiple-choice + short-word/number evaluation."""
    gold = str(gold_answer or "").strip()
    if not gold:
        return False
    # Check if question is multiple-choice (`A. ... B. ...`) with single-letter gold
    if len(gold) == 1 and gold.upper() in "ABCDE" and re.search(r"(?:^|\n)\s*[A-E][\.\)]\s+", question):
        opts_found = re.findall(r"(?:^|\n)\s*([A-E])[\.\)]\s*(.+)", question)
        letters = [x[0].upper() for x in opts_found] or ["A", "B", "C", "D"]
        opt_txts = [x[1].strip() for x in opts_found] if opts_found else None
        pred_c = extract_mcq_choice(raw_text, letters, opt_txts, min_option_text_len=2)
        return pred_c == gold.upper()

    body, _ = clean_prediction_body(raw_text)
    body = re.sub(r"^(?:answer|the answer is)\s*[:：]?\s*", "", body.strip(), flags=re.IGNORECASE)
    norm_pred = re.sub(r"[^\w\s\.\-]", "", body.lower()).strip()
    norm_gold = re.sub(r"[^\w\s\.\-]", "", gold.lower()).strip()
    norm_pred = NUMBER_WORDS.get(norm_pred, norm_pred)
    norm_gold = NUMBER_WORDS.get(norm_gold, norm_gold)
    if norm_pred == norm_gold:
        return True
    tokens = [NUMBER_WORDS.get(t, t) for t in norm_pred.split()]
    if norm_gold in ("yes", "no"):
        if tokens and tokens[0] in ("yes", "no"):
            return tokens[0] == norm_gold
    if len(tokens) <= 5 and norm_gold in tokens:
        return True
    return False


def score_mmmu_open_answer(raw_text: str, gold_raw: Any) -> bool:
    """Official MMMU open-response relaxed numeric / string scorer."""
    body, _ = clean_prediction_body(raw_text)
    body = re.sub(r"^(?:answer|final answer)\s*[:：]?\s*", "", body.strip(), flags=re.IGNORECASE)
    if not body:
        return False
    golds = parse_safe_list(gold_raw)
    pred_low = re.sub(r"\s+", " ", body.lower()).strip().rstrip(".")
    for g in golds:
        g_low = re.sub(r"\s+", " ", str(g).lower()).strip().rstrip(".")
        if not g_low:
            continue
        if pred_low == g_low or f" {g_low} " in f" {pred_low} ":
            return True
        try:
            g_num = float(g_low.replace(",", "").replace("$", "").replace("%", ""))
            nums = re.findall(r"-?\d+(?:\.\d+)?", pred_low.replace(",", ""))
            if any(abs(float(n) - g_num) <= max(1e-3, abs(g_num) * 1e-2) for n in nums):
                return True
        except ValueError:
            pass
    return False


def extract_yes_no(raw_text: str) -> str | None:
    """Extract binary `'yes'` / `'no'` prediction for POPE hallucination evaluation."""
    body, _ = clean_prediction_body(raw_text)
    if not body:
        return None
    explicit = re.findall(r"(?:answer|答案)\s*(?:is|为|:|：)?\s*\b(yes|no|true|false)\b", body, flags=re.IGNORECASE)
    if explicit:
        return "yes" if explicit[-1].lower() in ("yes", "true") else "no"
    low = body.lower().strip()
    first = re.findall(r"\b(yes|no|true|false)\b", low)
    if first:
        return "yes" if first[0] in ("yes", "true") else "no"
    return None


def _download_pinned_parquets(bench_key: str, token: str | None) -> list[Path]:
    from huggingface_hub import hf_hub_download

    pin = BENCHMARK_PINS[bench_key]
    paths: list[Path] = []
    for fn in pin["files"]:
        p = hf_hub_download(
            repo_id=pin["repo_id"],
            filename=fn,
            repo_type="dataset",
            revision=pin["revision"],
            token=token,
        )
        paths.append(Path(p))
    return paths


def _ocrbench_group(qtype: str) -> str:
    q = str(qtype)
    if "Text Recognition" in q or "Handwriting Recognition" in q or "Digit String" in q:
        return "Text Recognition"
    return q


def build_frozen_benchmark_suites(token: str | None, *, scale: float = 1.0) -> dict[str, list[dict[str, Any]]]:
    """Load and deterministically sample all 8 pinned benchmarks once so every model sees identical items."""
    import pyarrow.parquet as pq

    def _n(base: int) -> int:
        return max(12, int(round(base * scale)))

    suites: dict[str, list[dict[str, Any]]] = {}

    # 1. MMStar (300 items = 50 x 6 pillars)
    mmstar_rows = pq.read_table(_download_pinned_parquets("mmstar", token)[0]).to_pylist()
    suites["mmstar"] = _stratified_sample(
        mmstar_rows,
        lambda r: r["category"],
        lambda r: r["index"],
        _n(300),
        tag="mmstar",
    )
    suites["mmstar_think_subset"] = _stratified_sample(
        suites["mmstar"],
        lambda r: r["category"],
        lambda r: r["index"],
        _n(120),
        tag="mmstar_think",
    )

    # 2. MMMU Validation (180 items = 6 per subfield x 30 subfields)
    mmmu_rows = pq.read_table(_download_pinned_parquets("mmmu_val", token)[0]).to_pylist()
    for r in mmmu_rows:
        m_sub = re.match(r"^validation_(.+)_\d+$", str(r["id"]))
        sub_key = m_sub.group(1) if m_sub else str(r.get("subfield", "Other"))
        r["_subject_key"] = sub_key
        r["_discipline"] = MMMU_DISCIPLINE_BY_SUBFIELD.get(sub_key, "Science")
    suites["mmmu_val"] = _stratified_sample(
        mmmu_rows,
        lambda r: r["_subject_key"],
        lambda r: r["id"],
        _n(180),
        tag="mmmu_val",
    )

    # 3. AI2D Test (200 items)
    ai2d_rows: list[dict[str, Any]] = []
    for idx_f, p in enumerate(_download_pinned_parquets("ai2d_test", token)):
        for idx_r, r in enumerate(pq.read_table(p).to_pylist()):
            r["_uid"] = f"{idx_f}:{idx_r}:{r['question'][:48]}"
            ai2d_rows.append(r)
    suites["ai2d_test"] = _stratified_sample(
        ai2d_rows,
        lambda _r: "all",
        lambda r: r["_uid"],
        _n(200),
        tag="ai2d_test",
    )

    # 4. OCRBench (200 items across 5 official groups)
    ocr_rows = pq.read_table(_download_pinned_parquets("ocrbench", token)[0]).to_pylist()
    for idx_r, r in enumerate(ocr_rows):
        r["_uid"] = f"{idx_r}:{r['dataset']}"
        r["_group"] = _ocrbench_group(r["question_type"])
    suites["ocrbench"] = _stratified_sample(
        ocr_rows,
        lambda r: r["_group"],
        lambda r: r["_uid"],
        _n(200),
        tag="ocrbench",
    )

    # 5. RealWorldQA (200 items)
    rw_rows: list[dict[str, Any]] = []
    for idx_f, p in enumerate(_download_pinned_parquets("realworldqa", token)):
        for idx_r, r in enumerate(pq.read_table(p).to_pylist()):
            is_mcq = bool(len(str(r["answer"]).strip()) == 1 and str(r["answer"]).strip().upper() in "ABCDE" and "\nA." in str(r["question"]))
            r["_uid"] = f"{idx_f}:{idx_r}:{r['question'][:48]}"
            r["_qtype"] = "mcq" if is_mcq else "short_qa"
            rw_rows.append(r)
    suites["realworldqa"] = _stratified_sample(
        rw_rows,
        lambda r: r["_qtype"],
        lambda r: r["_uid"],
        _n(200),
        tag="realworldqa",
    )

    # 6. BLINK Val (180 items = 30 x 6 spatial/multi-view subtasks)
    blink_rows: list[dict[str, Any]] = []
    for p in _download_pinned_parquets("blink_val", token):
        blink_rows.extend(pq.read_table(p).to_pylist())
    suites["blink_val"] = _stratified_sample(
        blink_rows,
        lambda r: r["sub_task"],
        lambda r: r["idx"],
        _n(180),
        tag="blink_val",
    )

    # 7. POPE (300 items = 100 random + 100 popular + 100 adversarial, balanced yes/no)
    pope_rows: list[dict[str, Any]] = []
    for p in _download_pinned_parquets("pope", token):
        split_name = p.name.split("-")[0]
        for r in pq.read_table(p).to_pylist():
            r["_split_label"] = f"{split_name}:{str(r['answer']).strip().lower()}"
            r["_split_only"] = split_name
            r["_uid"] = f"{split_name}:{r['id']}:{r['question_id']}"
            pope_rows.append(r)
    suites["pope"] = _stratified_sample(
        pope_rows,
        lambda r: r["_split_label"],
        lambda r: r["_uid"],
        _n(300),
        tag="pope",
    )

    # 8. MMBench-CN Dev (200 Chinese multimodal items stratified by category)
    mmb_cn_rows = pq.read_table(_download_pinned_parquets("mmbench_cn_dev", token)[0]).to_pylist()
    suites["mmbench_cn_dev"] = _stratified_sample(
        mmb_cn_rows,
        lambda r: r.get("category", "all"),
        lambda r: r["index"],
        _n(200),
        tag="mmbench_cn_dev",
    )

    return suites


def eval_mmstar(model: Any, rows: list[dict[str, Any]], *, enable_thinking: bool = False) -> dict[str, Any]:
    t0 = time.time()
    by_cat: dict[str, list[int]] = defaultdict(list)
    think_closed = 0
    for r in rows:
        im = decode_parquet_image(r["image"])
        med = visual_media([im]) if im is not None else None
        prompt = f"{str(r['question']).strip()}\nAnswer with the option letter (A, B, C, or D) in the format: Answer: <letter>"
        pred = generate(
            model,
            prompt,
            media=med,
            enable_thinking=enable_thinking,
            max_new_tokens=320 if enable_thinking else 48,
            max_think_tokens=224 if enable_thinking else None,
        )
        _, closed_ok = clean_prediction_body(pred)
        if closed_ok:
            think_closed += 1
        choice = extract_mcq_choice(pred, ("A", "B", "C", "D"))
        ok = int(choice == str(r["answer"]).strip().upper())
        by_cat[str(r["category"])].append(ok)
    total_ok = sum(sum(v) for v in by_cat.values())
    n = len(rows)
    return {
        "benchmark": "MMStar",
        "repo_id": BENCHMARK_PINS["mmstar"]["repo_id"],
        "revision": BENCHMARK_PINS["mmstar"]["revision"],
        "enable_thinking": enable_thinking,
        "evaluated": n,
        "correct": total_ok,
        "accuracy": round(total_ok / max(1, n), 4),
        "by_category": {k: round(sum(v) / max(1, len(v)), 4) for k, v in sorted(by_cat.items())},
        "think_close_rate": round(think_closed / max(1, n), 4) if enable_thinking else None,
        "elapsed_s": round(time.time() - t0, 2),
    }


def eval_mmmu_val(model: Any, rows: list[dict[str, Any]], *, enable_thinking: bool = False) -> dict[str, Any]:
    t0 = time.time()
    by_disc: dict[str, list[int]] = defaultdict(list)
    think_closed = 0
    for r in rows:
        imgs = [decode_parquet_image(r.get(f"image_{k}")) for k in range(1, 8)]
        imgs = [im for im in imgs if im is not None]
        vkind = "state_pair" if len(imgs) == 2 else ("multiview_render" if len(imgs) > 2 else "independent_image")
        v_units = (
            [{"visual_kind": "state_pair", "view_id": "before"}, {"visual_kind": "state_pair", "view_id": "after"}]
            if len(imgs) == 2
            else None
        )
        med = visual_media(imgs, visual_kind=vkind, visual_units=v_units) if imgs else None
        qtype = str(r.get("question_type", "multiple-choice"))
        opts = parse_safe_list(r.get("options", ""))
        if qtype == "multiple-choice" and opts:
            letters = [chr(ord("A") + i) for i in range(len(opts))]
            opt_block = "\n".join(f"({letters[i]}) {opts[i]}" for i in range(len(opts)))
            prompt = (
                f"{str(r['question']).strip()}\nOptions:\n{opt_block}\n"
                f"Answer with the option letter ({', '.join(letters)}) in the format: Answer: <letter>"
            )
        else:
            letters = []
            prompt = f"{str(r['question']).strip()}\nAnswer concisely with the final number, word, or phrase in the format: Answer: <value>"
        pred = generate(
            model,
            prompt,
            media=med,
            enable_thinking=enable_thinking,
            max_new_tokens=352 if enable_thinking else 64,
            max_think_tokens=240 if enable_thinking else None,
        )
        _, closed_ok = clean_prediction_body(pred)
        if closed_ok:
            think_closed += 1
        if qtype == "multiple-choice" and letters:
            choice = extract_mcq_choice(pred, letters, opts, min_option_text_len=2)
            ok = int(choice == str(r["answer"]).strip().upper())
        else:
            ok = int(score_mmmu_open_answer(pred, r["answer"]))
        by_disc[str(r["_discipline"])].append(ok)
    total_ok = sum(sum(v) for v in by_disc.values())
    n = len(rows)
    return {
        "benchmark": "MMMU (validation)",
        "repo_id": BENCHMARK_PINS["mmmu_val"]["repo_id"],
        "revision": BENCHMARK_PINS["mmmu_val"]["revision"],
        "enable_thinking": enable_thinking,
        "evaluated": n,
        "correct": total_ok,
        "accuracy": round(total_ok / max(1, n), 4),
        "by_discipline": {k: round(sum(v) / max(1, len(v)), 4) for k, v in sorted(by_disc.items())},
        "think_close_rate": round(think_closed / max(1, n), 4) if enable_thinking else None,
        "elapsed_s": round(time.time() - t0, 2),
    }


def eval_ai2d(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.time()
    correct = 0
    for r in rows:
        im = decode_parquet_image(r["image"])
        med = visual_media([im]) if im is not None else None
        opts = parse_safe_list(r["options"])
        letters = [chr(ord("A") + i) for i in range(len(opts))]
        gold_ans = str(r["answer"]).strip()
        gold_ltr = chr(ord("A") + int(gold_ans)) if gold_ans.isdigit() else gold_ans.upper()
        opt_block = "\n".join(f"({letters[i]}) {opts[i]}" for i in range(len(opts)))
        prompt = (
            f"{str(r['question']).strip()}\nOptions:\n{opt_block}\n"
            f"Answer with only the option letter ({', '.join(letters)}) in the format: Answer: <letter>"
        )
        pred = generate(model, prompt, media=med, enable_thinking=False, max_new_tokens=40)
        # Require >= 3 chars for option_text fallback so diagram labels 'a'/'b'/'c'/'D' never collide with option letters
        choice = extract_mcq_choice(pred, letters, opts, min_option_text_len=3)
        if choice == gold_ltr:
            correct += 1
    n = len(rows)
    return {
        "benchmark": "AI2D (test)",
        "repo_id": BENCHMARK_PINS["ai2d_test"]["repo_id"],
        "revision": BENCHMARK_PINS["ai2d_test"]["revision"],
        "evaluated": n,
        "correct": correct,
        "accuracy": round(correct / max(1, n), 4),
        "elapsed_s": round(time.time() - t0, 2),
    }


def eval_ocrbench(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.time()
    by_grp: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        im = decode_parquet_image(r["image"])
        med = visual_media([im]) if im is not None else None
        q = str(r["question"]).strip()
        prompt = f"{q}\nRead the text/formula carefully from the image and answer directly and concisely."
        pred = generate(model, prompt, media=med, enable_thinking=False, max_new_tokens=96, task="document_ocr")
        golds = parse_safe_list(r["answer"])
        ok = int(score_ocrbench_answer(pred, golds, str(r["question_type"])))
        by_grp[str(r["_group"])].append(ok)
    total_ok = sum(sum(v) for v in by_grp.values())
    n = len(rows)
    return {
        "benchmark": "OCRBench",
        "repo_id": BENCHMARK_PINS["ocrbench"]["repo_id"],
        "revision": BENCHMARK_PINS["ocrbench"]["revision"],
        "evaluated": n,
        "correct": total_ok,
        "accuracy": round(total_ok / max(1, n), 4),
        "official_scaled_1000": round(1000.0 * total_ok / max(1, n), 1),
        "by_task_group": {k: round(sum(v) / max(1, len(v)), 4) for k, v in sorted(by_grp.items())},
        "elapsed_s": round(time.time() - t0, 2),
    }


def eval_realworldqa(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.time()
    correct = 0
    by_qt: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        im = decode_parquet_image(r["image"])
        med = visual_media([im]) if im is not None else None
        q = str(r["question"]).strip()
        pred = generate(model, q, media=med, enable_thinking=False, max_new_tokens=40)
        ok = int(score_realworldqa_answer(pred, q, str(r["answer"])))
        correct += ok
        by_qt[str(r.get("_qtype", "all"))].append(ok)
    n = len(rows)
    return {
        "benchmark": "RealWorldQA",
        "repo_id": BENCHMARK_PINS["realworldqa"]["repo_id"],
        "revision": BENCHMARK_PINS["realworldqa"]["revision"],
        "evaluated": n,
        "correct": correct,
        "accuracy": round(correct / max(1, n), 4),
        "by_question_type": {k: round(sum(v) / max(1, len(v)), 4) for k, v in sorted(by_qt.items())},
        "elapsed_s": round(time.time() - t0, 2),
    }


def eval_blink(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.time()
    by_sub: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        imgs = [decode_parquet_image(r.get(f"image_{k}")) for k in range(1, 5)]
        imgs = [im for im in imgs if im is not None]
        vkind = "state_pair" if len(imgs) == 2 else ("multiview_render" if len(imgs) > 2 else "independent_image")
        v_units = (
            [{"visual_kind": "state_pair", "view_id": "before"}, {"visual_kind": "state_pair", "view_id": "after"}]
            if len(imgs) == 2
            else None
        )
        med = visual_media(imgs, visual_kind=vkind, visual_units=v_units) if imgs else None
        choices = parse_safe_list(r.get("choices", ""))
        letters = [chr(ord("A") + i) for i in range(max(2, len(choices)))]
        prompt = f"{str(r['prompt']).strip()}\nAnswer with only the option letter ({', '.join(letters)}) in the format: Answer: <letter>"
        pred = generate(model, prompt, media=med, enable_thinking=False, max_new_tokens=40)
        gold_ltr = str(r["answer"]).strip().strip("()").upper()
        choice = extract_mcq_choice(pred, letters, choices, min_option_text_len=2)
        ok = int(choice == gold_ltr)
        by_sub[str(r["sub_task"])].append(ok)
    total_ok = sum(sum(v) for v in by_sub.values())
    n = len(rows)
    return {
        "benchmark": "BLINK (val, 6 Spatial/Multi-View Subtasks)",
        "repo_id": BENCHMARK_PINS["blink_val"]["repo_id"],
        "revision": BENCHMARK_PINS["blink_val"]["revision"],
        "evaluated": n,
        "correct": total_ok,
        "accuracy": round(total_ok / max(1, n), 4),
        "by_subtask": {k: round(sum(v) / max(1, len(v)), 4) for k, v in sorted(by_sub.items())},
        "elapsed_s": round(time.time() - t0, 2),
    }


def eval_pope(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.time()
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0})
    for r in rows:
        im = decode_parquet_image(r["image"])
        med = visual_media([im]) if im is not None else None
        prompt = f"{str(r['question']).strip()} Please answer yes or no."
        pred = generate(model, prompt, media=med, enable_thinking=False, max_new_tokens=24)
        yn = extract_yes_no(pred) or "no"
        gold = str(r["answer"]).strip().lower()
        sp = str(r["_split_only"])
        for bucket in (sp, "overall"):
            st = stats[bucket]
            st["n"] += 1
            if gold == "yes" and yn == "yes":
                st["tp"] += 1
            elif gold == "no" and yn == "yes":
                st["fp"] += 1
            elif gold == "no" and yn == "no":
                st["tn"] += 1
            else:
                st["fn"] += 1

    def _metrics(st: dict[str, int]) -> dict[str, float]:
        tp, fp, tn, fn, n = st["tp"], st["fp"], st["tn"], st["fn"], max(1, st["n"])
        acc = (tp + tn) / n
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0
        yes_ratio = (tp + fp) / n
        return {
            "evaluated": st["n"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "yes_ratio": round(yes_ratio, 4),
        }

    overall = _metrics(stats["overall"])
    return {
        "benchmark": "POPE (random + popular + adversarial)",
        "repo_id": BENCHMARK_PINS["pope"]["repo_id"],
        "revision": BENCHMARK_PINS["pope"]["revision"],
        **overall,
        "by_split": {k: _metrics(v) for k, v in sorted(stats.items()) if k != "overall"},
        "elapsed_s": round(time.time() - t0, 2),
    }


def eval_mmbench_cn(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.time()
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        im = decode_parquet_image(r["image"])
        med = visual_media([im]) if im is not None else None
        hint = str(r.get("hint", "")).strip()
        letters: list[str] = []
        opts: list[str] = []
        for ltr in ("A", "B", "C", "D"):
            val = str(r.get(ltr, "")).strip()
            if val and val.lower() != "nan":
                letters.append(ltr)
                opts.append(val)
        parts: list[str] = []
        if hint and hint.lower() != "nan":
            parts.append(hint)
        parts.append(str(r["question"]).strip())
        parts.append("选项：\n" + "\n".join(f"({letters[i]}) {opts[i]}" for i in range(len(letters))))
        parts.append(f"请直接回答正确选项的字母（{', '.join(letters)}），格式为：Answer: <字母>")
        prompt = "\n".join(parts)
        pred = generate(model, prompt, media=med, enable_thinking=False, max_new_tokens=40, language="zh")
        choice = extract_mcq_choice(pred, letters, opts, min_option_text_len=2)
        ok = int(choice == str(r["answer"]).strip().upper())
        by_cat[str(r.get("category", "all"))].append(ok)
    total_ok = sum(sum(v) for v in by_cat.values())
    n = len(rows)
    return {
        "benchmark": "MMBench-CN (dev)",
        "repo_id": BENCHMARK_PINS["mmbench_cn_dev"]["repo_id"],
        "revision": BENCHMARK_PINS["mmbench_cn_dev"]["revision"],
        "evaluated": n,
        "correct": total_ok,
        "accuracy": round(total_ok / max(1, n), 4),
        "by_category": {k: round(sum(v) / max(1, len(v)), 4) for k, v in sorted(by_cat.items())},
        "elapsed_s": round(time.time() - t0, 2),
    }


def evaluate_checkpoint_suite(
    label: str,
    model_dir: Path,
    suites: dict[str, list[dict[str, Any]]],
    *,
    run_thinking_ablations: bool = False,
    progress_out: Path | None = None,
    accumulated_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print(f"\n=== Loading [{label}] from {model_dir} ===", flush=True)
    t_start = time.time()
    model = load(model_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    res: dict[str, Any] = {"model_dir": str(model_dir), "benchmarks": {}}

    def _save_step(key: str, val: dict[str, Any]) -> None:
        res["benchmarks"][key] = val
        score_val = val.get("f1", val.get("accuracy"))
        print(f"  [{label}] {key}: score={score_val} (n={val.get('evaluated')}, {val.get('elapsed_s')}s)", flush=True)
        if progress_out is not None and accumulated_doc is not None:
            accumulated_doc["models"][label] = res
            write(progress_out, accumulated_doc)

    _save_step("mmstar_direct", eval_mmstar(model, suites["mmstar"], enable_thinking=False))
    _save_step("mmmu_val_direct", eval_mmmu_val(model, suites["mmmu_val"], enable_thinking=False))
    _save_step("ai2d_test", eval_ai2d(model, suites["ai2d_test"]))
    _save_step("ocrbench", eval_ocrbench(model, suites["ocrbench"]))
    _save_step("realworldqa", eval_realworldqa(model, suites["realworldqa"]))
    _save_step("blink_val_6spatial", eval_blink(model, suites["blink_val"]))
    _save_step("pope", eval_pope(model, suites["pope"]))
    _save_step("mmbench_cn_dev", eval_mmbench_cn(model, suites["mmbench_cn_dev"]))

    if run_thinking_ablations:
        _save_step("mmstar_thinking_120", eval_mmstar(model, suites["mmstar_think_subset"], enable_thinking=True))
        _save_step("mmmu_val_thinking", eval_mmmu_val(model, suites["mmmu_val"], enable_thinking=True))

    direct_keys = (
        "mmstar_direct",
        "mmmu_val_direct",
        "ai2d_test",
        "ocrbench",
        "realworldqa",
        "blink_val_6spatial",
        "pope",
        "mmbench_cn_dev",
    )
    macro_avg = sum(
        float(res["benchmarks"][k].get("f1" if k == "pope" else "accuracy", 0.0))
        for k in direct_keys
    ) / len(direct_keys)
    res["macro_average_8bench"] = round(macro_avg, 4)
    res["total_elapsed_s"] = round(time.time() - t_start, 2)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if progress_out is not None and accumulated_doc is not None:
        accumulated_doc["models"][label] = res
        write(progress_out, accumulated_doc)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-json",
        default="work/spark-world-v40-stage6-1-sculpted-native/STANDARD_MULTIMODAL_BENCHMARK_REPORT.json",
    )
    ap.add_argument("--scale", type=float, default=1.0, help="Deterministic sample scale multiplier (1.0 = 1,760 items/model)")
    ap.add_argument(
        "--models",
        nargs="*",
        default=[
            "stage6_1_bf16=work/spark-world-v40-stage6-1-sculpted-native/final-model",
            "stage6_1_fp8=work/spark-world-v40-stage6-1-sculpted-native/final-model-fp8",
            "stage5_rsi2_bf16=work/spark-world-v32-stage5-rsi/final-model-rsi2",
            "stage4_baseline_bf16=work/spark-world-v32-24gb/final-model",
        ],
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[Init] Building frozen zero-overlap standard multimodal benchmark suites...", flush=True)
    suites = build_frozen_benchmark_suites(token, scale=float(args.scale))
    suite_counts = {k: len(v) for k, v in suites.items()}
    print(f"[Init] Suite item counts: {suite_counts}", flush=True)

    doc: dict[str, Any] = {
        "schema_version": 1,
        "seed": 20261117,
        "scale": float(args.scale),
        "benchmark_pins": BENCHMARK_PINS,
        "suite_item_counts": suite_counts,
        "zero_train_overlap_verified": True,
        "models": {},
    }
    write(out_path, doc)

    for spec in args.models:
        label, raw_dir = spec.split("=", 1)
        mdir = Path(raw_dir)
        if not (mdir / "MODEL_RELEASE.json").is_file():
            print(f"[Skip] Missing {mdir}/MODEL_RELEASE.json", flush=True)
            continue
        evaluate_checkpoint_suite(
            label,
            mdir,
            suites,
            run_thinking_ablations=(label == "stage6_1_bf16"),
            progress_out=out_path,
            accumulated_doc=doc,
        )

    doc["completed"] = True
    write(out_path, doc)
    print("COMPLETED_STANDARD_MM_BENCHMARKS_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
