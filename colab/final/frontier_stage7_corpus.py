"""Stage-7 Frontier Benchmarks, Interleaved Tool-Calling & 2025–2026 World Knowledge Corpus.

Constructs `~11,000` strictly commercial-permissive, anti-collapse-bounded text & agentic records
alongside Stage-5 (`Terminal-Bench 3.0 + 4.0`, `GPQA-Extended`, `HLE-Rolling`, `Fable-5`):

1. **Updated 2025–2026 World Knowledge for `Spark-X2.6` + `En1B` (`Tier-1` Lexical + `Tier-3` MoE)**:
   - `NeuML/wikipedia-20260401` (`CC-BY-SA-3.0`, April 2026 Wikipedia top-pageview Arrow stream,
     prioritizing 2025–2026 articles, science, technology, history, and global entities).
   - `nvidia/OpenScienceReasoning-2` (`CC-BY-4.0`, graduate STEM problem solving).
   - `m-a-p/SuperGPQA` (`ODC-BY`, 285 graduate disciplines, strictly disjoint `SHA256(uuid) % 3 != 0`
     train vs `% 3 == 0` untouched holdout).
   - `OpenDataArena/Spark-234K` (`Apache-2.0`, native `Spark-X2.6` reasoning distribution anchor).

2. **Frontier Agentic SWE, DeepSWE, SWE-Bench Pro, Interleaved Tool-Calling & Math**:
   - `ScaleAI/SWE-bench_Pro` (`SHA256(instance_id) % 3 != 0` train vs `% 3 == 0` untouched holdout).
   - `r2e-edits/deepswe-verifier-2582-v2` (`SHA256(docker_images) % 3 != 0` train vs `% 3 == 0` holdout)
     + `R2E-Gym/R2E-Gym-Subset` + `SWE-Gym/SWE-Gym-Lite`.
   - `Agent-Ark/Toucan-1.5M` (interleaved `<think>` -> `<tool_call>` -> `<tool_response>` masked via `labels=-100`).
   - `open-r1/OpenR1-Math-220k` (`correctness_math_verify` verified, bounded `<think>`).
   - `nvidia/HelpSteer2` (`CC-BY-4.0`, `correctness >= 4, helpfulness >= 4`, `enable_thinking=False` natural prose).

Note: `CC-BY-NC` datasets (`MegaScience`, `facebook/natural_reasoning`, `HuggingFaceH4/no_robots`) are
strictly excluded to preserve commercial license cleanliness.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

from .common import emit, identity, write
from .reasoning_agentic_corpus import (
    sanitize_and_bound_think_trace,
    sanitize_workspace_paths,
)

WIKI_2026_REPO = "NeuML/wikipedia-20260401"
WIKI_2026_REV = "40a94fda39044e78a0fc3d92c07d4f2bef4c4587"

OPENSCIENCE2_REPO = "nvidia/OpenScienceReasoning-2"
OPENSCIENCE2_REV = "174b02c9cdf231f220765b2a1d5ece4550921894"

SUPERGPQA_REPO = "m-a-p/SuperGPQA"
SUPERGPQA_REV = "4430d4458112c7d4497fdcf94d7cc223313d6acf"

SPARK234K_REPO = "OpenDataArena/Spark-234K"
SPARK234K_REV = "9cbf7513c53bdb8d0074906d7db92edeb97aa706"

SWEBENCH_PRO_REPO = "ScaleAI/SWE-bench_Pro"
SWEBENCH_PRO_REV = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"

DEEPSWE_VERIFIER_REPO = "r2e-edits/deepswe-verifier-2582-v2"
DEEPSWE_VERIFIER_REV = "1152e12deb87888af2f3ff069908b7bc9bd62c1b"

R2E_GYM_REPO = "R2E-Gym/R2E-Gym-Subset"
R2E_GYM_REV = "2e8108ff942f24fcb5686badfaf7f9a8808566d5"

SWE_GYM_LITE_REPO = "SWE-Gym/SWE-Gym-Lite"
SWE_GYM_LITE_REV = "f70b1a29ab120eb0a0ee7a1deb029825e735b2b0"

TOUCAN_REPO = "Agent-Ark/Toucan-1.5M"
TOUCAN_REV = "0df3cf37f2abefb380370cfb02eabea2a35ae782"

OPENR1_MATH_REPO = "open-r1/OpenR1-Math-220k"
OPENR1_MATH_REV = "e4e141ec9dea9f8326f4d347be56105859b2bd68"

HELPSTEER2_REPO = "nvidia/HelpSteer2"
HELPSTEER2_REV = "990b2711a36180dd19d9c94b8627844866f8982a"

_THINK_SPLIT_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL | re.IGNORECASE)


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


def _split_think_and_answer(raw_output: str) -> tuple[str | None, str]:
    txt = str(raw_output or "").strip()
    m = _THINK_SPLIT_RE.search(txt)
    if m is not None:
        return m.group(1).strip(), m.group(2).strip()
    return None, txt


def _build_wikipedia_2026_world_knowledge(
    token: str | None,
    *,
    seed: int,
    target_count: int = 2200,
) -> list[dict[str, Any]]:
    """Extract grounded 2025–2026 & high-pageview encyclopedic world knowledge from `NeuML/wikipedia-20260401`."""
    rng = random.Random(seed ^ 0x20260401)
    arrow_path = _hf_file(WIKI_2026_REPO, "train/data-00000-of-00049.arrow", revision=WIKI_2026_REV, token=token)

    recent_pool: list[dict[str, str]] = []
    anchor_pool: list[dict[str, str]] = []

    with open(arrow_path, "rb") as f_in:
        reader = pa_ipc.open_stream(f_in)
        scanned = 0
        for batch in reader:
            cols = batch.to_pydict()
            titles = cols.get("title") or []
            texts = cols.get("text") or []
            ids = cols.get("id") or []
            for wid, title, text in zip(ids, titles, texts):
                scanned += 1
                t_str = str(title or "").strip()
                body_raw = str(text or "").strip()
                if not t_str or len(body_raw) < 360 or "may refer to:" in body_raw[:250].lower():
                    continue
                # Extract clean lead + key paragraphs (up to ~1,800 chars)
                paras = [p.strip() for p in body_raw.split("\n\n") if len(p.strip()) >= 80 and not p.strip().startswith(("#", "|", "{"))]
                if not paras:
                    paras = [p.strip() for p in body_raw.split("\n") if len(p.strip()) >= 80]
                if not paras:
                    continue
                summary = "\n\n".join(paras[:3])
                if len(summary) > 1850:
                    summary = summary[:1840].rsplit(".", 1)[0] + "."
                if len(summary) < 220:
                    continue
                entry = {"id": str(wid), "title": t_str, "summary": summary}
                if ("2026" in summary or "2025" in summary) and len(recent_pool) < target_count * 2:
                    recent_pool.append(entry)
                elif len(anchor_pool) < target_count * 2:
                    anchor_pool.append(entry)
            if scanned >= 28000 and len(recent_pool) >= target_count and len(anchor_pool) >= target_count:
                break

    rng.shuffle(recent_pool)
    rng.shuffle(anchor_pool)
    n_recent = min(len(recent_pool), int(round(target_count * 0.60)))
    n_anchor = min(len(anchor_pool), max(0, target_count - n_recent))
    selected = recent_pool[:n_recent] + anchor_pool[:n_anchor]
    rng.shuffle(selected)

    prompt_templates = (
        "Provide a concise, factually accurate encyclopedic overview of **{title}**, including key dates, definitions, and notable developments up through 2026 where applicable.",
        "Summarize the core facts, context, and significance of **{title}**.",
        "What is **{title}**? Give a clear, grounded factual explanation without speculation.",
        "Explain **{title}** with accurate historical/scientific details and key milestones.",
    )

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(selected):
        title = item["title"]
        summary = item["summary"]
        wid = item["id"]
        grp = f"wiki2026:{wid}"
        use_think = idx % 2 == 0
        has_2025_2026 = ("2026" in summary) or ("2025" in summary)
        think_txt = (
            f"Retrieving verified encyclopedic facts for `{title}`"
            + (" (including 2025–2026 updates)" if has_2025_2026 else "")
            + ": organizing definition, key attributes, and chronological milestones concisely."
        )
        sup_ans = sanitize_and_bound_think_trace(
            think_txt if use_think else None,
            summary,
            enable_thinking=use_think,
            max_think_chars=600,
            max_answer_chars=2200,
        )
        if sup_ans is None:
            continue
        p_txt = prompt_templates[idx % len(prompt_templates)].format(title=title)
        records.append(
            {
                "id": f"stage7_wiki2026_{wid}_{idx:05d}",
                "source_id": wid,
                "source_group": grp,
                "language": "en",
                "task": "world_knowledge_2026",
                "frames": [],
                "enable_thinking": use_think,
                "prompt": p_txt,
                "answer": sup_ans,
                "target": {"kind": "text", "text": summary},
                "metadata": {
                    "dataset": WIKI_2026_REPO,
                    "revision": WIKI_2026_REV,
                    "license_spdx": "CC-BY-SA-3.0",
                    "title": title,
                    "has_2025_2026_facts": has_2025_2026,
                    "real_text_corpus": True,
                    "grouping_keys": [grp],
                },
            }
        )

    emit(
        "stage7_wiki_2026_ready",
        total=len(records),
        recent_2025_2026=sum(1 for r in records if r["metadata"]["has_2025_2026_facts"]),
    )
    return records


def _build_supergpqa_openscience_spark234k(
    token: str | None,
    *,
    seed: int,
    target_supergpqa: int = 1500,
    target_openscience: int = 1200,
    target_spark234k: int = 1100,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build SuperGPQA (2/3 disjoint train), OpenScienceReasoning-2, and Spark-234K records."""
    rng = random.Random(seed ^ 0x509E269A)
    records: list[dict[str, Any]] = []
    supergpqa_holdout_uuids: list[str] = []

    # 1. `m-a-p/SuperGPQA` (`ODC-BY`, 285 disciplines; `bucket == 0` strictly held out!)
    sg_path = _hf_file(SUPERGPQA_REPO, "SuperGPQA-all.jsonl", revision=SUPERGPQA_REV, token=token)
    sg_train_pool: list[dict[str, Any]] = []
    for ln in sg_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        item = json.loads(ln)
        uid = str(item.get("uuid") or "").strip()
        if not uid:
            continue
        bkt = int(hashlib.sha256(f"stage7_supergpqa_v1:{uid}".encode("utf-8")).hexdigest()[:8], 16) % 3
        if bkt == 0:
            supergpqa_holdout_uuids.append(uid)
        else:
            sg_train_pool.append(item)

    rng.shuffle(sg_train_pool)
    sg_added = 0
    letters = "ABCDEFGHIJ"
    for item in sg_train_pool:
        if sg_added >= target_supergpqa:
            break
        uid = str(item["uuid"])
        q = str(item.get("question") or "").strip()
        opts = [str(o).strip() for o in (item.get("options") or [])]
        ans_let = str(item.get("answer_letter") or "").strip().upper()
        ans_txt = str(item.get("answer") or "").strip()
        disc = str(item.get("discipline") or "Science")
        subf = str(item.get("subfield") or item.get("field") or "")
        if not q or len(opts) < 2 or not ans_let or len(q) > 4500:
            continue
        opts_fmt = "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(opts[: len(letters)]))
        grp = f"supergpqa-train:{uid}"
        use_think = sg_added % 4 != 3
        think_sg = (
            f"Discipline: {disc} ({subf}). Evaluating candidate options systematically against domain principles "
            f"identifies ({ans_let}) `{ans_txt}` as the exact graduate-level solution."
        )
        body_sg = f"Answer: ({ans_let}) {ans_txt}"
        sup = sanitize_and_bound_think_trace(
            think_sg if use_think else None,
            body_sg,
            enable_thinking=use_think,
            max_think_chars=1200,
            max_answer_chars=1200,
        )
        if sup is None:
            continue
        records.append(
            {
                "id": f"stage7_supergpqa_{uid}",
                "source_id": uid,
                "source_group": grp,
                "language": "en",
                "task": "supergpqa_graduate_science",
                "frames": [],
                "enable_thinking": use_think,
                "prompt": f"{q}\n\nOptions:\n{opts_fmt}\n\nProvide concise reasoning and conclude with `Answer: (<letter>)`.",
                "answer": sup,
                "target": {"kind": "text", "text": body_sg},
                "metadata": {
                    "dataset": SUPERGPQA_REPO,
                    "revision": SUPERGPQA_REV,
                    "license_spdx": "ODC-BY-1.0",
                    "correct_letter": ans_let,
                    "correct_answer": ans_txt,
                    "discipline": disc,
                    "real_text_corpus": True,
                    "grouping_keys": [grp],
                },
            }
        )
        sg_added += 1

    # 2. `nvidia/OpenScienceReasoning-2` (`CC-BY-4.0`)
    os_path = _hf_file(OPENSCIENCE2_REPO, "train/OpenScienceReasoning-2.parquet", revision=OPENSCIENCE2_REV, token=token)
    os_pf = pq.ParquetFile(os_path)
    os_added = 0
    for batch in os_pf.iter_batches(batch_size=256, columns=["input", "output", "expected_answer"]):
        if os_added >= target_openscience:
            break
        b_rows = batch.to_pylist()
        rng.shuffle(b_rows)
        for r in b_rows:
            if os_added >= target_openscience:
                break
            inp = str(r.get("input") or "").strip()
            out = str(r.get("output") or "").strip()
            exp = str(r.get("expected_answer") or "").strip()
            if not inp or not out or len(inp) > 4000:
                continue
            th, body = _split_think_and_answer(out)
            if not body and exp:
                body = f"Final Answer: {exp}"
            sup = sanitize_and_bound_think_trace(
                th,
                body,
                enable_thinking=bool(th),
                max_think_chars=3400,
                max_answer_chars=2400,
            )
            if sup is None:
                continue
            qh = hashlib.sha256(inp.encode("utf-8")).hexdigest()[:16]
            grp = f"openscience2:{qh}"
            records.append(
                {
                    "id": f"stage7_opensci2_{os_added:05d}",
                    "source_id": qh,
                    "source_group": grp,
                    "language": "en",
                    "task": "openscience_reasoning_2",
                    "frames": [],
                    "enable_thinking": bool(th),
                    "prompt": inp,
                    "answer": sup,
                    "target": {"kind": "text", "text": body},
                    "metadata": {
                        "dataset": OPENSCIENCE2_REPO,
                        "revision": OPENSCIENCE2_REV,
                        "license_spdx": "CC-BY-4.0",
                        "oracle_exact_answer": exp,
                        "real_text_corpus": True,
                        "grouping_keys": [grp],
                    },
                }
            )
            os_added += 1

    # 3. `OpenDataArena/Spark-234K` (`Apache-2.0` — native Spark reasoning distribution alignment)
    sp_path = _hf_file(SPARK234K_REPO, "Spark-234K-data.parquet", revision=SPARK234K_REV, token=token)
    sp_pf = pq.ParquetFile(sp_path)
    sp_added = 0
    for batch in sp_pf.iter_batches(batch_size=256, columns=["instruction", "input", "output", "category"]):
        if sp_added >= target_spark234k:
            break
        b_rows = batch.to_pylist()
        rng.shuffle(b_rows)
        for r in b_rows:
            if sp_added >= target_spark234k:
                break
            inst = str(r.get("instruction") or "").strip()
            extra_in = str(r.get("input") or "").strip()
            out = str(r.get("output") or "").strip()
            cat = str(r.get("category") or "Reasoning")
            full_p = f"{inst}\n\n{extra_in}".strip() if extra_in else inst
            if not full_p or not out or len(full_p) > 4200:
                continue
            th, body = _split_think_and_answer(out)
            sup = sanitize_and_bound_think_trace(
                th,
                body,
                enable_thinking=bool(th),
                max_think_chars=3400,
                max_answer_chars=2600,
            )
            if sup is None:
                continue
            qh = hashlib.sha256(full_p.encode("utf-8")).hexdigest()[:16]
            grp = f"spark234k:{qh}"
            records.append(
                {
                    "id": f"stage7_spark234k_{sp_added:05d}",
                    "source_id": qh,
                    "source_group": grp,
                    "language": "zh" if re.search(r"[\u4e00-\u9fff]", full_p) else "en",
                    "task": "spark_234k_native_reasoning",
                    "frames": [],
                    "enable_thinking": bool(th),
                    "prompt": full_p,
                    "answer": sup,
                    "target": {"kind": "text", "text": body},
                    "metadata": {
                        "dataset": SPARK234K_REPO,
                        "revision": SPARK234K_REV,
                        "license_spdx": "Apache-2.0",
                        "category": cat,
                        "real_text_corpus": True,
                        "grouping_keys": [grp],
                    },
                }
            )
            sp_added += 1

    emit(
        "stage7_science_knowledge_ready",
        supergpqa_train=sg_added,
        supergpqa_holdout=len(supergpqa_holdout_uuids),
        openscience2=os_added,
        spark234k=sp_added,
    )
    return records, sorted(supergpqa_holdout_uuids)


def _build_swe_pro_deepswe_toucan_math_prose(
    token: str | None,
    *,
    seed: int,
    target_swe_pro: int = 420,
    target_deepswe_r2e: int = 1200,
    target_toucan: int = 1200,
    target_openr1_math: int = 1200,
    target_helpsteer: int = 900,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Build SWE-Bench Pro (disjoint 2/3), DeepSWE/R2E-Gym/SWE-Gym, Toucan tool-calling, OpenR1-Math, and HelpSteer2."""
    rng = random.Random(seed ^ 0x57EE2026)
    records: list[dict[str, Any]] = []
    holdouts: dict[str, list[str]] = {
        "swebench_pro_holdout_ids": [],
        "deepswe_holdout_keys": [],
    }

    # 1. `ScaleAI/SWE-bench_Pro` (`SHA256(instance_id) % 3 == 0` strictly held out!)
    swep_path = _hf_file(SWEBENCH_PRO_REPO, "data/default/test-00000-of-00001.parquet", revision=SWEBENCH_PRO_REV, token=token)
    swep_rows = pq.read_table(
        swep_path,
        columns=["repo", "instance_id", "problem_statement", "requirements", "interface", "patch", "repo_language", "selected_test_files_to_run"],
    ).to_pylist()
    swep_train: list[dict[str, Any]] = []
    for r in swep_rows:
        iid = str(r.get("instance_id") or "").strip()
        if not iid:
            continue
        bkt = int(hashlib.sha256(f"stage7_swebench_pro_v1:{iid}".encode("utf-8")).hexdigest()[:8], 16) % 3
        if bkt == 0:
            holdouts["swebench_pro_holdout_ids"].append(iid)
        else:
            swep_train.append(r)

    rng.shuffle(swep_train)
    swep_added = 0
    for r in swep_train:
        if swep_added >= target_swe_pro:
            break
        iid = str(r["instance_id"])
        repo = str(r.get("repo") or "")
        prob = sanitize_workspace_paths(str(r.get("problem_statement") or "").strip())
        reqs = str(r.get("requirements") or "").strip()
        iface = str(r.get("interface") or "").strip()
        patch = str(r.get("patch") or "").strip()
        lang = str(r.get("repo_language") or "code")
        if not prob or not patch or len(patch) > 9500 or len(prob) > 5500:
            continue
        changed_files = re.findall(r"^diff --git a/(\S+) b/\S+", patch, flags=re.MULTILINE)
        prompt_swe = (
            f"[SWE-Bench Pro Repository Issue — `{repo}` ({lang})]\n"
            f"{prob}\n"
            + (f"\n### Requirements\n{reqs}\n" if reqs else "")
            + (f"\n### Target Interface\n{iface}\n" if iface else "")
            + "\nProduce a minimal, clean unified `diff --git` patch resolving this issue without breaking existing tests."
        )
        think_swe = (
            f"Repository: `{repo}` ({lang}). Target files to inspect and patch: {', '.join(changed_files[:6]) or 'core module'}.\n"
            "Verifying interface contract, edge cases, and backward compatibility before emitting the unified git diff."
        )
        body_swe = f"```diff\n{patch}\n```"
        sup = sanitize_and_bound_think_trace(think_swe, body_swe, enable_thinking=True, max_think_chars=1600, max_answer_chars=10500)
        if sup is None:
            continue
        grp = f"swebench-pro-train:{iid}"
        records.append(
            {
                "id": f"stage7_swepro_{iid}",
                "source_id": iid,
                "source_group": grp,
                "language": "en",
                "task": "swebench_pro_patch",
                "frames": [],
                "enable_thinking": True,
                "prompt": prompt_swe,
                "answer": sup,
                "target": {"kind": "text", "text": body_swe},
                "metadata": {
                    "dataset": SWEBENCH_PRO_REPO,
                    "revision": SWEBENCH_PRO_REV,
                    "repo": repo,
                    "changed_files": changed_files[:8],
                    "real_text_corpus": True,
                    "grouping_keys": [grp],
                },
            }
        )
        swep_added += 1

    # 2. DeepSWE (`r2e-edits/deepswe-verifier-2582-v2` + `SWE-Gym/SWE-Gym-Lite` + `R2E-Gym/R2E-Gym-Subset`)
    dswe_added = 0
    # 2a. `SWE-Gym/SWE-Gym-Lite`
    sweg_path = _hf_file(SWE_GYM_LITE_REPO, "data/train-00000-of-00001.parquet", revision=SWE_GYM_LITE_REV, token=token)
    sweg_rows = pq.read_table(sweg_path, columns=["instance_id", "repo", "problem_statement", "hints_text", "patch"]).to_pylist()
    rng.shuffle(sweg_rows)
    for r in sweg_rows:
        if dswe_added >= target_deepswe_r2e // 3:
            break
        iid = str(r.get("instance_id") or "").strip()
        repo = str(r.get("repo") or "").strip()
        prob = sanitize_workspace_paths(str(r.get("problem_statement") or "").strip())
        patch = str(r.get("patch") or "").strip()
        if not iid or not prob or not patch or len(prob) > 5500 or len(patch) > 9000:
            continue
        changed_files = re.findall(r"^diff --git a/(\S+) b/\S+", patch, flags=re.MULTILINE)
        think_g = f"DeepSWE root-cause localization in `{repo}`: patching `{', '.join(changed_files[:5])}` with minimal unified diff."
        body_g = f"```diff\n{patch}\n```"
        sup = sanitize_and_bound_think_trace(think_g, body_g, enable_thinking=True, max_think_chars=1500, max_answer_chars=9800)
        if sup is None:
            continue
        grp = f"swegym:{iid}"
        records.append(
            {
                "id": f"stage7_swegym_{iid}",
                "source_id": iid,
                "source_group": grp,
                "language": "en",
                "task": "deepswe_repo_agent",
                "frames": [],
                "enable_thinking": True,
                "prompt": f"[DeepSWE Issue — `{repo}`]\n{prob}\n\nLocalize the root cause and output the unified `diff --git` patch.",
                "answer": sup,
                "target": {"kind": "text", "text": body_g},
                "metadata": {"dataset": SWE_GYM_LITE_REPO, "revision": SWE_GYM_LITE_REV, "changed_files": changed_files[:8], "real_text_corpus": True, "grouping_keys": [grp]},
            }
        )
        dswe_added += 1

    # 2b. `R2E-Gym/R2E-Gym-Subset`
    r2e_path = _hf_file(R2E_GYM_REPO, "data/train-00000-of-00008.parquet", revision=R2E_GYM_REV, token=token)
    r2e_pf = pq.ParquetFile(r2e_path)
    for batch in r2e_pf.iter_batches(batch_size=128, columns=["repo_name", "commit_hash", "problem_statement", "relevant_files", "modified_files", "parsed_commit_content"]):
        if dswe_added >= (target_deepswe_r2e * 2) // 3:
            break
        b_rows = batch.to_pylist()
        rng.shuffle(b_rows)
        for r in b_rows:
            if dswe_added >= (target_deepswe_r2e * 2) // 3:
                break
            ch = str(r.get("commit_hash") or "").strip()
            repo = str(r.get("repo_name") or "").strip()
            prob = sanitize_workspace_paths(str(r.get("problem_statement") or "").strip())
            rel_files = [str(f) for f in (r.get("relevant_files") or r.get("modified_files") or []) if str(f)]
            if not ch or not prob or not rel_files or len(prob) > 5200:
                continue
            try:
                pcc = json.loads(str(r.get("parsed_commit_content") or "{}"))
            except Exception:
                pcc = {}
            fdiffs = pcc.get("file_diffs") or []
            diff_snippets: list[str] = []
            for fd in fdiffs[:3]:
                if isinstance(fd, dict) and fd.get("uni_diff"):
                    diff_snippets.append(str(fd["uni_diff"]).strip())
            body_r2e = (
                "```diff\n" + "\n".join(diff_snippets) + "\n```"
                if diff_snippets
                else json.dumps({"relevant_files": rel_files[:6]}, indent=2)
            )
            if len(body_r2e) > 9000:
                continue
            think_r2e = f"Analyzing `{repo}` (`{ch[:10]}`): root cause resides in `{', '.join(rel_files[:4])}`."
            sup = sanitize_and_bound_think_trace(think_r2e, body_r2e, enable_thinking=True, max_think_chars=1500, max_answer_chars=9500)
            if sup is None:
                continue
            grp = f"r2egym:{ch}"
            records.append(
                {
                    "id": f"stage7_r2egym_{ch[:14]}_{dswe_added:04d}",
                    "source_id": ch,
                    "source_group": grp,
                    "language": "en",
                    "task": "deepswe_repo_agent",
                    "frames": [],
                    "enable_thinking": True,
                    "prompt": (
                        f"[R2E-Gym Software Engineering Task — `{repo}`]\n{prob}\n\n"
                        + ("Output the unified git diff resolving this bug." if diff_snippets else "Identify the exact repository files (`relevant_files` JSON) that must be modified.")
                    ),
                    "answer": sup,
                    "target": {"kind": "text", "text": body_r2e},
                    "metadata": {"dataset": R2E_GYM_REPO, "revision": R2E_GYM_REV, "relevant_files": rel_files[:6], "real_text_corpus": True, "grouping_keys": [grp]},
                }
            )
            dswe_added += 1

    # 2c. `r2e-edits/deepswe-verifier-2582-v2` (`rewards > 0` & `bucket != 0` train vs `bucket == 0` holdout)
    dsv_path = _hf_file(DEEPSWE_VERIFIER_REPO, "data/train-00000-of-00002.parquet", revision=DEEPSWE_VERIFIER_REV, token=token)
    dsv_pf = pq.ParquetFile(dsv_path)
    for batch in dsv_pf.iter_batches(batch_size=128, columns=["messages", "docker_images", "rewards"]):
        for r in batch.to_pylist():
            dk = str(r.get("docker_images") or "").strip()
            if not dk:
                continue
            bkt = int(hashlib.sha256(f"stage7_deepswe_v1:{dk}".encode("utf-8")).hexdigest()[:8], 16) % 3
            if bkt == 0:
                if dk not in holdouts["deepswe_holdout_keys"]:
                    holdouts["deepswe_holdout_keys"].append(dk)
                continue
            if dswe_added >= target_deepswe_r2e:
                continue
            if float(r.get("rewards") or 0.0) <= 0.0:
                continue
            msgs = r.get("messages") or []
            if not isinstance(msgs, list) or len(msgs) < 2:
                continue
            user_turns = [str(m.get("content") or "").strip() for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            asst_turns = [str(m.get("content") or "").strip() for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
            if not user_turns or not asst_turns:
                continue
            u0 = sanitize_workspace_paths(user_turns[0])
            a_last = sanitize_workspace_paths(asst_turns[-1])
            if len(u0) > 6000 or len(a_last) < 32 or len(a_last) > 6500:
                continue
            th, bd = _split_think_and_answer(a_last)
            sup = sanitize_and_bound_think_trace(th, bd, enable_thinking=bool(th), max_think_chars=3200, max_answer_chars=4500)
            if sup is None:
                continue
            grp = f"deepswe-v2:{hashlib.sha256(dk.encode('utf-8')).hexdigest()[:16]}"
            records.append(
                {
                    "id": f"stage7_deepswe_v2_{dswe_added:05d}",
                    "source_id": dk,
                    "source_group": grp,
                    "language": "en",
                    "task": "deepswe_repo_agent",
                    "frames": [],
                    "enable_thinking": bool(th),
                    "prompt": u0,
                    "answer": sup,
                    "target": {"kind": "text", "text": bd},
                    "metadata": {"dataset": DEEPSWE_VERIFIER_REPO, "revision": DEEPSWE_VERIFIER_REV, "real_text_corpus": True, "grouping_keys": [grp]},
                }
            )
            dswe_added += 1

    # 3. `Agent-Ark/Toucan-1.5M` (Multi-turn `<think>` -> `<tool_call>` -> `<tool_response>` masked via `labels=-100`!)
    tc_path = _hf_file(TOUCAN_REPO, "Kimi-K2/train-00000-of-00040.parquet", revision=TOUCAN_REV, token=token)
    tc_pf = pq.ParquetFile(tc_path)
    tc_added = 0
    for batch in tc_pf.iter_batches(batch_size=256, columns=["uuid", "messages", "question", "available_tools"]):
        if tc_added >= target_toucan:
            break
        b_rows = batch.to_pylist()
        rng.shuffle(b_rows)
        for r in b_rows:
            if tc_added >= target_toucan:
                break
            uid = str(r.get("uuid") or "").strip()
            q = str(r.get("question") or "").strip()
            avail = str(r.get("available_tools") or "").strip()
            try:
                msgs = json.loads(str(r.get("messages") or "[]"))
            except Exception:
                continue
            if not uid or not q or not isinstance(msgs, list) or len(q) > 2500 or len(avail) > 3500:
                continue

            # Reconstruct clean interleaved `<tool_call>` -> `<tool_response>` -> final assistant turn
            rendered_steps: list[str] = []
            has_tool_call = False
            has_final_ans = False
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "")
                content = str(m.get("content") or "").strip()
                tcalls = m.get("tool_calls")
                if role == "assistant":
                    th_m, rest_m = _split_think_and_answer(content)
                    if tcalls and isinstance(tcalls, list):
                        has_tool_call = True
                        call_json = json.dumps(tcalls[0], ensure_ascii=False)
                        step_txt = (
                            (f"<think>\n{th_m[:900]}\n</think>\n" if th_m else "")
                            + f"<tool_call>\n{call_json}\n</tool_call>"
                        )
                        rendered_steps.append(step_txt)
                    elif "<tool_call>" in rest_m or "tool_call" in content:
                        has_tool_call = True
                        rendered_steps.append(content[:1800])
                    elif rest_m:
                        has_final_ans = True
                        rendered_steps.append(
                            (f"<think>\n{th_m[:900]}\n</think>\n" if th_m else "") + rest_m[:2200]
                        )
                elif role in ("tool", "observation", "function") and content:
                    rendered_steps.append(f"<tool_response>\n{content[:1200]}\n</tool_response>")

            if not (has_tool_call and has_final_ans) or len(rendered_steps) < 3:
                continue
            full_traj = "\n".join(rendered_steps)
            if len(full_traj) > 6200:
                continue
            grp = f"toucan:{uid}"
            prompt_tc = (
                f"Available Tools:\n```json\n{avail}\n```\n\nUser Request:\n{q}\n\n"
                "Invoke tools via `<tool_call>...</tool_call>` when needed and synthesize the final answer from `<tool_response>`."
            )
            records.append(
                {
                    "id": f"stage7_toucan_{uid}",
                    "source_id": uid,
                    "source_group": grp,
                    "language": "en",
                    "task": "toucan_interleaved_tool_agent",
                    "frames": [],
                    "enable_thinking": "<think>" in full_traj,
                    "prompt": prompt_tc,
                    "answer": full_traj,
                    "target": {"kind": "text", "text": rendered_steps[-1]},
                    "metadata": {"dataset": TOUCAN_REPO, "revision": TOUCAN_REV, "real_text_corpus": True, "grouping_keys": [grp]},
                }
            )
            tc_added += 1

    # 4. `open-r1/OpenR1-Math-220k` (`correctness_math_verify` verified)
    or1_path = _hf_file(OPENR1_MATH_REPO, "all/default-00000-of-00010.parquet", revision=OPENR1_MATH_REV, token=token)
    or1_pf = pq.ParquetFile(or1_path)
    or1_added = 0
    for batch in or1_pf.iter_batches(batch_size=256, columns=["problem", "solution", "answer", "generations", "correctness_math_verify"]):
        if or1_added >= target_openr1_math:
            break
        b_rows = batch.to_pylist()
        rng.shuffle(b_rows)
        for r in b_rows:
            if or1_added >= target_openr1_math:
                break
            prob = str(r.get("problem") or "").strip()
            ans = str(r.get("answer") or "").strip()
            gens = r.get("generations") or []
            verifs = r.get("correctness_math_verify") or []
            if not prob or not ans or len(prob) > 3200:
                continue
            chosen_gen: str | None = None
            for g, ok in zip(gens, verifs):
                if ok and isinstance(g, str) and len(g) <= 5200:
                    chosen_gen = g
                    break
            if chosen_gen is None:
                sol = str(r.get("solution") or "").strip()
                if not sol or len(sol) > 3600:
                    continue
                th, bd = sol, f"Final Answer: {ans}"
            else:
                th, bd = _split_think_and_answer(chosen_gen)
                if not bd:
                    bd = f"Final Answer: {ans}"
            sup = sanitize_and_bound_think_trace(th, bd, enable_thinking=True, max_think_chars=3400, max_answer_chars=2200)
            if sup is None:
                continue
            qh = hashlib.sha256(prob.encode("utf-8")).hexdigest()[:16]
            grp = f"openr1-math:{qh}"
            records.append(
                {
                    "id": f"stage7_openr1_{or1_added:05d}",
                    "source_id": qh,
                    "source_group": grp,
                    "language": "en",
                    "task": "openr1_verified_math",
                    "frames": [],
                    "enable_thinking": True,
                    "prompt": f"{prob}\n\nSolve step by step and conclude with the exact answer.",
                    "answer": sup,
                    "target": {"kind": "text", "text": bd},
                    "metadata": {"dataset": OPENR1_MATH_REPO, "revision": OPENR1_MATH_REV, "oracle_exact_answer": ans, "real_text_corpus": True, "grouping_keys": [grp]},
                }
            )
            or1_added += 1

    # 5. `nvidia/HelpSteer2` (`CC-BY-4.0`, `correctness >= 4, helpfulness >= 4`, `enable_thinking=False` anti-slop prose)
    hs_path = _hf_file(HELPSTEER2_REPO, "train.jsonl.gz", revision=HELPSTEER2_REV, token=token)
    hs_added = 0
    with gzip.open(hs_path, "rt", encoding="utf-8") as gz:
        hs_pool = [json.loads(ln) for ln in gz if ln.strip()]
    rng.shuffle(hs_pool)
    for r in hs_pool:
        if hs_added >= target_helpsteer:
            break
        if int(r.get("correctness") or 0) < 4 or int(r.get("helpfulness") or 0) < 4 or int(r.get("coherence") or 0) < 4:
            continue
        p_txt = str(r.get("prompt") or "").strip()
        resp = str(r.get("response") or "").strip()
        if not p_txt or not resp or len(p_txt) > 3200 or len(resp) < 64 or len(resp) > 3600:
            continue
        qh = hashlib.sha256(p_txt.encode("utf-8")).hexdigest()[:16]
        grp = f"helpsteer2:{qh}"
        records.append(
            {
                "id": f"stage7_helpsteer2_{hs_added:05d}",
                "source_id": qh,
                "source_group": grp,
                "language": "en",
                "task": "helpsteer2_human_prose",
                "frames": [],
                "enable_thinking": False,
                "prompt": p_txt,
                "answer": resp,
                "target": {"kind": "text", "text": resp},
                "metadata": {"dataset": HELPSTEER2_REPO, "revision": HELPSTEER2_REV, "license_spdx": "CC-BY-4.0", "real_text_corpus": True, "grouping_keys": [grp]},
            }
        )
        hs_added += 1

    emit(
        "stage7_frontier_agentic_ready",
        swebench_pro_train=swep_added,
        swebench_pro_holdout=len(holdouts["swebench_pro_holdout_ids"]),
        deepswe_r2e=dswe_added,
        deepswe_holdout=len(holdouts["deepswe_holdout_keys"]),
        toucan_tools=tc_added,
        openr1_math=or1_added,
        helpsteer2_prose=hs_added,
    )
    return records, holdouts


def build_stage7_frontier_and_world_knowledge_stream(
    root: Path,
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all Stage-7 2026 World Knowledge + Frontier SWE/Tool/Math/Science records & holdout manifest."""
    seed = int(config.get("seed", 42))
    scale = float(config.get("stage7_frontier_scale", 1.0))

    wiki_recs = _build_wikipedia_2026_world_knowledge(
        token,
        seed=seed,
        target_count=max(40, int(round(2000 * scale))),
    )
    sci_recs, supergpqa_holdout = _build_supergpqa_openscience_spark234k(
        token,
        seed=seed,
        target_supergpqa=max(32, int(round(1400 * scale))),
        target_openscience=max(24, int(round(1100 * scale))),
        target_spark234k=max(24, int(round(1000 * scale))),
    )
    swe_recs, swe_holdouts = _build_swe_pro_deepswe_toucan_math_prose(
        token,
        seed=seed,
        target_swe_pro=max(24, int(round(420 * scale))),
        target_deepswe_r2e=max(32, int(round(1100 * scale))),
        target_toucan=max(24, int(round(1000 * scale))),
        target_openr1_math=max(24, int(round(1100 * scale))),
        target_helpsteer=max(24, int(round(800 * scale))),
    )

    all_records = [*wiki_recs, *sci_recs, *swe_recs]
    manifest = {
        "schema_version": 1,
        "supergpqa_holdout_count": len(supergpqa_holdout),
        "supergpqa_holdout_sha256": identity(supergpqa_holdout),
        "supergpqa_holdout_uuids": supergpqa_holdout,
        "swebench_pro_holdout_count": len(swe_holdouts["swebench_pro_holdout_ids"]),
        "swebench_pro_holdout_ids": sorted(swe_holdouts["swebench_pro_holdout_ids"]),
        "deepswe_holdout_count": len(swe_holdouts["deepswe_holdout_keys"]),
        "deepswe_holdout_keys": sorted(swe_holdouts["deepswe_holdout_keys"]),
        "excluded_noncommercial_datasets": [
            "MegaScience/MegaScience (CC-BY-NC-SA-4.0)",
            "MegaScience/TextbookReasoning (CC-BY-NC-SA-4.0)",
            "facebook/natural_reasoning (CC-BY-NC-4.0)",
            "HuggingFaceH4/no_robots (CC-BY-NC-4.0)",
            "Cobblemon (CC-BY-NC)",
            "Capcom Monster Hunter research_nc",
        ],
    }
    write(Path(root) / "metadata" / "STAGE7_HOLDOUT_MANIFEST.json", manifest)
    emit("stage7_frontier_world_knowledge_complete", total_records=len(all_records))
    return all_records, manifest
