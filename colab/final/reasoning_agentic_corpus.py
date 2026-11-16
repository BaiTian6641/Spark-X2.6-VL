"""Stage-5 Advanced Reasoning, Agentic SWE, Terminal-Bench 3.0, GPQA & HLE Corpus Builder.

Integrates frontier `<think>` chain-of-thought, autonomous coding/terminal tool-use,
and graduate/frontier multimodal + scientific reasoning while enforcing:

1. **Strict License Hygiene**:
   - `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` (`Apache-2.0`, unperturbed `idx < 1000` ONLY)
   - `TheFusionCube/Fable-5-CoT-Traces` (`MIT`, non-decoy untruncated ONLY)
   - `armand0e/claude-fable-5-claude-code` (`MIT`, sanitized real-repo coding turns)
   - `harborframework/terminal-bench-3.0` (`Apache-2.0`, disjoint 2/3 train vs 1/3 held-out tasks)
   - `open-thoughts/OpenThoughts-114k` (`Apache-2.0`, concise non-looping code/math/science CoT)
   - `Idavidrein/gpqa` (`CC-BY-4.0`, strictly `gpqa_extended` MINUS `gpqa_diamond` — 0% Diamond overlap!)
   - `cais/hle` (`MIT`, strictly disjoint 80% train vs 20% stratified held-out text + multimodal vision!)
   - Copyleft (`AGPL-3.0`) and NC (`CC-BY-NC-*`) sources are strictly excluded.

2. **Five Hard Anti-Collapse Controls for 4B VLMs**:
   - Bounded `<think>` trace length (`<= 3,800` chars / `~1,050` tokens) — never raw 12k-token circular monologues.
   - 4-gram repetition filter (`rep_4gram <= 0.10`) and self-doubt backtrack cap (`<= 4`).
   - Guaranteed closed `<think>\\n...\\n</think>\\n` followed by a substantive non-empty final answer.
   - Explicit dual-mode supervision (`~70% enable_thinking=True`, `~30% enable_thinking=False`) so non-thinking
     generation stays fast, direct, and collapse-free.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import re
from pathlib import Path
import random
from typing import Any

from .common import emit, identity, write

FABLE51_REPO_ID = "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x"
FABLE51_REVISION = "4904ca0c09044f447aff16e4f66b6f2f53c9f702"

FABLE5_COT_REPO_ID = "TheFusionCube/Fable-5-CoT-Traces"
FABLE5_COT_REVISION = "3624e3a8aab36a89dc3b89aebef0e034a64ea04e"

CLAUDE_FABLE5_CODE_REPO_ID = "armand0e/claude-fable-5-claude-code"
CLAUDE_FABLE5_CODE_REVISION = "7388ef96f3179f23b6b3618b96572f1aa9b84510"

TERMINAL_BENCH3_REPO_ID = "harborframework/terminal-bench-3.0"
TERMINAL_BENCH3_REVISION = "2ee5a897d6be85513b41f44b945c4ff4d3e04a1d"

OPENTHOUGHTS_REPO_ID = "open-thoughts/OpenThoughts-114k"
OPENTHOUGHTS_REVISION = "bd093c3994fd54d2390985b66988ddf282a55eb6"

GPQA_REPO_ID = "Idavidrein/gpqa"
GPQA_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"

HLE_REPO_ID = "cais/hle"
HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"

STAGE5_PURE_TEXT_TASKS: tuple[str, ...] = (
    "fable5_swe_agent",
    "fable5_reasoning",
    "claude_code_agent",
    "terminal_bench_agent",
    "openthoughts_reasoning",
    "gpqa_extended_science",
    "hle_text_reasoning",
)

_THOUGHT_TAG_RE = re.compile(r"<thought>\s*(.*?)\s*</thought>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_FINAL_ANS_TAG_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL | re.IGNORECASE)
_OT_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>\s*(.*?)\s*<\|end_of_thought\|>\s*(?:<\|begin_of_solution\|>\s*(.*?)\s*(?:<\|end_of_solution\|>|$)|(.*))",
    re.DOTALL,
)
_USER_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+")
_BACKTRACK_RE = re.compile(
    r"\b(?:wait,\s*let\s*me|wait\s+a\s+second|hmm,\s*wait|actually,\s*wait|let\s+me\s+re-?read)\b",
    re.IGNORECASE,
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


def sanitize_workspace_paths(text: str) -> str:
    """Replace host-specific `/Users/<name>` or `/home/<name>` prefixes with `/workspace`."""
    return _USER_PATH_RE.sub("/workspace", str(text or ""))


def compute_ngram_repetition_ratio(text: str, n: int = 4) -> float:
    """Return fraction of repeated word/character n-grams in `[0.0, 1.0]` (`0.0` = all unique)."""
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return 0.0
    tokens = cleaned.split(" ")
    if len(tokens) < max(12, n * 3):
        # Fallback to character 12-grams for CJK or compact code/JSON strings
        k = 12
        if len(cleaned) < k * 3:
            return 0.0
        grams = [cleaned[i : i + k] for i in range(0, len(cleaned) - k + 1, 3)]
    else:
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    unique_cnt = len(set(grams))
    return round(max(0.0, 1.0 - (unique_cnt / len(grams))), 5)


def check_generation_collapse_health(
    text: str,
    *,
    expect_thinking: bool = False,
    max_rep_4gram: float = 0.10,
    max_think_chars: int = 4800,
) -> dict[str, Any]:
    """Audit a model or dataset completion for `<think>` closure, non-empty answer, and repetition health."""
    raw = str(text or "")
    stripped = raw.strip()
    if not stripped:
        return {"healthy": False, "reason": "empty_completion", "rep_4gram": 1.0, "think_closed": False}

    has_open = "<think>" in stripped
    has_close = "</think>" in stripped
    think_body = ""
    answer_body = stripped

    if has_open or has_close:
        if not (has_open and has_close and stripped.find("<think>") < stripped.find("</think>")):
            return {
                "healthy": False,
                "reason": "unclosed_or_malformed_think_tag",
                "rep_4gram": compute_ngram_repetition_ratio(stripped),
                "think_closed": False,
            }
        m = _THINK_TAG_RE.search(stripped)
        if m is None:
            return {
                "healthy": False,
                "reason": "malformed_think_span",
                "rep_4gram": compute_ngram_repetition_ratio(stripped),
                "think_closed": False,
            }
        think_body = m.group(1).strip()
        answer_body = (stripped[: m.start()] + stripped[m.end() :]).strip()
        if len(think_body) > max_think_chars:
            return {
                "healthy": False,
                "reason": f"excessive_think_length:{len(think_body)}",
                "rep_4gram": compute_ngram_repetition_ratio(think_body),
                "think_closed": True,
            }
    elif expect_thinking:
        return {
            "healthy": False,
            "reason": "missing_expected_think_block",
            "rep_4gram": compute_ngram_repetition_ratio(stripped),
            "think_closed": False,
        }

    if not answer_body:
        return {
            "healthy": False,
            "reason": "empty_post_think_answer",
            "rep_4gram": compute_ngram_repetition_ratio(stripped),
            "think_closed": has_close,
        }

    rep_think = compute_ngram_repetition_ratio(think_body) if think_body else 0.0
    rep_ans = compute_ngram_repetition_ratio(answer_body)
    # Code/DSL blocks legitimately repeat structural keywords (`bone`, `attach`, `size`); allow higher ceiling on code
    is_structured_dsl_or_code = "```" in answer_body or "@rig" in answer_body or "def " in answer_body
    eff_ans_cap = max(max_rep_4gram, 0.28) if is_structured_dsl_or_code else max_rep_4gram
    if rep_think > max_rep_4gram:
        return {
            "healthy": False,
            "reason": f"repetitive_think_loop:{rep_think:.3f}",
            "rep_4gram": rep_think,
            "think_closed": True,
        }
    if rep_ans > eff_ans_cap:
        return {
            "healthy": False,
            "reason": f"repetitive_answer_loop:{rep_ans:.3f}",
            "rep_4gram": rep_ans,
            "think_closed": not has_open or has_close,
        }

    return {
        "healthy": True,
        "reason": "ok",
        "rep_4gram": round(max(rep_think, rep_ans), 5),
        "think_chars": len(think_body),
        "answer_chars": len(answer_body),
        "think_closed": (has_open and has_close) if has_open else True,
    }


def sanitize_and_bound_think_trace(
    think_text: str | None,
    answer_text: str,
    *,
    enable_thinking: bool = True,
    max_think_chars: int = 3800,
    max_answer_chars: int = 12000,
    max_rep_4gram: float = 0.10,
    max_backtracks: int = 4,
) -> str | None:
    """Validate and format a supervised `<think>\\n...\\n</think>\\n{answer}` string, returning `None` if unsafe."""
    ans = sanitize_workspace_paths(answer_text).strip()
    ans = _FINAL_ANS_TAG_RE.sub(r"\1", ans).strip()
    # Strip any stray nested think/thought tags from the visible final answer
    ans = _THOUGHT_TAG_RE.sub("", ans)
    ans = _THINK_TAG_RE.sub("", ans).strip()
    if not ans or len(ans) > max_answer_chars:
        return None

    if not enable_thinking:
        health = check_generation_collapse_health(ans, expect_thinking=False, max_rep_4gram=max_rep_4gram)
        return ans if health["healthy"] else None

    th = sanitize_workspace_paths(think_text or "").strip()
    th = _THOUGHT_TAG_RE.sub(r"\1", th)
    th = _THINK_TAG_RE.sub(r"\1", th).strip()
    if not th or len(th) < 16 or len(th) > max_think_chars:
        return None
    if len(_BACKTRACK_RE.findall(th)) > max_backtracks:
        return None

    formatted = f"<think>\n{th}\n</think>\n{ans}"
    health = check_generation_collapse_health(
        formatted,
        expect_thinking=True,
        max_rep_4gram=max_rep_4gram,
        max_think_chars=max_think_chars,
    )
    return formatted if health["healthy"] else None


def _normalize_question_key(q: str) -> str:
    return re.sub(r"\s+", " ", str(q or "").strip().lower())


def _format_tool_calls_markdown(tool_calls: list[dict[str, Any]]) -> str:
    rendered = []
    for tc in tool_calls[:4]:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str(fn.get("name") or tc.get("name") or "tool")
        raw_args = fn.get("arguments", tc.get("input", {}))
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = raw_args
        else:
            parsed_args = raw_args
        rendered.append({"name": name, "arguments": parsed_args})
    return "```json\n" + json.dumps({"tool_calls": rendered}, ensure_ascii=False, indent=2) + "\n```"


def _build_fable5_swe_and_cot_records(
    token: str | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract unperturbed Fable-5.1 SWE agent turns (`Apache-2.0`), Fable-5 CoT (`MIT`), and Claude-Code (`MIT`)."""
    seed = int(config["seed"])
    records: list[dict[str, Any]] = []

    # 1. `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` (`10k/train.jsonl`, strictly `idx < 1000` unperturbed!)
    f10_path = _hf_file(FABLE51_REPO_ID, "10k/train.jsonl", revision=FABLE51_REVISION, token=token)
    swe_cap = int(config.get("stage5_fable51_swe_cap", 260))
    seen_ticket_hashes: set[str] = set()

    with f10_path.open("r", encoding="utf-8") as fh:
        for idx, ln in enumerate(fh):
            if idx >= 1000 or len(records) >= swe_cap:
                break
            if not ln.strip():
                continue
            item = json.loads(ln)
            msgs = list(item.get("messages") or ())
            if len(msgs) < 3:
                continue
            sys_msg = next((str(m.get("content", "")) for m in msgs if m.get("role") == "system"), "").strip()
            user_msg = next((str(m.get("content", "")) for m in msgs if m.get("role") == "user"), "").strip()
            if not user_msg:
                continue
            u_low = user_msg.lower()
            if "seed author" in u_low or "authoring-agent preamble" in u_low or "wave-" in u_low[:80]:
                continue
            t_hash = hashlib.sha256(user_msg[:200].strip().lower().encode("utf-8")).hexdigest()[:16]
            if t_hash in seen_ticket_hashes:
                continue
            seen_ticket_hashes.add(t_hash)

            tools_list = list(item.get("tools") or ())
            tool_names = [
                str((t.get("function") or {}).get("name") or t.get("name") or "")
                for t in tools_list
                if isinstance(t, dict)
            ]
            sys_compact = (
                "You are an autonomous software-engineering and terminal agent. Inspect files before editing, "
                "emit structured JSON `tool_calls` when invoking tools, and verify all fixes with tests."
            )
            if tool_names:
                sys_compact += f" Available tools: {', '.join(x for x in tool_names if x)[:240]}."

            group_id = f"fable51-swe:{t_hash}"

            # Extract (a) initial planning + tool call turn, (b) mid-trajectory patch/command turn, and (c) final verified summary
            asst_indices = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            chosen_turn_slots: list[tuple[str, int]] = []
            if asst_indices:
                chosen_turn_slots.append(("init", asst_indices[0]))
            if len(asst_indices) >= 4:
                chosen_turn_slots.append(("mid", asst_indices[len(asst_indices) // 2]))
            if len(asst_indices) >= 2:
                chosen_turn_slots.append(("final", asst_indices[-1]))

            for slot_tag, a_pos in chosen_turn_slots:
                am = msgs[a_pos]
                raw_c = str(am.get("content") or "")
                th_m = _THOUGHT_TAG_RE.search(raw_c) or _THINK_TAG_RE.search(raw_c)
                thought_str = th_m.group(1).strip() if th_m else ""
                visible_str = _THOUGHT_TAG_RE.sub("", _THINK_TAG_RE.sub("", raw_c)).strip()
                tcs = list(am.get("tool_calls") or ())
                if tcs:
                    tc_block = _format_tool_calls_markdown(tcs)
                    ans_body = (visible_str + "\n\n" + tc_block).strip() if visible_str else tc_block
                else:
                    ans_body = _FINAL_ANS_TAG_RE.sub(r"\1", visible_str).strip()
                if not ans_body:
                    continue

                # Build compact context window preceding `a_pos` so prompt fits comfortably in 8k context
                if slot_tag == "init":
                    prompt_txt = sanitize_workspace_paths(user_msg)
                else:
                    recent_ctx: list[str] = [f"[Task]\n{sanitize_workspace_paths(user_msg[:1800])}"]
                    start_ctx = max(1, a_pos - 4)
                    for prev_m in msgs[start_ctx:a_pos]:
                        pr = str(prev_m.get("role") or "")
                        pc = sanitize_workspace_paths(str(prev_m.get("content") or ""))
                        pc = _THOUGHT_TAG_RE.sub("", pc).strip()
                        if pr == "tool":
                            tname = str(prev_m.get("name") or "tool")
                            recent_ctx.append(f"[Tool Output: {tname}]\n{pc[:1100]}")
                        elif pr == "assistant":
                            ptc = prev_m.get("tool_calls")
                            if ptc:
                                fn0 = ((ptc[0].get("function") or {}).get("name")) if isinstance(ptc[0], dict) else "tool"
                                recent_ctx.append(f"[Previous Agent Step]\nCalled `{fn0}`. {pc[:400]}".strip())
                            elif pc:
                                recent_ctx.append(f"[Previous Agent Step]\n{pc[:500]}")
                    if tcs:
                        recent_ctx.append("Decide and emit the next agentic `tool_calls` JSON step.")
                    else:
                        recent_ctx.append("Provide the verified engineering summary of the completed fix.")
                    prompt_txt = "\n\n".join(recent_ctx)

                use_think = bool(thought_str) and ((idx + len(records)) % 4 != 3)
                if use_think:
                    sup_ans = sanitize_and_bound_think_trace(
                        thought_str,
                        ans_body,
                        enable_thinking=True,
                        max_think_chars=3200,
                        max_answer_chars=6500,
                    )
                else:
                    sup_ans = sanitize_and_bound_think_trace(
                        None,
                        ans_body,
                        enable_thinking=False,
                        max_answer_chars=6500,
                    )
                if sup_ans is None or len(prompt_txt) > 9500:
                    continue

                records.append(
                    {
                        "id": f"fable51-swe-{idx:04d}-{slot_tag}",
                        "source_id": f"fable51-{idx:04d}",
                        "source_group": group_id,
                        "language": "en",
                        "task": "fable5_swe_agent",
                        "frames": [],
                        "enable_thinking": use_think,
                        "system_prompt": sys_compact,
                        "prompt": prompt_txt,
                        "answer": sup_ans,
                        "target": {"kind": "text", "text": ans_body},
                        "metadata": {
                            "dataset": FABLE51_REPO_ID,
                            "revision": FABLE51_REVISION,
                            "license_spdx": "Apache-2.0",
                            "real_text_corpus": True,
                            "slot": slot_tag,
                            "grouping_keys": [group_id],
                        },
                    }
                )
                if len(records) >= swe_cap:
                    break

    # 2. `TheFusionCube/Fable-5-CoT-Traces` (`MIT`, strictly `category != 'decoy'` and `not truncated`)
    fc_path = _hf_file(FABLE5_COT_REPO_ID, "train.jsonl", revision=FABLE5_COT_REVISION, token=token)
    fc_cap = int(config.get("stage5_fable5_cot_cap", 220))
    fc_candidates: list[dict[str, Any]] = []
    with fc_path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            d = json.loads(ln)
            if str(d.get("category") or "").lower() == "decoy" or d.get("truncated"):
                continue
            fc_candidates.append(d)
    fc_candidates.sort(key=lambda x: identity([seed, "f5_cot", x.get("index"), x.get("session_id")]))

    fc_added = 0
    for d in fc_candidates:
        if fc_added >= fc_cap:
            break
        p_txt = sanitize_workspace_paths(str(d.get("prompt") or "")).strip()
        r_txt = sanitize_workspace_paths(str(d.get("response") or "")).strip()
        if len(p_txt) < 20 or len(p_txt) > 7000 or len(r_txt) < 40 or len(r_txt) > 9000:
            continue
        th_m = _THINK_TAG_RE.search(r_txt) or _THOUGHT_TAG_RE.search(r_txt)
        if th_m:
            th_part = th_m.group(1).strip()
            ans_part = (_THINK_TAG_RE.sub("", _THOUGHT_TAG_RE.sub("", r_txt))).strip()
            use_think = True
        else:
            th_part = None
            ans_part = r_txt
            use_think = False
        sup_ans = sanitize_and_bound_think_trace(
            th_part,
            ans_part,
            enable_thinking=use_think,
            max_think_chars=3400,
            max_answer_chars=8500,
        )
        if sup_ans is None:
            continue
        gid = f"fable5-cot:{d.get('session_id') or d.get('index')}"
        records.append(
            {
                "id": f"fable5-cot-{int(d.get('index', fc_added)):04d}",
                "source_id": str(d.get("session_id") or f"fc-{fc_added}"),
                "source_group": gid,
                "language": "en",
                "task": "fable5_reasoning",
                "frames": [],
                "enable_thinking": use_think,
                "prompt": p_txt,
                "answer": sup_ans,
                "target": {"kind": "text", "text": ans_part},
                "metadata": {
                    "dataset": FABLE5_COT_REPO_ID,
                    "revision": FABLE5_COT_REVISION,
                    "license_spdx": "MIT",
                    "category": str(d.get("category") or "general"),
                    "real_text_corpus": True,
                    "grouping_keys": [gid],
                },
            }
        )
        fc_added += 1

    # 3. `armand0e/claude-fable-5-claude-code` (`MIT`, real software-engineering session turns)
    cc_cap = int(config.get("stage5_claude_code_cap", 120))
    if cc_cap > 0:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        info = api.dataset_info(CLAUDE_FABLE5_CODE_REPO_ID, revision=CLAUDE_FABLE5_CODE_REVISION)
        session_files = sorted(s.rfilename for s in (info.siblings or ()) if s.rfilename.endswith(".jsonl"))
        session_files.sort(key=lambda fn: identity([seed, "cc_fable5", fn]))
        cc_added = 0
        for s_file in session_files[:24]:
            if cc_added >= cc_cap:
                break
            local_s = _hf_file(
                CLAUDE_FABLE5_CODE_REPO_ID,
                s_file,
                revision=CLAUDE_FABLE5_CODE_REVISION,
                token=token,
            )
            sid = s_file.rsplit(".", 1)[0]
            gid = f"claude-fable5-code:{sid}"
            last_user_text = ""
            with local_s.open("r", encoding="utf-8") as fh:
                for line_i, ln in enumerate(fh):
                    if cc_added >= cc_cap:
                        break
                    if not ln.strip():
                        continue
                    ev = json.loads(ln)
                    ev_type = ev.get("type")
                    msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
                    if ev_type == "user":
                        c = msg.get("content")
                        if isinstance(c, str) and len(c.strip()) >= 24:
                            last_user_text = sanitize_workspace_paths(c.strip())
                    elif ev_type == "assistant" and last_user_text:
                        blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
                        text_parts: list[str] = []
                        tool_uses: list[dict[str, Any]] = []
                        think_parts: list[str] = []
                        for b in blocks:
                            if not isinstance(b, dict):
                                continue
                            bt = b.get("type")
                            if bt == "thinking" and b.get("thinking"):
                                think_parts.append(str(b["thinking"]).strip())
                            elif bt == "text" and b.get("text"):
                                text_parts.append(str(b["text"]).strip())
                            elif bt == "tool_use" and b.get("name"):
                                tool_uses.append(
                                    {
                                        "name": str(b["name"]),
                                        "input": json.loads(sanitize_workspace_paths(json.dumps(b.get("input", {})))),
                                    }
                                )
                        ans_chunks = list(text_parts)
                        if tool_uses:
                            ans_chunks.append(_format_tool_calls_markdown(tool_uses))
                        ans_body = sanitize_workspace_paths("\n\n".join(x for x in ans_chunks if x)).strip()
                        if len(ans_body) < 40 or len(last_user_text) > 6500:
                            continue
                        th_joined = "\n".join(x for x in think_parts if x).strip()
                        use_think = bool(th_joined)
                        sup_ans = sanitize_and_bound_think_trace(
                            th_joined if use_think else None,
                            ans_body,
                            enable_thinking=use_think,
                            max_think_chars=3000,
                            max_answer_chars=6500,
                        )
                        if sup_ans is None:
                            continue
                        records.append(
                            {
                                "id": f"claude-fable5-{sid[:12]}-{line_i:04d}",
                                "source_id": sid,
                                "source_group": gid,
                                "language": "en",
                                "task": "claude_code_agent",
                                "frames": [],
                                "enable_thinking": use_think,
                                "prompt": last_user_text,
                                "answer": sup_ans,
                                "target": {"kind": "text", "text": ans_body},
                                "metadata": {
                                    "dataset": CLAUDE_FABLE5_CODE_REPO_ID,
                                    "revision": CLAUDE_FABLE5_CODE_REVISION,
                                    "license_spdx": "MIT",
                                    "real_text_corpus": True,
                                    "grouping_keys": [gid],
                                },
                            }
                        )
                        cc_added += 1
                        last_user_text = ""

    return records


def _build_terminal_bench_records(
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build Terminal-Bench 3.0 (`Apache-2.0`) trajectories while holding out 1/3 of tasks strictly for evaluation."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.dataset_info(TERMINAL_BENCH3_REPO_ID, revision=TERMINAL_BENCH3_REVISION)
    by_task: dict[str, set[str]] = defaultdict(set)
    for sib in info.siblings or ():
        parts = sib.rfilename.split("/")
        if len(parts) >= 3 and parts[0] == "tasks":
            by_task[parts[1]].add(sib.rfilename)

    holdout_tasks: list[str] = []
    train_tasks: list[str] = []
    for tid in sorted(by_task):
        bucket = int(hashlib.sha256(f"tb3_holdout_v1:{tid}".encode("utf-8")).hexdigest()[:8], 16) % 3
        if bucket == 0:
            holdout_tasks.append(tid)
        else:
            train_tasks.append(tid)

    records: list[dict[str, Any]] = []
    for t_idx, tid in enumerate(train_tasks):
        files = by_task[tid]
        inst_rel = f"tasks/{tid}/instruction.md"
        if inst_rel not in files:
            continue
        sol_candidates = sorted(
            f for f in files if f.startswith(f"tasks/{tid}/solution/") and f.endswith((".sh", ".py"))
        )
        if not sol_candidates:
            continue

        inst_txt = _hf_file(
            TERMINAL_BENCH3_REPO_ID,
            inst_rel,
            revision=TERMINAL_BENCH3_REVISION,
            token=token,
        ).read_text(encoding="utf-8", errors="replace").strip()

        sol_blocks: list[str] = []
        sol_names: list[str] = []
        for s_rel in sol_candidates[:2]:
            s_txt = _hf_file(
                TERMINAL_BENCH3_REPO_ID,
                s_rel,
                revision=TERMINAL_BENCH3_REVISION,
                token=token,
            ).read_text(encoding="utf-8", errors="replace").strip()
            if not s_txt or len(s_txt) > 9000:
                continue
            lang_tag = "python" if s_rel.endswith(".py") else "bash"
            sname = s_rel.rsplit("/", 1)[-1]
            sol_names.append(sname)
            sol_blocks.append(f"### `{sname}`\n```{lang_tag}\n{s_txt}\n```")

        if not inst_txt or not sol_blocks:
            continue
        ans_body = "\n\n".join(sol_blocks)
        think_summary = (
            f"Terminal engineering task: `{tid}`.\n"
            "1. Inspect the workspace layout, environment constraints, and expected output contract in the specification.\n"
            f"2. Implement a deterministic, non-interactive solution ({', '.join(sol_names)}) handling edge cases cleanly.\n"
            "3. Validate outputs, file paths, and exit status `0`."
        )
        use_think = (t_idx % 4 != 3)
        sup_ans = sanitize_and_bound_think_trace(
            think_summary if use_think else None,
            ans_body,
            enable_thinking=use_think,
            max_think_chars=1800,
            max_answer_chars=10500,
        )
        if sup_ans is None or len(inst_txt) > 7500:
            continue
        gid = f"terminal-bench-3:{tid}"
        records.append(
            {
                "id": f"tb3-train-{tid}",
                "source_id": tid,
                "source_group": gid,
                "language": "en",
                "task": "terminal_bench_agent",
                "frames": [],
                "enable_thinking": use_think,
                "prompt": (
                    "[Terminal Agent Task]\n"
                    f"{inst_txt}\n\n"
                    "Author a complete, reproducible bash/Python terminal solution script that satisfies all task requirements."
                ),
                "answer": sup_ans,
                "target": {"kind": "text", "text": ans_body},
                "metadata": {
                    "dataset": TERMINAL_BENCH3_REPO_ID,
                    "revision": TERMINAL_BENCH3_REVISION,
                    "license_spdx": "Apache-2.0",
                    "terminal_task_id": tid,
                    "real_text_corpus": True,
                    "grouping_keys": [gid],
                },
            }
        )

    return records, holdout_tasks


def _build_openthoughts_records(
    token: str | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Sample concise, non-repetitive code, math, and science `<think>` traces from `OpenThoughts-114k` (`Apache-2.0`)."""
    import pyarrow.parquet as pq

    cap = int(config.get("stage5_openthoughts_cap", 360))
    if cap <= 0:
        return []
    seed = int(config["seed"])
    pq_path = _hf_file(
        OPENTHOUGHTS_REPO_ID,
        "data/train-00000-of-00006.parquet",
        revision=OPENTHOUGHTS_REVISION,
        token=token,
    )
    pf = pq.ParquetFile(pq_path)
    candidates: list[tuple[int, str, str, str]] = []
    row_i = 0
    for batch in pf.iter_batches(batch_size=512, columns=["conversations"]):
        for item in batch.to_pylist():
            row_i += 1
            convs = list(item.get("conversations") or ())
            if len(convs) < 2:
                continue
            u_val = str(convs[0].get("value") or "").strip()
            a_val = str(convs[1].get("value") or "").strip()
            if not u_val or len(u_val) > 4500 or len(a_val) > 8500:
                continue
            m = _OT_THOUGHT_RE.search(a_val)
            if m is None:
                continue
            th = m.group(1).strip()
            sol = (m.group(2) or m.group(3) or "").strip()
            if not (160 <= len(th) <= 3200) or not (40 <= len(sol) <= 5000):
                continue
            if len(_BACKTRACK_RE.findall(th)) > 2:
                continue
            candidates.append((row_i, u_val, th, sol))
            if len(candidates) >= cap * 4:
                break
        if len(candidates) >= cap * 4:
            break

    candidates.sort(key=lambda t: identity([seed, "openthoughts_concise", t[0]]))
    records: list[dict[str, Any]] = []
    for idx, (orig_i, u_val, th, sol) in enumerate(candidates):
        if len(records) >= cap:
            break
        # 75% `<think>` mode, 25% direct non-thinking solution to prevent `<think>` mode lock-in
        use_think = (idx % 4 != 3)
        sup_ans = sanitize_and_bound_think_trace(
            th if use_think else None,
            sol,
            enable_thinking=use_think,
            max_think_chars=3200,
            max_answer_chars=5200,
            max_rep_4gram=0.09,
            max_backtracks=2,
        )
        if sup_ans is None:
            continue
        gid = f"openthoughts:{orig_i:06d}"
        records.append(
            {
                "id": f"ot-concise-{orig_i:06d}",
                "source_id": f"ot-{orig_i:06d}",
                "source_group": gid,
                "language": "en",
                "task": "openthoughts_reasoning",
                "frames": [],
                "enable_thinking": use_think,
                "prompt": u_val,
                "answer": sup_ans,
                "target": {"kind": "text", "text": sol},
                "metadata": {
                    "dataset": OPENTHOUGHTS_REPO_ID,
                    "revision": OPENTHOUGHTS_REVISION,
                    "license_spdx": "Apache-2.0",
                    "real_text_corpus": True,
                    "grouping_keys": [gid],
                },
            }
        )
    return records


def _build_gpqa_non_diamond_records(
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Load strictly non-Diamond `gpqa_extended` (`CC-BY-4.0`) while keeping 100% of `gpqa_diamond` held out."""
    cap = int(config.get("stage5_gpqa_extended_cap", 348))
    if cap <= 0:
        return [], 0
    seed = int(config["seed"])
    dia_path = _hf_file(GPQA_REPO_ID, "gpqa_diamond.csv", revision=GPQA_REVISION, token=token)
    ext_path = _hf_file(GPQA_REPO_ID, "gpqa_extended.csv", revision=GPQA_REVISION, token=token)

    with dia_path.open("r", encoding="utf-8") as fh:
        dia_rows = list(csv.DictReader(fh))
    diamond_keys = {_normalize_question_key(r.get("Question", "")) for r in dia_rows if r.get("Question")}

    with ext_path.open("r", encoding="utf-8") as fh:
        ext_rows = list(csv.DictReader(fh))

    records: list[dict[str, Any]] = []
    for r_idx, r in enumerate(ext_rows):
        if len(records) >= cap:
            break
        q = str(r.get("Question") or "").strip()
        if not q or _normalize_question_key(q) in diamond_keys:
            continue
        correct = str(r.get("Correct Answer") or "").strip()
        wrongs = [
            str(r.get(f"Incorrect Answer {k}") or "").strip()
            for k in (1, 2, 3)
            if str(r.get(f"Incorrect Answer {k}") or "").strip()
        ]
        expl = str(r.get("Explanation") or r.get("Pre-Revision Explanation") or "").strip()
        if not correct or len(wrongs) < 3:
            continue

        rng = random.Random(int(identity([seed, "gpqa_ext_mc", r_idx])[:12], 16))
        opts = [correct, *wrongs[:3]]
        rng.shuffle(opts)
        letters = ("A", "B", "C", "D")
        correct_letter = letters[opts.index(correct)]
        opt_lines = "\n".join(f"({letters[i]}) {opts[i]}" for i in range(4))
        prompt = (
            f"{q}\n\nChoices:\n{opt_lines}\n\n"
            "Analyze the scientific principles carefully and conclude with `Answer: (X)` where X is A, B, C, or D."
        )
        ans_body = f"Answer: ({correct_letter}) {correct}"
        # Trim overly long raw literature dumps to concise PhD reasoning
        if len(expl) > 3000:
            paras = [p.strip() for p in expl.split("\n") if p.strip()]
            trimmed: list[str] = []
            cur_c = 0
            for p in paras:
                if cur_c + len(p) > 2800:
                    break
                trimmed.append(p)
                cur_c += len(p) + 1
            expl = "\n".join(trimmed).strip()

        use_think = bool(expl) and (len(records) % 5 != 4)
        sup_ans = sanitize_and_bound_think_trace(
            expl if use_think else None,
            ans_body,
            enable_thinking=use_think,
            max_think_chars=3200,
            max_answer_chars=1200,
        )
        if sup_ans is None:
            continue
        q_hash = hashlib.sha256(_normalize_question_key(q).encode("utf-8")).hexdigest()[:16]
        gid = f"gpqa-ext:{q_hash}"
        records.append(
            {
                "id": f"gpqa-ext-{r_idx:04d}",
                "source_id": f"gpqa-ext-{q_hash}",
                "source_group": gid,
                "language": "en",
                "task": "gpqa_extended_science",
                "frames": [],
                "enable_thinking": use_think,
                "prompt": prompt,
                "answer": sup_ans,
                "target": {"kind": "text", "text": ans_body},
                "metadata": {
                    "dataset": GPQA_REPO_ID,
                    "revision": GPQA_REVISION,
                    "license_spdx": "CC-BY-4.0",
                    "correct_letter": correct_letter,
                    "correct_answer": correct,
                    "diamond_disjoint_verified": True,
                    "real_text_corpus": True,
                    "grouping_keys": [gid],
                },
            }
        )

    return records, len(diamond_keys)


def _decode_hle_image_bytes(raw_image: Any) -> tuple[bytes, str] | None:
    """Extract raw PNG/JPEG bytes from HLE `image` data-URI or dict column."""
    import base64

    payload: bytes | None = None
    if isinstance(raw_image, dict) and raw_image.get("bytes"):
        payload = bytes(raw_image["bytes"])
    elif isinstance(raw_image, (bytes, bytearray)) and len(raw_image) > 16:
        payload = bytes(raw_image)
    elif isinstance(raw_image, str) and raw_image.strip():
        s = raw_image.strip()
        if s.startswith("data:image/") and ";base64," in s:
            b64_part = s.split(";base64,", 1)[1]
            try:
                payload = base64.b64decode(b64_part)
            except Exception:
                return None
    if not payload or len(payload) < 32:
        return None
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        return payload, "png"
    if payload[:3] == b"\xff\xd8\xff":
        return payload, "jpg"
    # Re-encode other PIL-readable formats (e.g. WebP/GIF) cleanly to PNG
    try:
        import io
        from PIL import Image

        im = Image.open(io.BytesIO(payload))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue(), "png"
    except Exception:
        return None


def _build_hle_disjoint_records(
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes], list[str]]:
    """Build strictly disjoint HLE (`MIT`) multimodal + text `<think>` records while holding out 20% for evaluation."""
    import pyarrow.parquet as pq

    cap_total = int(config.get("stage5_hle_train_cap", 460))
    if cap_total <= 0:
        return [], {}, []
    seed = int(config["seed"])
    hle_path = _hf_file(
        HLE_REPO_ID,
        "data/test-00000-of-00001.parquet",
        revision=HLE_REVISION,
        token=token,
    )
    pf = pq.ParquetFile(hle_path)

    holdout_ids: list[str] = []
    multimodal_pool: list[dict[str, Any]] = []
    text_pool_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for batch in pf.iter_batches(
        batch_size=128,
        columns=["id", "question", "image", "answer", "answer_type", "rationale", "category", "raw_subject"],
    ):
        for item in batch.to_pylist():
            hid = str(item.get("id") or "").strip()
            if not hid:
                continue
            # Deterministic 20% stratified holdout (`bucket == 0` NEVER enters training)
            bucket = int(hashlib.sha256(f"hle_holdout_v1:{hid}".encode("utf-8")).hexdigest()[:8], 16) % 5
            if bucket == 0:
                holdout_ids.append(hid)
                continue
            q = str(item.get("question") or "").strip()
            ans = str(item.get("answer") or "").strip()
            rat = str(item.get("rationale") or "").strip()
            if not q or not ans or len(rat) < 32 or len(rat) > 3600 or len(q) > 5500:
                continue
            if item.get("image"):
                multimodal_pool.append(item)
            else:
                text_pool_by_cat[str(item.get("category") or "Other")].append(item)

    multimodal_pool.sort(key=lambda r: identity([seed, "hle_mm", r["id"]]))
    for cat in text_pool_by_cat:
        text_pool_by_cat[cat].sort(key=lambda r: identity([seed, "hle_txt", cat, r["id"]]))

    chosen_items: list[dict[str, Any]] = list(multimodal_pool[: min(len(multimodal_pool), 180)])
    cat_keys = sorted(text_pool_by_cat)
    round_i = 0
    while len(chosen_items) < cap_total:
        progressed = False
        for ck in cat_keys:
            if round_i < len(text_pool_by_cat[ck]):
                chosen_items.append(text_pool_by_cat[ck][round_i])
                progressed = True
                if len(chosen_items) >= cap_total:
                    break
        if not progressed:
            break
        round_i += 1

    blob_map: dict[str, bytes] = {}
    out_records: list[tuple[dict[str, Any], list[str]]] = []

    for idx, item in enumerate(chosen_items):
        hid = str(item["id"])
        q = str(item["question"]).strip()
        ans = str(item["answer"]).strip()
        rat = str(item["rationale"]).strip()
        cat = str(item.get("category") or "Science")
        ans_body = f"Final Answer: {ans}"
        use_think = (idx % 6 != 5)
        sup_ans = sanitize_and_bound_think_trace(
            rat if use_think else None,
            ans_body,
            enable_thinking=use_think,
            max_think_chars=3600,
            max_answer_chars=1500,
            max_rep_4gram=0.10,
        )
        if sup_ans is None:
            continue

        gid = f"hle-train:{hid}"
        decoded_img = _decode_hle_image_bytes(item.get("image")) if item.get("image") else None
        frame_paths: list[str] = []
        if decoded_img is not None:
            img_bytes, ext = decoded_img
            m_path = f"hle_mm/{hid}.{ext}"
            blob_map[m_path] = img_bytes
            frame_paths = [m_path]

        rec: dict[str, Any] = {
            "id": f"hle-train-{hid}",
            "source_id": hid,
            "source_group": gid,
            "language": "en",
            "task": "hle_multimodal_reasoning" if frame_paths else "hle_text_reasoning",
            "frames": frame_paths,
            "enable_thinking": use_think,
            "prompt": f"{q}\n\nProvide a rigorous derivation and end with `Final Answer: <answer>`.",
            "answer": sup_ans,
            "target": {"kind": "text", "text": ans_body},
            "metadata": {
                "dataset": HLE_REPO_ID,
                "revision": HLE_REVISION,
                "license_spdx": "MIT",
                "category": cat,
                "subject": str(item.get("raw_subject") or ""),
                "oracle_exact_answer": ans,
                "hle_holdout_disjoint_verified": True,
                "real_text_corpus": not bool(frame_paths),
                "grouping_keys": [gid],
            },
        }
        if frame_paths:
            rec["media_times_s"] = [0.0]
            rec["visual_kind"] = "independent_image"
            rec["visual_units"] = [{"visual_kind": "independent_image", "view_id": "single"}]
        out_records.append((rec, frame_paths))

    return out_records, blob_map, sorted(holdout_ids)


def build_stage5_reasoning_and_agentic_stream(
    root: Path,
    token: str | None,
    config: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, bytes], dict[str, Any]]:
    """Build all Stage-5 reasoning, coding, terminal, GPQA-non-Diamond, and HLE-disjoint records + holdout manifest."""
    emit("stage5_reasoning_stream_start")
    fable_records = _build_fable5_swe_and_cot_records(token, config)
    tb3_records, tb3_holdout_tasks = _build_terminal_bench_records(token, config)
    ot_records = _build_openthoughts_records(token, config)

    gpqa_records: list[dict[str, Any]] = []
    gpqa_diamond_count = 0
    if config.get("enable_stage5_gpqa_extended", True):
        try:
            gpqa_records, gpqa_diamond_count = _build_gpqa_non_diamond_records(token, config)
        except Exception as exc:
            emit("stage5_gpqa_skipped", reason=str(exc)[:240])

    hle_pairs: list[tuple[dict[str, Any], list[str]]] = []
    hle_blobs: dict[str, bytes] = {}
    hle_holdout_ids: list[str] = []
    if config.get("enable_stage5_hle_disjoint", True):
        try:
            hle_pairs, hle_blobs, hle_holdout_ids = _build_hle_disjoint_records(token, config)
        except Exception as exc:
            emit("stage5_hle_skipped", reason=str(exc)[:240])

    combined: list[tuple[dict[str, Any], list[str]]] = []
    for r in (*fable_records, *tb3_records, *ot_records, *gpqa_records):
        combined.append((r, []))
    combined.extend(hle_pairs)

    by_task = Counter(r["task"] for r, _ in combined)
    think_true = sum(1 for r, _ in combined if r.get("enable_thinking"))
    holdout_manifest = {
        "schema_version": 1,
        "gpqa_diamond_held_out_100_percent": True,
        "gpqa_diamond_question_count": gpqa_diamond_count,
        "terminal_bench_3_holdout_tasks": tb3_holdout_tasks,
        "hle_holdout_count": len(hle_holdout_ids),
        "hle_holdout_ids_sha256": identity(hle_holdout_ids),
        "hle_holdout_ids": hle_holdout_ids,
        "anti_collapse_policy": {
            "max_think_chars": 3800,
            "max_rep_4gram": 0.10,
            "thinking_fraction": round(think_true / max(1, len(combined)), 4),
            "non_thinking_fraction": round(1.0 - think_true / max(1, len(combined)), 4),
        },
        "stage5_task_counts": dict(sorted(by_task.items())),
    }
    write(Path(root) / "metadata" / "HELD_OUT_BENCHMARKS.json", holdout_manifest)
    emit(
        "stage5_reasoning_stream_ready",
        total_records=len(combined),
        multimodal_hle_images=len(hle_blobs),
        thinking_fraction=holdout_manifest["anti_collapse_policy"]["thinking_fraction"],
        tasks=dict(sorted(by_task.items())),
    )
    return combined, hle_blobs, holdout_manifest
