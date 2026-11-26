"""Online Visual-Causality-Gated Multimodal RLVR (`VC-GRPO`) + User-Feedback Preference Engine.

Implements genuine on-policy Group Relative Policy Optimization (GRPO) for `SparkNative` under 24 GB VRAM:
  1. Samples `K` live rollouts per multimodal/reasoning prompt via KV-cached decoding.
  2. Scores every rollout with **deterministic, non-hackable ground-truth verifiers** (`prism_exact`,
     `yesno_bool`, `numeric_exact`, `mcq_letter`, `ocr_cer`, `color_shape`, `dapo_math`, `synccraft_dsl`)
     plus anti-collapse health gating (repetition & `<think>` closure).
  3. Applies a **Visual-Causality Gate (`VC-Gate`)** on visual prompts by comparing real-image vs. blank-gray
     log-probability on the generated answer tokens (`delta_logp = mean_logp(y|I_real,q) - mean_logp(y|I_blank,q)`):
     visually grounded correct answers earn strictly higher group-relative advantage than blind guesses!
  4. Interleaves optional **User-Feedback Preference Pairs (`user_feedback.jsonl`)** via Direct Preference
     Optimization (`DPO`) margin updates so human multimodal feedback continuously shapes policy behavior.
  5. Enforces a one-sided visual RMS ceiling and KL anchor (`kl_beta`) against pre-update rollout log-probs
     with zero extra reference-model VRAM overhead, updating smoothly under `APOLLO+COAT`.
"""
from __future__ import annotations

import contextlib
import gc
import io
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F

from .common import emit
from .engine import ensure_coat_for_model, optimizer_for
from .inference import generate as spark_generate, visual_media
from .optimizer import MergedGradientAccumulator
from .reasoning_agentic_corpus import check_generation_collapse_health
from .synccraft_bridge import strip_think_block


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _extract_body(raw: str) -> tuple[str, bool]:
    txt = str(raw or "")
    if "<think>" in txt:
        if "</think>" in txt:
            _, body = strip_think_block(txt)
            return body.strip(), True
        return txt.split("<think>", 1)[1].strip().splitlines()[-1] if txt.split("<think>", 1)[1].strip() else "", False
    return txt.strip(), True


def verify_rlvr_prediction(
    pred_text: str,
    verifier: dict[str, Any],
    *,
    expect_thinking: bool = False,
) -> dict[str, Any]:
    """Compute deterministic verifiable reward in `[0.0, 1.0]` plus collapse penalty."""
    body, closed_ok = _extract_body(pred_text)
    health = check_generation_collapse_health(
        pred_text,
        expect_thinking=expect_thinking,
        max_rep_4gram=0.24,
        max_think_chars=2600,
    )
    if not body or not health["healthy"] or (expect_thinking and not closed_ok):
        return {"reward": 0.0, "exact": False, "penalty": -0.35, "healthy": False}

    kind = str(verifier.get("kind") or "prism_exact").lower()
    gold = str(verifier.get("gold") or "").strip()
    if not gold:
        return {"reward": 0.0, "exact": False, "penalty": 0.0, "healthy": True}

    clean_body = re.sub(r"^(?:answer|final answer|the answer is)\s*[:：]?\s*", "", body, flags=re.IGNORECASE).strip()
    clean_body_l = clean_body.lower().strip(" .!?,;:\"'")
    gold_l = gold.lower().strip(" .!?,;:\"'")

    if kind == "yesno_bool":
        yn_map = {"yes": "yes", "true": "yes", "no": "no", "false": "no"}
        g_norm = yn_map.get(gold_l, gold_l)
        toks = re.findall(r"\b(yes|no|true|false)\b", clean_body_l)
        p_norm = yn_map.get(toks[-1], "") if toks else ""
        ok = (p_norm == g_norm)
        return {"reward": 1.0 if ok else 0.0, "exact": ok, "penalty": 0.0, "healthy": True}

    if kind == "mcq_letter":
        m = re.findall(r"(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]*([A-J])[\)\]]*", body, flags=re.IGNORECASE)
        pred_l = m[-1].upper() if m else (clean_body[:1].upper() if clean_body[:1].upper() in "ABCDEFGHIJ" else "")
        ok = (pred_l == gold.upper()[:1])
        return {"reward": 1.0 if ok else 0.0, "exact": ok, "penalty": 0.0, "healthy": True}

    if kind in ("numeric_exact", "dapo_math"):
        g_num = re.sub(r"[,%\s]", "", gold_l)
        nums = re.findall(r"-?\d+(?:\.\d+)?", clean_body_l.replace(",", ""))
        if not nums:
            return {"reward": 0.0, "exact": False, "penalty": 0.0, "healthy": True}
        p_num = nums[-1]
        try:
            ok = abs(float(p_num) - float(g_num)) <= max(1e-3, 1e-2 * abs(float(g_num)))
        except ValueError:
            ok = (p_num == g_num)
        return {"reward": 1.0 if ok else 0.0, "exact": ok, "penalty": 0.0, "healthy": True}

    if kind == "ocr_cer":
        if gold_l in clean_body_l:
            return {"reward": 1.0, "exact": True, "penalty": 0.0, "healthy": True}
        d = _levenshtein(clean_body_l[: max(len(gold_l) * 2, 32)], gold_l)
        sim = max(0.0, 1.0 - d / max(1, len(gold_l)))
        return {"reward": round(0.85 * sim, 4), "exact": False, "penalty": 0.0, "healthy": True}

    if kind == "color_shape":
        g_parts = gold_l.split()
        hits = sum(1 for tok in g_parts if tok in clean_body_l)
        exact = (hits == len(g_parts)) and (len(clean_body_l.split()) <= len(g_parts) + 4)
        rew = 1.0 if exact else (0.4 * hits / max(1, len(g_parts)))
        return {"reward": round(rew, 4), "exact": exact, "penalty": 0.0, "healthy": True}

    # Default `prism_exact`: normalized exact or concise containment
    ok = (clean_body_l == gold_l) or (len(gold_l) >= 3 and gold_l in clean_body_l and len(clean_body_l) <= len(gold_l) + 16)
    return {"reward": 1.0 if ok else 0.0, "exact": ok, "penalty": 0.0, "healthy": True}


def compute_vc_grpo_rewards(
    ver_results: list[dict[str, Any]],
    delta_logps: list[float] | None = None,
    *,
    causal_bonus_weight: float = 0.22,
    causal_gate_floor: float = 0.35,
) -> tuple[list[float], list[float]]:
    """Combine verifiable rewards with Visual-Causality gating (`delta_logp = logp_real - logp_blank`)
    and compute group-normalized GRPO advantages.
    """
    raw_rewards: list[float] = []
    for idx, vr in enumerate(ver_results):
        base = float(vr.get("reward", 0.0))
        pen = float(vr.get("penalty", 0.0))
        if delta_logps is not None and idx < len(delta_logps):
            dl = float(delta_logps[idx])
            gate = causal_gate_floor + (1.0 - causal_gate_floor) / (1.0 + math.exp(-4.0 * (dl - 0.05)))
            bonus = causal_bonus_weight * max(-0.5, min(1.0, dl))
            r = base * gate + bonus + pen
        else:
            r = base + pen
        raw_rewards.append(round(r, 5))

    n = max(1, len(raw_rewards))
    mean_r = sum(raw_rewards) / n
    var_r = sum((x - mean_r) ** 2 for x in raw_rewards) / n
    std_r = math.sqrt(var_r)
    if std_r < 1e-3:
        return raw_rewards, [0.0 for _ in raw_rewards]
    advs = [max(-3.0, min(3.0, (x - mean_r) / (std_r + 1e-4))) for x in raw_rewards]
    return raw_rewards, advs


def load_user_feedback_jsonl(path: Path | None) -> list[dict[str, Any]]:
    """Load human multimodal feedback pairs from JSONL:
    `{"image_path": "...", "prompt": "...", "chosen": "...", "rejected": "...", "rating": 1..5, "note": "..."}`
    """
    if path is None or not path.is_file():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        prm = str(obj.get("prompt") or "").strip()
        chosen = str(obj.get("chosen") or "").strip()
        rejected = str(obj.get("rejected") or "").strip()
        if not prm or not chosen:
            continue
        img_p = obj.get("image_path")
        img_bytes: bytes | None = None
        if img_p and Path(str(img_p)).is_file():
            img_bytes = Path(str(img_p)).read_bytes()
        items.append({
            "prompt": prm,
            "chosen": chosen,
            "rejected": rejected,
            "image_bytes": img_bytes,
            "rating": float(obj.get("rating", 5.0)),
            "note": str(obj.get("note") or ""),
        })
    emit("stage13_user_feedback_loaded", path=str(path), count=len(items))
    return items


def _build_prefix_tensors(
    model: Any,
    *,
    prompt: str,
    media: dict[str, Any] | None,
    enable_thinking: bool,
    task: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Encode prompt + optional visual prefix once; returns `(pre_emb, vis_ctx, pre_tid, pre_tm, vis_rms)`."""
    device = model.device
    emb_layer = model.get_input_embeddings()
    row = {
        "_family": "image" if (media and media.get("images")) else "text",
        "language": "en",
        "task": task,
        "prompt": prompt,
        "enable_thinking": bool(enable_thinking),
    }
    if media and media.get("images"):
        pre_emb, vis_ctx, pre_tid, pre_tm = model.prefix_with_visual_context_and_ids(row, media, for_generation=True)
        vm = ~pre_tm[0]
        vis_rms = pre_emb[0, vm].float().pow(2).mean().add(1e-8).sqrt() if vm.any() else None
        return pre_emb, vis_ctx, pre_tid, pre_tm, vis_rms
    pre_ids, post_ids = model.chat.encode(prompt, enable_thinking=bool(enable_thinking), for_generation=True)
    full_pre = torch.tensor(pre_ids + post_ids, device=device, dtype=torch.long).unsqueeze(0)
    pre_emb = emb_layer(full_pre).to(model.embed_dtype)
    return pre_emb, None, full_pre, torch.ones_like(full_pre, dtype=torch.bool), None


def _sequence_answer_logp(
    model: Any,
    *,
    prompt: str,
    answer: str,
    media: dict[str, Any] | None,
    enable_thinking: bool,
    task: str,
    with_grad: bool,
    cached_prefix: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor | None]:
    """Compute exact per-token log-probs on `answer` tokens + optional visual RMS tensor."""
    device = model.device
    tok = model.tokenizer
    eos_id = int(tok.eos_token_id)
    emb_layer = model.get_input_embeddings()

    ans_ids = tok(answer, add_special_tokens=False).input_ids[:384] + [eos_id]
    if not ans_ids:
        return torch.tensor(0.0, device=device), 0, None

    ans_t = torch.tensor(ans_ids, device=device, dtype=torch.long)
    ans_in = ans_t[:-1]
    use_amp = torch.cuda.is_available() and str(device).startswith("cuda")
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    grad_ctx = contextlib.nullcontext() if with_grad else torch.no_grad()

    with grad_ctx, amp_ctx:
        if cached_prefix is not None and not with_grad:
            pre_emb, vis_ctx, pre_tid, pre_tm, vis_rms = cached_prefix
        else:
            pre_emb, vis_ctx, pre_tid, pre_tm, vis_rms = _build_prefix_tensors(
                model, prompt=prompt, media=media, enable_thinking=enable_thinking, task=task
            )

        if len(ans_in) > 0:
            ae = emb_layer(ans_in.unsqueeze(0)).to(model.embed_dtype)
            fe = torch.cat([pre_emb, ae], dim=1)
            tid = torch.cat([pre_tid, ans_in.unsqueeze(0)], dim=1)
            tm = torch.cat([pre_tm, torch.ones((1, len(ans_in)), device=device, dtype=torch.bool)], dim=1)
        else:
            fe, tid, tm = pre_emb, pre_tid, pre_tm

        am = torch.ones((1, fe.shape[1]), device=device, dtype=torch.long)
        out = model.forward_looped_model(
            inputs_embeds=fe,
            attention_mask=am,
            token_ids_for_engram=tid,
            text_token_mask=tm,
            visual_context=vis_ctx,
        )
        start_pos = pre_emb.shape[1] - 1
        logits = model.spark.lm_head(out.last_hidden_state[0, start_pos : start_pos + len(ans_ids)]).float()
        log_probs = F.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(-1, ans_t.unsqueeze(-1)).squeeze(-1)
        return tok_lp, len(ans_ids), vis_rms


def run_multimodal_vc_grpo_rlvr(
    model: Any,
    *,
    base_cfg: dict[str, Any],
    rlvr_items: list[tuple[dict[str, Any], dict[str, bytes]]],
    user_feedback_items: list[dict[str, Any]] | None = None,
    max_prompts: int = 750,
    group_size: int = 4,
    grad_accum_prompts: int = 4,
    max_wall_hours: float = 5.5,
    lr: float = 3.0e-6,
    connector_lr: float = 1.2e-5,
    kl_beta: float = 0.04,
    clip_eps: float = 0.20,
    rms_cap: float = 0.10,
    rms_weight: float = 5.0,
    freeze_vision_projector: bool = False,
    checkpoint_dir: Path | None = None,
    seed: int = 20261399,
) -> dict[str, Any]:
    """Execute wall-clock-budgeted online Visual-Causality-Gated Multimodal `VC-GRPO` + User-Feedback DPO."""
    from .engine import _save_model_safetensors

    rng = random.Random(seed)
    t0 = time.monotonic()
    deadline = t0 + max(0.02, float(max_wall_hours)) * 3600.0

    pool = list(rlvr_items)
    rng.shuffle(pool)
    fb_pool = list(user_feedback_items or [])

    # Configure trainable parameters cleanly via `set_stage` (disables B50 vision backward while preserving B50 XPU forward!)
    if hasattr(model, "set_stage"):
        model.set_stage(alignment_only=False, stage_mode="full_spark", unfreeze_vision_last_n_layers=0)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        if getattr(model, "vision", None) is not None:
            for p in model.vision.parameters():
                p.requires_grad_(False)
    vp = getattr(model, "vision_projector", None)
    if vp is not None:
        if freeze_vision_projector:
            vp.requires_grad_(False)
        elif hasattr(vp, "meta_scale"):
            vp.meta_scale.requires_grad_(False)

    eff_cfg = dict(base_cfg)
    eff_cfg.update({
        "optimizer_profile": "apollo_coat",
        "learning_rate": float(lr),
        "backbone_lr": float(lr),
        "apollo_learning_rate": float(lr),
        "connector_learning_rate": float(connector_lr),
        "adapter_lr": float(connector_lr),
        "apollo_connector_learning_rate": float(connector_lr),
        "engram_lr": float(lr) * 3.0,
        "weight_decay": 0.01,
        "clip_norm": 0.8,
        "visual_rms_ceiling_weight": float(rms_weight),
        "visual_rms_ceiling_cap": float(rms_cap),
    })

    coat_ctrl = ensure_coat_for_model(model, eff_cfg)
    inv_fn = coat_ctrl.invalidate_weight_caches if (coat_ctrl is not None and coat_ctrl.enabled) else None
    optimizer = optimizer_for(model, eff_cfg)
    planned_updates = max(1, math.ceil(min(max_prompts, len(pool)) / max(1, grad_accum_prompts)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=planned_updates, eta_min=float(lr) * 0.15)
    accumulator = MergedGradientAccumulator(
        model,
        accum_mode="bf16_apollo_fp32_small",
        min_bf16_accum_dim=128,
        allow_embeddings_in_apollo=bool(eff_cfg.get("apollo_include_embeddings", False)),
    )

    stats = {
        "prompts_sampled": 0,
        "rollouts_generated": 0,
        "updates_applied": 0,
        "skipped_zero_var": 0,
        "sft_anchor_fallbacks": 0,
        "dpo_feedback_steps": 0,
        "ver_exact_hits": 0,
        "ver_reward_sum": 0.0,
        "causal_delta_logp_sum": 0.0,
        "causal_rollouts": 0,
        "kl_sum": 0.0,
        "tokens_scored": 0,
    }
    window_prompts = 0

    def _apply_step(denom_prompts: int) -> None:
        lr_ratio = 1.0
        if optimizer.param_groups and float(optimizer.param_groups[0].get("initial_lr", 0.0)) > 0.0:
            lr_ratio = float(optimizer.param_groups[0]["lr"]) / float(optimizer.param_groups[0]["initial_lr"])
        step_res = accumulator.step(
            optimizer,
            scheduler,
            max_grad_norm=0.8,
            post_scale=1.0 / float(max(1, denom_prompts)),
            invalidate_caches_fn=inv_fn,
        )
        if step_res.finite and step_res.applied:
            stats["updates_applied"] += 1
            if hasattr(model, "step_tier1b_sparse_adam"):
                model.step_tier1b_sparse_adam(lr=float(eff_cfg["engram_lr"]) * lr_ratio, denom=float(max(1, denom_prompts)))
        else:
            if hasattr(model, "zero_tier1b_buffers"):
                model.zero_tier1b_buffers()

    emit(
        "stage13_vc_grpo_start",
        pool_size=len(pool),
        user_feedback=len(fb_pool),
        max_prompts=max_prompts,
        group_size=group_size,
        max_wall_hours=max_wall_hours,
        freeze_vision_projector=freeze_vision_projector,
    )

    try:
        for p_idx, (rec, blobs) in enumerate(pool):
            if p_idx >= max_prompts or time.monotonic() >= deadline:
                break

            prm = str(rec.get("prompt") or "").strip()
            meta = rec.get("metadata") or {}
            verifier = meta.get("verifier") or {
                "kind": "prism_exact",
                "gold": str((rec.get("target") or {}).get("text") or rec.get("answer") or ""),
            }
            en_think = bool(rec.get("enable_thinking", False))
            task = str(rec.get("task") or "general_vqa")
            frames = rec.get("frames") or []
            im_real: Image.Image | None = None
            med_real: dict[str, Any] | None = None
            med_blank: dict[str, Any] | None = None
            if frames and blobs.get(frames[0]):
                try:
                    im_real = Image.open(io.BytesIO(blobs[frames[0]])).convert("RGB")
                    med_real = visual_media([im_real])
                    med_blank = visual_media([Image.new("RGB", im_real.size, (128, 128, 128))])
                except Exception:
                    continue

            # 1. On-policy rollout sampling (`K = group_size`) + single-pass prefix caching for `no_grad` scoring!
            model.eval()
            max_nt = 224 if en_think else 48
            rollouts: list[str] = []
            ver_res: list[dict[str, Any]] = []
            old_lps: list[torch.Tensor] = []
            delta_lps: list[float] | None = [] if med_real is not None else None

            use_amp = torch.cuda.is_available() and str(model.device).startswith("cuda")
            amp_ng = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with torch.no_grad():
                for k in range(group_size):
                    pred_k = spark_generate(
                        model,
                        prm,
                        media=med_real,
                        enable_thinking=en_think,
                        max_new_tokens=max_nt,
                        do_sample=True,
                        temperature=0.72 if k > 0 else 0.35,
                        top_p=0.92,
                    ).strip()
                    if not pred_k:
                        pred_k = "unknown"
                    rollouts.append(pred_k)

                with amp_ng:
                    cached_real = _build_prefix_tensors(
                        model, prompt=prm, media=med_real, enable_thinking=en_think, task=task
                    )
                    cached_blank = (
                        _build_prefix_tensors(
                            model, prompt=prm, media=med_blank, enable_thinking=en_think, task=task
                        )
                        if med_blank is not None
                        else None
                    )

                kept_rollouts: list[str] = []
                for pred_k in rollouts:
                    vr = verify_rlvr_prediction(pred_k, verifier, expect_thinking=en_think)
                    lp_real_toks, n_tok, _ = _sequence_answer_logp(
                        model,
                        prompt=prm,
                        answer=pred_k,
                        media=med_real,
                        enable_thinking=en_think,
                        task=task,
                        with_grad=False,
                        cached_prefix=cached_real,
                    )
                    if n_tok <= 0 or not torch.isfinite(lp_real_toks).all():
                        continue
                    kept_rollouts.append(pred_k)
                    ver_res.append(vr)
                    old_lps.append(lp_real_toks.detach())
                    stats["rollouts_generated"] += 1
                    stats["tokens_scored"] += int(n_tok + cached_real[0].shape[1])
                    stats["ver_exact_hits"] += int(bool(vr["exact"]))
                    stats["ver_reward_sum"] += float(vr["reward"])
                    if cached_blank is not None and delta_lps is not None:
                        lp_blank_toks, _, _ = _sequence_answer_logp(
                            model,
                            prompt=prm,
                            answer=pred_k,
                            media=med_blank,
                            enable_thinking=en_think,
                            task=task,
                            with_grad=False,
                            cached_prefix=cached_blank,
                        )
                        dl = float((lp_real_toks.mean() - lp_blank_toks.mean()).item())
                        delta_lps.append(dl if math.isfinite(dl) else 0.0)
                        stats["causal_delta_logp_sum"] += dl if math.isfinite(dl) else 0.0
                        stats["causal_rollouts"] += 1
                cached_real = cached_blank = None

            rollouts = kept_rollouts
            stats["prompts_sampled"] += 1
            if len(rollouts) < 2:
                continue

            raw_r, advs = compute_vc_grpo_rewards(ver_res, delta_lps)
            coat_ctx = (
                coat_ctrl.activation_context(training=True)
                if (coat_ctrl is not None and coat_ctrl.enabled)
                else contextlib.nullcontext()
            )
            if max(abs(a) for a in advs) < 1e-3:
                gold_ans = str(rec.get("answer") or "").strip()
                if sum(v["reward"] for v in ver_res) == 0.0 and gold_ans and len(gold_ans) <= 240:
                    model.train()
                    with coat_ctx:
                        lp_g, ng, vrms = _sequence_answer_logp(
                            model, prompt=prm, answer=gold_ans, media=med_real, enable_thinking=en_think, task=task, with_grad=True
                        )
                        if ng > 0 and torch.isfinite(lp_g).all():
                            loss_g = -0.35 * lp_g.mean()
                            if vrms is not None:
                                loss_g = loss_g + rms_weight * F.relu(vrms - rms_cap).pow(2)
                            with accumulator.microbatch(weight=1.0, supervised_units=1):
                                loss_g.backward()
                            stats["sft_anchor_fallbacks"] += 1
                            window_prompts += 1
                else:
                    stats["skipped_zero_var"] += 1
            else:
                model.train()
                with coat_ctx:
                    for ans_k, old_lp_k, adv_k in zip(rollouts, old_lps, advs):
                        if abs(adv_k) < 1e-3:
                            continue
                        cur_lp_k, n_tok, vrms = _sequence_answer_logp(
                            model, prompt=prm, answer=ans_k, media=med_real, enable_thinking=en_think, task=task, with_grad=True
                        )
                        if n_tok != len(old_lp_k) or not torch.isfinite(cur_lp_k).all():
                            continue
                        log_ratio = (cur_lp_k - old_lp_k).clamp(-4.0, 4.0)
                        ratio = torch.exp(log_ratio)
                        adv_t = torch.tensor(float(adv_k), device=cur_lp_k.device, dtype=torch.float32)
                        pg1 = ratio * adv_t
                        pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
                        kl_k = (torch.exp(-log_ratio) + log_ratio - 1.0).clamp(0.0, 10.0).mean()
                        loss_k = (-torch.minimum(pg1, pg2).mean() + kl_beta * kl_k) / float(len(rollouts))
                        if vrms is not None:
                            loss_k = loss_k + (rms_weight / float(len(rollouts))) * F.relu(vrms - rms_cap).pow(2)
                        if torch.isfinite(loss_k):
                            with accumulator.microbatch(weight=1.0, supervised_units=1):
                                loss_k.backward()
                            stats["kl_sum"] += float(kl_k.detach().item())
                window_prompts += 1

            # 3. Interleave User-Feedback DPO Preference step if available!
            if fb_pool and (stats["prompts_sampled"] % 4 == 0):
                fb = fb_pool[stats["dpo_feedback_steps"] % len(fb_pool)]
                fb_med = None
                if fb.get("image_bytes"):
                    try:
                        fb_med = visual_media([Image.open(io.BytesIO(fb["image_bytes"])).convert("RGB")])
                    except Exception:
                        fb_med = None
                model.train()
                with coat_ctx:
                    lp_pos, np_t, vrms_p = _sequence_answer_logp(
                        model, prompt=fb["prompt"], answer=fb["chosen"], media=fb_med, enable_thinking=False, task="user_feedback", with_grad=True
                    )
                    if np_t > 0 and torch.isfinite(lp_pos).all():
                        if fb.get("rejected"):
                            lp_neg, nn_t, _ = _sequence_answer_logp(
                                model, prompt=fb["prompt"], answer=fb["rejected"], media=fb_med, enable_thinking=False, task="user_feedback", with_grad=True
                            )
                            if nn_t > 0 and torch.isfinite(lp_neg).all():
                                margin = 0.25 * (lp_pos.mean() - lp_neg.mean())
                                loss_fb = -F.logsigmoid(margin) - 0.15 * lp_pos.mean()
                            else:
                                loss_fb = -0.30 * lp_pos.mean()
                        else:
                            loss_fb = -0.30 * lp_pos.mean()
                        if vrms_p is not None:
                            loss_fb = loss_fb + rms_weight * F.relu(vrms_p - rms_cap).pow(2)
                        if torch.isfinite(loss_fb):
                            with accumulator.microbatch(weight=1.0, supervised_units=1):
                                loss_fb.backward()
                            stats["dpo_feedback_steps"] += 1

            if window_prompts >= grad_accum_prompts:
                _apply_step(window_prompts)
                window_prompts = 0

                if stats["updates_applied"] == 1 or stats["updates_applied"] % 10 == 0:
                    rn = max(1, stats["rollouts_generated"])
                    cn = max(1, stats["causal_rollouts"])
                    emit(
                        "stage13_vc_grpo_progress",
                        updates=stats["updates_applied"],
                        prompts=stats["prompts_sampled"],
                        rollouts=stats["rollouts_generated"],
                        exact_rate=round(stats["ver_exact_hits"] / rn, 4),
                        mean_reward=round(stats["ver_reward_sum"] / rn, 4),
                        mean_causal_delta_logp=round(stats["causal_delta_logp_sum"] / cn, 4),
                        dpo_fb_steps=stats["dpo_feedback_steps"],
                        elapsed_min=round((time.monotonic() - t0) / 60.0, 2),
                    )
                if checkpoint_dir is not None and stats["updates_applied"] > 0 and stats["updates_applied"] % 40 == 0:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    if hasattr(model, "sync_intel_vision_to_host"):
                        model.sync_intel_vision_to_host()
                    _save_model_safetensors(model, checkpoint_dir / "model.safetensors")

        # Flush any final partial gradient accumulation window
        if window_prompts > 0:
            _apply_step(window_prompts)
            window_prompts = 0
    finally:
        accumulator.close()
        del optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "sync_intel_vision_to_host"):
            model.sync_intel_vision_to_host()
        _save_model_safetensors(model, checkpoint_dir / "model.safetensors")

    rn = max(1, stats["rollouts_generated"])
    cn = max(1, stats["causal_rollouts"])
    summary = {
        **stats,
        "exact_rate": round(stats["ver_exact_hits"] / rn, 4),
        "mean_reward": round(stats["ver_reward_sum"] / rn, 4),
        "mean_causal_delta_logp": round(stats["causal_delta_logp_sum"] / cn, 4),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    emit("stage13_vc_grpo_complete", **summary)
    return summary
