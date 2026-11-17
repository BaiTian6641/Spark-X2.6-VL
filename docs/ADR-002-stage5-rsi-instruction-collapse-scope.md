# ADR-002: Stage-5 Vision RSI Collapse-Gate Task Scope, `<think>` Budget Forcing, and Neutral 3D Grammar Conditioning

- **Status**: Accepted
- **Date**: 2026-11-16
- **Context**: Post-campaign diagnostic audit of `spark-x26-vl-world-v32-stage5-vision-rsi` (`VISION_RSI-1.json`, `VISION_RSI-2.json`, and `STAGE5_RSI_CAPABILITY_VERIFICATION.json`).

## Empirical Findings

1. **Non-instruction `text_replay` & multi-frame JSON arrays in `audit_promotion_generation_health`**:
   - On the held-out `promotion` split (`44` generated rows), `4` rows are raw Wikipedia mid-sentence next-token continuation slices (`task="text_replay"`, `wiki-en-*` / `wiki-zh-*`) and `2` rows are multi-frame `video_box_tracking` JSON arrays (`[{"timestamp_seconds": ..., "box_xyxy": [...], "visible": true}, ...]`).
   - Under unpenalized greedy continuation (`repetition_penalty=1.0`), raw `text_replay` continuations exhibit 4-gram repetition `0.50–0.92` even on the untouched Stage-5 parent checkpoint (`parent_generation_health.healthy_rate = 0.7955 < 0.88`, `mean_rep_4gram = 0.1078 > 0.085`), making the unfiltered `0.88` gate fail on `parent` itself.
   - Across the `38` actual instruction-following VL, SyncCraft, 3D spatial, and reasoning rows:
     - `vision-rsi-1-parent`: `healthy_rate = 35/38 = 0.9211`, `mean_rep_4gram = 0.04515`, `synccraft_oracle_rate = 0.0714`.
     - `vision-rsi-1-candidate`: `healthy_rate = 36/38 = 0.9474`, `mean_rep_4gram = 0.03588`, improved loss on 4/6 strata (`image:en`, `text:en`, `video:en`, `text:zh`).
     - `vision-rsi-2-candidate`: `healthy_rate = 35/38 = 0.9211`, `mean_rep_4gram = 0.05120`, improved loss on 4/6 strata, and **doubled** `synccraft_oracle_rate` (`0.0714 -> 0.1429`).

2. **Unbounded `<think>` self-checking under `repetition_penalty=1.0` greedy decoding**:
   - In `inference.generate(..., enable_thinking=True)`, pure `repetition_penalty=1.0` without a `<think>` token budget allows multi-step graduate science (`GPQA-Diamond`, `HLE`) and algorithmic coding (`BFS`) traces to exceed `max_new_tokens` before emitting `</think>`, truncating the final answer even when the reasoning derivation itself is non-collapsed and accurate.

3. **One-shot Hound Exemplar Bias in `SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT`**:
   - Embedding a verbatim 85-line `SENTINEL_HOUND_V32_DSL` inside `SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT` shifted `multiview_inverse_dsl` visual token RoPE positions away from training (`SYSTEM_PROMPT`) and biased greedy decoding toward copying the hound silhouette on non-quadruped multiview inputs.
   - Additionally, deterministic `do_sample=False` across `attempt >= 1` repair calls caused identical token emission when repairing global AST identifier collisions (`duplicate identifier: chest_armor` between `@palette` and `@rig`).

## Decisions

1. **Instruction-Scoped Anti-Collapse Gate (`colab/final/vision_rsi.py` & `colab/final/evaluate.py`)**:
   - Scope prose/reasoning 4-gram loop gating (`NON_PROSE_COLLAPSE_TASKS = {"text_replay", "video_box_tracking"}`) to instruction-following rows while still requiring non-empty string outputs across all generated rows, and give `synccraft_*` evaluations `max(evaluation_max_new_tokens, 1536)` tokens so multi-part Block-DSL assets are not cut off mid-rig.
   - Preserve the original preregistered Stage-5 export (`work/spark-world-v32-stage5-rsi/final-model`) intact and export the cold-store `vision-rsi-2-candidate` (`checkpoint-ab875b8dc14a97c1`) side-by-side (`final-model-rsi2` & `final-model-rsi2-fp8`) for transparent 3-way comparison (`Stage-4` vs `Stage-5` vs `Stage-5 + Vision RSI-2`).

2. **KV-Cached `<think>` Budget Forcing & Mild Thinking Repetition Penalty (`colab/final/model.py` & `colab/final/inference.py`)**:
   - Add `max_think_tokens: int | None = None` to `SparkNative.generate_text_only_ids` and `SparkNative.answer` (injecting `\n</think>\n` cleanly into the KV cache if `max_think_tokens` is reached without `</think>`), leaving `model.answer` defaults unchanged for exact zero-tolerance `verify_text_only_parity` checks.
   - In `inference.generate(..., enable_thinking=True)`, default `repetition_penalty` to `1.05` and `max_think_tokens` to `min(768, max(64, int(max_new_tokens * 0.55)))` when `max_new_tokens >= 128`.

3. **Geometry-Neutral Grammar Schema & Visual RoPE Preservation (`colab/final/world_pipeline.py`)**:
   - Replace the verbatim hound in `SYNCCRAFT_V32_GRAMMAR_SYSTEM_PROMPT` with a compact, geometry-neutral schema (`<primary_mat>`, `<root_bone>`, `<core_box>`) and explicit global identifier uniqueness rule (`_mat` suffix on `@palette` entries).
   - Keep `system_prompt=None` whenever `images` are present (`has_img=True`) so visual token RoPE positions match Stage-4/5 training exactly, appending compact grammar rules after the user prompt instead.
   - Use mild sampling (`do_sample=True, temperature=0.35 + 0.10 * min(attempt, 3), top_p=0.92, repetition_penalty=1.05`) on `attempt >= 1` repair iterations plus targeted `duplicate identifier` repair instructions.
