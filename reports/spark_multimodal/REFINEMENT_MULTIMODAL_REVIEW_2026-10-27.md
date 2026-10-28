# Multimodal review of the SparkX v5.1 refinement proposal

Reviewed 27 October 2026. Scope is the user-supplied [24_SparkX_2_6_v5_1_Refinement_Plan.md](../../24_SparkX_2_6_v5_1_Refinement_Plan.md), sections 2 and 4, the multimodal portions of sections 7–8 and their relevant references. The separate project's native memory, nested learning, model editing and online optimizer designs are excluded. This review changes documentation only; it does not certify implementation, revise the trainer or launch model/API work.

## Decision

Adopt the proposal's stronger visual input, explicit media clock, staged alignment/full fine-tuning, evidence-linked outputs and modality-ablation evaluation. Modify the token arithmetic, baseline description, frozen-stage replay claim and audio scope. Keep grounded GRPO as a separately budgeted experiment after the supervised model is competent. Do not import the other project's online optimizer-reset or memory mechanisms into resumable full fine-tuning.

| Proposal item | Decision | Required interpretation for this project |
|---|---|---|
| So400m NaFlex, thumbnail and detail tiles | Adopt with corrected budgets | Selected pinned NaFlex tower; count real patch grids and post-reducer tokens |
| Spatial pixel shuffle + two-layer MLP | Modify terminology and masks | 2×2 spatial-to-channel **downshuffle/unshuffle**, four 1152-channel states → 4608 channels → Spark width2560 |
| Text timestamps and two-second AV chunks | Adopt as a proposed format | Preserve original timing and availability; no future chunks during streaming evaluation |
| 12.5-Hz audio reference | Adopt as a candidate, not a native encoder property | Learned rate adaptation, with25-Hz comparison for fine acoustic perception |
| Image → video → audio → AV curriculum | Adopt | Warmup projectors first; full Spark updates in principal multimodal stages |
| Text share in a frozen-backbone F1 stage | Modify | Pure-text replay does not train a frozen Spark or unused multimodal projectors |
| Point/box/interval evidence schema | Adopt multimodal subset | Explicit coordinate/time conventions and source-feature linkage; memory-record kind excluded |
| Grounded GRPO / T-GRPO | Defer | Competent SFT baseline, trustworthy rewards, multiple-rollout resource profile and separate budget first |
| Native memory and nested self-learning stages | Exclude | Belong to the user's other project |
| Speech and general sound output | Retain from original scope | Proposal mainly covers understanding; output branches and their data/evaluation remain mandatory |

## Correct the vision baseline and token accounting

The proposal repeatedly calls the old baseline **SigLIP2-base224px/64 tokens**. This workspace's qualified encoder is **`google/siglip2-base-patch16-256`**, revision **`3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`**: **256px,16×16 patches,256 raw patch tokens**. The technical qualification's grid8 path reduces that to64 decoder tokens; that technical token reduction must not be confused with the encoder's native input size or full raw grid. Correct G-V1 to compare against the actual256px baseline and name the reducer configuration. A14-example memorization diagnostic tests learnability only; it does not establish generalization or formal corpus quality.

The selected new checkpoint is **`google/siglip2-so400m-patch16-naflex`**, revision **`cc24074f717b612951c2dead130904ab9b65a81e`**. Its processor defaults to **256 raw patches**. For a square16×16 patch grid, **2×2 downshuffle gives64 tokens**, and4×4 would give16. Therefore the proposal's “about256 tokens per crop via2×2 or4×4 shuffle” is incomplete: 256 reduced tokens need1024 raw patches with2×2 reduction, or4096 with4×4. Neither is the default NaFlex path. [Pinned NaFlex processor](https://huggingface.co/google/siglip2-so400m-patch16-naflex/blob/cc24074f717b612951c2dead130904ab9b65a81e/preprocessor_config.json).

Use the bounded design in [VISION_ENCODER_REVIEW_2026-10-27.md](VISION_ENCODER_REVIEW_2026-10-27.md): aspect-aware thumbnail≤256 raw patches,0/2/4 detail tiles, and an initial ceiling of **768 actual post-reducer visual tokens per image sample**. Square tiles at256 or576 raw patches yield64 or144 tokens. An arbitrary thumbnail can need up to128 reduced tokens after odd-grid padding; four144-token tiles plus that thumbnail fit within704. Extra patch/tile configurations are candidates to profile, not established safe training settings.

NaFlex returns patchified pixels, `pixel_attention_mask` and per-sample `spatial_shapes`. Build the real Hpatch×Wpatch grid before downshuffle; do not reshape padded sequences by square root. Preserve crop/resize transforms, validity masks, 2D coordinates and tile boundaries. This matters directly for point/box supervision. The existing Transformers4.57.3 includes this interface; actual masked forward/backward behavior with this project's Unsloth integration still requires qualification. [Pinned implementation](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/siglip2/modeling_siglip2.py).

The proposed4608→2560→2560 two-layer MLP contains approximately **18.35M weight parameters**, plus biases and normalization, consistent with the proposal's18.4M estimate. This count is for2×2 downshuffle only. A4×4 version changes the input to18432 channels and substantially increases projector parameters. Encoder, projector and optimizer census must be recorded together.

Keep thumbnail/detail tile ordering and any image-span bidirectional attention as controlled ablations. A768-token visual budget can already exceed Spark's512-token local window; global layers may communicate across it, but that does not prove tile-order invariance or fine-grounding retention. Preserve an unchanged-attention control and measure effects before accepting mask surgery.

## Media clock: adopt the idea, preserve what the cited systems actually do

**Text timestamps are supported by primary evidence.** Qwen3-VL§2.3 prefixes video temporal patches with formatted time strings and trains seconds/HMS representations. However, Qwen3-VL also retains **interleaved MRoPE** in§2.1. Its textual timestamps do not prove that replacing all multidimensional positional mechanisms is equivalent. For Spark, retaining its existing positional computation and adding canonical textual timestamps is a project-specific low-surgery choice. Plain timestamp markup can tokenize into several existing vocabulary tokens; do not call it a newly trained special token. [Qwen3-VL report](https://arxiv.org/html/2511.21631v1#S2.SS3).

**Two-second AV interleave is also verified.** Qwen2.5-Omni§2.2 explicitly groups inputs by two seconds of actual time, placing visual features before audio features inside each chunk. It additionally uses TMRoPE. Copying the chunk schedule does not reproduce that model's complete temporal architecture. [Qwen2.5-Omni report](https://arxiv.org/html/2503.20215v1#S2.SS2).

Store source-media timestamps, presentation timestamps, intervals, sampling FPS, timestamp units, speed transformations and `available_at`. A two-second chunk containing observations through12s cannot be available at10s in a causal evaluation. Separate playback time from real elapsed time in time-lapse media. Random source-video windows must stay in their source video's dataset partition. For temporal sensitivity, compare ordered, reversed, shuffled, single-frame and timestamp-perturbed inputs where the question genuinely depends on order or speed.

Apollo provides primary evidence favoring FPS-based sampling over fixed-count uniform sampling for its investigated video models. This supports a bounded1–2FPS coarse semantic stream with dense local motion windows and explicit timestamps; it does not prove that1–2FPS can capture every short event. Retain a motion branch and task-specific dense windows. [Apollo paper](https://arxiv.org/abs/2412.10360), [author project page](https://apollo-lmms.github.io/).

## Audio:12.5Hz is a useful rate target, not a complete design

The proposal's rate references are verified: Qwen3-Omni's AuT produces12.5Hz input representations (about80ms/frame); Kimi-Audio combines12.5Hz semantic tokens with Whisper features adapted from50Hz to12.5Hz; Qwen2.5-Omni uses roughly40ms audio representations, equivalent to25Hz. [Qwen3-Omni§2.2](https://arxiv.org/html/2509.17765v1#S2.SS2), [Kimi-Audio§2.2](https://arxiv.org/html/2504.18425v1#S2.SS2), [Qwen2.5-Omni§2.2](https://arxiv.org/html/2503.20215v1#S2.SS2).

Adopt12.5Hz as a learned adapter's planning reference and compare25Hz on short events, overlap, pitch, loudness and sound localization. Do not relabel a SPEAR/Whisper encoder output as12.5Hz without implementing and validating the rate adapter. Preserve sample rate, channel layout, waveform time alignment and any loudness normalization; independent normalization can erase the target information in loudness-comparison tasks. ASR competence is not environmental-sound competence. English and Mandarin need separately reported speech slices, alongside general sound/music slices.

Keep Whisper as a practical speech alignment candidate; SPEAR is a candidate general-audio representation that still needs an exact checkpoint/license/interface review and an alignment stage. Do not silently replace previously selected audio components from this proposal alone.

**Restore the missing output requirements explicitly:** English/Chinese speech generation, general sound generation, native conditioning from Spark states, output codec/decoder objectives, paired output data, intelligibility/fidelity/alignment tests and gradient/update evidence. Audio understanding, ASR, audio captions and a text-to-external-TTS demo do not satisfy those requirements. The input12.5Hz choice does not determine the output codec rate, codebook count, waveform rate or decoder architecture.

## Curriculum and retention corrections

Keep projector-only F1 as an optional finite alignment warmup. In a pure-text batch with Spark frozen and media projectors unused, the supervised text loss has no trainable path to those components. Spending≥20% of F1 positions on such replay cannot preserve a backbone that is not being updated; use text-only evaluation there, or state the actual trainable path/objective if replay is retained. Frozen Spark may generate versioned retention targets locally before renting the GPU, subject to original model terms.

In F2a–d, preserve **full Spark parameter fine-tuning** with the enabled modality encoders/projectors/output branches explicitly listed. The proposal's≥25–30% text replay shares are candidate charged-position budgets; they are not validated optimal mixtures. Track actual text/media/output positions, source categories, language, unique media identities, loss masks and per-branch gradient coverage. Select mixtures using preregistered held-out retention and modality results. A two-percentage-point retention threshold needs named benchmarks, baseline uncertainty and sufficient evaluation size; it is not a universal tolerance for every metric.

Every main-training resumable checkpoint must restore optimizer, scheduler, RNG, data/shard cursor and exact parameter mapping. **Do not reset Adam momentum at instructor calls, RSI rounds or teaching events.** The proposal's reset concerns the other project's online inner loop; importing it into ordinary recovery would break continuity. Any deliberate optimizer change requires a separately named training stage and recorded lineage, rather than being called an exact resume.

## Grounded reasoning:retain the schema, gate RL

Adopt a bounded multimodal evidence schema for point, box, video interval and audio interval, with source/frame IDs, coordinates/time units, transform provenance and confidence. Omit memory-record references from this project's implementation scope. Evaluate source-feature linkage and geometry, not just valid-looking tags. Short verified evidence and final answers are sufficient targets where long reasoning traces provide no demonstrated benefit.

The proposal's broad RL direction has primary support, but its results are task/backbone-specific. Thinking with Visual Grounding improves Gemma3-4B-IT on counting/spatial tasks using point/box supervision and grounding-aware GRPO; its appendix uses **eight rollouts per prompt** and reports about **400H200 GPU-hours for training and evaluation**. These figures are evidence of nontrivial resource requirements, not a forecast for Spark or this G4. [Thinking with Visual Grounding,§5/B.1/B.4](https://arxiv.org/html/2606.16122v1).

Video-R1 uses ordered/shuffled sequences for T-GRPO and a mixed image/video cold start; this supports temporal shortcut controls. Time-R1 supports verifiable temporal-grounding rewards; cite **arXiv:2503.13377v3**, becausev1 was titled TimeZero. The audio-QA report finds no significant benefit from explicit reasoning in its studied setting, which supports leaving lengthy thinking optional rather than declaring it necessary for every audio task. [Video-R1](https://arxiv.org/html/2503.21776v1), [Time-R1v3](https://arxiv.org/abs/2503.13377v3), [Audio-QA RL reportv4](https://arxiv.org/abs/2503.11197v4).

Before a GRPO stage, require: a functioning supervised multimodal model; reliable point/box/interval validators and reviewed answers; leakage-free evaluation; inference/rollout-engine compatibility; actual peak VRAM and generation throughput; bounded rollout count/length; policy/reference/optimizer storage accounting; and matched-compute answer-only controls. Keep budget separate from the original data preparation and instructed RSI plan. Ordinary instructor-corrected supervised updates are **not GRPO**, and any training improvement is **not automatic recursive self-improvement or automatic promotion**.

## Evaluation and data admission

Adopt G-F0, corrected G-V1, G-A1, G-AV1 and G-T as proposed gates, with explicit datasets, metrics and thresholds before selection. Add speech/general-sound **output** gates and recovery/rolling-cache gates omitted from the proposal. Preserve source-grouped train/dev/promotion/audit partitions and keep protected answers outside public training shards. Compare fullAV, audio-only, video-only and text-only; evidence-dependent examples should demonstrate real modality use. Avoid requiring AV>eachunimodal on items where one modality already contains all information.

Molmo/PixMo/video, VisualCoT, structured audio reasoning and omni instruction datasets are source candidates. Each needs exact revision, annotation provenance, media-source rights, transformations, record counts, language/category coverage, split exclusions and per-source citation before admission or Hugging Face redistribution. An annotation license does not automatically license every referenced image/video/audio payload. A “research” lineage must not become the default redistributed product corpus merely because the paper is public.

Native memory gates, memory-grounded R5, F3/F3b/F4, Meta-SGD, model editing and online learning gates remain excluded. No other-project files were read or modified for this review. This document supplies decisions for revising our multimodal plan; it does not turn the whole v5.1 proposal into an authorized implementation specification.
