# Spark X2.6-4B full multimodal Colab handoff — corrected full-Git plan

## 1. Current decision

Use `colab/Spark_Formal_Full_Finetune.ipynb` with `colab/final`. The notebook clones the **full Git repository** and uses that checkout for the environment, source locks, corpus, model, tests, checkpoints, evaluation, and export. Do not use the replacement `colab/oneshot` architecture or its three-stage WSD/mixture schedule.

The target is a full-parameter multimodal experiment on one RTX PRO 6000 Blackwell with a cumulative **110-GPU-hour hard budget** and a conservative **230-decimal-GB local-disk envelope**. A one-click run can span several Colab sessions; one uninterrupted 110-hour VM is neither assumed nor required.

The authoritative design is the original campaign contract in `COLAB_AGENT_HANDOFF.md`, `reports/spark_multimodal/COLAB_CAMPAIGN_PLAN.md`, `REFINEMENT_MULTIMODAL_REVIEW_2026-10-27.md`, `HF_ROLLING_DATASET_PLAN_2026-10-27.md`, `FORMAL_NOTEBOOK_RELEASE_2026-10-27.md`, and `FULL_DATASET_RELEASE_2026-10-27.md`.

## 2. What was wrong with the replacement plan

| Replacement change | Restored contract |
| --- | --- |
| Time-based `align → main → anneal` mixture sampling | Complete staged epochs: bridge, vision, temporal/AV, audio understanding, audio generation, joint tuning. **603,668 planned exposures before RSI**, not a sampled subset. |
| Twenty rapidly added data sources | The reviewed **304,861-instruction / 315,359-view** corpus with source-grouped train/dev/promotion/audit partitions. No unreviewed expansion. |
| `colab/oneshot` as a second architecture | One `SparkNative` implementation, with narrowly scoped correctness and throughput fixes. |
| Bypass whole-model qualification and recovery | Mandatory all-family CUDA profile plus two continuation updates in a **fresh process**. |
| 12 GB free floor and 105 GB data budget | **44 GB free floor**, two measured checkpoint slots, **23 GB** bounded TAR cache, and the 230 GB envelope. |
| Trainable-only checkpoint and squashed HF history | Full model/optimizer/scheduler/RNG/cursor checkpoints. Preserve HF history for parent/candidate/control recovery. |
| One optional replay round and mean-loss rollback | **Two serial instructor rounds**, each with matched candidate and uniform-control arms from the same parent and optimizer moments. |
| Claims of an unlocated `test_e2e_pipeline.py` and unverified ingest success | Only repository tests and observed receipts count. No quality or full-GPU success is inferred from static checks. |

Useful engineering corrections were retained: media inside Spark's first user turn with `enable_thinking=False`, batched NaFlex views, cheaper video timestamp demuxing, worker-owned media readers, shared Spark text microbatches, KV-cached inference, and Drive-free HF persistence. These do not alter the staged data or promotion contracts.

## 3. Model and native objectives

- **Backbone:** `XHToken/Spark-X2.6-4B`, pinned revision `66e3ae7bee54e0a523d692cb8ef841f35409807b`. All Spark parameters train in principal stages. FP32 parameter storage, BF16 compute, AdamW 8-bit, no LoRA or weight quantization.
- **Vision:** `google/siglip2-so400m-patch16-naflex`, revision `cc24074f717b612951c2dead130904ab9b65a81e`. Real variable grids, thumbnail + bounded detail tiles, 2×2 spatial downshuffle, crop-to-original coordinates, and a hard visual-token ceiling.
- **Temporal:** pinned V-JEPA 2.1 ViT-B/16 at 384 pixels with a bounded 16-frame adaptation, semantic frames, presentation timestamps, ordered motion resampling, and separate known/unknown capture clocks.
- **Speech input:** multilingual `openai/whisper-small` encoder plus learned temporal resampling. **Sound input:** `laion/clap-htsat-unfused` plus its own temporal resampler. ASR competence is not treated as environmental-sound competence.
- **Speech output:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` talker and real 12.5 Hz / 16-codebook objective. The Spark-state bridge and learned default voice are trained. **No target-waveform speaker cue** is used. Prefix construction follows the official non-streaming SFT layout; residual codebooks use the hidden state after their own frame.
- **Sound/music output:** `stabilityai/stable-audio-open-1.0` DiT + Oobleck VAE. The bridge and DiT receive gradients into Spark. Training and inference use the pinned Diffusers **EDM** scheduler and v-prediction equations, not the replacement cosine scheduler.
- **Frozen infrastructure only:** speech codec and audio autoencoder. Input encoders, connectors, Spark, and applicable generation branches are explicitly listed in each stage's gradient/writeback qualification.

**Dependency correction:** the environment pins `diffusers==0.38.0`. The older 0.35.2 path hits the documented TorchAO 0.17 import incompatibility, and versions before 0.38.0 retain reported `trust_remote_code` vulnerabilities. Native EDM tests pass with the locked TorchAO 0.17 API.

**FlashAttention:** `requirements-flash-attn.lock` adds the separate Blackwell/CUDA-13 CuTeDSL environment (`flash-attn-4[cu13]==4.0.0b36`). It is compiled with `requirements.lock` as a constraint, so shared packages retain the main environment's versions; in particular, `datasets==4.3.0` keeps its required `fsspec==2025.9.0`. The CUTE API returns `(out, lse)`; the health check backpropagates through `out` and requires finite, nonzero input gradients. `direct_doctor` verifies `pip check`, CUTE import, and an actual BF16 backward kernel on the assigned GPU; installation alone is not treated as proof of acceleration.

Primary references: [Spark model card](https://huggingface.co/XHToken/Spark-X2.6-4B), [SigLIP2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2), [V-JEPA 2](https://github.com/facebookresearch/vjepa2), [Qwen3-TTS report](https://arxiv.org/html/2601.15621v1), [Qwen SFT code](https://github.com/QwenLM/Qwen3-TTS), [Stable Audio card](https://huggingface.co/stabilityai/stable-audio-open-1.0), [Diffusers Stable Audio](https://github.com/huggingface/diffusers/tree/v0.38.0/src/diffusers/pipelines/stable_audio), and [NVIDIA RTX PRO 6000 specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/).

## 4. Corpus decision

The existing corpus is used; no new source is added in this correction.

| Component | Media packets | Instruction variants / views | Important boundary |
| --- | ---: | ---: | --- |
| DOCCI | 8,776 | 8,776 | Prior exposure is recorded; not a fresh benchmark. |
| Open Images / Localized Narratives | 100,000 | 100,000 | Human captions and individual CC BY 2.0 attribution. |
| COCO / COCO-CN / controlled geometry | 7,192 | 42,215 | Human boxes/counts plus renderer truth; not calibrated general 3D. |
| SA-V / PLM video | 919 | 2,358 | Real clips with tracks/localization; silent. |
| CREMA-D AV | 2,100 | 2,100 | Synchronized acted affect; narrow sentences/domain. |
| LibriTTS-R / AISHELL-3 | 32,890 | 65,780 | Genuine English/Mandarin ASR and native speech targets. |
| FSD50K | 9,911 | 19,822 | Environmental sound understanding/generation; weak clip labels. |
| MusicNet / open-licensed music | 1,027 | 4,108 | Narrow classical/instrument coverage. |
| NASA/JSC time lapse | 72 | 360 | Narrow Earth-observation coverage; derivative capture clocks may be unknown. |
| Bilingual OCR | 9,671 | 19,342 | Synthetic documents; not general natural-document or calibrated-depth proof. |
| EN/ZH text replay | 20,000 article groups | 40,000 passages | Language retention, not new media diversity. |

The full source contains **304,861 original instructions**, expanded to **315,359 bounded views**, with **290,685 training views**. Splitting joins source groups, speaker/book/work/session identities, known photo/article identities, and exact media sets before variants. Promotion and audit remain outside instructor input.

The earlier `training_ready=false` handoff remains a warning about uncompleted gates, not permission to call the corpus or model qualified. Narrow AV, time-lapse, music, natural-document, and calibrated 3D-depth coverage remains an explicit limitation. Adding SAT, ChronoMagic, VoiceAssistant, Belle, AudioCaps, LLaVA-Video, or other supplements would require a separate source/revision/rights/split/throughput review; their earlier unchecked inclusion is not retained.

## 5. Disk and wall-budget enforcement

The launcher computes actual bytes before qualification and again after measuring a complete checkpoint:

| Reservation | Contract |
| --- | ---: |
| Existing environment, repository, assets, metadata | Measured once, including hardlinks only once |
| Two local checkpoint slots | Initially `2 × 44 GB`; replaced by measured complete size + 10% |
| Bounded media TAR cache | `10 × 2.3 GB = 23 GB` |
| Serialization/transfer scratch | 4 GB |
| Non-borrowable free floor | 44 GB |
| **Total** | **Must remain ≤ 230 GB and actual free capacity** |

Checkpoints include all model branches, optimizer, scheduler, RNG, exact source/data/policy binding, cumulative wall/CU accounting, and optimizer-committed cursor. Older verified HF revisions remain available in the same repository. Sanitized qualification, recovery, instructor, RSI and publication ledgers are committed under `ledger/` for replacement-VM recovery. Verification uses tree/HEAD LFS hashes and streamed SHA256 for ordinary Git blobs, so large text metadata does not require a separate repository or skipped verification. Local retention is at most two verified slots; unverified artifacts are never deleted.

The cumulative wall budget is **110 hours**. Whole-model qualification measures per-family p90 forward/backward/update cost and projects the full curriculum with a 25% transfer/evaluation/teacher/control reserve. If the projection does not fit remaining time, formal training stops before launch. Actual time is also checked at optimizer boundaries. No epoch, family, or RSI arm is silently shortened.

A session limit defaults to 22 hours. Exit `3` means a complete checkpoint exists and a later `Run all` resumes. Exit `10` means whole-model qualification completed and the notebook automatically starts the required fresh recovery process. Exit `0` requires staged training, both RSI rounds, protected audit, offline export reload, and verified sanitized public-model upload. Any other stop is a failure or an explicitly incomplete result.

Colab itself documents variable runtime limits and nonpersistent VMs; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html). This is why the flow uses local working files plus verified HF persistence, not Drive-mounted live I/O.

## 6. One-click execution contract

1. **Clone and bind source.** Clone/fetch `BaiTian6641/Spark-X2.6-VL`, verify required tracked files and a clean tree, record `HEAD`, and hash every tracked path and byte. Optional `REPO_COMMIT` accepts a full immutable commit.
2. **Install the locked environment.** `colab/final/bootstrap.py` creates/repairs an isolated CPython 3.13 CUDA environment from the hash lock, including the single reviewed SoX source-build exception.
3. **Prepare assets and original data.** Download revision-pinned model components and the original corpus without a second HF payload cache. Require the user's Stable Audio gate and a write-capable `HF_TOKEN` for account `TassP0lar`.
4. **Qualify the complete model.** Exercise every family and the real Studio loop, check finite gradients and parameter writeback, measure VRAM/throughput/checkpoint bytes, and project the complete campaign.
5. **Prove recovery in a fresh process.** Restore full model/optimizer/scheduler/RNG state and reproduce two continuation updates before restoring the original initial snapshot.
6. **Run the staged campaign.** Preserve every stage, effective eight-record optimizer batch, source-grouped order, and family/language loss accounting. Worker-prefetched media does not advance the committed cursor.
7. **Run two controlled RSI rounds.** `OPENCODE_API_KEY` is required. Each round uses 32 reviewed training records, typed replay-priority decisions, matched candidate/uniform-control replay, inherited optimizer moments, and promotion/rollback. Uncertain billing stops new requests.
8. **Audit and publish.** Run the protected audit, preserve pending human-audio review, reload the closed export offline, then upload and verify the sanitized public model. Do not infer broad capability acceptance from completion. Protected evaluation reports and account metadata are excluded from public artifacts.

## 7. Verification boundary

The focused local suite exercises actual behavior: official Spark chat layout, Qwen prefix/sub-talker alignment and cached frames, Stable Audio EDM equations, Spark microbatch and cached-generation parity, source-group splitting, target-role isolation, contiguous-prefix worker commits, checkpoint write/retry/restore/purge behavior, full-Git source hashing, and expected subprocess exit codes. These are CPU/source tests, not proof of full-model G4 residency, throughput, quality, or fresh-VM recovery.

**Observed during this correction:** **95 focused tests passed**, including direct HF lineage recovery, guarded same-VM qualification retry, CLAP reconstruction, final-publication corruption/reconciliation and large-file HEAD/stream verification, public budget restore, and Drive-free instructor durability. BF16 kernel parity checks sampled gradients with 15% relative tolerance and a 0.0005 near-zero absolute floor, plus a full-tensor gradient-norm gate. The actual Studio/TRL trainer completed initialization and one optimizer update with the ordered cursor dataset and tensor-anchored native batches. The supplemental install passed `pip check`; CUTE import and a local-HTTP checkpoint HEAD/stream smoke passed. These checks used the pinned API stack where required; no full Colab G4 campaign was launched.

The first G4 run remains the required hardware gate. A completed notebook must not be described as validated unless its actual receipts show whole-model qualification, fresh-process recovery, all stages, both RSI rounds, protected audit, offline reload, and verified upload. Human speech intelligibility/tone review, sound/music listening, broad spatial/temporal benchmarks, and independent 3D evaluation remain separate acceptance work.

## 8. Operator steps

1. Open `colab/Spark_Formal_Full_Finetune.ipynb` from branch `codex/formal-colab` on a G4 RTX PRO 6000.
2. Enable `HF_TOKEN` and `OPENCODE_API_KEY` in Colab Secrets. The HF token needs component read access and model/checkpoint write access. Accept Stable Audio's gate on the same account beforehand. `INSTRUCTOR_MODEL` defaults to **`mimo-v2.6-pro`**, which is an allowed operator selection, not an independently verified provider-eligibility or multimodal-competence claim. Returned-model and billing checks still apply. Public artifact repositories are sanitized; protected evaluation and account metadata are excluded.
3. Optionally enter the current Colab CU balance/rate. The 110-hour wall budget applies regardless; valid CU observations add a second budget check.
4. Click **Runtime → Run all**. Keep the streamed output. If exit `3` occurs, start a new G4 session and run all again. Do not start a second writer against the same work root or checkpoint repo.
5. Inspect `ONESHOT_RECEIPT.json`, `QUALIFICATION.json`, `CAMPAIGN.json`, per-stage evaluations, and `MODEL_UPLOAD_RECEIPT.json`. Do not treat an incomplete/stopped run as a completed model.

Canonical runtime: `colab/final`. Historical planning and qualification notebooks remain for evidence, not as alternate launchers.
