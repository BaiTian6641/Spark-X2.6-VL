# Spark multimodal Colab campaign

Prepared 27 October 2026. This is the revised implementation and execution plan for the complete English and Chinese multimodal prototype. Infrastructure preflight has passed. The separate Stage 1 notebook now executes the actual Unsloth Studio backend on CUDA and has passed a bounded image/frame full-parameter training and process-resume test. Native audio, dense temporal integration, the formal corpus and instructed RSI still require the gates below before the end-to-end campaign can launch. Stage 1 is a subset of C1/C3, not completion of those whole-model gates.

## Decisions and intended result

Use Colab G4 compute and Google Drive as the permanent artifact store, with ephemeral local disk for active training. Allocate at most **2,000,000,000,000 bytes (2 TB decimal)** of Drive to this project and **US$50** to the instructor campaign. The successful preflight recorded a proposed 1,000-CU project allocation. At approximately 12:45 CDT on 27 October 2026, the connected Chrome Resources panel showed **1,110.18 available units**, **approximately 8.9 units/hour**, **one active session**, and the **Colab Pro+** label. This point-in-time observation covers the proposed 1,000-unit allocation; it does not establish the identity of a subscription purchase or allocate the rest of the account balance. Refresh the balance/rate before later sessions and never sum grants automatically.

**Keep every existing workstation checkpoint local.** Do not upload checkpoint 2751, its alignment parent, optimizer states, the local environment or the local secret file. Start the new training lineage from the original pretrained **XHToken/Spark-X2.6-4B**, downloaded directly inside Colab at a pinned revision. Download the selected pretrained encoders/generators there too. This is fresh multimodal alignment of pretrained components, not language-model pretraining from random weights. Existing evaluation reports can accompany the handoff as historical evidence; they are not new CUDA baseline measurements.

Pin Spark to **`66e3ae7bee54e0a523d692cb8ef841f35409807b`**, the revision recorded by the local experiment. Its public repository metadata and custom configuration/model source remain accessible, verified on 27 October 2026 in `evidence/colab-original-spark-revision.json`. The current upstream head has changed; do not silently follow `main`. Stage 1 subsequently downloaded and verified the actual Spark and SigLIP weights inside Colab against the pinned per-file LFS SHA-256 or Git blob identities in `colab/stage1/model_sources.json`.

The completed target includes:

- English/Chinese image understanding, OCR, points, boxes, counts and evidence-grounded spatial relations.
- Real video with entity identity, tracks, occlusion, event order and time lapse, distinguishing media timestamps from physical capture time.
- English/Mandarin speech understanding, environmental sound understanding and synchronized audiovisual questions.
- Native speech output and short general-sound generation conditioned on Spark hidden states, including Chinese instructions.
- Two bounded instructed recursive-improvement rounds, with independent verification, replay, controlled comparisons and rollback.
- A complete reloadable model package, resumable optimizer checkpoint, capability report and evidence for each claimed branch.

A single campaign can span several Colab sessions. Its controller resumes completed stages and committed checkpoints. It cannot guarantee one uninterrupted VM, a quality pass within one subscription grant, or recovery of work after the last committed checkpoint. If a gate or budget fails, export the best validated checkpoint with an explicit partial-capability report. Do not label an image-only result as the complete target.

## What the workspace documents establish

Read together: [architecture and curriculum](REPORT.md), [current implementation](IMPLEMENTATION_STATUS.md), [formal training](FORMAL_TRAINING.md), [checkpoint 2751 evaluation](EVALUATION_CHECKPOINT_2751.md), [operations](FORMAL_RUNBOOK.md), [RSI design](RECURSIVE_SELF_IMPROVEMENT.md), [instructed RSI](INSTRUCTED_RECURSIVE_SELF_IMPROVEMENT.md) and [teacher qualification](TEACHER_QUALIFICATION.md).

The implemented model is Spark plus the headless SigLIP 2 vision tower and learned position/media connectors: 4,201,669,760 parameters. Current full stages store parameters in FP32, compute with BF16 AMP without autocast weight caching, and use AdamW8bit with FP32 state exceptions. The paused ROCm run consumed 22,008 records and peaked at 44.04 GiB allocated and 46.57 GiB reserved. It proved execution and checkpointing; grounding gates did not pass. Chinese retention perplexity regressed 14.22% on the fixed diagnostic bank. The earlier 1,024-record alignment is not sufficient evidence of competent visual conditioning.

The historical trainer remains unchanged: `scripts/train_multimodal_studio.py` explicitly asserts `torch.version.hip`, and `scripts/env.sh` contains workstation paths. `colab/stage1/port_trainer.py` creates a separate guarded CUDA entrypoint and strengthens CUDA resume bindings without editing that historical source. The tested Stage 1 runtime uses Python 3.13.16, CUDA PyTorch 2.11.0, Transformers 4.57.3, and the official Unsloth/Studio 2026.10.9 wheel in a local virtual environment. The full Studio web UI extra has incompatible dependency requirements; the actual headless Studio trainer is installed and its source hash is checked. xFormers falls back to PyTorch attention because the available binary targets older Torch/CUDA. Exact source/runtime/data identities remain required for resume.

The live qualification reached step 8, saved a complete 27.36-GB checkpoint, resumed in a new process and reached step 10. It verified all 4,201,669,760 parameters were trainable and covered by the optimizer, real media-to-Spark gradients, sampled component updates, and sampled optimizer/scheduler/RNG restoration. The first trainer interval took 18.46 seconds; post-warmup short-fixture steps averaged 0.251 seconds, with 42.12 GiB peak allocated GPU memory. Initial local checkpoint serialization took 73.66 seconds. This is technical image/frame evidence; no formal quality, audio/video capacity, uninterrupted-trajectory equivalence or fresh-VM recovery claim follows. Receipts are under `evidence/colab-stage1-*.json`; the live notebook and operation details are indexed by `colab/README.md`.

The first large checkpoint upload exposed an HTTP transport bug after a small-file preflight had passed. The notebook now uses Google's `build_http()` transport so HTTP 308 is handled as a resumable-upload status. The existing file IDs and private upload session were retained for recovery. This demonstrates why the large-checkpoint test is necessary before formal training; only a successful final `COMMITTED.json` receipt establishes a durable checkpoint. [Google API Python client transport implementation](https://github.com/googleapis/google-api-python-client/blob/main/googleapiclient/http.py).

The corrected commit subsequently passed: all 26 payloads, totaling
27,359,363,974 bytes, were verified remotely before the final commit manifest.
The successful commit operation took 551.28 seconds including local hashing,
transfer and verification. The final qualification receipt and commit manifest
were downloaded and SHA-256 verified in the workspace. Their IDs/hashes and the
private checkpoint folder are recorded in `colab/README.md` and
`evidence/colab_stage1_drive_release.json`. This establishes a durable uploaded
checkpoint; recovery of that checkpoint in a new VM remains untested.

The older `multimodal/README.md` contains historical S0 paths and pooled-grid descriptions. Use the implementation status, formal protocol and actual model configuration for current behavior. The 80k multimodal expansion in the original plan is different from the 79,706-record image/frame corpus that actually ran.

## Subscription and budget accounting

Google's current help page lists **1,000 or 2,000 Colab compute units for Ultra**, depending on the tier, plus background execution and premium GPU access. Its storage description currently says **20 or 30 TB**, not 10 TB. The account checkout and Colab Subscription/Resources panels settle the actual entitlement; free trials and ordinary family membership do not automatically confer the manager's paid Colab benefit. The authorized project allocation remains 2 TB even if the account has more. [Google AI Ultra benefits](https://support.google.com/googleone/answer/16286513?hl=en).

At the user's observed **8.71 compute units per G4 hour**:

| Allocated units | Arithmetic maximum G4 runtime | With 15% held back for recovery and uncertainty |
|---|---:|---:|
| 500 | 57.41 hours | 48.79 hours |
| 1,000, if the grant is active and allocated | 114.81 hours | 97.59 hours |
| 1,500, only if both grants are actually available and allocated | 172.22 hours | 146.38 hours |

These include the entire assigned GPU runtime: installation, compilation, downloads, idle API waits, training, checkpoint transfer and evaluation. They are not training-throughput estimates or a promise that G4 stays at this rate. Google subscription benefits can stack, but do not assume an upgrade retains a previous monthly grant or doubles an advertised allowance. [Google announcement](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/), [Colab FAQ](https://research.google.com/colaboratory/faq.html).

At the later live rate of approximately **8.9 CU/hour**, 1,000 allocated units cover about **112.36 hours**, or **95.51 hours** after a 15% reserve. The 94-hour planning schedule below must therefore be re-fitted after complete multimodal profiling; the earlier 8.71-CU arithmetic is retained as the original planning scenario.

Before every session, record the visible balance, current hourly rate, active assignments and subscription. Use the official Colab CLI's documented `colab usage` if it is separately authenticated; otherwise record the Resources panel values. Account-level usage includes other active runtimes. Do not scrape undocumented account credentials or bypass timeouts. [Official CLI usage documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md).

For a conditional 1,000-unit campaign, the following are **planning allocations to fit after profiling**, not measured durations:

| Work | G4-hour planning allowance |
|---|---:|
| CUDA compatibility, whole-model profiling and fresh-session recovery | 4 |
| Baselines and GPU-dependent teacher qualification work | 2 |
| Stronger connector and input alignment | 10 |
| Formal image, OCR and spatial full fine-tuning | 20 |
| Temporal/video full fine-tuning | 12 |
| Speech/sound input and AV alignment | 10 |
| Speech and general-sound output integration training | 14 |
| Joint full-parameter consolidation | 8 |
| Two RSI rounds and matched continued/static control work | 8 |
| Final evaluation and export | 6 |
| **Scheduled work** | **94** |
| **Remaining runtime allowance** | **20.81** |

CPU-only corpus construction and instructor requests should be completed before assigning a G4 where possible. Actual timed batches and transfers determine whether a complete fixed corpus fits. If a stage cannot fit its complete-epoch protocol, revise the campaign budget before launch; never silently shorten its corpus or call an unfinished epoch complete. The 500-unit profile is a smaller initial tranche, not a claim that all branches fit in 57 hours. The original plan already anticipated several weeks of integration work; additional units do not implement missing architecture.

## Drive and local disk design

The user's latest storage instruction is explicit: prioritize local disk and use
Drive as permanent storage with minimal random read/write. The new preparation
release implements deterministic 1-GiB local TAR packages, sequential resumable
Drive commits with the final commit marker written last, and verified local
restore/extraction. It rejects mounted Drive as a working path and requires
archive plus extracted payload space plus a 40-GiB reserve. Source R3 and the CPU
notebook have been uploaded privately; 18 focused host integrity tests passed.
An actual Drive round trip of these new dataset helpers remains pending.
See [the concrete implementation](../../colab/formal/README.md),
[storage contract](../../colab/formal/storage_policy.json) and
[release record](../../evidence/formal-v2-storage-release.json).

The first revised source acquisition is running on workstation CPU, targeting
102,000 distinct PixMo-Cap images. It is an English detailed-caption alignment
source, not the full bilingual multimodal mixture. The caption restore helper
requires that this source fit on local disk. A future bounded rolling loader
must stage video/audio shards locally, mix categories/tasks across staged shards,
and checkpoint exact sample/shard order; it has not yet been implemented.
Google similarly advises copying archives from Drive and unpacking on the VM,
because many small mounted-Drive reads can fail or throttle.
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

The private project folder created for this handoff is [Spark Multimodal Colab](https://drive.google.com/drive/folders/1ELmWWHxpCb3IGM2LZDQV-u7NiLxK12T_). Its metadata showed owner-only access and `shared=false` at preparation. Notebook authorization is separate from this desktop connector's authorization.

Use stable Drive folder/file IDs for transfers. A readable folder layout is:

```text
Spark Multimodal Colab/
  handoff/                       source and plan releases, with hashes
  assets/models/<repo>/<sha>/     pretrained downloads, configurations and licenses
  datasets/<version>/            immutable manifests and media shards
  teacher/<campaign>/            sanitized requests, results, verified targets and costs
  campaigns/<campaign>/
    protocol/                    pinned stage graph, budgets and split identities
    environment/                 resolved lock, wheels manifest and runtime receipts
    sessions/<uuid>/             local progress, CU observations and recovery receipts
    checkpoints/<stage>/<uuid>/  immutable payloads and final commit receipt
    evaluations/                 baseline, development, promotion and final audit
    exports/                     best validated inference and resumable packages
  preflight/                     bounded Drive transport probes
```

The folders are an organization convention, not an access-control sandbox. Drive mount authorization can expose broader Drive contents to notebook code. Only reviewed project code receives access; the teacher receives allowlisted media and summaries, not filesystem access or arbitrary executable tools. Keep the final test bank unavailable to training/teacher workers and give it only to the separate final evaluator.

Use `/content/spark_mm` for the code, isolated Python environment, active source shards, decoded caches and newly written checkpoints. Store `HF_HOME`, Torch/Triton compilation caches and temporary files locally. Google specifically recommends copying archive files to the VM and extracting locally when mounted Drive small-file operations are problematic. [Colab Drive guidance](https://research.google.com/colaboratory/faq.html).

Do not train directly against tens of thousands of Drive JPEG/WAV files; do not place a virtualenv, live optimizer tensors, a SQLite database or a hot Hugging Face cache on the mounted filesystem. Use immutable media shards, initially about 1–2 GiB each, copied and checksum-verified before extraction. Keep only the current/next shard group locally. Do not compress already compressed JPEG/video/audio indiscriminately. Sharding must preserve exact sample order, source identities and transform metadata for resume.

Use the Drive API's resumable upload protocol for large model files, with bounded chunks and remote size/checksum verification. Mounting Drive is optional for browsing small configuration files. A completed mount write or same-session readback is not by itself a tested fresh-session restore. [Drive upload API](https://developers.google.com/workspace/drive/api/guides/manage-uploads).

At the user's request, the qualification notebook also includes the standard
`from google.colab import drive; drive.mount('/content/drive')` cell after the
verified checkpoint commit. It does not change the active local training paths.

### Project storage envelope

The following are capacity reservations in decimal GB, not existing downloads:

| Purpose | Maximum planning allocation |
|---|---:|
| Pinned pretrained components and environment artifacts | 160 GB |
| Curated image/document data | 120 GB |
| Selected video and time-lapse data | 500 GB |
| Selected speech and sound data | 300 GB |
| New complete checkpoints and selected exports | 300 GB |
| Instructor receipts, evaluation media and reports | 40 GB |
| In-progress uploads and temporary durable staging | 250 GB |
| Unallocated headroom | 330 GB |
| **Total project allocation** | **2,000 GB** |

Drive capacity does not enlarge Colab local disk. Compute a stage-specific local requirement from active model files, data working set, decoded cache, two checkpoint snapshots, transfer scratch and 20% headroom. Profile the small probe first; a 140 GiB local-free target and 500 GiB Drive-free target are conservative starting checks, not universal requirements for every stage. Do not download whole video collections to satisfy a row quota.

### Checkpoint commit and recovery

The old image model's resumable bundle was about 25.5 GiB; future audio-inclusive checkpoints will be larger. Set cadence from measured checkpoint/write/upload time, not the old 250-step interval. Start with a 90-minute durable checkpoint target and stage boundaries. Increase the interval if transfer cost is excessive, and explicitly report the resulting recovery-point exposure. At 20 MiB/s, a 25.5 GiB payload alone takes about 21.8 minutes; at 100 MiB/s it takes about 4.4 minutes, before verification overhead. These are transfer scenarios, not measured Drive speeds.

1. Save locally at an optimizer boundary into a unique snapshot directory. Include all model branches, optimizer/scheduler, AMP state if applicable, RNG, sampler state, shard cursor, exposure counters, exact architecture/processor/tokenizer, source and environment hashes, stage contract and cumulative budget ledgers.
2. Close the snapshot and compute a manifest of file sizes and SHA-256 digests. Preserve tied parameter aliases. A weights-only file is not a resumable checkpoint.
3. Upload immutable payload files under unique Drive IDs. Persist resumable-upload state privately; upload session URLs are sensitive and must not enter public logs. If the original local bytes disappear with the VM, an incomplete upload cannot reconstruct them: fall back to the last committed checkpoint.
4. Verify remote size and provider checksum. Publish a small `COMMITTED.json` receipt only after every required payload is present. The receipt binds payload IDs, hashes, stage/step and parent identity. Treat a mutable `latest.json` as a hint only; validate the referenced immutable receipt and payloads before using it.
5. Retain two latest complete checkpoints, the best validated checkpoint and the current RSI parent/control as needed, subject to the project cap. Proposed future retention applies only to this new campaign, after remote verification. No old local checkpoint deletion is authorized here.
6. Before long training, terminate a test runtime, reconnect, rebuild the environment from the lock, download a committed test checkpoint, verify SHA-256 and restore the next sample IDs, optimizer/scheduler counters and RNG. Compare a short uninterrupted trajectory against stop/resume under the same CUDA runtime, with prespecified numerical tolerances. Record full numerical differences rather than claiming cross-platform bitwise identity.

Only one writer may own a campaign. Notebook execution must reject a second active owner; Drive directory names and local `flock` do not provide distributed locking. The runner needs a conditional ownership protocol or explicit operator-controlled session handoff. Each session has a unique ID. On upload failure, keep the previous committed pointer, stop expansion and preserve the complete local snapshot if possible. Retry transport with bounded backoff, never treat a partial directory as recovery.

Use conservative 8–10-hour session chunks until actual entitlement is observed; with confirmed longer background execution, extend chunks while leaving measured upload and shutdown time. Background execution does not make VM storage permanent. Reconnection and account consent may still require the user. No keepalive or quota-evasion mechanism belongs in the notebook.

## Environment and model portability gate

The notebook preflight intentionally does not replace Colab packages. It records the base Python, GPU, driver, CUDA/Torch versions and storage state. Build the training runtime in a separate local virtual environment and invoke that Python explicitly from notebook cells/subprocesses.

1. Verify the assigned accelerator using `nvidia-smi` and PyTorch. Target one RTX PRO 6000 Blackwell with at least approximately 90 GiB reported memory, available BF16 and CUDA rather than HIP. Google Cloud's G4 family also has fractional configurations; the G4 label alone is insufficient. [Google G4 specifications](https://docs.cloud.google.com/compute/docs/gpus#G4).
2. Resolve a CUDA PyTorch/torchvision/audio stack and Unsloth/Studio revision supporting the assigned GPU. Pin Python minor version, CUDA wheel build, driver observation, Torch, Triton, Unsloth, Zoo, Studio, Transformers, TRL, bitsandbytes and decoding libraries. Verify imported file paths and `pip check`. Archive the final lock and wheel hashes only after successful qualification. An untested list of version guesses is not a lock.
3. Preserve the custom Spark attention, cache positions, tied embeddings and FP32 parameter policy. Start from the known Transformers 4.57.1 Spark behavior. The original plan identifies a 4.57.3 requirement in Qwen3-TTS; resolve and test a common version or isolate component adapters without detaching the training graph. Do not combine incompatible dependencies by copying the ROCm overlay or using unchecked `--no-deps` installs.
4. Install the actual Studio backend used by `UnslothTrainer`, not merely the Unsloth library. Pin its source/wheel identity and test its imports. Standard Colab is not assumed to support a privileged Docker daemon. Use an isolated Python runtime; Docker is an alternative only if the assigned platform supports it.
5. Port the HIP-only assertion, configurable model/data/output roots, hardware telemetry and stop handling in a separate source release. Keep the original ROCm source intact. Preserve the tested custom collator, alias-aware save/load and loss-mask behavior. The notebook uses the Studio backend directly; it does not claim submission through Studio's UI.
6. Fetch original pretrained assets by pinned repository revision and verify hashes. Capture required custom code and original licenses. Derive the headless SigLIP tower reproducibly using `prepare_vision_component.py`; its pinned upstream revision is already recorded in that script. No local trained checkpoint is an input.
7. On real small samples, compare reference and optimized forward/loss paths, finite backward gradients and actual changes for every declared trainable group. Use microbatch 1 initially. Keep BF16 compute and FP32 storage until another precision policy earns a separate numerical/quality result.
8. Run 5–10 warm-up and at least 30 timed steps per active task family, then a representative mixed cycle and complete save/restore. Measure actual optimizer-state bytes, allocated/reserved memory, host available RAM, local free disk, sequence/codec targets and media duration. Batch size, resolution or sequence extensions require another profile.

Unsloth documents RTX PRO 6000/Blackwell support, but that does not qualify this custom Spark wrapper or native audio losses. [Unsloth Blackwell guide](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth.md).

## Architecture implementation requirements

Preserve the component choices and interfaces in the main plan:

| Branch | Implementation target | Evidence required before training scale-up |
|---|---|---|
| Images | Headless SigLIP 2 plus spatial/resampling connectors into Spark's 2,560-dimensional input | Correct masks/coordinates, image-conditioned answers and bilingual retention |
| Video | SigLIP semantics plus V-JEPA 2.1 base temporal features, persistent entity IDs and two clocks | Track/occlusion checks; reversal/shuffle controls; capture-time unknown handling |
| Speech input | Multilingual Whisper-small encoder plus learned temporal resampler | English WER, Mandarin CER and speech QA using actual waveforms |
| Sound input | Qualified BEATs checkpoint or a separately reviewed sound encoder | Permitted checkpoint and sound-event/temporal controls; no implicit substitution |
| Speech output | Qwen3-TTS pretrained talker/codec with a learned Spark-hidden-state bridge | Codec objective reaches Spark; intelligibility, Chinese tones and detached-bridge ablation |
| Sound output | Accessible pretrained short-sound generator, initially Stable Audio Open Small candidate | Supported training objective and conditioning; finite gradients into Spark; bilingual prompt-match audio |
| Joint model | One versioned package and task router with alternating applicable losses | All designated parameter groups update over a complete task cycle and reload together |

Spark's sliding/full attention pattern and 512-position sliding window require actual media-access tests. Its final decoder-state access needs an explicit tested implementation for output conditioning. Adding 3D position IDs does not implement spatial-temporal RoPE. Keep the original backbone positional scheme initially; nested-learning and attention redesign stay with the separate architecture project.

Use the original video budget as a candidate: 8 frames × 64 semantic positions, about 256 motion and 128 event/time positions, with the rest of a 2,048-position budget for language. This requires new compression code; the current wrapper supports up to four dense frames and must not be relabeled as the eight-frame path. Increase resolution only after checking target visibility and coordinate transforms.

The estimated whole system is 5.5–6.5B resident parameters, pending an actual census. Freezing codecs/autoencoders remains an explicit infrastructure exception. Freezing Spark in initial bridge alignment is allowed; the main full stages and RSI updates train all Spark parameters and all applicable designated components. A frozen-feature cache is valid only while its encoder is frozen. A text response forwarded to an external TTS service is a comparison baseline, not native speech integration.

## Revised data contract

Expand the vision curriculum to 100k–200k distinct-image alignment pairs and 200k–300k curated image/spatial/document instructions, subject to measured complete-stage budgets. Start with DOCCI/COCO/COCO-CN and vetted PixMo captions, pointing and questions. Add verified relations and document/OCR targets. Distinguish human annotation, LLM-edited human descriptions, reviewed translation, renderer truth and teacher estimates. File/source license and annotation terms travel with the manifest.

Target approximately 50:50 English/Chinese within reasoning tasks, while also reporting unique source media and supervised positions. Do not inflate Chinese coverage by repeatedly translating a few captions or attach a translated transcript to an unchanged waveform and call it Chinese audio. Sample images/source groups before instruction variants, normalize overlapping category labels, preserve unmapped examples, gate positive quotas on query polarity and cap repeats. Keep count support balanced across sources where correct labels are available.

Use selected Molmo2 video description/QA, point and track data for real spatial-temporal tasks; preserve raw-to-annotation timestamps and uncertainty flags. Controlled renders support geometry and counterfactuals but cannot replace real clips. Genuine time-lapse examples need actual capture metadata for physical-duration targets; otherwise the target must say elapsed capture time is unknown.

Use LibriSpeech/AISHELL-1 for speech input, LibriTTS-R/AISHELL-3 for output and a curated FSD50K subset for sound events. Sound generation needs waveform-checked descriptions. Keep the same original recording/book, speaker/session, video, image family and all transformed/translated variants in one split. Preserve official evaluation splits and check cross-corpus overlap.

The native audio/video expansion follows the main plan's quality-first pools, but freeze its actual audio hours, unique speakers/clips, target positions and full-stage epoch counts after ingestion and profiling. The earlier 10k proof and 80k expansion are not sufficient grounds to claim broad quality. Per-family counts are not interchangeable with image-row counts.

Maintain separate train, diagnosis development, promotion and final audit manifests. Teacher-assisted curation can use training media and diagnosis examples to identify weaknesses; final audit items and labels are excluded. Newly constructed items cannot reuse diagnosis/promotion/test media as training examples. Review 5–10% per stratum and a critical 500-example set as in the original plan; high-impact unresolved audio semantics need qualified listeners, not a self-declared automatic pass.

## Instructor and two-round RSI contract

Preferred instructor remains OpenCode `mimo-v2.6-pro`, stored as `OPENCODE_API_KEY` in Colab Secrets. The local `runtime/secrets/opencode.env` stays local. Use Colab's `userdata.get` only in the interactive notebook, then pass the value in process memory to the client. Do not serialize it, include it in notebook output, archive the environment, or mount Drive as a secret store. Optional Hugging Face access uses a separate `HF_TOKEN` secret only if required.

The prior Go endpoint passed small image/video/audio fixtures, not broad instructor competence. Its documentation describes coding-agent traffic. Confirm that the selected product/route allows the proposed batch teaching workload; a successful HTTP response is not that confirmation. Do not impersonate another client, silently switch models or assume a Go subscription is a pay-as-you-go account. If route eligibility is unresolved, complete independent preparation and stop instructor production at that gate. [OpenCode Go](https://opencode.ai/v2/docs/console/go/), [Console models](https://opencode.ai/v2/docs/console/models/).

Allocate the authorized $50 conservatively: $3 broad qualification, $22 targeted data review/bilingual corrections, $10 round one, $10 round two and $5 reserved for explicitly accounted repair/uncertain outcomes. Reallocate only within the total recorded cap. Cash, list-equivalent cost and Go allowance remain separate ledgers; unavailable billing is unknown, not zero. Local reservation logic is not a provider-enforced hard cap. Prefer an account-side spending ceiling as well when available.

Use one concurrent request initially. Bound media size, input/output tokens, requests and total reserved exposure before dispatch. Cache by provider/endpoint/returned version, prompt/schema, exact submitted media, preprocessing and decoding settings. A timeout with an unknown accepted/billed outcome stops new spending until reconciled. Do not blindly retry it. Treat model aliases as movable; requalify a changed serving identity. Never store provider hidden reasoning or credentials in receipts.

Before a teacher can correct a family, qualify it on independent labeled cases for that family; use the original plan's 200-source-group target and competence-gap checks. If it fails, it may help with wording or construction proposals but cannot certify the answers. Keep the bulk corpus grounded in existing human/source/renderer truth. The $50 instructor allowance is for targeted work, not an assumption that it can annotate an unlimited multimodal corpus.

One round is: diagnose permitted development failures → receive typed bounded instructions → construct/select new training media → Spark attempts → teacher atomic critique → independent truth/visibility/timing verification → full-parameter update plus replay → blind promotion evaluation → keep the child or retain the parent. Only allowlisted construction functions run; teacher output cannot execute Python or alter evaluators, budgets, package dependencies or test access.

Retain `teacher_proposed_answer` separately from `verified_target`; ambiguous semantic corrections go to review and stay outside training until resolved. Use the original 60% new verified / 30% multimodal replay / 10% bilingual-text recipe as a starting record mixture, then record actual positions and durations. Normalize losses by task family, including codec and diffusion objectives, instead of making long audio targets dominate language losses.

Run two adaptive rounds only after competent base media conditioning exists. From the same pre-RSI parent, run matched static/continued-training controls with the same learner updates, allowed source pool, replay and evaluation budget. If the budget cannot support a control, report teacher-assisted improvement without claiming an advantage from recursion. The original three-arm/three-round self-only/static/adaptive study remains a later stronger experiment; this two-round campaign does not claim to reproduce it.

Use the existing proposed promotion rule: at least +5 percentage points on the declared normalized primary score, paired source-group confidence interval above zero, maintained media dependence, no established family/language loss over 3 points, and text perplexity within the 5% retention target. For error metrics define the higher-is-better mapping before results. Failed or inconclusive rounds keep the parent. Never use the final audit to select an RSI recipe or perform another update.

## Execution graph and deliverables

| Gate | Work and dependency | Completion artifact |
|---|---|---|
| C0 | Account, actual GPU, storage/API access, secrets presence, budget observations | Infrastructure preflight receipt; no model-quality claim |
| C1 | CUDA port, common environment, all native branches, source tests and component objectives | Pinned source/environment release and real mixed-task gradient tests |
| C2 | New corpus, source grouping, target visibility, instructor eligibility and verification | Immutable data manifests, shards, data card, instructor eligibility receipts |
| C3 | Whole-model benchmark and fresh-session checkpoint restore | Throughput/capacity estimates and recovery comparison |
| S1 | Strong connector/conditioning alignment from original pretrained components | Media-dependent alignment checkpoint and baseline report |
| S2 | Formal image/spatial/OCR full fine-tuning | Grounded bilingual vision checkpoint |
| S3 | Real video and two-clock temporal training | Track/time/occlusion results and checkpoint |
| S4 | Speech/sound input and synchronized AV training | WER/CER/event/AV results and checkpoint |
| S5 | Native speech and sound generation training | Audio-gradient, waveform and blinded-review results |
| S6 | Joint full fine-tuning with earlier-task retention | Complete native multimodal checkpoint |
| R1/R2 | Qualified teacher-guided verified curricula and matched control | Candidate/parent/control metrics, budget ledger and promotion decisions |
| S7 | Lock winner and run final independent audit once | Final capability matrix, outputs, inference package and resumable state |

The controller must persist state transitions as immutable receipts bound to source, environment, input manifest and parent checkpoint hashes. Re-running a notebook resumes the same declared job; it must not silently start another epoch or spend the instructor budget again. Protocol changes start a new stage identity with weights-only initialization where appropriate. It serializes GPU jobs and leaves CPU/API preparation off the paid GPU where possible.

Final deliverables are: original-pretrained/text and aligned multimodal baselines; pre-RSI and round-one/two candidates; matched controls; the best validated full package; precise capability flags including any failed branch; EN/ZH and per-task scores with intervals; natural-media ablations; audio files and video overlays; complete training/exposure and billing records; committed checkpoint manifests; an inference notebook and a resume notebook using the proven environment.

## Acceptance and operator handoff

The full-capability claim requires the original plan's separate image grounding, video, time-lapse, speech, general-sound, generation and language-retention gates. Report at least 200–500 held-out cases per major family when feasible, real source-group counts and paired intervals. Test blank/swapped images, shuffled/reversed frames, wrong/missing capture clocks, muted/swapped/misaligned audio and detached output conditioning. Correctly invariant tasks should remain invariant under the appropriate transformation.

Use official/reference metrics where applicable: PCK/IoU and point matching, track identity and event boundaries, known-time error and unknown-time abstention, English WER and Mandarin CER, sound macro-F1, human speech intelligibility and blind sound prompt correspondence. Automated ASR or audio-embedding similarity is a screen, not complete perceptual acceptance. Final model outputs and human-review pending states must be saved even when a gate fails.

Immediate operator steps:

1. Use the same Google Account for the private project folder and Colab. Confirm actual paid entitlement and CU balance after any upgrade. Allocate the campaign budget explicitly; do not buy or activate a subscription through this handoff.
2. Open the supplied `Spark_Colab_Preflight.ipynb` in Colab. Start on CPU for Drive authorization and transport checks; select G4 only for the hardware/CUDA portion. It does not install a training stack, call the instructor or launch training.
3. Add `OPENCODE_API_KEY` in Colab Secrets and grant this notebook access. Add `HF_TOKEN` only for any chosen gated component. Do not upload local secrets or old checkpoints.
4. Run the Drive probe; retain the returned receipt and payload IDs. In a fresh runtime, set the restore ID/hash and rerun the recovery cell. This tests small-file transport/recovery, not a 25–40 GiB model checkpoint or optimizer continuation.
5. Hand the source bundle, this plan and `COLAB_AGENT_HANDOFF.md` to the implementation session. Complete C1–C3 before enabling the full campaign launcher. The source bundle intentionally retains original ROCm code as a reference and is not a CUDA-qualified training release.
6. After the implementation and real checkpoint-recovery gates pass, launch the budgeted stage graph. At session rollover, reconnect, authorize, verify the receipt and resume from the latest committed checkpoint. Keep the strongest completed model when a branch fails instead of assuming later epochs repair it.

No GPU runtime, instructor request, model training, subscription purchase, old-checkpoint upload or old-checkpoint deletion was performed while preparing this plan. The handoff's static validation and private Drive file placement are distinct from execution evidence on Colab.
