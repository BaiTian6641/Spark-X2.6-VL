# Spark Colab handoff

## Full-Git multimodal fine-tune (`colab/final`)

For the corrected one-click G4 campaign, open [`Spark_Formal_Full_Finetune.ipynb`](Spark_Formal_Full_Finetune.ipynb). The notebook clones the **full Git repository** and runs the original staged curriculum; it does not use the replacement `colab/oneshot` WSD schedule or its unreviewed data expansion.

- [`ONESHOT_HANDOFF_AND_NEXT_STEPS.md`](ONESHOT_HANDOFF_AND_NEXT_STEPS.md) — corrected architecture, restored contracts, corpus boundaries, 230 GB / 110-hour enforcement, verification scope, and operator steps.
- [`DIRECT_FULL_MULTIMODAL.md`](DIRECT_FULL_MULTIMODAL.md) — concise launch, resume, storage, and exit-code guide.

The flow is full Git checkout → locked environment → original corpus → whole-model qualification → fresh-process recovery → six staged curricula → two controlled instructor RSI rounds → protected audit → offline reload → verified private HF upload. No Drive mount or API is used. Complete epochs and matched RSI controls are never silently shortened.

The older gated launcher remains archived as [`Spark_Formal_Omni_Gated.ipynb`](Spark_Formal_Omni_Gated.ipynb).

## Current formal preparation, 27 October 2026

At approximately **15:03 CDT**, the known qualification notebook was disconnected;
Colab showed **0 active sessions, approximately 0 CU/hour and 1,098.65 CU**.
The fresh-VM recovery has no observed PASS, and its follow-on checkpoint/Drive/
instructor-readiness cells remain unexecuted. The complete technical checkpoint
and original immutable source remain on Drive. No new paid GPU was allocated for
the CPU source/cache work below.

The active source route is now **local construction → pinned Hugging Face →
bounded local rolling cache**, with Drive as cold storage. The published
[DOCCI training source](https://huggingface.co/datasets/TassP0lar/SparkX2.6-MM/tree/a1190c8098162cb45630b0f8ceb23d96b7cf3664/releases/docci-training-source-v1)
contains 8,776 original-caption/image pairs in two TARs totaling 2.104 GB. It
preserves historical exposure and is not new evaluation data or complete
multimodal/EN-ZH coverage.

Use [Spark_HF_Rolling_Cache_CPU.ipynb](Spark_HF_Rolling_Cache_CPU.ipynb) for the
next public dataset/storage check in a **CPU** runtime. Its source ZIP, dependency
lock, tests, data indices and full-source commit are pinned. It requires no
Drive access, instructor secret or HF token and cannot launch training. The
source bundle passed 26 tests after actual verified unpacking, and public HTTP
interruption/range recovery passed for the fixture and first full-source TAR on
the workstation. A fresh Colab CPU execution remains a separate pending check.
See `evidence/hf-cache-locked-cpu-verification.json`, `formal/README.md` and
`reports/spark_multimodal/HF_ROLLING_DATASET_PLAN_2026-10-27.md`.

In parallel, the workstation CPU is acquiring a new pinned PixMo-Cap caption
source, targeting 102,000 distinct processed images from 200,000 frozen candidates.
Live progress is `data/colab_formal_v2/pixmo_cap/acquisition-progress.json`; it is
not a formal training counter. A local CPU continuation worker waits on the acquisition
owner lock, then verifies/group-splits and packages the successful source; it
stops on insufficient images and makes no teacher/training/upload calls. The new source is English caption alignment only,
with separate human-quality, EN/ZH-mixture and native-branch gates still pending.

The new [formal preparation source](https://drive.google.com/file/d/1Nkdl2hr9v7YXV7OvXNbTFSnTOmFJK13r/view)
and [CPU data/storage notebook](https://drive.google.com/file/d/1qDyHtHmeSdx6nT1XcXPYglco5Mz5hz-F/view)
are private permanent Drive artifacts. Source SHA-256 is
`cb78a609ee22a2debf95462d6c8362e513e48b9efb7f93b52264130d4761d29d`.
All 23 source files were verified after unpacking, and 18 focused host tests
passed from that unpacked release in the isolated CPU environment. The notebook's
code cells compile; it has not been run in Colab and does not train models.
Do not launch a duplicate caption acquisition there while the workstation owns
the running writer. See `formal/README.md`, `formal/storage_policy.json` and
`evidence/formal-v2-storage-release.json`.

Random reads/writes, decoded media, caches, optimizer state and checkpoint
serialization use local disk. A conservative 230-decimal-GB plan reserves two
complete checkpoint slots and 44 GB free; actual native module/checkpoint sizes
still require profiling. HF indexed TAR reads use no extracted dataset duplicate.
Drive stores closed checkpoints/backups/receipts with checksum verification and
`COMMITTED.json` written last. The older R4 Drive restore/unpack route remains
separate and its live dataset round trip is pending. The generic rolling TAR
cache is implemented; its real trainer/native media/RNG adapter remains pending.

Formal full fine-tuning has not started. Native motion, audio input, native
speech/general-sound generation, mixed-task profiling, the complete bilingual
corpus and instructed RSI remain incomplete. Preserve all old local checkpoints.

## Technical qualification evidence

This package prepares the full multimodal campaign described in `reports/spark_multimodal/COLAB_CAMPAIGN_PLAN.md`. Infrastructure preflight has passed on G4. The next notebook, **Spark_Studio_CUDA_Qualification.ipynb**, installs a hash-pinned isolated environment and tests the actual Studio backend, full-parameter image/frame training, optimizer continuation, and a new checkpoint commit to Drive. See `stage1/README.md` for its scope and recovery procedure. This remains a qualification stage; the complete native audio/temporal/RSI campaign is not yet implemented.

The live qualification notebook is https://colab.research.google.com/drive/1aerQV3Aod7tjQ7RgvkWqKpcKiu80GWqY . Its private source release and initial metadata checks are recorded in `evidence/colab_stage1_drive_release.json`. The live run on 27 October 2026 passed dependency checks, verified the official Studio backend source, and completed eight full-parameter updates followed by a new-process continuation to step ten. All 4,201,669,760 parameters were trainable and covered by the optimizer; no tensor was frozen or offloaded. Media-to-Spark gradients and component updates were observed. The continuation matched sampled saved parameter/optimizer values, scheduler/global step, and CPU/Python/CUDA RNG states. See `evidence/colab-stage1-initial-training-receipt.json` and `evidence/colab-stage1-resume-training-receipt.json`. These technical-fixture checks do not establish model quality or fresh-VM checkpoint recovery. xFormers 0.0.35 cannot load its older CUDA/Torch binary in this environment; Unsloth selected PyTorch attention.

The first eight updates took 18.46 seconds inside the trainer; the post-warmup mean was 0.251 seconds per update for this short, variable image/frame fixture, with 42.12 GiB peak allocated GPU memory. This is not a formal-corpus or audio/video throughput forecast. A complete checkpoint is approximately 27.36 GB, and the first local serialization took 73.66 seconds.

The initial large Drive upload exposed `RedirectMissingLocation`: raw `httplib2.Http` treated Drive's HTTP 308 resumable status as a redirect. The notebook bootstrap and checkpoint cell now use Google's `googleapiclient.http.build_http()`, which excludes HTTP 308 from redirect handling. The same known file IDs and saved upload session were reused; no model training was repeated. The failure receipt remains in `evidence/colab-stage1-upload-failure-receipt.json`. The source ZIP and trained code are unchanged; the correction is in notebook HTTP initialization. The corrected clean notebook is generated by `scripts/build_colab_stage1.py` using the original source ZIP ID/hash.

The retry **completed successfully**: all 26 checkpoint payloads, totaling
27,359,363,974 bytes, passed remote size/MD5 verification before `COMMITTED.json`
was written. The successful commit operation took 551.28 seconds, including
hashing, upload and verification. The private checkpoint folder is
https://drive.google.com/drive/folders/1K356Xqn917p0mxKUjAQlSMy7dsclQXqE .
The final report and commit manifest were downloaded into
`evidence/colab-stage1-complete-receipt.json` and
`evidence/colab-stage1-COMMITTED.json`, and their SHA-256 hashes were verified.
For the later fresh-VM restore test, use:

```python
RESTORE_COMMIT_ID = "1noBVM6Dvtr7ecn6xINuSRlk7ga0p0o5a"
RESTORE_COMMIT_SHA256 = "6694bc5fc6736dcc4c2bc51f7e086e68cf9ee06265f906fc26025ab6b7c30acd"
```

The user-requested `drive.mount('/content/drive')` cell is included after the
commit cell in both the live notebook and the maintained clean template. The
mount has a separate Google Drive for desktop sign-in. The first attempt timed
out while that popup was pending. After the user completed sign-in, rerunning
only the mount cell succeeded. At 18:16 UTC on 27 October 2026, the project was
readable at `/content/drive/MyDrive/Spark Multimodal Colab`, and the committed
checkpoint manifest read through the mount matched its expected SHA-256. The
verified receipt is `evidence/colab-stage1-drive-mount-receipt.json`. Training and
checkpoint upload were not repeated. Active training and caches stay under
`/content/spark_mm`; mounted Drive is for browsing the persistent project files.
This manifest readback does not establish full checkpoint recovery in a new VM.

Open `Spark_Colab_Preflight.ipynb` in Google Colab. Drive authentication happens in the notebook with the same account that owns the private project folder. The desktop Drive connector's login does not authorize the Colab runtime. The notebook performs a new 64 MiB random-file round trip and saves a sanitized receipt. It leaves its small probe files in the project folder and never edits existing artifacts. Increase the probe to 1 GiB only when deliberately measuring sustained transfer; a tiny probe does not predict full-checkpoint upload speed.

Run Drive checks on CPU first. Use G4 to run the GPU portion, then disconnect the GPU when finished to avoid idle compute-unit use. Keep the project storage cap at 2 TB and the instructor cap at US$50. At approximately 12:45 CDT on 27 October 2026, the Colab Resources panel showed 1,110.18 units available, approximately 8.9 units/hour, and one active session. The proposed project allocation remains 1,000 units, not the entire account balance. This is a point-in-time observation; later runs must refresh it.

Optional instructor readiness: add `OPENCODE_API_KEY` in the Secrets panel and enable access for this notebook. The key is read only when the explicit check is enabled, kept in memory and never printed. Do not paste it into a code cell or upload `runtime/secrets/opencode.env`. `HF_TOKEN` is separate and only needed for selected gated assets.

For a fresh-runtime storage test, save the returned probe file ID, SHA-256 and runtime boot ID, terminate/reconnect the runtime, fill the three restore fields and rerun the restore cell. This proves persistence of that byte probe only. The campaign still needs a real model/optimizer stop-resume comparison.

Old checkpoints remain local, by user request. Stage 1 downloads original pretrained model files directly in Colab at pinned revisions. The source bundles exclude all weights, optimizer state, datasets, secrets, environments and caches. `HANDOFF_MANIFEST.json` describes the historical planning bundle; the separate Stage 1 ZIP contains its own `RELEASE_MANIFEST.json`. The campaign's remaining implementation packet is `COLAB_AGENT_HANDOFF.md`.
