# Spark X2.6 multimodal formal full-tuning pipeline

[**Open the formal notebook in Colab**](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/05a7ad773723e31e4d09f7282b75f93142be4437/colab/Spark_Formal_Full_Finetune.ipynb) · [Review notebook source](https://github.com/BaiTian6641/Spark-X2.6-VL/blob/05a7ad773723e31e4d09f7282b75f93142be4437/colab/Spark_Formal_Full_Finetune.ipynb) · [Complete operator guide](colab/final/RUN.md) · [Release report](reports/spark_multimodal/FORMAL_HF_CHECKPOINT_RELEASE_2026-10-27.md)

The complete formal script for the English/Chinese multimodal Spark prototype is ready to run through its hardware and recovery checks. It uses Unsloth Studio's actual **headless training backend**. Its graphical UI is not launched, and custom-model speedup is measured during qualification.

The notebook is only **23,071 bytes**. It downloads a separate 5,019,537-byte source ZIP from runtime commit `669d9ce6ddbfa2d08d31bb4745008fdcd522eedc`, verifies SHA256 `7be91268cc09294d9aa29b9b9e6bcd447268a48764ddbbabf2f27f95d7be2332` and all 43 source files before executing. No large source archive or model weights are embedded in the notebook. [The release receipt](releases/RELEASE.json) binds the source and launcher.

## Complete configured workflow

1. CPU environment, model access, source-grouped data/context audit and private checkpoint destination preflight.
2. G4 CUDA/Studio whole-model qualification and fresh-VM full optimizer recovery.
3. Finite bridge alignment, then full Spark vision tuning, temporal/AV tuning, speech/sound understanding, native speech/sound/music generation, and joint full tuning.
4. Two MiMo-V2.6-Pro instructed replay rounds with matched uniform-replay controls and protected promotion evaluation.
5. Final protected audit and native samples, offline export reload, then private HF model upload using Colab Secret `HF_TOKEN`.

The architecture retains SigLIP2 So400m NaFlex with tiles and 2x2 spatial downshuffle; V-JEPA2.1 motion and explicit media clocks; Whisper/CLAP understanding; Qwen native speech output; and Stable Audio native sound/music output. Full Spark parameters train in the principal stages. The short bridge alignment freezes pretrained networks; codec/autoencoder infrastructure remains frozen. No LoRA or reduced-curriculum fallback is used.

All 304,861 published original instructions are represented by 315,359 complete bounded views, split by connected source groups. There are 290,685 training views and 603,668 configured stage exposures before RSI. The [dataset card](https://huggingface.co/datasets/TassP0lar/SparkX2.6-MM) preserves source citations, licenses and known coverage limits. Repeated training exposures are not additional media diversity. The [multimodal refinement review](reports/spark_multimodal/REFINEMENT_MULTIMODAL_REVIEW_2026-10-27.md) defines this project's scope; unrelated nested-learning work is excluded.

## Storage for the 230 GB Colab disk

| Artifact | Destination | Retention |
|---|---|---|
| Published training data | HF dataset `TassP0lar/SparkX2.6-MM` | Pinned remote corpus;23 GB rolling local cache |
| Full resumable checkpoints | Private HF model repo `TassP0lar/SparkX2.6-MM-checkpoints` | Atomically updated `checkpoint/`, exact commit receipts |
| Finished inference model | Private HF model repo `TassP0lar/SparkX2.6-MM` | Clean export, separate from optimizer state |
| Receipts, reports, instructor ledger | Existing cold Drive project | Small immutable uploads,2 TB project cap |

**After every successful checkpoint upload and verification, older local snapshots are purged.** One newest snapshot stays local; two slots may coexist during saving/restoring. The initial reservation is 55 GB per slot, replaced by the measured size plus margin after qualification. A 44 GB free floor is preserved. No whole-dataset LFS clone, checkpoint-repository clone, duplicate HF snapshot cache or random training I/O on Drive is used.

Verification covers every remote file: size and LFS SHA256 for large objects, downloaded SHA256 for small Git files. A small receipt is backed up to Drive before deletion is acknowledged. Upload, checksum, quota or receipt-backup failure preserves the local fallback and new snapshot. Use the notebook's CPU-only upload-retry cell after the training process exits; it reconciles a lost response and does not repeat optimizer steps. A process lock prevents simultaneous training/retry/cleanup.

Restore uses the receipt's exact HF revision and complete manifest, including weights, optimizer, scheduler, RNG, stage, frozen split and committed data cursor. Old remote revisions remain available for recovery and RSI. **Local cleanup does not reclaim HF history storage.** This script never squashes remote history or deletes unrelated data. HF private-storage quota is separate from local disk and Drive capacity; a rejected upload stops cleanup. Original workstation checkpoints remain local and untouched.

## Start the notebook

1. Open the pinned link above on **CPU**. Enable notebook access to `HF_TOKEN` and `OPENCODE_API_KEY` in Colab Secrets. HF needs component read access and write access to both private model repositories; do not put either secret in cells or files.
2. Run source bootstrap, operator configuration, Google authentication and CPU setup/preflight in order. Keep active work under `/content/spark-formal`. Drive mounting is optional for browsing cold results.
3. Select the G4 Blackwell runtime, enter its **current** CU balance/rate, enable `INSTALL_CUDA_ENVIRONMENT`, then run `RUN_QUALIFICATION` once. Preserve its emitted profile receipt.
4. On a fresh VM, repeat setup and select `RUN_MODE='train'` with that profile receipt. The complete optimizer continuation check runs before the formal stages. Later sessions use `RUN_MODE='resume'` plus the exact verified checkpoint receipt.
5. If upload failed while local files still exist, use `RETRY_CHECKPOINT_DIRECTORY` in the CPU recovery cell first. `DISCOVER_CHECKPOINTS` lists cold receipts and the latest verified HF checkpoint.
6. The completed campaign uploads the private inference package automatically. `MODEL_UPLOAD_RECEIPT.json` identifies the exact verified final commit.

The $50 instructor and 1000 CU campaign limits remain configured. OpenCode Go's provider/account dataset-review eligibility field is still explicit in the guide; the script does not invent evidence or switch the chosen provider. Full CUDA qualification and recovery alone do not require instructor calls. GPU flags and `RUN_MODE='inspect'` initially prevent an accidental launch merely by opening the notebook.

## Validation and current evidence

**71 focused tests passed for this release:** [68 runtime/data/native-component/checkpoint/notebook tests](validation/colab-formal-hf-checkpoint-tests-2026-10-27.log) and [three real isolated CPU bootstrap tests](validation/colab-formal-hf-bootstrap-tests-2026-10-27.log). Coverage includes real CPU optimizer serialization, repeated saves after cleanup, simulated Hub upload failures and lost responses, every-file remote verification, older-revision restore, receipt-backup failure, concurrency exclusion, CPU-only retry, cursor/cache behavior, native component gradients and the actual packaged notebook's hashes/import closure. One upstream Diffusers deprecation warning remains.

The previous source release passed actual Colab CPU preflight and its chat-template audit. This source revision adds the checkpoint destination and must pass its own CPU preflight. The full native G4 campaign, live multi-gigabyte HF checkpoint transfer and trained multimodal quality are **not yet qualified**. No GPU training, instructor request or trained-model upload was performed while constructing this release. The final model card reports actual results and pending human audio review; it does not equate curriculum completion with broad capability acceptance.

The earlier `ensurepip` and Jinja2 failures remain repaired: a hash-pinned external uv bootstrap creates/repairs isolated environments and synchronizes complete CPU/CUDA locks without replacing kernel packages. Child stdout/stderr are captured and sanitized, stopped reasons are visible, and Ctrl-C drains the checkpoint output. [Historical repair evidence](reports/spark_multimodal/COLAB_PREFLIGHT_REPAIR_2026-10-27.md) is retained separately.

## Rebuild and provenance

Runtime files, component/data/environment locks, tests, the builder and the pinned upstream motion source archive are reviewable here. This repository contains no training media, credentials, notebook execution outputs or model weights.

First run `python scripts/build_final_colab.py --runtime-only`, copy its ZIP and manifest into `releases/` and commit the runtime. Then use `python scripts/build_final_colab.py --runtime-revision FULL_RUNTIME_COMMIT`, test and commit the small notebook and release receipt. The notebook always pins a full source commit rather than a moving branch.

See the [campaign plan with current storage amendment](reports/spark_multimodal/COLAB_CAMPAIGN_PLAN.md), [operator guide and component citations](colab/final/RUN.md), [HF upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload), [atomic commit API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.create_commit), and [Unsloth Studio documentation](https://unsloth.ai/docs/new/unsloth-studio). Component and dataset terms apply separately; the composite is not relicensed wholesale.
