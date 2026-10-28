# Formal full-tuning script with Hugging Face checkpoints

Prepared 27 October 2026 from the existing campaign plan, multimodal refinement review, completed dataset release and repaired CPU bootstrap. The latest user instruction changes checkpoint storage and cleanup; this document and `colab/final/RUN.md` supersede the older Drive-only checkpoint design. The notebook remains a small launcher for reviewed source at an immutable GitHub commit. Exact final notebook/runtime identities are recorded in `releases/RELEASE.json` in the published repository and `output/colab-formal-final/RELEASE.json` locally.

## Delivered workflow

The notebook retains the complete six-stage curriculum and both controlled instructor rounds. Nothing in this storage revision reduces the run to a trial, removes a modality, or changes the dataset, component revisions, learning rates or epochs.

| Stage | Configured work |
|---|---|
| Bridge alignment | Finite alignment of trainable modality bridges; pretrained networks frozen for this stage |
| Vision full tuning | Full Spark updates with image/spatial tasks and text retention |
| Temporal full tuning | Video and audiovisual tasks, actual media clocks, motion branch and retention |
| Audio understanding full tuning | English/Chinese speech and general sound inputs, with retention |
| Audio generation full tuning | Native speech and sound/music objectives that backpropagate into Spark |
| Joint full tuning | Complete declared epoch across all eight families |
| Two instructed replay rounds | MiMo-V2.6-Pro priorities, matched uniform-replay controls, held-out promotion |
| Final evaluation/export | Protected audit, native samples, offline export reload and private HF upload |

The data remain pinned to the published `TassP0lar/SparkX2.6-MM` media revision. The original304,861 instructions produce315,359 complete views and290,685 training views after the frozen source-group splits. The configured six stages total603,668 record exposures before RSI; repeated exposures do not create new media diversity. Architecture remains SigLIP2 So400m NaFlex with tiles/spatial downshuffle, V-JEPA motion, Whisper/CLAP understanding, Qwen native speech output and Stable Audio native sound/music output. Codec/autoencoder infrastructure is explicitly frozen. No LoRA fallback or unrelated nested-learning implementation is introduced.

## Destination and disk contract

| Purpose | Destination |
|---|---|
| Data, source notices and citations | Existing HF **dataset** `TassP0lar/SparkX2.6-MM` |
| Full resumable snapshots | New configured private HF **model** repo `TassP0lar/SparkX2.6-MM-checkpoints` |
| Completed inference model | Configured private HF **model** repo `TassP0lar/SparkX2.6-MM` |
| Small receipts, reports and instructor ledger | Existing cold Drive project,2TB project cap |

The script uses Colab Secrets `HF_TOKEN` and `OPENCODE_API_KEY`. The HF token needs component read access and checkpoint/final-model write access. It is passed in memory to API clients, never saved into a model, dataset, config, argv or log. The checkpoint destination is private, account checked, and bound to this exact campaign/source; unrelated or public repositories are refused without overwriting them. The dataset repository is not modified.

Active model/data I/O stays on VM-local disk. The media cache stays bounded at23GB; the free floor stays44GB. Two complete checkpoint slots can coexist while saving or restoring, initially reserved at55GB each and replaced with the measured reservation after qualification. The newest verified snapshot stays local. Large downloads go directly to their destination with pinned content checks, without a Git/LFS checkout or a second HF snapshot cache.

Every checkpoint contains complete safetensor weights, optimizer, scheduler, RNG, stage, data cursor/split identity, parameter mapping and reload assets/runtime. The HF client preuploads file objects and atomically replaces `checkpoint/` plus its `COMMITTED.json` manifest. A parent-commit comparison prevents an unexpected branch update from being overwritten. Verification checks every remote size and LFS SHA256, or downloads ordinary small Git files to verify their content SHA256. Only after verification and a small Drive receipt backup is a local deletion acknowledgement written. Then older verified campaign checkpoint directories are purged.

Lost commit responses reconcile the same transaction instead of generating an unexamined duplicate. Upload failure, remote checksum failure, quota rejection or Drive receipt failure leaves the older fallback and the new local snapshot intact. A CPU-only `upload-checkpoint` command/cell retries a closed snapshot after the training process exits; it does not take optimizer steps or call the instructor. A process lock prevents training, restore and upload retry from racing over local files.

Restore always uses an immutable HF commit receipt, verifies the full inventory, and retains one verified local fallback until restoration finishes. Earlier remote revisions remain available for qualification and RSI parent/control/candidate selection. **Updating the remote branch does not free HF history storage.** The script does not squash history, remove old remote revisions, delete unrelated files or touch the workstation's original checkpoints. HF private-storage quota is a separate account constraint from the local230GB and Drive2TB limits.

Two recovery defects were corrected during this integration: later saves now copy preprocessors/codecs from the retained snapshot rather than a purged restored directory; restoring an older RSI parent can no longer rewind the campaign's recorded compute-use counter.

## Operation

1. Open the new pinned notebook on CPU, enable the two Secrets, review destinations, and run bootstrap/configuration/authentication/CPU preflight in order. Environment recovery retains the earlier uv/ensurepip fix, hashed locks and the Jinja2 chat-template fix.
2. Select G4, enter the actual current compute balance and rate, enable CUDA setup and run qualification once. The real Unsloth Studio headless training backend is used; its graphical UI is not launched.
3. Preserve the profile receipt. On a fresh VM, repeat setup and select `RUN_MODE='train'`; the script performs complete checkpoint recovery and continuation checks before the formal stages.
4. After later interruptions, use `RUN_MODE='resume'` with the exact HF snapshot receipt. If only upload failed and the VM still has the closed directory, use the CPU upload-retry cell first.
5. Completed stages and both controlled RSI rounds lead to final protected evaluation and an offline-reloaded private model export. Optimizer and private training ledgers are excluded from the inference model repository. Inspect `MODEL_UPLOAD_RECEIPT.json` for the verified final HF commit.

The instructor remains OpenCode Go/MiMo-V2.6-Pro with the existing$50 ceiling. Its provider/account dataset-review eligibility field remains explicit; this release does not invent that evidence or silently change the user's provider. Current CU observations and the existing1000CU ceiling remain enforced. GPU qualification/recovery alone need no instructor calls.

## Evidence boundary

The regression suite exercises real CPU optimizer serialization, three successive checkpoint saves, actual pinned Hub operation/data classes at a simulated remote boundary, failed upload/hash/receipt cases, lost-response reconciliation, old-revision restore, local retention, asset rebinding, concurrency exclusion and CPU-only retry. Existing native-component, data, cursor, cache, logging, bootstrap and notebook checks are retained. The final validation log and immutable release identity accompany the published source.

These tests do not establish a live multi-gigabyte HF checkpoint upload, complete G4 execution or trained-model quality. The previous release passed actual Colab CPU preflight; this source revision adds a new checkpoint destination and must run its own CPU preflight. No new GPU allocation, instructor request, checkpoint upload or trained-model publication was performed merely to construct this release.

The expected result after successful execution is a reloadable multimodal Spark prototype with the included native branches and measured evaluation reports. Broad spatial/video reasoning, bilingual speech quality, music quality and interleaved omni-dialogue competence require actual evaluation; completion is not substituted for those measurements.

## References

- [Current operator guide](../../colab/final/RUN.md), [campaign plan with storage amendment](COLAB_CAMPAIGN_PLAN.md), [multimodal refinement review](REFINEMENT_MULTIMODAL_REVIEW_2026-10-27.md).
- [HF atomic commit API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.create_commit) and [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload). Implementation uses the pinned0.36.2 library APIs.
- [Unsloth Studio](https://unsloth.ai/docs/new/unsloth-studio), [OpenCode Go](https://opencode.ai/v2/docs/console/go), [OpenCode inference/accounting](https://opencode.ai/v2/docs/console/api/inference).
- Component/model and dataset licenses remain separately attributed in the operator guide, component lock and [published dataset card](https://huggingface.co/datasets/TassP0lar/SparkX2.6-MM).
