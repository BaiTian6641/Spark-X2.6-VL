# Full native multimodal Spark fine-tune — direct one-shot Colab

Use [`Spark_Formal_Full_Finetune.ipynb`](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/codex/formal-colab/colab/Spark_Formal_Full_Finetune.ipynb). It replaces the former CPU-preflight/qualification launcher, archived as [`Spark_Formal_Omni_Gated.ipynb`](Spark_Formal_Omni_Gated.ipynb).

## Scope

The `colab/final` native runtime is retained in full:

| Stage | Families |
|---|---|
| Bridge alignment | every modality bridge; pretrained towers frozen |
| Image full | image + text |
| Temporal full | video + audiovisual |
| Audio understanding full | speech input + general sound |
| Audio generation full | native speech + native sound/music output |
| Joint full | all eight families |
| Instructor RSI | two 32-example rounds with matched uniform replay controls |
| Final | protected audit, offline export reload, private model upload |

All Spark parameters train in the principal stages. Codecs/autoencoders remain frozen infrastructure. No LoRA or quantized fallback is introduced. The dataset is the full pinned `TassP0lar/SparkX2.6-MM` corpus: **304,861 instructions, 315,359 views, 290,685 training views**, with image, video, audiovisual, speech, sound, music, OCR/text and text replay.

## Removed gates

The direct path specifically removes:

- CPU runtime preflight (`run_campaign('cpu')` / `PREFLIGHT.json`).
- Complete-model qualification (`QUALIFICATION.json`, profile receipt).
- New-process/prior-VM recovery (`FULL_RECOVERY_PROOF.json`).
- Drive authentication and Drive artifact storage.
- Colab CU balance/rate observation.
- Instructor route-eligibility receipt.

It retains the HF token, gated-model access, instructor key, $50 instructor ledger, checkpoint binding, disk floor, source/asset hashes, checkpoint verification and final private model publication.

## One-shot flow

1. Clone the reviewed GitHub branch.
2. Install one isolated hash-pinned CUDA environment.
3. Download native components and the full HF dataset catalogue.
4. Build the native model and run bridge + all full stages + two instructor RSI rounds.
5. Upload complete checkpoints to `TassP0lar/SparkX2.6-MM-checkpoints` and, only after a complete curriculum/final audit, the private model package to `TassP0lar/SparkX2.6-MM`.

No file from a previous Colab VM is read. The notebook starts a new timestamped run and a new local `/content/spark-mm-direct` tree.

## Requirements

- One CUDA GPU with **≥90 GB** and BF16; G4 / RTX PRO 6000 Blackwell is the target.
- Colab Secrets `HF_TOKEN` and `OPENCODE_API_KEY`.
- HF account `TassP0lar` with Stable Audio Open 1.0 access and write access to both private model repositories.
- Local `/content` disk sufficient for the bounded 23 GB media cache, assets, one or two complete checkpoints, and the 44 GB free floor.

`DIRECT_WALL_HOURS` is the internal graceful-stop budget. `0` disables it. HF checkpoints are written at the configured checkpoint interval and at stage boundaries. A curriculum that does not finish in one session leaves the latest complete checkpoint in the private HF repository; the next notebook run does not depend on the prior VM.

The full curriculum is large (603,668 configured exposures before RSI). A single Colab session is not guaranteed to finish it; completion is not inferred from a stop or from checkpoint availability.

## Local validation

Static notebook/schema checks, direct-config validation, disabled budget behavior, instructor eligibility behavior, full runtime compilation and focused tests pass. This workstation did not execute the full native CUDA campaign or upload checkpoints. The first G4 run remains the environment/model gate.
