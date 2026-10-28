# One-shot image/frame full fine-tune on Colab

Open [`Spark_Formal_Full_Finetune.ipynb`](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/codex/formal-colab/colab/Spark_Formal_Full_Finetune.ipynb) in Colab. The source file is [`colab/Spark_Formal_Full_Finetune.ipynb`](Spark_Formal_Full_Finetune.ipynb). It replaces the former gated all-modality launcher, now archived as [`Spark_Formal_Omni_Gated.ipynb`](Spark_Formal_Omni_Gated.ipynb), and does **not** call its CPU preflight, qualification, instructor, Drive, or checkpoint-recovery gates.

## What it runs

Fresh GPU VM → clone GitHub → install one isolated CUDA environment → download original Spark-X2.6-4B + SigLIP2 base-256 → build a pinned public image/frame corpus → connector alignment (256 updates) → full-parameter fine-tune (all 4,201,669,760 parameters) → upload the complete checkpoint to Hugging Face.

No local state is assumed. `RUN_ID` is timestamped, `/content/spark-mm` is recreated, and the notebook does not read a checkpoint from a previous VM.

## GitHub synchronization

Push these paths to the branch configured in the notebook (`REPO_REF`, default `codex/formal-colab`):

```text
multimodal/__init__.py
multimodal/model.py
multimodal/collator.py
multimodal/data.py
multimodal/bundle.py
scripts/train_multimodal_studio.py
scripts/prepare_vision_component.py
scripts/package_colab_image_frame_data.py
colab/stage1/__init__.py
colab/stage1/common.py
colab/stage1/environment.py
colab/stage1/environment_spec.json
colab/stage1/candidate-cu130-py313.lock
colab/stage1/prepare_models.py
colab/stage1/port_trainer.py
colab/stage1/prepare_public_image_frame.py
colab/Spark_Formal_Full_Finetune.ipynb
```

The notebook hard-checks these files and stops with the missing list. Do not include downloaded weights, datasets, checkpoints, secrets, or the workspace `runtime/` directory.

## Data

The default `DATASET_SOURCE="public_hf"` downloads and verifies two pinned components of public dataset `TassP0lar/SparkX2.6-MM` at revision `f472f980a6c95d3e9844f27e6045c8c09b21536a`:

- `releases/docci-training-source-v1`
- `releases/full-ft-v1/vision-core`

Result: **50,991 records, 17,120 media files, 3.05 GB**, across factual captions, point/box, counts, temporal order, occlusion and capture elapsed. The 2% development split is deterministic and source-group isolated. No test split is created; this is not a sealed-test score.

For the exact frozen formal-v1 corpus, set `DATASET_SOURCE="local"` and `DATASET_LOCAL_DIR` to a root containing `train.jsonl`, `dev.jsonl`, `dataset_manifest.json`, `media_manifest.json` and `media/`.

## Launch

1. Push the source paths above.
2. Open the notebook from GitHub in Colab.
3. Runtime → Change runtime type → one GPU with **≥46 GiB** and BF16. G4 / RTX PRO 6000 is the tested target.
4. Add `HF_TOKEN` to Colab Secrets and enable notebook access. It needs write access to `TassP0lar/SparkX2.6-MM-checkpoints`.
5. Run all cells. The notebook uses one isolated environment for the whole run.

`WALL_HOURS=9.0` bounds the full stage to leave time for checkpoint upload. If the epoch completes earlier, `epoch_plan_complete=true`; otherwise the complete optimizer checkpoint is still saved and uploaded. Set `PUSH_CHECKPOINT=False` only if you accept losing the checkpoint when the VM ends.

## Local verification performed

- Public builder read and hash-verified the pinned DOCCI and vision-core TARs; output validated with `multimodal.data.validate_dataset`.
- 50,991 built records; 17,120 media files; train 50,045 / dev 946; 37,436 English / 13,555 Chinese.
- Spark tokenizer audit: maximum 1,084 positions at 256 tokens/frame, below the 2,048 limit; zero over-budget records.
- Collator smoke test on dev records with 1 and 4 frames; real SigLIP processor and Spark tokenizer; no truncation.
- `scripts/package_colab_image_frame_data.py` round-trip on `data/multimodal_s0`.
- `python -m unittest tests.test_colab_one_shot` passes, including notebook AST/schema checks and the generic CUDA guard.
- No CUDA model training or checkpoint upload was run on this workstation for this notebook; that remains the Colab GPU gate.
