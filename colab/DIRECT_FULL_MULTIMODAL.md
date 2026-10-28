# Full native multimodal Spark fine-tune — full-Git one-click Colab

Use [`Spark_Formal_Full_Finetune.ipynb`](Spark_Formal_Full_Finetune.ipynb). Read [`ONESHOT_HANDOFF_AND_NEXT_STEPS.md`](ONESHOT_HANDOFF_AND_NEXT_STEPS.md) for the corrected audit, exact architecture, corpus boundaries, and evidence requirements.

## What this runs

`colab/final` executes the original staged campaign from a **full Git repository checkout**:

1. Locked CUDA environment and revision-pinned components.
2. Original `TassP0lar/SparkX2.6-MM` corpus: 304,861 instructions / 315,359 bounded views.
3. Whole-model CUDA qualification and fresh-process optimizer/RNG recovery.
4. Bridge alignment → vision → temporal/AV → audio understanding → native speech/sound generation → joint full tuning.
5. Two instructor-controlled candidate-versus-uniform-control RSI rounds.
6. Protected audit, offline export reload, and verified sanitized public HF upload.

The replacement `colab/oneshot` WSD schedule, 20-source expansion, 12 GB reserve, history squashing, and single optional replay round are **not** used.

## Requirements

- G4 / RTX PRO 6000 Blackwell with at least 90,000,000,000 GPU bytes and BF16.
- `flash-attn-4[cu13]`: supplemental Blackwell/CUDA-13 environment with import and BF16 backward-kernel checks.
- `INSTRUCTOR_MODEL`: defaults to `mimo-v2.6-pro`; this is an allowed operator selection, not independently verified provider eligibility or instructor competence.
- Conservative 230 GB local-disk envelope; 44 GB non-borrowable free floor.
- `HF_TOKEN`: component read access plus model/checkpoint write access for `TassP0lar`; public snapshots are sanitized.
- `OPENCODE_API_KEY`: required for both instructor rounds; $50 cumulative ledger and uncertain-outcome stop remain enforced.
- Stable Audio Open 1.0 access accepted on the same HF account before launch.

## Launch and resume

Open the notebook on branch `codex/formal-colab`, select the G4 runtime, and click **Run all**. The notebook records the actual Git `HEAD` and hashes every tracked file. Set `REPO_COMMIT` to a full 40-character commit when an immutable launch is required.

If an existing checkpoint repository contains an incompatible trained payload, choose a new empty repository. The runner will not overwrite or rebind that history. The current default is a new sanitized public `TassP0lar/SparkX2.6-MM-checkpoints-g4-v8` repository because the residual-codebook and time-conditioning repairs change the exact source binding.

| Exit | Meaning | Action |
| ---: | --- | --- |
| `10` | Whole-model qualification completed | The notebook automatically starts fresh-process recovery; if stopped after this phase, run all again. |
| `3` | Session deadline reached after a complete checkpoint | Start a new G4 session and click Run all to resume. |
| `0` | Stages, both RSI rounds, audit, offline reload, and upload completed | Inspect `ONESHOT_RECEIPT.json` and `MODEL_UPLOAD_RECEIPT.json`. |
| Other | Failure or explicitly incomplete campaign | Inspect logs/receipts; do not call the model complete. |

The cumulative budget is **110 GPU-hours**, including setup, qualification, training, RSI, evaluation, and export. Qualification projects the complete 603,668-exposure curriculum from measured per-family p90 timings plus a 25% reserve. An over-budget projection stops rather than shortening any epoch or RSI arm.

## Storage

- No Drive mount or Drive API.
- Local active files stay under `/content/spark-formal`.
- Two measured checkpoint slots, initially 44 GB each, plus a 10% post-qualification margin.
- Ten bounded 2.3 GB TAR caches; no full dataset clone or second HF payload cache.
- Full checkpoints go to public sanitized `TassP0lar/SparkX2.6-MM-checkpoints-g4-v8`; HF history is preserved for recovery and RSI parent/control snapshots. Protected evaluation rows and account metadata are excluded.
- Final inference weights go to public `TassP0lar/SparkX2.6-MM-g4-v2` after offline reload and remote hash verification; protected evaluation reports are excluded.

## Evidence limits

Repository tests do not establish full-GPU throughput, model quality, or broad capability. The first G4 run is the real hardware gate. Human speech intelligibility/tone review, sound/music listening, external spatial/temporal/3D benchmarks, and final acceptance remain separate. Narrow AV, time-lapse, music, natural-document, and calibrated-depth coverage is documented in the handoff rather than silently patched with unreviewed data.
