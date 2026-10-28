# Spark X2.6 multimodal formal Colab pipeline

[**Open the formal notebook in Colab**](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/fab1fa3b6e7428c79a5c496160ae871ccdf7a047/colab/Spark_Formal_Full_Finetune.ipynb) · [Review the notebook](https://github.com/BaiTian6641/Spark-X2.6-VL/blob/fab1fa3b6e7428c79a5c496160ae871ccdf7a047/colab/Spark_Formal_Full_Finetune.ipynb) · [Operator guide](colab/final/RUN.md)

Runtime source for the experimental full-parameter Spark X2.6 4B multimodal campaign. It combines SigLIP2 So400m NaFlex vision, tiles and spatial downshuffle; V-JEPA2.1 video features and explicit clocks; Whisper/CLAP audio understanding; native Qwen speech output; Stable Audio sound/music output; and two bounded instructor-guided replay comparisons. English and Chinese are the planned language coverage.

**The complete native CUDA campaign has not yet passed its hardware qualification.** CPU source/component tests and an install dry run are not evidence of a trained multimodal model. The notebook keeps the actual full-model profile and fresh-VM recovery gates before long formal training.

The notebook is a small launcher. It fetches `releases/spark-formal-runtime.zip` from a **full Git commit**, checks its SHA256 and exact source manifest, and extracts it to a release-specific local source directory. No source ZIP or model weights are embedded in the notebook. Runtime files are also available individually for review.

## Start or recover setup

Open the pinned Colab link above. The launcher is **19,828 bytes**. Its runtime commit is `6f925694cf6cb8e43ca8a4c2791903c76a50e6b1`; the separately downloaded 5,010,530-byte source ZIP has SHA256 `c88f69aca289f5cb57919ca09daac9a9ae4db74086652d58cea4202b37200dce`. The notebook commit follows the runtime commit so it can pin immutable source without a circular build dependency. [The release receipt](releases/RELEASE.json) binds both artifacts.

1. Use a **CPU runtime** for setup/preflight. Enable notebook access to Colab Secrets `HF_TOKEN` and `OPENCODE_API_KEY`. Do not paste either key in notebook cells, Git or chat.
2. Run the source bootstrap, configuration and normal Google authentication cells. Keep the configured `/content/spark-formal` work root and intended Drive project folder. Review [the complete operator guide](colab/final/RUN.md).
3. Run CPU environment setup and preflight. The setup repairs a previous isolated `env-cpu` containing Python but no pip. It validates isolation, synchronizes the complete hash-locked requirements and writes a readiness receipt only after `pip check`. It does not clear directories or delete data, checkpoints or Drive files.
4. Only after CPU preflight, select the reviewed G4 Blackwell target, enter the actual current CU balance/rate and explicitly enable CUDA installation and qualification. GPU installation uses the same repair helper with the separate CUDA lock. The real Studio headless backend is used; the graphical UI is not launched.
5. Follow the emitted qualification and cold-storage receipts for fresh-VM recovery, formal training, interrupted-session resume and final evaluation. `RUN_MODE='inspect'` and GPU-install/qualification flags are initially off. Opening this notebook does not start training or spend the instructor budget.

Read the existing provider/account eligibility gate in the guide before instructor dataset review. The API key alone does not resolve that gate. The campaign retains the $50 cumulative instructor ceiling and the authorized 1000 CU envelope, and never silently shortens the curriculum to fit.

## Visible preflight failures

An `environment_setup_complete` event means package setup passed; model, corpus and Drive checks still follow. The 2026-10-27 failure was HTTP403 `GatedRepo` from Stable Audio Open1.0. The user completed its access form, and the same Colab token then passed a read-only metadata probe.

The next actual Colab step exposed a missing CPU dependency: Transformers chat-template rendering requires Jinja2. The CPU lock now includes `Jinja2==3.1.6` and `MarkupSafe==3.0.4`, using the already reviewed CUDA versions/hashes. The isolated-environment test exercises `apply_chat_template` directly.

The notebook now captures child stdout **and** stderr, prints progress and the exact stopped reason, reports the exit code, and saves bounded sanitized logs under `/content/spark-formal/logs`. It removes known secret values and signed URL queries before display/write. Ctrl-C forwards SIGINT once and continues reading the runner's checkpoint output until exit. The Hub access check verifies pinned metadata without following a signed CDN HEAD redirect; ordinary Hub/transport errors are no longer mislabeled as missing license approval. A failed retry invalidates earlier local readiness, and GPU commands require a successful preflight for the current source revision.

For a `GatedRepo` error, review/request access at [Stable Audio Open1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) using the HF account that owns the Colab token. After approval, ensure the token permits reading that gated model and rerun the CPU cell. The pipeline never accepts terms or edits token scope. A new VM or environment reinstall is unnecessary for that access error.

The repaired release has now completed the **actual CPU preflight in Colab**: 304,861 original instructions, 315,359 views, zero context-audit failures, a committed 1,276-byte preflight artifact in Drive, and child exit code 0. Model access passed; local free storage was 218.05 decimal GB. No GPU training or instructor call occurred. [Repair report](reports/spark_multimodal/COLAB_PREFLIGHT_REPAIR_2026-10-27.md) records the evidence boundary and next gates.

## Bootstrap fix

The previous notebook used `venv.EnvBuilder(with_pip=True)` and failed in the Colab system interpreter's `ensurepip` subprocess. The traceback did not include that subprocess's output, so the exact underlying `ensurepip` failure is unknown. A second bug accepted an existing Python executable as a completed environment.

`colab/final/bootstrap.py` uses only the standard library to fetch the pinned **uv 0.13.5** wheel, verifies both wheel and extracted executable hashes, and creates a pipless environment. `uv pip sync --require-hashes --only-binary :all:` installs the complete lock, including **pip 26.2.2**, independently of system `ensurepip`. Repeated setup audits matching packages; a failed sync invalidates a stale ready receipt. Installer state and temporary downloads stay under the local work root, outside Drive and the kernel environment. PyTorch packages use `--torch-backend cu130`; unrelated packages remain on PyPI.

The original ensurepip repair added `pip` to both dependency locks. This follow-up additionally adds the missing `Jinja2`/`MarkupSafe` CPU dependencies; existing training package versions remain unchanged. The CUDA environment's new install command passes a dry run; its installation and whole-model CUDA execution still need the notebook's hardware gates.

The archive/export packaging also includes the instructor accounting helper used by the RSI stage. Static import-closure checks and an isolated subprocess import verify that those modules are available outside the author's workspace.

## Storage, data and model publication

- Training uses the pinned [TassP0lar/SparkX2.6-MM dataset](https://huggingface.co/datasets/TassP0lar/SparkX2.6-MM). Its README preserves source citations and per-source terms. This code repository contains no training-media shards, datasets or model weights.
- Active files use VM-local disk, a bounded 23GB rolling media cache, a 44GB free floor and two measured complete checkpoint slots. Do not clone all dataset LFS files into Colab's 230GB disk.
- Google Drive is **cold storage** for closed, verified checkpoints and receipts, within the 2TB project cap. Training does not perform random model/data writes through a Drive mount.
- The final completed model is uploaded using in-memory `HF_TOKEN` to the configured private **model** repository, default `TassP0lar/SparkX2.6-MM`. The dataset and model namespaces are separate. Resumable optimizer checkpoints remain in Drive.
- Original workstation checkpoints remain local. A new runtime has a new source binding and is not interchangeable with an older trained checkpoint's source identity.

## Review, tests and rebuild

`releases/SOURCE_MANIFEST.json` lists every runtime file and SHA256. `colab/final/components.lock.json`, `dataset.lock.json`, the package locks and `bootstrap.lock.json` bind model sources, data, dependencies and installer separately. The upstream vision source is the pinned archive `colab/final/vendor/vjepa2-source.zip`; its MIT license and bundled third-party notices apply. Component models and datasets retain their own upstream licenses and access requirements; this repository grants no additional rights to their weights or data.

The focused bootstrap tests require public PyPI access, CPython 3.13, and local test storage with at least 46GB free. They create/repair real environments, install CPU tooling, reject a bad package hash and verify that unrelated saved artifacts survive. They make no GPU, Drive or instructor requests. The notebook tests check schema, source inventory, immutable URL, code syntax, source tamper rejection and secret transport.

The final repair passed **59 focused tests** (56 runtime/package/data tests plus three real isolated-environment tests), including actual subprocess stdout/stderr capture, secret redaction, native signal exit, graceful interrupt drain, changed pinned-file rejection, preflight readiness invalidation, notebook/package verification and existing native-component/storage/data tests. See the [repair test log](validation/colab-preflight-final-tests-2026-10-27.log). One upstream Diffusers deprecation warning remains. The earlier [49-test baseline](validation/colab-github-formal-final-tests-2026-10-27.log), [real bootstrap recovery tests](validation/colab-bootstrap-repair-tests-2026-10-27.log) and [CUDA lock dry run](validation/colab-cuda-lock-dry-run-2026-10-27.log) are also retained. The [updated CPU install/chat-template tests](validation/colab-chat-template-bootstrap-tests-2026-10-27.log) passed after adding Jinja2/MarkupSafe; the CUDA lock and installer are unchanged. These results do not establish full-model CUDA readiness or multimodal training quality.

```bash
UV_CACHE_DIR="$PWD/.test-cache" uv run --no-project --python 3.13 \
  --with pytest==8.4.2 --with nbformat==5.10.4 \
  python -m pytest -q tests/test_final_bootstrap.py tests/test_final_notebook.py \
  --basetemp "$PWD/.test-tmp"
```

To release changes, first run `python scripts/build_final_colab.py --runtime-only`, copy its source ZIP/manifest from `output/colab-formal-final/` into `releases/`, and commit the reviewed runtime files and archive. Then run `python scripts/build_final_colab.py --runtime-revision FULL_RUNTIME_COMMIT`, copy `output/colab-formal-final/RELEASE.json` into `releases/`, test and commit the small notebook. Always publish the runtime commit as well as the launcher commit. Never replace the full source commit with a moving branch URL inside the notebook.

## Plans and references

The [formal campaign plan](reports/spark_multimodal/COLAB_CAMPAIGN_PLAN.md), [multimodal refinement review](reports/spark_multimodal/REFINEMENT_MULTIMODAL_REVIEW_2026-10-27.md) and [operator guide with component references](colab/final/RUN.md) describe the experimental scope, objective/data contracts and pending evaluations. Nested learning and unrelated architecture changes from the separate project are outside this runtime.

Installer references: [uv environments](https://docs.astral.sh/uv/pip/environments/), [uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/), [pinned uv metadata](https://pypi.org/pypi/uv/0.13.5/json), [pinned pip metadata](https://pypi.org/pypi/pip/26.2.2/json).
