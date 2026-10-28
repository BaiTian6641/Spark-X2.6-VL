# Colab CPU preflight repair — 27 October 2026

This repair addresses the user's successful CPU package setup followed by a generic `Campaign stopped` exception. It does not change the model architecture, corpus, source splits, training schedule or instructor budget.

## Observed causes and resolution

1. The original subprocess inherited stdout/stderr. Its useful error event did not appear in the Colab cell, leaving only a wrapper traceback. A captured-output diagnostic showed that the actual error was access to `stabilityai/stable-audio-open-1.0`.
2. A read-only probe in the existing Colab VM verified the expected HF publishing account and a fine-grained token. The Hub itself returned HTTP403 with `X-Error-Code: GatedRepo`, explicitly reporting that the account was outside the authorized list. This was not a signed-download redirect error. The user completed the model access form. The same token then obtained HTTP302 from the Hub and successful pinned-file metadata, without downloading model weights.
3. Continuing preflight reached the corpus audit and exposed a missing optional CPU dependency: `apply_chat_template requires jinja2`. The CPU input/lock now includes `Jinja2==3.1.6` and `MarkupSafe==3.0.4`, using the versions and wheel hashes already present in the CUDA lock. Existing package versions and the CUDA lock remain unchanged.

The observed model access requirement matches the [upstream Stable Audio model page](https://huggingface.co/stabilityai/stable-audio-open-1.0). It requires the account holder's access form/license decision. No terms were accepted by the runtime and no token scope was edited by this repair.

## Runtime changes

`colab/final/launcher.py` routes both child file descriptors through Python output so Colab captures them. It redacts known credential values, bearer tokens and signed URL queries before displaying or writing output. Each invocation receives a unique bounded sanitized log under `/content/spark-formal/logs`; raw credentials remain in memory/stdin. Oversized output lines are discarded in their entirety to avoid leaking a secret across read boundaries. Errors include the child exit code and actual stopped reason. Ctrl-C signals the runner once and drains output until its checkpoint/shutdown path exits.

Model access now uses Hub file metadata, checking pinned revision, byte count and SHA256 instead of following the signed CDN URL with HEAD. Gated-repository denial, ordinary Hub errors, transport failures and metadata mismatch have different diagnostics. The deprecated coarse token-permission call is replaced with the `whoami` token-role field; fine-grained tokens are not treated as proof of model-publication write permission.

Preflight prints stage progress and corpus counters. A failed retry invalidates an earlier successful local readiness receipt. Training/qualification requires a successful receipt bound to the current source archive. A readable diagnostic is not itself a successful preflight.

## Published artifacts

- Repository: [BaiTian6641/Spark-X2.6-VL](https://github.com/BaiTian6641/Spark-X2.6-VL), branch `codex/formal-colab`.
- [Immutable Colab launcher](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/fab1fa3b6e7428c79a5c496160ae871ccdf7a047/colab/Spark_Formal_Full_Finetune.ipynb), notebook commit `fab1fa3b6e7428c79a5c496160ae871ccdf7a047`.
- Runtime commit: `6f925694cf6cb8e43ca8a4c2791903c76a50e6b1`.
- Notebook: 19,828 bytes, SHA256 `47a21bea3318548f77e6df7e4c49c402ccd2edc631b3111d8db5b1cd89f14de2`.
- Runtime ZIP: 5,010,530 bytes, 41 exact source files, SHA256 `c88f69aca289f5cb57919ca09daac9a9ae4db74086652d58cea4202b37200dce`.
- CPU lock SHA256: `5e7f42b31f74192a22d0f7ba7457e77815381a14f6d51182606c4ae43f336992`.

Both public artifacts were downloaded and checked against their local SHA256; every runtime manifest file matched the reviewed source. The current user-owned Colab notebook was updated in place, preserving its configuration, authenticated session and local data. Notebook source extraction verified the new archive before execution. The temporary access probe was replaced with a read-only results summary. GPU execution switches remain off.

## Verification

The final repair passed 59 distinct focused tests: 56 subprocess/runtime/native-component/notebook/data/storage tests and three real isolated-CPU environment tests. The latter install the exact lock and execute `PreTrainedTokenizerFast.apply_chat_template`, covering the optional dependency missed by imports and `pip check` alone. The tests also exercise child stdout/stderr capture under Python output capture, secret redaction, native signal death, graceful interrupt drain, frozen metadata rejection and readiness invalidation. One upstream Diffusers deprecation warning remains.

Logs: `evidence/colab-preflight-final-tests-2026-10-27.log` and `evidence/colab-chat-template-bootstrap-tests-2026-10-27.log`. Publication binding: `evidence/colab-preflight-repair-publication-2026-10-27.json`.

## Live Colab result

**PASS in the user's existing CPU Colab VM.** The final source-bound invocation emitted `cpu_preflight_complete` with `passed: true`, committed its `formal-preflight` artifact to the intended Drive project and exited with code 0. The queued read-only summary confirmed the same source archive, 304,861 original instructions, 315,359 derived training views and zero context-audit failures. Local free storage was 218,051,256,320 bytes (218.05 decimal GB).

| Family | Checked views |
| --- | ---: |
| Image | 164,189 |
| Video | 8,862 |
| Audiovisual | 2,100 |
| Speech understanding | 32,890 |
| Speech generation | 32,890 |
| Sound understanding | 22,463 |
| Sound/music generation | 11,965 |
| Text retention | 40,000 |

The frozen split membership SHA256 is `3b9dad9b1b4be8c3c82ec82e7ed7c36c34b500bd3f68517e6e8ee6167cc80d63`. The Drive artifact contains 1,276 bytes; the committed receipt is retained in the existing notebook output and local `evidence/colab-preflight-live-2026-10-27.json`. Private Drive identifiers are not published in this code repository. A screenshot is retained locally at `evidence/colab-preflight-passed-2026-10-27.jpg`.

This confirms the metadata/source-split/token-length audit and small closed-artifact Drive commit. It does not measure large optimizer-checkpoint bandwidth, complete GPU memory use or trained multimodal quality. `training_ready` remains false until the listed subsequent gates pass.

Full-model CUDA qualification, fresh-VM optimizer continuation, full formal training, instructor-guided RSI and model-quality acceptance remain separate requirements. No GPU training or instructor request was made during this repair.
