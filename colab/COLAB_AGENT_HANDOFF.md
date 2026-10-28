# Direct full multimodal one-shot — current

Use [`Spark_Formal_Full_Finetune.ipynb`](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/codex/formal-colab/colab/Spark_Formal_Full_Finetune.ipynb) and [`DIRECT_FULL_MULTIMODAL.md`](DIRECT_FULL_MULTIMODAL.md). It runs `python -m colab.final.run direct`: full HF dataset and native components, all eight families, both instructor RSI rounds, HF checkpoints. It does not require CPU preflight, qualification/profile, Drive, or a prior-VM recovery proof. The former gated launcher is archived as [`Spark_Formal_Omni_Gated.ipynb`](Spark_Formal_Omni_Gated.ipynb). Historical notes below describe the gated path.

# Current direct-G4 repair — 27 October 2026

Notebook `97c90b4e95bb39e1a330367567d230a6f99a0b7b` pins runtime `4e90988168355a8a2de7025ed3dbf3bae5804cca`, ZIP `7df35726d7a16d8fb032b517aa1231efc9b168433712c57af93ede7bb7381148`. Start directly on G4, reuse one CUDA environment; do not replace a working VM after qualification. New-process full checkpoint continuation retains actual optimizer/RNG checks and records `fresh_vm` honestly. PROFILE_RECEIPT auto-loads after qualification; receipt fields require complete JSON objects, never folder IDs.

Actual user notebook1V01dsUf9HfjGUAnrmvvW1DlJFibDjqeS runs G4 RTX PRO6000 Blackwell. Reproduced missing SoX wheel, installed all179 locked packages with a hash-checked source-only SoX exception, and passed pip check, native BF16 CUDA gradients and Studio backend import.53 tests passed. No model training or instructor call occurred. Local original workstation checkpoints and HF dataset are unchanged. Read reports/spark_multimodal/G4_SETUP_REPAIR_2026-10-27.md and colab/final/RUN.md. Earlier notes below are historical.

---

# Current formal script with HF checkpoint retention — 27 October 2026

Use [the formal notebook](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/05a7ad773723e31e4d09f7282b75f93142be4437/colab/Spark_Formal_Full_Finetune.ipynb). Notebook commit `05a7ad773723e31e4d09f7282b75f93142be4437` pins runtime `669d9ce6ddbfa2d08d31bb4745008fdcd522eedc`, ZIP SHA256 `7be91268cc09294d9aa29b9b9e6bcd447268a48764ddbbabf2f27f95d7be2332`. Launcher23,071bytes; sourceZIP5,019,537bytes;43boundruntimefiles. The script retains the complete six-stage multimodal curriculum and two instructed RSI rounds.

The latest user instruction supersedes Drive-only full checkpoints: complete snapshots now go to private HF model repo `TassP0lar/SparkX2.6-MM-checkpoints`; their small receipts/reports/ledger stay cold in Drive; the final model remains private model repo `TassP0lar/SparkX2.6-MM`. After full remote file verification and Drive receipt backup, older local checkpoints are purged, keeping the newest and at most two slots during save/restore. Remote history is not squashed. Read `colab/final/RUN.md` and `reports/spark_multimodal/FORMAL_HF_CHECKPOINT_RELEASE_2026-10-27.md`.

71focusedtestspassed (68runtime/package/native/data/checkpoint tests and3realCPUbootstraptests). HF boundary tests are simulated remote failures plus real local CPU optimizer saves, not liveHFlarge-transfer evidence. This construction did not allocate a GPU, access Colab Secrets, send instructor requests, upload checkpoints or train a model. The existing live notebook was not changed in this turn; its earlier CPU PASS is for the older source. Open the new pinned notebook and run its current CPU preflight before qualification. No trained full-model checkpoint exists from this formal native source, so no source-bound training resume is claimed. All earlier notes below are historical.

---

# Current CPU preflight repair and live PASS — 27 October 2026

The user's `environment_setup_complete` followed by generic `Campaign stopped` error is resolved. Actual Colab diagnostics found HTTP403 `GatedRepo` for Stable Audio Open1.0. The user completed its access form and the same token then passed. Continuing exposed missing Jinja2 in the CPU lock; pinned Jinja2/MarkupSafe were added, with a real isolated chat-template test. The launcher now captures/redacts stdout+stderr, records bounded logs, reports the actual stopped reason, and preserves SIGINT drain. Failed preflight retries invalidate readiness.

Use [the latest immutable notebook](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/fab1fa3b6e7428c79a5c496160ae871ccdf7a047/colab/Spark_Formal_Full_Finetune.ipynb). Notebook commit `fab1fa3b6e7428c79a5c496160ae871ccdf7a047` pins runtime `6f925694cf6cb8e43ca8a4c2791903c76a50e6b1`, ZIP SHA256 `c88f69aca289f5cb57919ca09daac9a9ae4db74086652d58cea4202b37200dce`. The 19,828-byte launcher verifies 41 runtime files. Exact remote bytes were verified. 59 distinct tests passed across the 56-test focused suite and three real CPU bootstrap tests.

The existing user notebook `1W6nTYT-svs7Nko4HyJF2U1InJlRtVFiC` was updated in place and its final CPU run passed: 304,861 original instructions, 315,359 views, zero context failures, model access verified, 1,276-byte formal-preflight artifact committed to Drive, exit 0. Local free disk 218.05 decimal GB. The final read-only summary is visible in the notebook. CPU VM remains connected; GPU flags remain off. No instructor call or GPU training was made. The active source/config/Secrets and downloaded metadata are preserved.

Read `reports/spark_multimodal/COLAB_PREFLIGHT_REPAIR_2026-10-27.md`, `evidence/colab-preflight-repair-publication-2026-10-27.json`, and private local `evidence/colab-preflight-live-2026-10-27.json` (contains the exact Drive receipt). Next: full CUDA/Studio qualification, fresh-VM continuation, actual CU balance/rate and curriculum budget, provider/account eligibility for OpenCode Go dataset review. CPU success is not full training readiness. Keep Drive cold and preserve the local 44 GB free floor. All earlier notes below are historical.

---

# Current GitHub-backed formal notebook — 27 October 2026

The ensurepip setup failure and large embedded notebook are superseded by the reviewed lightweight release. Use [the immutable Colab notebook](https://colab.research.google.com/github/BaiTian6641/Spark-X2.6-VL/blob/bac6320946527102627a4682dffd34443993b6f0/colab/Spark_Formal_Full_Finetune.ipynb) from `git@github.com:BaiTian6641/Spark-X2.6-VL.git`, default branch `codex/formal-colab`. Notebook commit `bac6320946527102627a4682dffd34443993b6f0` pins runtime commit `b2bd280613bf1b82727513265cc5d154e1b7d8bd`, ZIP SHA256 `1d290e5e373e80b27c81b20eb43e97f1880a8c8f7a4d9b81a4d99ae39d2f0fdc`. The19,679-byte notebook downloads/verifies40runtime files rather than embedding the5MB ZIP.

Both environment setups now use the hash-pinned external uv installer instead of system ensurepip and repair an existing Python-without-pip environment.49focused tests passed; the remote notebook/ZIP and actual remote bootstrap extraction were verified. The new notebook loaded in connected Chrome and was left disconnected. No GPU training, Drive mutation, instructor request or model upload occurred for this repair. The full CUDA/Studio profile, fresh-VM recovery and model-quality gates remain pending.

See `reports/spark_multimodal/COLAB_GITHUB_BOOTSTRAP_REPAIR_2026-10-27.md` and `evidence/colab-github-publication-2026-10-27.json` for current identity/evidence. Local clone: `output/github-spark-x26-vl`. Only this nested repo was committed/pushed. Do not recreate the old base64 notebook. To rebuild, commit `--runtime-only` output first, then pass that complete commit to `scripts/build_final_colab.py --runtime-revision`. The earlier release and operational notes below are historical.

---

# Current formal notebook handoff — 2026-10-27

The new native multimodal notebook is `colab/Spark_Formal_Full_Finetune.ipynb`, with runtime `colab/final`, source bundle/release in `output/colab-formal-final`, and operator guide `colab/final/RUN.md`. This supersedes the older image-only qualification as the proposed runtime, not as already-observed CUDA evidence. All304,861 original instructions have passed the variant/role/context CPU audit, with315,359 complete bounded-score views. Data media stay pinned to `f472f980a6c95d3e9844f27e6045c8c09b21536a`; the separate variant catalogue revision is recorded in `colab/final/dataset.lock.json`.

No full native GPU run or instructor call was made while constructing this notebook. Begin on CPU, then earn the complete-model Studio/CUDA profile and fresh-VM continuation gates. The default final destination is the private **model** repo `TassP0lar/SparkX2.6-MM`, using Colab Secret `HF_TOKEN`. The public dataset repo contains data/notices and the new metadata companion only. Existing workstation checkpoints remain local. See `reports/spark_multimodal/FORMAL_NOTEBOOK_RELEASE_2026-10-27.md` for current evidence and remaining constraints; historical notes follow.

---

# Implement the complete Spark multimodal Colab campaign

## Latest full dataset release — 27 October 2026

The complete admitted source corpus is published and anonymously verified at `TassP0lar/SparkX2.6-MM`, commit `f472f980a6c95d3e9844f27e6045c8c09b21536a`. It contains **304,861 instruction variants/text passages**, **172,558 media packets**, and **59.157 GB of dataset artifacts** across ten multimodal components plus text replay. The current tree contains no notebooks, Python/runtime bundles or model weights. Existing workstation finetuning checkpoints remain local.

Read `reports/spark_multimodal/FULL_DATASET_RELEASE_2026-10-27.md` and `colab/formal/HF_FULL_DATASET_HANDOFF.json` for exact prefixes/index hashes, source coverage, waveform-target roles, license evidence and remaining gates. The final publication receipt is `evidence/hf-full-dataset-publication-2026-10-27.json`. Root README citations and all source-specific terms are preserved. The separate PixMo acquisition is not part of this admitted publication.

Dataset publication is complete; **full native training readiness remains false**. Qualify the role-aware grouped reader and larger index handling before allocating a paid GPU. Open Images RELEASE is 29,889,888 bytes and exceeds the old 16 MiB fixture preflight bound. Preserve the immutable old qualification ZIP; prepare a reviewed new runtime with all native modality branches, actual full-optimizer/decoder gradients, memory profiling and fresh-VM recovery. Keep active data on local disk and Drive cold, preserving the 44 GB free floor and two measured checkpoint slots. No new GPU training or instructor call was performed for this dataset release.

## Earlier qualification and preparation history

The observations below are historical and were not refreshed by the CPU dataset-publication work. The full-source state above supersedes the older DOCCI-only or pending-corpus descriptions.

Current continuation (27 October; browser observation approximately 15:03 CDT):
Colab showed **zero active sessions**, approximately zero CU/hour, 1,098.65 CU
and a disconnected notebook. The fresh-VM restore that began at
18:34:08.948645 UTC has **no observed PASS**. Its continuation, new snapshot,
Drive-mount and instructor-readiness cells were unexecuted at that observation.
Do not call a disconnected runtime successful recovery or infer why it ended.
The previous complete checkpoint remains committed in Drive. No new instructor
request or formal training has been observed. Refresh actual runtime/balance
before any new GPU allocation; CPU corpus work requires no G4.

The user's new storage route is authoritative: build sources locally, publish
admitted training media to `TassP0lar/SparkX2.6-MM`, use a bounded rolling cache
on Colab local disk and keep Drive as **cold** storage. The public full source
at commit `a1190c8098162cb45630b0f8ceb23d96b7cf3664` contains 8,776 original
DOCCI captions/images (strict local-train ∩ upstream-train), in two indexed TARs
totaling 2,104,033,280 bytes. Historical exposure is preserved; this is not fresh
evaluation data, complete EN/ZH coverage or a full multimodal training corpus.
The separate 64-image fixture intentionally overlaps it. PixMo media are not
approved for wholesale redistribution from the annotation license alone.

Read `reports/spark_multimodal/HF_ROLLING_DATASET_PLAN_2026-10-27.md` and
`reports/spark_multimodal/REFINEMENT_MULTIMODAL_REVIEW_2026-10-27.md` before
continuing. Only multimodal parts of supplied plan 24 apply here. Select pinned
So400m NaFlex with real-grid 2×2 downshuffle and bounded tiles; retain native
speech/general-sound output. The 230-decimal-GB disk profiles reserve two measured
checkpoint slots and a 44-GB free floor. Cache/transport host tests do not pass
trainer integration, fresh-VM recovery, native AV, generation, quality or RSI gates.

The real larger caption-source CPU acquisition is running locally, exec session
89375. It uses pinned PixMo-Cap metadata and 200,000 frozen candidates, targeting
102,000 distinct processed images. Read its live progress file instead of
relaunching acquisition. Finalization requires completed acquisition, excludes
historical media identities, groups near duplicates before splitting and stops
if fewer than 100,000 train plus three 500-image held-out sets remain. Caption
proxies are weak labels. This English source does not complete the EN/ZH corpus.

The new formal preparation R4 ZIP is `1Nkdl2hr9v7YXV7OvXNbTFSnTOmFJK13r`, SHA
`cb78a609ee22a2debf95462d6c8362e513e48b9efb7f93b52264130d4761d29d`.
The CPU storage notebook is `1qDyHtHmeSdx6nT1XcXPYglco5Mz5hz-F`; it has not run.
Its source identity is separate from the immutable CUDA qualification source.
R4 includes deterministic local TAR packaging, verified sequential Drive commits
and local restore, with 18 host tests passed. The new dataset helper's actual
Drive round trip is still pending. `drive_download.py` improves future transfers;
it has NOT been patched into the active restore. Use local random I/O and keep
Drive for closed permanent artifacts, per the latest user instruction. Preserve
2 TB / US$50 / proposed 1,000-CU limits. See `formal/README.md`, `campaign_plan.json`
and `evidence/formal-v2-storage-release.json` for the source-bound state.

Read `reports/spark_multimodal/COLAB_CAMPAIGN_PLAN.md` first, then the architecture, implementation status, checkpoint-2751 evaluation and both RSI documents. This source bundle is a planning/reference release. It is not a CUDA-qualified full training application.

The subsequent bounded Stage 1 release is separate. The live notebook at
https://colab.research.google.com/drive/1aerQV3Aod7tjQ7RgvkWqKpcKiu80GWqY
has passed the actual Studio import, CUDA optimizer test, original-model file
verification, eight full-parameter updates and a new-process continuation to step
ten. See `colab/README.md` and `evidence/colab_stage1_drive_release.json` for the
latest checkpoint persistence status and receipt identities. Reuse that evidence;
do not repeat training to work around an upload failure. The original source ZIP
is immutable; later notebook fixes use the same archive ID/hash and do not alter
the trained implementation. Fresh-VM checkpoint recovery, uninterrupted-trajectory
comparison, whole-model profiling, native audio/temporal branches, the formal
corpus and RSI remain incomplete.

The user requires the full English/Chinese multimodal endpoint: images and spatial primitives; real video, tracks, occlusion and time lapse; speech and general-sound understanding; native speech and short-sound output; then two bounded instructed RSI rounds. Use the Unsloth Studio backend and full Spark parameter updates in the main stages. Nested-learning architecture changes remain separate.

Keep old trained checkpoints on the workstation. Do not request or upload them. Download the original pretrained Spark-X2.6-4B and selected pretrained components directly inside Colab at pinned revisions; create a fresh multimodal alignment lineage. The checkpoint-2751 evaluation is historical context, not a new paired CUDA baseline. Never include `runtime/secrets/opencode.env`, credentials, browser profiles or local virtualenvs in uploads.

The original Spark revision is `66e3ae7bee54e0a523d692cb8ef841f35409807b`. Its public metadata and custom source access were verified for this handoff; see `evidence/colab-original-spark-revision.json`. Do not follow the changed upstream `main` without a separately reviewed baseline change.

The authorized project storage allocation is 2 TB of Google Drive. The teacher allowance is US$50 total. The proposed project compute allocation is 1,000 CU. At approximately 12:45 CDT on 27 October 2026, the live Resources panel showed 1,110.18 available CU, a rate of approximately 8.9 CU/hour, one active session and the Colab Pro+ label. This does not allocate the remaining account balance or prove a particular subscription purchase. Refresh the balance and rate before subsequent sessions; do not silently sum grants. The older 8.71-CU rate is a planning scenario. Storage benefits do not make Colab runtime disk persistent.

Start with the supplied preflight notebook. It performs real infrastructure checks and a bounded Drive upload/download test without training or teacher API calls. Confirm the same Google Account can write the explicit project folder, check quota and local free disk, verify assigned GPU memory and BF16, and repeat the probe restore in a fresh runtime. An infrastructure pass is not a Spark CUDA or optimizer-resume pass.

Implementation order:

1. Create a separate Colab source/runtime release. Port the HIP-only requirement in `scripts/train_multimodal_studio.py` and the fixed roots in `scripts/env.sh`. Preserve historical ROCm source and exact-resume checks. Install the actual Studio backend, and qualify Spark's custom code against a pinned CUDA/Python/Transformers/Unsloth stack. Resolve the Spark 4.57.1 versus Qwen3-TTS 4.57.3 compatibility question explicitly. Do not generate a fake environment lock or copy ROCm wheels.
2. Implement Drive storage and recovery as specified by the plan: stable IDs, immutable sharded datasets, local working set, complete local snapshots, resumable uploads, size/hash checks and commit receipts. Only verified commits are recoverable. Prevent concurrent campaign writers, account for storage/upload quotas and test interruption/corruption cases. The supplied preflight helpers are not the production checkpoint transaction manager.
3. Add and test the temporal branch, audio input modules, Spark final hidden-state access and both native audio-output bridges. Verify losses reach Spark and preserve aliases/masks/timestamps. Profile all components together. No detached external text-to-speech substitute can pass the native-output gate.
4. Prepare the larger category/task/source/language-balanced corpus with independent source grouping, real-media supervision, visibility checks and separate development/promotion/final-test data. Preserve existing frozen records; use a new version. Keep final test inputs unavailable to teacher/curation workers.
5. Implement the instructor's typed instructions, request/billing guards, cache, independent target verification and rejection log. Use Colab Secrets for `OPENCODE_API_KEY`. Confirm the selected OpenCode route supports the batch teaching workload; retain `mimo-v2.6-pro` unless the human changes the choice. Broader teacher competence is unproven. Unknown request/billing outcomes must be reconciled before more calls.
6. Implement an idempotent, budget-aware stage runner with source/data/environment/parent bindings and exact sample/shard resume state. Add the main training and evaluation notebook only after real CUDA gradient, checkpoint and fresh-runtime continuation tests pass. Until then, report missing prerequisites explicitly instead of enabling a launcher that skips them.
7. Run alignment, image/spatial/OCR, real temporal/video, audio input/AV, audio output and joint full training with capability gates. Then perform the two adaptive instructed rounds plus matched static/continued-training controls, preserving the pre-RSI parent. The teacher cannot modify verifiers, dependency installation or promotion criteria.
8. Freeze the winner and run the untouched final audit once. Export all implemented branches, processors and configurations, an inference entrypoint, a resumable checkpoint and the complete capability/cost report. Failed stages remain marked failed or incomplete. The original three-arm/three-round RSI research comparison is not claimed by this smaller campaign.

Use real source/media/runtime evidence. Host unit tests, small random tensor checks and a Drive byte probe cannot establish model quality, audio intelligibility or full-checkpoint recovery. Required human audio review remains an explicit pending gate until performed. Advance without repeated permission for already authorized reversible preparation, but do not purchase plans, change account sharing, expand budgets or publish models.


The local CPU follow-on worker is exec session 79020, running
`scripts/prepare_formal_caption_release.py` with the R4-verified source under
`/tmp/spark-formal-r4-verified` and the isolated hash-locked CPU venv. It waits on
the real acquisition lock, then verifies/groups, freezes 100000/500/500/500 and
packages three separate role archives. Its mutable status is
`data/colab_formal_v2/release-preparation-progress.json`; its log is
`data/colab_formal_v2/release-preparation.log`. It never uploads or starts training.
A failed acquisition or insufficient grouped population stops it explicitly.
The previous R3 source/notebook and status artifacts remain preserved; use R4.
Automatic approval review rejected overwriting the earlier Drive notebook, so
R4 is a separately uploaded file, not an in-place replacement.
