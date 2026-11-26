#!/usr/bin/env bash
set -euo pipefail

cd /home/baitian6641/Spark-X2.6-VL

if [[ -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY=".venv-blackwell/bin/python"
OUT_ROOT="work/spark-x26-4b-loop48-en1b-vl-stage13-prism-rlvr"
BF16_DIR="${OUT_ROOT}/final-model-stage13-bf16"
FP8_DIR="${OUT_ROOT}/final-model-stage13-fp8"
mkdir -p "${OUT_ROOT}"

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage-13 24h Multimodal + VC-GRPO Campaign Starting ==="

"${PY}" scripts/run_stage13_24h_multimodal_rlvr.py \
  --init-release-dir work/spark-x26-4b-loop48-en1b-vl-stage12-frontier-vl/final-model-stage12-bf16 \
  --output-root "${OUT_ROOT}" \
  --release-tag stage13 \
  --phase13a-updates 10500 \
  --phase13a-batch-size 8 \
  --phase13a-max-hours 9.0 \
  --phase13b-updates 950 \
  --grad-accum 12 \
  --unfreeze-vision-layers 8 \
  --visual-margin-every-n 3 \
  --visual-margin-target 0.35 \
  --visual-margin-boost 1.6 \
  --phase13c-max-prompts 1200 \
  --phase13c-group-size 4 \
  --phase13c-grad-accum-prompts 4 \
  --phase13c-max-hours 5.5 \
  --intel-vision xpu \
  --vision-optimizer muon8bit

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage-13 Training & Export Complete -> Running Independent Counterfactual Audit ==="

"${PY}" scripts/verify_stage12_visual_causality.py \
  --model-dir "${BF16_DIR}" \
  --out-json "${OUT_ROOT}/STAGE13_VISUAL_CAUSAL_REVERIFY.json" \
  --scale 0.6

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running Standard Multimodal Benchmark Suite (BF16 + FP8+q4g128, scale=0.6) ==="

"${PY}" scripts/eval_standard_multimodal_benchmarks.py \
  --out-json "${OUT_ROOT}/STAGE13_STANDARD_MULTIMODAL_BENCHMARK_REPORT.json" \
  --scale 0.6 \
  --models "stage13_bf16=${BF16_DIR}" "stage13_fp8=${FP8_DIR}"

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running 3D Spatial + Frontier Visual/SyncCraft Suite (BF16) ==="

"${PY}" scripts/eval_stage9_frontier_and_3d_suite.py \
  --model "stage13_bf16=${BF16_DIR}" \
  --output-dir "${OUT_ROOT}/frontier_3d_eval" \
  --benchmarks "cvbench3d,3dsrbench,mmmu_pro,v28_synccraft,tb41,gpqa,simpleqa"

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage-13 24h Campaign & All Evaluations Finished Successfully! ==="
