#!/usr/bin/env bash
# ==============================================================================
# Automated Setup, Qualification & Full-Finetune Script for Spark-X2.6-VL + SyncCraft v3.2
# Target Workstation: NVIDIA RTX PRO 4000 Blackwell SFF (24 GB GDDR7, SM120)
# Backend: APOLLO + COAT (FP8 E4M3 activations) + Merged Gradient Accumulator
#
# Security & Hygiene:
# - Never logs or echoes HF_TOKEN / OPENCODE_API_KEY; secrets pass via stdin JSON.
# - Keeps checkpoints, datasets, caches, and model exports local by default.
# - Records exact hardware & pip dependency locks under <WORK_ROOT>/locks/.
# ==============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_SYNCCRAFT_DIR="$(cd "${REPO_ROOT}/.." && pwd)/SyncCraft"

MODE="${MODE:-all}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/work/spark-synccraft-24gb}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-blackwell}"
SYNCCRAFT_DIR="${SYNCCRAFT_DIR:-${DEFAULT_SYNCCRAFT_DIR}}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/colab/final/campaign_rtx_pro_4000_24gb.json}"
DATASET_MODE="${DATASET_MODE:-local_synccraft}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-}"
ABLATE_STEPS="${ABLATE_STEPS:-10}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_ABLATE="${SKIP_ABLATE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_workstation_rtx_pro_4000.sh [OPTIONS]

Automated venv bootstrap, SM120 hardware qualification, 4-rung APOLLO+COAT ablation,
and resumable multi-stage training for Spark-X2.6-VL + SyncCraft v3.2 on a 24 GB
NVIDIA RTX PRO 4000 Blackwell SFF workstation.

Options:
  --mode <mode>            Execution mode (default: all):
                             setup-only   Create venv, install CUDA 12.8+/SM120 deps & SyncCraft
                             smoke-cpu    Setup venv + run fast CPU unit & SyncCraft SAT suites
                             ablate-only  Setup + doctor + assets + live SM120 4-rung VRAM ablation
                             pilot        Run Stage-1 frozen-Spark visual connector pilot only
                             all | full   Full pipeline: setup -> doctor -> data/assets -> ablate
                                          -> qualify -> cross-process recovery -> all 4 stages
                                          -> protected audit -> local final-model export
  --work-root <path>       Campaign NVMe working directory (default: ./work/spark-synccraft-24gb)
  --venv <path>            Python virtualenv path (default: ./.venv-blackwell)
  --synccraft-dir <path>   Path to sibling SyncCraft repo (default: ../SyncCraft)
  --config <path>          Campaign JSON config (default: colab/final/campaign_rtx_pro_4000_24gb.json)
  --dataset-mode <mode>    local_synccraft (default) or hf_full (304k HF multimodal corpus)
  --torch-index-url <url>  PyTorch CUDA wheel index (default: https://download.pytorch.org/whl/cu128)
  --python <bin>           Base Python >=3.11 interpreter to create venv
  --ablate-steps <int>     Optimizer updates per profile during live VRAM ablation (default: 10)
  --skip-install           Reuse an already-populated virtualenv without running pip install
  --skip-tests             Skip preflight pytest suite
  --skip-ablate            Skip 4-rung optimizer ablation if you want to jump straight to training
  -h, --help               Show this help message

Environment Variables:
  HF_TOKEN                 Hugging Face read token (required on first run to fetch Spark-X2.6-4B
                           and SigLIP2-SO400M-NaFlex weights; never written to disk or argv)
  OPENCODE_API_KEY         Optional; only needed if instructor_rounds > 0 is enabled in config
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)            MODE="$2"; shift 2 ;;
    --work-root|--root) WORK_ROOT="$2"; shift 2 ;;
    --venv)            VENV_DIR="$2"; shift 2 ;;
    --synccraft-dir)   SYNCCRAFT_DIR="$2"; shift 2 ;;
    --config)          CONFIG_PATH="$2"; shift 2 ;;
    --dataset-mode)    DATASET_MODE="$2"; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL="$2"; shift 2 ;;
    --python)          PYTHON_BIN="$2"; shift 2 ;;
    --ablate-steps)    ABLATE_STEPS="$2"; shift 2 ;;
    --skip-install)    SKIP_INSTALL=1; shift ;;
    --skip-tests)      SKIP_TESTS=1; shift ;;
    --skip-ablate)     SKIP_ABLATE=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  setup-only|smoke-cpu|ablate-only|pilot|all|full) ;;
  *)
    echo "[ERROR] Unsupported --mode '${MODE}'. Choose: setup-only, smoke-cpu, ablate-only, pilot, all, full." >&2
    exit 2
    ;;
esac

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/locks"
WORK_ROOT="$(cd "${WORK_ROOT}" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${WORK_ROOT}/logs/workstation_${MODE}_${TIMESTAMP}.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}"
}

die() {
  printf '[%s] [FATAL] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}" >&2
  exit 1
}

resolve_base_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Requested Python binary not found: ${PYTHON_BIN}"
    echo "${PYTHON_BIN}"
    return
  fi
  local candidate
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        echo "${candidate}"
        return
      fi
    fi
  done
  die "Python >= 3.11 is required on the workstation (python3.12 recommended)."
}

log "======================================================================"
log "Spark-X2.6-VL + SyncCraft v3.2 Workstation Runner (mode=${MODE})"
log "  Repo root     : ${REPO_ROOT}"
log "  SyncCraft dir : ${SYNCCRAFT_DIR}"
log "  Virtualenv    : ${VENV_DIR}"
log "  Work root     : ${WORK_ROOT}"
log "  Config        : ${CONFIG_PATH}"
log "  Dataset mode  : ${DATASET_MODE}"
log "  Log file      : ${LOG_FILE}"
log "======================================================================"

[[ -f "${CONFIG_PATH}" ]] || die "Campaign config not found at ${CONFIG_PATH}"
[[ -f "${SYNCCRAFT_DIR}/pyproject.toml" ]] || die "SyncCraft repository not found at ${SYNCCRAFT_DIR} (override with --synccraft-dir /path/to/SyncCraft)"

# ------------------------------------------------------------------------------
# Phase 1: Virtualenv Setup & Dependency Installation
# ------------------------------------------------------------------------------
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  BASE_PY="$(resolve_base_python)"
  log "[Phase 1/6] Creating Python virtualenv at ${VENV_DIR} using ${BASE_PY}..."
  "${BASE_PY}" -m venv "${VENV_DIR}"
else
  log "[Phase 1/6] Reusing existing virtualenv at ${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
export PYTHON="${VENV_PY}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="false"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  log "[Phase 1/6] Upgrading pip, setuptools, and wheel..."
  "${VENV_PY}" -m pip install --upgrade pip setuptools wheel >>"${LOG_FILE}" 2>&1

  NEED_TORCH=0
  if ! "${VENV_PY}" -c 'import torch; raise SystemExit(0 if (torch.cuda.is_available() or __import__("sys").platform == "darwin") else 1)' 2>/dev/null; then
    NEED_TORCH=1
  fi

  if [[ "${NEED_TORCH}" == "1" ]]; then
    if [[ "$(uname -s)" == "Linux" ]]; then
      log "[Phase 1/6] Installing CUDA 12.8+ Blackwell (SM120) PyTorch from ${TORCH_INDEX_URL}..."
      "${VENV_PY}" -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}" 2>&1 | tee -a "${LOG_FILE}"
    else
      log "[Phase 1/6] Non-Linux host detected ($(uname -s)); installing standard PyTorch..."
      "${VENV_PY}" -m pip install --upgrade torch 2>&1 | tee -a "${LOG_FILE}"
    fi
  else
    log "[Phase 1/6] Compatible PyTorch already installed in ${VENV_DIR}"
  fi

  log "[Phase 1/6] Installing Spark-X2.6-VL vision_text + SyncCraft runtime dependencies..."
  "${VENV_PY}" -m pip install \
    "transformers>=4.51.0,<5.0.0" \
    "tokenizers>=0.21.0" \
    "safetensors>=0.4.5" \
    "huggingface-hub>=0.28.0" \
    "accelerate>=1.2.0" \
    "sentencepiece>=0.2.0" \
    "einops>=0.8.0" \
    "timm>=1.0.12" \
    "Pillow>=10.4.0,<14.0.0" \
    "numpy>=1.26.4,<3.0.0" \
    "scipy>=1.13.0" \
    "psutil>=6.0.0" \
    "requests>=2.32.0" \
    "jinja2>=3.1.4" \
    "markupsafe>=2.1.5" \
    "lark>=1.1.9,<2.0.0" \
    "pyarrow>=15.0.0" \
    "nbformat>=5.10.0,<6.0.0" \
    "pytest>=8.0.0,<10.0.0" 2>&1 | tee -a "${LOG_FILE}"

  log "[Phase 1/6] Installing SyncCraft v3.2 in editable mode (--no-deps to preserve CUDA PyTorch)..."
  "${VENV_PY}" -m pip install --no-build-isolation --no-deps -e "${SYNCCRAFT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
else
  log "[Phase 1/6] Skipping package installation (--skip-install)"
fi

# Record exact hardware & dependency lock per SyncCraft AGENTS.md reproducibility rule
"${VENV_PY}" -m pip freeze > "${WORK_ROOT}/locks/pip_freeze_${TIMESTAMP}.txt"
cp -f "${WORK_ROOT}/locks/pip_freeze_${TIMESTAMP}.txt" "${WORK_ROOT}/locks/pip_freeze_latest.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "${WORK_ROOT}/locks/nvidia_smi_${TIMESTAMP}.txt" || true
fi

if [[ "${MODE}" == "setup-only" ]]; then
  log "Setup complete (--mode setup-only). Lock recorded at ${WORK_ROOT}/locks/pip_freeze_latest.txt"
  exit 0
fi

# ------------------------------------------------------------------------------
# Phase 2: Pinned Source Manifest, Analytical 24 GB Budget & Fast Unit Suite
# ------------------------------------------------------------------------------
log "[Phase 2/6] Verifying source lineage manifest and 24 GB VRAM budget..."
"${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" manifest | tee "${WORK_ROOT}/locks/source_manifest.json" | tee -a "${LOG_FILE}" >/dev/null
"${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" budget | tee "${WORK_ROOT}/locks/analytical_vram_budget_24gb.json" | tee -a "${LOG_FILE}" >/dev/null

if [[ "${SKIP_TESTS}" != "1" ]]; then
  log "[Phase 2/6] Running APOLLO+COAT+MergedGrad & Spark-Vision-SyncCraft verification tests..."
  "${VENV_PY}" -m pytest \
    "${REPO_ROOT}/tests/test_apollo_coat_merged.py" \
    "${REPO_ROOT}/tests/test_spark_vision_synccraft.py" \
    -q 2>&1 | tee -a "${LOG_FILE}"
else
  log "[Phase 2/6] Skipping unit tests (--skip-tests)"
fi

if [[ "${MODE}" == "smoke-cpu" ]]; then
  log "CPU smoke verification passed (--mode smoke-cpu)."
  exit 0
fi

# ------------------------------------------------------------------------------
# Phase 3: Blackwell SM120 GPU, BF16, COAT FP8 Kernel & NVMe Envelope Doctor
# ------------------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Run GPU modes on your RTX PRO 4000 Blackwell workstation."
log "[Phase 3/6] Running SM120 Blackwell GPU + COAT FP8 kernel doctor..."
"${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" doctor \
  --config "${CONFIG_PATH}" \
  --work-root "${WORK_ROOT}" \
  --dataset-mode "${DATASET_MODE}" 2>&1 | tee -a "${LOG_FILE}"

# ------------------------------------------------------------------------------
# Phase 4: Clean-Room SyncCraft v3.2 Shards & Pretrained Spark/SigLIP2 Weights
# ------------------------------------------------------------------------------
if [[ "${DATASET_MODE}" == "local_synccraft" ]]; then
  log "[Phase 4/6] Preparing deterministic clean-room SyncCraft v3.2 TAR shards & frozen SQLite splits..."
  "${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" prepare-synccraft \
    --config "${CONFIG_PATH}" \
    --work-root "${WORK_ROOT}" \
    --dataset-mode "${DATASET_MODE}" 2>&1 | tee -a "${LOG_FILE}"
fi

# Check whether weights need initial download; never print HF_TOKEN
set +x
if [[ ! -f "${WORK_ROOT}/asset-receipt.json" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  die "HF_TOKEN is not set in environment and ${WORK_ROOT}/asset-receipt.json does not exist yet. Export HF_TOKEN=<your_hf_read_token> for the initial Spark-X2.6-4B + SigLIP2 download."
fi

log "[Phase 4/6] Verifying / fetching pinned Spark-X2.6-4B & SigLIP2-SO400M-NaFlex assets (vision_text mode)..."
"${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" prepare-assets \
  --config "${CONFIG_PATH}" \
  --work-root "${WORK_ROOT}" \
  --dataset-mode "${DATASET_MODE}" 2>&1 | tee -a "${LOG_FILE}"

# ------------------------------------------------------------------------------
# Phase 5: Live SM120 4-Rung Optimizer & Memory Ablation
# ------------------------------------------------------------------------------
if [[ "${MODE}" == "ablate-only" ]] || { [[ "${SKIP_ABLATE}" != "1" ]] && [[ ! -f "${WORK_ROOT}/ABLATION_PROFILES.json" ]]; }; then
  log "[Phase 5/6] Running live SM120 4-rung matched ablation (steps=${ABLATE_STEPS})..."
  "${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" ablate \
    --config "${CONFIG_PATH}" \
    --work-root "${WORK_ROOT}" \
    --dataset-mode "${DATASET_MODE}" \
    --steps "${ABLATE_STEPS}" 2>&1 | tee -a "${LOG_FILE}"
  log "[Phase 5/6] Live ablation saved to ${WORK_ROOT}/ABLATION_PROFILES.json"
else
  log "[Phase 5/6] Reusing existing ${WORK_ROOT}/ABLATION_PROFILES.json (or --skip-ablate set)"
fi

if [[ "${MODE}" == "ablate-only" ]]; then
  log "SM120 optimizer/memory ablation complete (--mode ablate-only)."
  exit 0
fi

# ------------------------------------------------------------------------------
# Phase 6: Qualification, Cross-Process Recovery Proof & Staged APOLLO+COAT Run
# ------------------------------------------------------------------------------
if [[ "${MODE}" == "pilot" ]]; then
  log "[Phase 6/6] Running Stage-1 Visual Connector Pilot (qualify -> recover -> pilot)..."
  if [[ ! -f "${WORK_ROOT}/QUALIFICATION.json" ]]; then
    "${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" qualify \
      --config "${CONFIG_PATH}" \
      --work-root "${WORK_ROOT}" \
      --dataset-mode "${DATASET_MODE}" 2>&1 | tee -a "${LOG_FILE}"
  fi
  if [[ ! -f "${WORK_ROOT}/FULL_RECOVERY_PROOF.json" ]]; then
    "${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" recover \
      --config "${CONFIG_PATH}" \
      --work-root "${WORK_ROOT}" \
      --dataset-mode "${DATASET_MODE}" 2>&1 | tee -a "${LOG_FILE}"
  fi
  "${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" pilot \
    --config "${CONFIG_PATH}" \
    --work-root "${WORK_ROOT}" \
    --dataset-mode "${DATASET_MODE}" 2>&1 | tee -a "${LOG_FILE}"
  log "Pilot stage complete. Artifacts stored under ${WORK_ROOT}"
  exit 0
fi

# Full automated campaign supervisor loop:
# - Exit code 10 (DIRECT_QUALIFICATION_EXIT): qualification finished; relaunch in a fresh
#   OS process to prove cross-process APOLLO+COAT checkpoint recovery before formal training.
# - Exit code 3  (DIRECT_SESSION_EXIT): session wall-clock slice reached; relaunch in a fresh
#   OS process to resume seamlessly from the latest committed local checkpoint.
log "[Phase 6/6] Launching full resumable APOLLO+COAT campaign supervisor..."
SESSION_IDX=0
while true; do
  SESSION_IDX=$((SESSION_IDX + 1))
  log "[Supervisor] Starting campaign process #${SESSION_IDX}..."
  set +e
  "${VENV_PY}" "${REPO_ROOT}/scripts/launch_spark_synccraft.py" direct \
    --config "${CONFIG_PATH}" \
    --work-root "${WORK_ROOT}" \
    --dataset-mode "${DATASET_MODE}" 2>&1 | tee -a "${LOG_FILE}"
  RC="${PIPESTATUS[0]}"
  set -e

  if [[ "${RC}" -eq 0 ]]; then
    log "======================================================================"
    log "Campaign completed successfully!"
    log "  Closed inference model : ${WORK_ROOT}/final-model"
    log "  Release manifest       : ${WORK_ROOT}/final-model/MODEL_RELEASE.json"
    log "  Protected audit report : ${WORK_ROOT}/evaluations/final-protected-audit/REPORT.json"
    log "  SM120 ablation report  : ${WORK_ROOT}/ABLATION_PROFILES.json"
    log "======================================================================"
    exit 0
  elif [[ "${RC}" -eq 10 ]]; then
    log "[Supervisor] Qualification finished (rc=10); spawning fresh process for checkpoint recovery proof & training..."
    continue
  elif [[ "${RC}" -eq 3 ]]; then
    log "[Supervisor] Session slice boundary reached (rc=3); spawning fresh process to resume from committed checkpoint..."
    continue
  else
    die "Campaign process exited with non-resumable status code ${RC}. Inspect ${LOG_FILE} and re-run this exact command to resume safely."
  fi
done
