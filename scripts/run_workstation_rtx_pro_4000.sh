#!/usr/bin/env bash
# ==============================================================================
# Automated Setup, Qualification & Full-Finetune Script for Spark-X2.6-VL + SyncCraft v3.2
# Target Workstation: NVIDIA RTX PRO 4000 Blackwell SFF (24 GB GDDR7, SM120)
# Backend: APOLLO + COAT (FP8 E4M3 activations) + Merged Gradient Accumulator
#
# Self-Healing CUDA / Blackwell Features:
# - Auto-detects driver CUDA version (13.x -> cu130, 12.8/12.9 -> cu128, with fallback).
# - Initializes /dev/nvidia-uvm via nvidia-modprobe & diagnoses nvidia-open / group issues.
# - Locks installed +cu130/+cu128 torch via pip constraints so PyPI deps never clobber it.
# - Never logs or echoes HF_TOKEN / OPENCODE_API_KEY; secrets pass via stdin JSON.
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
TORCH_INDEX_URL="${TORCH_INDEX_URL:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"
ABLATE_STEPS="${ABLATE_STEPS:-10}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_ABLATE="${SKIP_ABLATE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
FORCE_REINSTALL_TORCH="${FORCE_REINSTALL_TORCH:-0}"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_workstation_rtx_pro_4000.sh [OPTIONS]

Automated venv bootstrap, SM120 hardware qualification, 4-rung APOLLO+COAT ablation,
and resumable multi-stage training for Spark-X2.6-VL + SyncCraft v3.2 on a 24 GB
NVIDIA RTX PRO 4000 Blackwell SFF workstation.

Options:
  --mode <mode>            Execution mode (default: all):
                             setup-only   Create venv, install CUDA/SM120 deps & SyncCraft
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
  --torch-index-url <url>  PyTorch CUDA wheel index ('auto' [default] probes nvidia-smi for cu130/cu128)
  --force-reinstall-torch  Force clean reinstall of CUDA PyTorch wheels inside --venv
  --python <bin>           Base Python 3.11-3.13 interpreter to create venv
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
    --mode)                  MODE="$2"; shift 2 ;;
    --work-root|--root)      WORK_ROOT="$2"; shift 2 ;;
    --venv)                  VENV_DIR="$2"; shift 2 ;;
    --synccraft-dir)         SYNCCRAFT_DIR="$2"; shift 2 ;;
    --config)                CONFIG_PATH="$2"; shift 2 ;;
    --dataset-mode)          DATASET_MODE="$2"; shift 2 ;;
    --torch-index-url)       TORCH_INDEX_URL="$2"; shift 2 ;;
    --force-reinstall-torch) FORCE_REINSTALL_TORCH=1; shift ;;
    --python)                PYTHON_BIN="$2"; shift 2 ;;
    --ablate-steps)          ABLATE_STEPS="$2"; shift 2 ;;
    --skip-install)          SKIP_INSTALL=1; shift ;;
    --skip-tests)            SKIP_TESTS=1; shift ;;
    --skip-ablate)           SKIP_ABLATE=1; shift ;;
    -h|--help)               usage; exit 0 ;;
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

# Sanitize environment variables that commonly mask GPUs or shadow libcuda.so.1
if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == "" || "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
  log "[WARN] CUDA_VISIBLE_DEVICES was set to '${CUDA_VISIBLE_DEVICES}', which hides all GPUs. Unsetting it."
  unset CUDA_VISIBLE_DEVICES
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]] && [[ "${LD_LIBRARY_PATH}" == *"/stubs"* ]]; then
  log "[WARN] Removing CUDA '/stubs' directory from LD_LIBRARY_PATH so real libcuda.so.1 is used."
  CLEAN_LD=""
  IFS=':' read -ra LD_PARTS <<< "${LD_LIBRARY_PATH}"
  for part in "${LD_PARTS[@]}"; do
    if [[ -n "${part}" && "${part}" != *"/stubs"* ]]; then
      CLEAN_LD="${CLEAN_LD:+${CLEAN_LD}:}${part}"
    fi
  done
  export LD_LIBRARY_PATH="${CLEAN_LD}"
fi

resolve_base_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Requested Python binary not found: ${PYTHON_BIN}"
    echo "${PYTHON_BIN}"
    return
  fi
  local candidate
  # Prefer CPython 3.12 / 3.13 first where PyTorch cu130/cu128 binary wheels are guaranteed.
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
        echo "${candidate}"
        return
      fi
    fi
  done
  # Fallback to >=3.11 if only newer Python is installed
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo "python3"
    return
  fi
  die "Python 3.11-3.13 is required on the workstation (install python3.12-venv or python3.13-venv)."
}

DRIVER_CUDA_VER=""
PRIMARY_TORCH_URL="https://download.pytorch.org/whl/cu130"
FALLBACK_TORCH_URL="https://download.pytorch.org/whl/cu128"

probe_linux_nvidia_driver() {
  [[ "$(uname -s)" == "Linux" ]] || return 0
  log "[Phase 0/6] Probing NVIDIA kernel driver & RTX PRO 4000 Blackwell SFF device nodes..."

  if [[ -e /dev/dxg && -d /usr/lib/wsl/lib ]]; then
    log "  WSL2 GPU paravirtualization (/dev/dxg) detected; ensuring /usr/lib/wsl/lib is in PATH & LD_LIBRARY_PATH."
    export PATH="/usr/lib/wsl/lib:${PATH}"
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi

  if command -v lspci >/dev/null 2>&1; then
    PCI_NV="$(lspci | grep -iE 'vga|3d|display' | grep -i nvidia || true)"
    if [[ -n "${PCI_NV}" ]]; then
      log "  PCI GPU detected: ${PCI_NV}"
    fi
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi is not installed. On Ubuntu/Debian for RTX PRO 4000 Blackwell SFF, install the OPEN kernel driver (>=570):
  sudo apt update && sudo apt install -y nvidia-driver-580-open nvidia-modprobe && sudo reboot"
  fi

  set +e
  SMI_OUT="$(nvidia-smi 2>&1)"
  SMI_RC=$?
  set -e

  if [[ "${SMI_RC}" -ne 0 ]]; then
    log "[ERROR] nvidia-smi failed (exit ${SMI_RC}):"
    printf '%s\n' "${SMI_OUT}" | tee -a "${LOG_FILE}" >&2
    if command -v mokutil >/dev/null 2>&1; then
      SB_STATE="$(mokutil --sb-state 2>/dev/null || true)"
      [[ -n "${SB_STATE}" ]] && log "  Secure Boot state: ${SB_STATE}"
    fi
    if [[ -f /proc/driver/nvidia/version ]]; then
      NV_MOD_LINE="$(head -n 1 /proc/driver/nvidia/version)"
      log "  Loaded kernel module: ${NV_MOD_LINE}"
      if [[ "${NV_MOD_LINE}" != *"Open Kernel Module"* ]]; then
        log "  [CAUSE DETECTED] Legacy closed nvidia.ko is loaded instead of nvidia-open!"
      fi
    fi
    die "The Linux kernel NVIDIA driver is not bound to your RTX PRO 4000 Blackwell SFF!
Common Blackwell (SM120 / GB203) fixes on Linux:
  1. Blackwell GPUs REQUIRE NVIDIA Open Kernel Modules ('-open'), NOT legacy closed nvidia.ko:
       sudo apt update && sudo apt install -y nvidia-driver-580-open nvidia-modprobe && sudo reboot
     (or nvidia-driver-570-open / 575-open depending on your distro).
  2. If UEFI Secure Boot is enabled, enroll the MOK key at boot or disable Secure Boot so nvidia-open can load.
  3. If you recently upgraded the kernel or driver, reboot the workstation so the running kernel matches."
  fi

  GPU_LIST="$(nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true)"
  DRIVER_CUDA_VER="$(printf '%s\n' "${SMI_OUT}" | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
  log "  nvidia-smi GPU   : ${GPU_LIST:-detected}"
  log "  Driver CUDA cap  : ${DRIVER_CUDA_VER:-unknown}"

  # Ensure /dev/nvidia-uvm exists on native Linux (often missing on headless Linux boot until nvidia-modprobe runs)
  if [[ ! -e /dev/dxg ]]; then
    if [[ -e /dev/nvidia0 && ! -e /dev/nvidia-uvm ]] && command -v nvidia-modprobe >/dev/null 2>&1; then
      log "  Initializing /dev/nvidia-uvm via nvidia-modprobe -u -c=0..."
      nvidia-modprobe -u -c=0 || true
    fi
    if [[ ! -e /dev/nvidia-uvm ]]; then
      log "  [WARN] /dev/nvidia-uvm is missing! PyTorch cuInit will fail without nvidia_uvm. Fix with: sudo apt install -y nvidia-modprobe && sudo modprobe nvidia_uvm && sudo nvidia-modprobe -u -c=0"
    fi
    if [[ -e /dev/nvidiactl ]] && [[ ! -r /dev/nvidiactl || ! -w /dev/nvidiactl ]]; then
      die "Current user '$(id -un)' lacks read/write permission on /dev/nvidiactl. Fix with:
  sudo usermod -aG video,render $(id -un) && sudo reboot"
    fi
  fi

  # Select primary/fallback PyTorch wheel index matching driver CUDA capability
  if [[ "${TORCH_INDEX_URL}" != "auto" ]]; then
    PRIMARY_TORCH_URL="${TORCH_INDEX_URL}"
    FALLBACK_TORCH_URL=""
  elif [[ -n "${DRIVER_CUDA_VER}" ]]; then
    DRV_MAJOR="${DRIVER_CUDA_VER%%.*}"
    DRV_MINOR="${DRIVER_CUDA_VER#*.}"
    DRV_MINOR="${DRV_MINOR%%.*}"
    if (( DRV_MAJOR >= 13 )); then
      PRIMARY_TORCH_URL="https://download.pytorch.org/whl/cu130"
      FALLBACK_TORCH_URL="https://download.pytorch.org/whl/cu128"
    elif (( DRV_MAJOR == 12 && DRV_MINOR >= 8 )); then
      PRIMARY_TORCH_URL="https://download.pytorch.org/whl/cu128"
      FALLBACK_TORCH_URL="https://download.pytorch.org/whl/cu130"
    else
      die "NVIDIA driver reports CUDA ${DRIVER_CUDA_VER}, which is too old for Blackwell SM120 (requires CUDA >= 12.8 / driver >= 570-open)."
    fi
  fi
  log "  PyTorch wheel URL: ${PRIMARY_TORCH_URL}${FALLBACK_TORCH_URL:+ (fallback: ${FALLBACK_TORCH_URL})}"
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

if [[ "${MODE}" != "smoke-cpu" ]] && [[ "$(uname -s)" == "Linux" ]]; then
  probe_linux_nvidia_driver
fi

# ------------------------------------------------------------------------------
# Phase 1: Virtualenv Setup & Self-Healing CUDA PyTorch Installation
# ------------------------------------------------------------------------------
if [[ -x "${VENV_DIR}/bin/python" ]] && [[ "$(uname -s)" == "Linux" ]] && [[ "${SKIP_INSTALL}" != "1" ]]; then
  # If an existing venv was created with Python >=3.14 (where CUDA wheels may be missing) and 3.12/3.13 exists, rebuild cleanly.
  if ! "${VENV_DIR}/bin/python" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
    CAND_PY="$(resolve_base_python)"
    if "${CAND_PY}" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      log "[Phase 1/6] Recreating ${VENV_DIR} with ${CAND_PY} (3.11-3.13) for prebuilt CUDA Blackwell wheel support..."
      rm -rf "${VENV_DIR}"
    fi
  fi
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  BASE_PY="$(resolve_base_python)"
  log "[Phase 1/6] Creating Python virtualenv at ${VENV_DIR} using ${BASE_PY}..."
  "${BASE_PY}" -m venv "${VENV_DIR}"
else
  log "[Phase 1/6] Reusing virtualenv at ${VENV_DIR} ($("${VENV_DIR}/bin/python" --version 2>&1))"
fi

VENV_PY="${VENV_DIR}/bin/python"
export PYTHON="${VENV_PY}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="false"

check_venv_cuda_torch() {
  "${VENV_PY}" -c '
import sys, torch
if sys.platform == "darwin":
    raise SystemExit(0)
if torch.version.hip or not torch.version.cuda:
    raise SystemExit(1)
if not torch.cuda.is_available():
    raise SystemExit(2)
x = torch.ones((8, 8), device="cuda", dtype=torch.bfloat16)
_ = float((x @ x).float().sum().item())
print(f"OK torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
' 2>/dev/null
}

install_cuda_torch_from_index() {
  local idx_url="$1"
  log "[Phase 1/6] Installing CUDA PyTorch wheels from ${idx_url}..."
  "${VENV_PY}" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  "${VENV_PY}" -m pip install --no-cache-dir --upgrade \
    torch torchvision torchaudio \
    --index-url "${idx_url}" 2>&1 | tee -a "${LOG_FILE}"
}

ACTIVE_TORCH_URL="${PRIMARY_TORCH_URL}"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  log "[Phase 1/6] Upgrading pip, setuptools, and wheel..."
  "${VENV_PY}" -m pip install --upgrade pip setuptools wheel >>"${LOG_FILE}" 2>&1

  NEED_TORCH=0
  if [[ "${FORCE_REINSTALL_TORCH}" == "1" ]]; then
    NEED_TORCH=1
  elif ! TORCH_PROBE_MSG="$(check_venv_cuda_torch)"; then
    NEED_TORCH=1
  else
    log "[Phase 1/6] Verified active PyTorch in venv: ${TORCH_PROBE_MSG}"
  fi

  if [[ "${NEED_TORCH}" == "1" ]]; then
    if [[ "$(uname -s)" == "Linux" ]]; then
      set +e
      install_cuda_torch_from_index "${PRIMARY_TORCH_URL}"
      TORCH_PROBE_MSG="$(check_venv_cuda_torch)"
      PROBE_RC=$?
      set -e
      if [[ "${PROBE_RC}" -ne 0 ]] && [[ -n "${FALLBACK_TORCH_URL}" ]]; then
        log "[WARN] PyTorch from ${PRIMARY_TORCH_URL} did not activate CUDA (rc=${PROBE_RC}); trying fallback ${FALLBACK_TORCH_URL}..."
        ACTIVE_TORCH_URL="${FALLBACK_TORCH_URL}"
        install_cuda_torch_from_index "${FALLBACK_TORCH_URL}"
      fi
    else
      log "[Phase 1/6] Non-Linux host detected ($(uname -s)); installing standard PyTorch..."
      "${VENV_PY}" -m pip install --upgrade torch 2>&1 | tee -a "${LOG_FILE}"
    fi
  fi

  # Pin the exact installed torch/torchvision/torchaudio versions so PyPI packages below
  # CANNOT replace +cu130/+cu128 torch with a CPU-only PyPI build!
  CONSTRAINT_FILE="${WORK_ROOT}/locks/torch_constraint.txt"
  "${VENV_PY}" -c '
import importlib.metadata as m
for pkg in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{pkg}=={m.version(pkg)}")
    except m.PackageNotFoundError:
        pass
' > "${CONSTRAINT_FILE}"
  log "[Phase 1/6] Pinned PyTorch constraint: $(tr '\n' ' ' < "${CONSTRAINT_FILE}")"

  EXTRA_IDX_ARGS=()
  if [[ "$(uname -s)" == "Linux" ]]; then
    EXTRA_IDX_ARGS=(--extra-index-url "${ACTIVE_TORCH_URL}")
  fi

  log "[Phase 1/6] Installing Spark-X2.6-VL vision_text + SyncCraft runtime dependencies (with torch locked)..."
  "${VENV_PY}" -m pip install \
    -c "${CONSTRAINT_FILE}" \
    "${EXTRA_IDX_ARGS[@]}" \
    "transformers>=4.51.0,<5.0.0" \
    "tokenizers>=0.21.0" \
    "safetensors>=0.4.5" \
    "huggingface-hub>=0.28.0" \
    "accelerate>=1.2.0" \
    "sentencepiece>=0.2.0" \
    "einops>=0.8.0" \
    "timm>=1.0.12" \
    "av>=13.0.0" \
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

# Verify CUDA PyTorch right now on Linux (before running unit tests) so any wheel/driver mismatch is caught immediately!
if [[ "${MODE}" != "smoke-cpu" ]] && [[ "$(uname -s)" == "Linux" ]]; then
  log "[Phase 1/6] Verifying PyTorch CUDA + BF16 + Blackwell SM120 visibility inside ${VENV_DIR}..."
  "${VENV_PY}" -c '
import torch
from colab.final.run import _diagnose_cuda_unavailable
if torch.version.hip or not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA check failed: " + _diagnose_cuda_unavailable(torch))
cap = torch.cuda.get_device_capability(0)
prop = torch.cuda.get_device_properties(0)
x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
print(f"  [CUDA VERIFIED] gpu={prop.name!r} vram_gib={prop.total_memory / (1024**3):.2f} capability={cap} torch={torch.__version__} cuda_runtime={torch.version.cuda} bf16_matmul_sum={y:.1f}")
' 2>&1 | tee -a "${LOG_FILE}"
fi

# Record exact hardware & dependency lock per SyncCraft AGENTS.md reproducibility rule
"${VENV_PY}" -m pip freeze > "${WORK_ROOT}/locks/pip_freeze_${TIMESTAMP}.txt"
cp -f "${WORK_ROOT}/locks/pip_freeze_${TIMESTAMP}.txt" "${WORK_ROOT}/locks/pip_freeze_latest.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "${WORK_ROOT}/locks/nvidia_smi_${TIMESTAMP}.txt" 2>&1 || true
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
