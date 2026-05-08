#!/usr/bin/env bash
set -euo pipefail

# Train a MoCapAct clip expert for the custom Exitvelo batting target.
#
# Expected usage from a fresh public clone on a RunPod/Linux machine:
#   git clone https://github.com/Dexin-Huang/exitvelo.git
#   cd exitvelo
#   BUILD_CUSTOM_MOCAP_HDF5=1 bash scripts/runpod/mocapact_train_clip_expert.sh
#
# Important defaults target the first physical imitation tracker:
#   CMU_124_07, MoCapAct steps 87-120, mapped from Exitvelo frames 260-360.

EXITVELO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${EXITVELO_ROOT}:${PYTHONPATH:-}"

MOCAPACT_ROOT="${MOCAPACT_ROOT:-/workspace/MoCapAct}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${EXITVELO_ROOT}/results/mocapact_clip_experts_custom}"
CLIP_ID="${CLIP_ID:-CMU_124_07}"
START_STEP="${START_STEP:-87}"
MAX_STEPS="${MAX_STEPS:-33}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-65536}"
N_WORKERS="${N_WORKERS:-8}"
N_STEPS="${N_STEPS:-1024}"
N_EPOCHS="${N_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_LAYERS="${N_LAYERS:-2}"
LAYER_SIZE="${LAYER_SIZE:-512}"
DEVICE="${DEVICE:-auto}"
MIN_STEPS="${MIN_STEPS:-1}"
TERMINATION_ERROR_THRESHOLD="${TERMINATION_ERROR_THRESHOLD:-0.3}"
ACT_NOISE="${ACT_NOISE:-0.1}"
EVAL_FREQ="${EVAL_FREQ:-8192}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
EVAL_MIN_STEPS="${EVAL_MIN_STEPS:-10}"
EVAL_PATIENCE="${EVAL_PATIENCE:-10}"
BUILD_CUSTOM_MOCAP_HDF5="${BUILD_CUSTOM_MOCAP_HDF5:-1}"
CUSTOM_MOCAP_PATH="${CUSTOM_MOCAP_PATH:-${EXITVELO_ROOT}/results/mocapact_custom_hdf5/CMU_124_07_from_amc_30hz.h5}"
CUSTOM_MOCAP_CONTROL_DT="${CUSTOM_MOCAP_CONTROL_DT:-0.03}"
MOCAP_PATH="${MOCAP_PATH:-${CUSTOM_MOCAP_PATH}}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

mkdir -p "${TRAIN_OUT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    echo "WARNING: ${PYTHON_BIN} not found; falling back to python3"
    PYTHON_BIN="python3"
  else
    echo "ERROR: neither ${PYTHON_BIN} nor python3 is available" >&2
    exit 1
  fi
fi

if [[ ! -d "${MOCAPACT_ROOT}/.git" ]]; then
  git clone https://github.com/microsoft/MoCapAct.git "${MOCAPACT_ROOT}"
fi

cd "${MOCAPACT_ROOT}"
if [[ ! -d ".venv" ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip wheel "setuptools<82"
python -m pip install -e .
python -m pip install "numpy==1.26.4" "protobuf==3.20.3" h5py

# The RunPod PyTorch 2.1 CUDA 11.8 image has a CUDA 12-capable driver, but
# MoCapAct's dependency chain may otherwise pull a CUDA 13 torch wheel that
# falls back to CPU. Reinstall a CUDA 12 wheel after dependency resolution.
python -m pip install --force-reinstall --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
python -m pip install "setuptools<82" "numpy==1.26.4" "protobuf==3.20.3" h5py

cd "${EXITVELO_ROOT}"
if [[ "${BUILD_CUSTOM_MOCAP_HDF5}" == "1" ]]; then
  python scripts/io/export_cmu_amc_to_mocapact_hdf5.py \
    --out "${CUSTOM_MOCAP_PATH}" \
    --summary "${CUSTOM_MOCAP_PATH%.h5}_summary.json" \
    --control-dt "${CUSTOM_MOCAP_CONTROL_DT}" \
    --overwrite
fi

cd "${MOCAPACT_ROOT}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the MoCapAct venv")
PY

RUN_DIR_NAME="${RUN_DIR_NAME:-${CLIP_ID}_${START_STEP}_${MAX_STEPS}_${TOTAL_TIMESTEPS}_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_PATH="${TRAIN_OUT_DIR}/${RUN_DIR_NAME}.log"
echo "Training log: ${LOG_PATH}"

python -m mocapact.clip_expert.train \
  --clip_id="${CLIP_ID}" \
  --mocap_path="${MOCAP_PATH}" \
  --start_step="${START_STEP}" \
  --max_steps="${MAX_STEPS}" \
  --log_root="${TRAIN_OUT_DIR}" \
  --total_timesteps="${TOTAL_TIMESTEPS}" \
  --n_workers="${N_WORKERS}" \
  --n_steps="${N_STEPS}" \
  --n_epochs="${N_EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --n_layers="${N_LAYERS}" \
  --layer_size="${LAYER_SIZE}" \
  --device="${DEVICE}" \
  --min_steps="${MIN_STEPS}" \
  --termination_error_threshold="${TERMINATION_ERROR_THRESHOLD}" \
  --act_noise="${ACT_NOISE}" \
  --eval.freq="${EVAL_FREQ}" \
  --eval.min_steps="${EVAL_MIN_STEPS}" \
  --eval.n_rsi_episodes="${EVAL_EPISODES}" \
  --eval.n_start_episodes="${EVAL_EPISODES}" \
  --eval.early_stop.patience="${EVAL_PATIENCE}" \
  2>&1 | tee "${LOG_PATH}"

ENV_PATH="${TRAIN_OUT_DIR}/${RUN_DIR_NAME}_environment.txt"
{
  echo "EXITVELO_COMMIT=$(cd "${EXITVELO_ROOT}" && git rev-parse --short HEAD)"
  echo "MOCAPACT_COMMIT=$(cd "${MOCAPACT_ROOT}" && git rev-parse --short HEAD)"
  python -m torch.utils.collect_env
  echo "PIP_FREEZE_BEGIN"
  python -m pip freeze
} > "${ENV_PATH}"

echo "Training complete."
echo "  log_root=${TRAIN_OUT_DIR}"
echo "  log=${LOG_PATH}"
echo "  environment=${ENV_PATH}"
