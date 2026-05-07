#!/usr/bin/env bash
set -euo pipefail

# RunPod/Linux MoCapAct probe for Exitvelo.
#
# Expected usage from the Exitvelo repo root:
#   bash scripts/runpod/mocapact_probe.sh
#
# Optional environment variables:
#   MOCAPACT_ROOT=/workspace/MoCapAct
#   OUT_DIR=/workspace/exitvelo/results/runpod_mocapact_probe
#   PYTHON_BIN=python3.8
#   CLIP_IDS="CMU_124_07 CMU_124_08 CMU_124_06 CMU_016_22"
#   START_STEP=260
#   END_STEP=360
#   ROLLOUT_STEPS=8
#   PROBE_CANDIDATES="CMU_124_07:260:360 CMU_124_08:260:360 CMU_124_06:0:189 CMU_016_22:0:82"
#   REF_STEPS=0
#   MOCAP_PATH=/path/to/custom/cmu_mocap.hdf5
#   MIN_STEPS=10
#   TERMINATION_ERROR_THRESHOLD=0.3

EXITVELO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MOCAPACT_ROOT="${MOCAPACT_ROOT:-/workspace/MoCapAct}"
OUT_DIR="${OUT_DIR:-${EXITVELO_ROOT}/results/runpod_mocapact_probe}"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"
START_STEP="${START_STEP:-260}"
END_STEP="${END_STEP:-360}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-8}"
REF_STEPS="${REF_STEPS:-0}"
MIN_STEPS="${MIN_STEPS:-10}"
GHOST_OFFSET="${GHOST_OFFSET:-1.0}"
TERMINATION_ERROR_THRESHOLD="${TERMINATION_ERROR_THRESHOLD:-0.3}"
ACT_NOISE="${ACT_NOISE:-0.0}"

DEFAULT_PROBE_CANDIDATES="CMU_124_07:260:360:primary_exitvelo_target CMU_124_08:260:360:secondary_exitvelo_raw_clip CMU_124_06:0:189:nearby_subject124_split_fallback CMU_016_22:0:82:known_mocapact_control"
PROBE_CANDIDATES="${PROBE_CANDIDATES:-${DEFAULT_PROBE_CANDIDATES}}"

mkdir -p "${OUT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ "${PYTHON_BIN}" == "python3.8" ]] && command -v conda >/dev/null 2>&1; then
    PY38_PREFIX="${MOCAPACT_ROOT}/.conda_py38"
    if [[ ! -x "${PY38_PREFIX}/bin/python" ]]; then
      conda create -y -p "${PY38_PREFIX}" python=3.8
    fi
    PYTHON_BIN="${PY38_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
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
python -m pip install -U pip wheel setuptools
python -m pip install -e .

cd "${EXITVELO_ROOT}"
ARGS=()
if [[ -n "${CLIP_IDS:-}" ]]; then
  for clip_id in ${CLIP_IDS}; do
    ARGS+=(--clip-id "${clip_id}")
  done
  ARGS+=(--start-step "${START_STEP}" --end-step "${END_STEP}")
else
  for candidate in ${PROBE_CANDIDATES}; do
    ARGS+=(--candidate "${candidate}")
  done
fi

if [[ -n "${MOCAP_PATH:-}" ]]; then
  ARGS+=(--mocap-path "${MOCAP_PATH}")
fi

if [[ "${ALWAYS_INIT_AT_CLIP_START:-0}" == "1" ]]; then
  ARGS+=(--always-init-at-clip-start)
fi

if [[ "${ZERO_ACTION:-0}" == "1" ]]; then
  ARGS+=(--zero-action)
fi

python scripts/runpod/mocapact_probe.py \
  --out-dir "${OUT_DIR}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --ref-steps "${REF_STEPS}" \
  --min-steps "${MIN_STEPS}" \
  --ghost-offset "${GHOST_OFFSET}" \
  --termination-error-threshold "${TERMINATION_ERROR_THRESHOLD}" \
  --act-noise "${ACT_NOISE}" \
  "${ARGS[@]}"

echo "Probe complete. Outputs:"
echo "  ${OUT_DIR}/mocapact_probe_report.json"
echo "  ${OUT_DIR}/*_rollout.json"
