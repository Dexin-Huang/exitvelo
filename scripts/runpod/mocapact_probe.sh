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
#   CLIP_IDS="CMU_124_07 CMU_124_08 CMU_016_22"
#   START_STEP=260
#   END_STEP=360
#   ROLLOUT_STEPS=8

EXITVELO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MOCAPACT_ROOT="${MOCAPACT_ROOT:-/workspace/MoCapAct}"
OUT_DIR="${OUT_DIR:-${EXITVELO_ROOT}/results/runpod_mocapact_probe}"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"
START_STEP="${START_STEP:-260}"
END_STEP="${END_STEP:-360}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-8}"
CLIP_IDS="${CLIP_IDS:-CMU_124_07 CMU_124_08 CMU_016_22}"

mkdir -p "${OUT_DIR}"

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
for clip_id in ${CLIP_IDS}; do
  ARGS+=(--clip-id "${clip_id}")
done

python scripts/runpod/mocapact_probe.py \
  --out-dir "${OUT_DIR}" \
  --start-step "${START_STEP}" \
  --end-step "${END_STEP}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  "${ARGS[@]}"

echo "Probe complete. Outputs:"
echo "  ${OUT_DIR}/mocapact_probe_report.json"
echo "  ${OUT_DIR}/*_rollout.json"
