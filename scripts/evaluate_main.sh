#!/usr/bin/env bash
# Evaluate one checkpoint. Fixed priors are the default; prior search must be requested explicitly.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to best_step_metric.pt}"
FEATURES_ROOT="${FEATURES_ROOT:?Set FEATURES_ROOT to extracted PitVis features}"
CSV_ROOT="${CSV_ROOT:?Set CSV_ROOT to the PitVis annotation directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/evaluation}"
GPU_IDS="${GPU_IDS:-0}"
PRIOR_MODE="${PRIOR_MODE:-fixed}"
mkdir -p "${OUTPUT_DIR}"
CMD=("${PYTHON_BIN}" "${PROJECT_ROOT}/ensemble_eval.py"
  --checkpoints "${CHECKPOINT}" --weights 1.0
  --features-root "${FEATURES_ROOT}" --csv-root "${CSV_ROOT}"
  --gpu-ids "${GPU_IDS}" --seed "${SEED:-42}" --deterministic
  --light-post-kernel-size 11 --min-duration-frames 30
  --position-prior-bins 8 --position-prior-weight 0.0
  --result-json "${OUTPUT_DIR}/metrics.json"
  --result-csv "${OUTPUT_DIR}/metrics.csv"
  --eval-frames-csv "${OUTPUT_DIR}/predictions.csv")
if [[ "${PRIOR_MODE}" == "search" ]]; then
  echo "Warning: emission priors are being tuned on the evaluated split." >&2
  CMD+=(--search-priors)
elif [[ "${PRIOR_MODE}" != "fixed" ]]; then
  echo "PRIOR_MODE must be fixed or search." >&2
  exit 2
fi
if [[ -n "${SPLIT_JSON:-}" ]]; then CMD+=(--split-json "${SPLIT_JSON}"); fi
"${CMD[@]}"
