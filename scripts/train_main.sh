#!/usr/bin/env bash
# Train the full-video temporal model with the main WF-PitSeg recipe.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FEATURES_ROOT="${FEATURES_ROOT:?Set FEATURES_ROOT to extracted PitVis features}"
CSV_ROOT="${CSV_ROOT:?Set CSV_ROOT to the PitVis annotation directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/main/seed_42}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-42}"
CMD=("${PYTHON_BIN}" "${PROJECT_ROOT}/train_pitvis_mainline.py"
  --config "${PROJECT_ROOT}/configs/main/wfpitseg.json"
  --features-root "${FEATURES_ROOT}" --csv-root "${CSV_ROOT}"
  --gpu-ids "${GPU_IDS}" --seed "${SEED}" --output-dir "${OUTPUT_DIR}"
  --deterministic)
if [[ -n "${SPLIT_JSON:-}" ]]; then CMD+=(--split-json "${SPLIT_JSON}"); fi
"${CMD[@]}"
