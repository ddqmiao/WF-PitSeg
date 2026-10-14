#!/usr/bin/env bash
# Fine-tune the visual backbone. The temporal head used here is discarded after feature extraction.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/vision/finetune_lemonfm_last_stage_224.json}"
IMAGES_ROOT="${IMAGES_ROOT:?Set IMAGES_ROOT to PitVis frame images}"
CSV_ROOT="${CSV_ROOT:?Set CSV_ROOT to the PitVis annotation directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/vision/backbone_finetune}"
GPU_IDS="${GPU_IDS:-0}"
CMD=("${PYTHON_BIN}" "${PROJECT_ROOT}/train_pitvis_mainline.py"
  --config "${CONFIG}" --images-root "${IMAGES_ROOT}" --csv-root "${CSV_ROOT}"
  --output-dir "${OUTPUT_DIR}" --gpu-ids "${GPU_IDS}")
if [[ -n "${LEMONFM_WEIGHTS:-}" ]]; then CMD+=(--weights "${LEMONFM_WEIGHTS}"); fi
"${CMD[@]}"
