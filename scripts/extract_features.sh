#!/usr/bin/env bash
# Extract one 1536-D feature vector per sampled frame from a fine-tuned checkpoint.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:?Set BACKBONE_CHECKPOINT}"
IMAGES_ROOT="${IMAGES_ROOT:?Set IMAGES_ROOT to PitVis frame images}"
CSV_ROOT="${CSV_ROOT:?Set CSV_ROOT to the PitVis annotation directory}"
FEATURES_ROOT="${FEATURES_ROOT:-${PROJECT_ROOT}/outputs/features_finetuned}"
GPU_ID="${GPU_ID:-0}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/extract_lemonfm_features.py" \
  --from-checkpoint --weights "${BACKBONE_CHECKPOINT}" \
  --images-root "${IMAGES_ROOT}" --csv-root "${CSV_ROOT}" \
  --output-dir "${FEATURES_ROOT}" --gpu-id "${GPU_ID}" \
  --batch-size "${BATCH_SIZE:-64}" --precision "${PRECISION:-bf16}"
