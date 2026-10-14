import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


LABELS12_DEFAULT = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14)


def frame_position_bin_indices(length: int, num_bins: int) -> np.ndarray:
    length = int(length)
    num_bins = int(num_bins)
    if length < 0:
        raise ValueError(f"length must be non-negative; received {length}")
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive; received {num_bins}")
    if length == 0:
        return np.zeros((0,), dtype=np.int64)
    rel = (np.arange(length, dtype=np.float64) + 0.5) / float(length)
    indices = np.floor(rel * num_bins).astype(np.int64)
    return np.clip(indices, 0, num_bins - 1)


def _resolve_annotation_files(csv_root: Path, video_ids: Optional[Sequence[int]]) -> Sequence[Path]:
    csv_root = Path(csv_root)
    if csv_root.is_file():
        return [csv_root]
    if video_ids is None:
        return sorted(csv_root.glob("annotations_*.csv"))
    return [csv_root / f"annotations_{int(video_id):02d}.csv" for video_id in video_ids]


def collect_position_bin_counts_from_csv_root(
    csv_root: Path,
    *,
    video_ids: Optional[Sequence[int]] = None,
    labels12: Sequence[int] = LABELS12_DEFAULT,
    num_bins: int = 8,
    normalize_per_video_bin: bool = True,
) -> np.ndarray:
    label_to_index = {int(raw_label): idx for idx, raw_label in enumerate(labels12)}
    counts = np.zeros((int(num_bins), len(labels12)), dtype=np.float64)

    for csv_path in _resolve_annotation_files(csv_root, video_ids):
        if not csv_path.is_file():
            raise FileNotFoundError(f"Annotation file does not exist: {csv_path}")

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        raw_steps = [int(row.get("int_step", -100)) for row in rows]
        bin_indices = frame_position_bin_indices(len(raw_steps), num_bins)
        video_counts = np.zeros_like(counts)

        for bin_idx, raw_label in zip(bin_indices, raw_steps):
            class_idx = label_to_index.get(int(raw_label))
            if class_idx is None:
                continue
            video_counts[int(bin_idx), int(class_idx)] += 1.0

        if normalize_per_video_bin:
            for bin_idx in range(video_counts.shape[0]):
                total = float(video_counts[bin_idx].sum())
                if total > 0:
                    counts[bin_idx] += video_counts[bin_idx] / total
        else:
            counts += video_counts

    return counts


def build_log_position_prior_numpy(
    bin_counts: np.ndarray,
    *,
    smoothing: float = 1e-4,
) -> np.ndarray:
    bin_counts = np.asarray(bin_counts, dtype=np.float64)
    if bin_counts.ndim != 2:
        raise ValueError(f"bin_counts must be a 2D matrix; received shape={bin_counts.shape}")
    num_bins, num_classes = bin_counts.shape
    log_prior = np.zeros((num_bins, num_classes), dtype=np.float64)
    for bin_idx in range(num_bins):
        row = bin_counts[bin_idx].astype(np.float64, copy=True)
        if float(row.sum()) <= 0.0:
            row.fill(1.0 / max(1, num_classes))
        else:
            row = row + float(smoothing)
            row = row / row.sum()
        log_prior[bin_idx] = np.log(np.clip(row, 1e-12, 1.0))
    return log_prior


def apply_position_log_prior_to_log_probs(
    log_probs: np.ndarray,
    *,
    log_position_prior_matrix: Optional[np.ndarray],
    position_prior_weight: float = 0.0,
) -> np.ndarray:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_position_prior_matrix is None or float(position_prior_weight) <= 0.0:
        return log_probs
    position_matrix = np.asarray(log_position_prior_matrix, dtype=np.float64)
    if position_matrix.ndim != 2:
        raise ValueError(
            f"log_position_prior_matrix must be a 2D matrix; received shape={position_matrix.shape}"
        )
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must be a 2D matrix; received shape={log_probs.shape}")
    if position_matrix.shape[1] != log_probs.shape[1]:
        raise ValueError(
            f"Position-prior class dimension mismatch: "
            f"{position_matrix.shape[1]} vs {log_probs.shape[1]}"
        )
    bin_indices = frame_position_bin_indices(log_probs.shape[0], position_matrix.shape[0])
    return log_probs + float(position_prior_weight) * position_matrix[bin_indices]


__all__ = [
    "LABELS12_DEFAULT",
    "apply_position_log_prior_to_log_probs",
    "build_log_position_prior_numpy",
    "collect_position_bin_counts_from_csv_root",
    "frame_position_bin_indices",
]
