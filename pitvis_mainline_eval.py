import csv
import gc
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from pitvis_eval_metrics import (
    apply_transition_tolerant_correction,
    bootstrap_confidence_interval,
    compute_global_classification_statistics,
    compute_transition_metrics,
    compute_variant_statistical_tests,
    compute_videowise_phase_metrics,
    export_metric_bundle,
)
from pitvis_error_profile import export_hard_class_error_profile
from pitvis_position_prior import apply_position_log_prior_to_log_probs
from pitvis_priors import build_emission_prior_by_index
from pitvis_mainline_common import (
    ALLOWED_TRANSITIONS_RAW,
    HARD_CLASS_RAW_LABELS,
    IGNORE_INDEX,
    LABELS12,
    LABEL_TO_INDEX as _L2I,
    build_log_transition_matrix_numpy,
    check_gpu_memory,
    check_ram_usage,
    check_resources_and_warn,
    step_index_to_raw_label,
)
from pitvis_mainline_model import LATE_CHAIN_CLASS_INDICES, LATE_CHAIN_RAW_LABELS


def _build_log_transition_matrix(num_classes: int) -> np.ndarray:
    if num_classes != len(LABELS12):
        raise ValueError(
            f"Only a {len(LABELS12)}-class transition matrix is supported; "
            f"received num_classes={num_classes}"
        )
    return build_log_transition_matrix_numpy()


def _viterbi_decode(log_probs: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Viterbi decoding.

    Args:
        log_probs: (T, C) log-probabilities per frame.
        log_trans: (C, C) log-transition matrix.
    Returns:
        (T,) decoded class indices.
    """
    T, C = log_probs.shape
    if T == 0:
        return np.array([], dtype=np.int64)

    viterbi = np.full((T, C), -np.inf, dtype=np.float64)
    backptr = np.zeros((T, C), dtype=np.int64)
    viterbi[0] = log_probs[0]

    for t in range(1, T):
        for j in range(C):
            scores = viterbi[t - 1] + log_trans[:, j]
            best_i = int(np.argmax(scores))
            viterbi[t, j] = scores[best_i] + log_probs[t, j]
            backptr[t, j] = best_i

    # Backtrace
    path = np.zeros(T, dtype=np.int64)
    path[T - 1] = int(np.argmax(viterbi[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def _online_max_decode(
    log_probs: np.ndarray,
    log_trans: np.ndarray,
    *,
    transition_alpha: float = 1.0,
) -> np.ndarray:
    """Strict-online max-product decoding with no future look-ahead or backtrace."""
    T, C = log_probs.shape
    if T == 0:
        return np.array([], dtype=np.int64)

    alpha = np.array(log_probs[0], copy=True, dtype=np.float64)
    path = np.zeros(T, dtype=np.int64)
    path[0] = int(np.argmax(alpha))
    transition_alpha = float(max(0.0, transition_alpha))

    for t in range(1, T):
        scores = alpha[:, np.newaxis] + transition_alpha * log_trans
        alpha = log_probs[t] + np.max(scores, axis=0)
        alpha -= np.max(alpha)
        path[t] = int(np.argmax(alpha))
    return path


def apply_late_chain_decoder_guard(
    log_probs: np.ndarray,
    *,
    late_chain_logits: Optional[np.ndarray] = None,
    late_chain_gate: Optional[np.ndarray] = None,
    late_chain_decoder_guard: float = 0.0,
) -> np.ndarray:
    if float(late_chain_decoder_guard) <= 0.0 or late_chain_logits is None:
        return log_probs

    guarded = np.array(log_probs, copy=True, dtype=np.float64)
    tail_logits = np.asarray(late_chain_logits, dtype=np.float64)
    if tail_logits.ndim != 2 or tail_logits.shape[0] != guarded.shape[0]:
        return guarded

    if late_chain_gate is None:
        gate = np.ones((guarded.shape[0],), dtype=np.float64)
    else:
        gate = np.asarray(late_chain_gate, dtype=np.float64).reshape(-1)
        if gate.shape[0] != guarded.shape[0]:
            return guarded
        gate = np.clip(gate, 0.0, 1.0)

    tail_logits = tail_logits - tail_logits.max(axis=1, keepdims=True)
    tail_probs = np.exp(tail_logits)
    tail_probs = tail_probs / np.clip(tail_probs.sum(axis=1, keepdims=True), 1e-12, None)

    idx_12_local = LATE_CHAIN_RAW_LABELS.index(12)
    idx_14_local = LATE_CHAIN_RAW_LABELS.index(14)
    idx_12_global = LATE_CHAIN_CLASS_INDICES[idx_12_local]
    idx_14_global = LATE_CHAIN_CLASS_INDICES[idx_14_local]
    support_12 = np.clip(tail_probs[:, idx_12_local] - tail_probs[:, idx_14_local], 0.0, 1.0)
    boost = float(late_chain_decoder_guard) * gate * support_12
    guarded[:, idx_12_global] += boost
    guarded[:, idx_14_global] -= boost
    return guarded


def apply_online_transition_postprocess(
    records: Sequence[Dict[str, object]],
    *,
    log_transition_matrix: Optional[np.ndarray] = None,
    transition_alpha: float = 1.0,
    ema_decay: float = 0.0,
) -> List[Dict[str, object]]:
    """Strict-online causal decoding with optional EMA smoothing and transition priors."""
    global _LOG_TRANS_CACHE
    num_classes = len(LABELS12)
    if log_transition_matrix is not None:
        log_trans = log_transition_matrix
    else:
        if _LOG_TRANS_CACHE is None:
            _LOG_TRANS_CACHE = _build_log_transition_matrix(num_classes)
        log_trans = _LOG_TRANS_CACHE

    emission_log_prior = np.zeros(num_classes, dtype=np.float64)
    for cls_idx, multiplier in _EMISSION_PRIOR.items():
        emission_log_prior[cls_idx] = math.log(max(1e-6, multiplier))

    by_video: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        by_video.setdefault(int(record["video_id"]), []).append(record)

    final_records: List[Dict[str, object]] = []
    ema_decay = float(min(max(ema_decay, 0.0), 0.999))
    for video_id in sorted(by_video):
        video_records = sorted(by_video[video_id], key=lambda item: int(item["int_time"]))
        has_probs = "step_prob_raw" in video_records[0] and torch.is_tensor(video_records[0]["step_prob_raw"])
        if not has_probs:
            for record in video_records:
                record["step_pred_online"] = int(record.get("step_pred_raw", 0))
                record["step_pred_online_label"] = int(step_index_to_raw_label(int(record["step_pred_online"])))
                final_records.append(record)
            continue

        probs = torch.stack([r["step_prob_raw"] for r in video_records], dim=0).numpy().astype(np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        if ema_decay > 0.0:
            smoothed = np.array(probs, copy=True)
            for t in range(1, smoothed.shape[0]):
                smoothed[t] = ema_decay * smoothed[t - 1] + (1.0 - ema_decay) * smoothed[t]
            probs = np.clip(smoothed, 1e-12, 1.0)

        log_probs = np.log(probs) + emission_log_prior[np.newaxis, :]
        decoded = _online_max_decode(log_probs, log_trans, transition_alpha=transition_alpha)
        for record, pred in zip(video_records, decoded):
            record["step_pred_online"] = int(pred)
            record["step_pred_online_label"] = int(step_index_to_raw_label(int(pred)))
            final_records.append(record)
    return sorted(final_records, key=lambda item: (int(item["video_id"]), int(item["int_time"])))
from pitvis_mainline_data import build_center_frame_weights
from pitvis_mainline_train import compute_mstcn_loss, compute_multitask_loss, maybe_make_progress_bar

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> torch.Tensor:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, pred in zip(y_true, y_pred):
        confusion[int(target), int(pred)] += 1
    return confusion


def compress_sequence(sequence: Sequence[int]) -> List[int]:
    values = [int(value) for value in sequence]
    if not values:
        return []
    compressed = [values[0]]
    for value in values[1:]:
        if value != compressed[-1]:
            compressed.append(value)
    return compressed


def levenshtein_distance(sequence_a: Sequence[int], sequence_b: Sequence[int]) -> int:
    a = list(sequence_a)
    b = list(sequence_b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for row_idx, value_a in enumerate(a, start=1):
        current = [row_idx]
        for col_idx, value_b in enumerate(b, start=1):
            insertion = current[col_idx - 1] + 1
            deletion = previous[col_idx] + 1
            substitution = previous[col_idx - 1] + (0 if value_a == value_b else 1)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def compute_edit_score(target_sequence: Sequence[int], predicted_sequence: Sequence[int]) -> float:
    target_comp = compress_sequence(target_sequence)
    pred_comp = compress_sequence(predicted_sequence)
    if not target_comp and not pred_comp:
        return 1.0
    denominator = max(len(target_comp), len(pred_comp), 1)
    distance = levenshtein_distance(target_comp, pred_comp)
    return float(1.0 - distance / denominator)


def _extract_segments(sequence: Sequence[int]) -> List[Tuple[int, int, int]]:
    values = [int(value) for value in sequence]
    if not values:
        return []
    segments: List[Tuple[int, int, int]] = []
    start = 0
    current = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != current:
            segments.append((current, start, index))
            start = index
            current = value
    segments.append((current, start, len(values)))
    return segments


def compute_segment_f1_counts(
    target_sequence: Sequence[int],
    predicted_sequence: Sequence[int],
    *,
    overlap_threshold: float,
) -> Tuple[int, int, int]:
    target_segments = _extract_segments(target_sequence)
    pred_segments = _extract_segments(predicted_sequence)
    if not target_segments and not pred_segments:
        return 0, 0, 0

    hits = np.zeros(len(target_segments), dtype=bool)
    true_positive = 0
    false_positive = 0
    for pred_label, pred_start, pred_end in pred_segments:
        best_iou = 0.0
        best_index = -1
        for index, (target_label, target_start, target_end) in enumerate(target_segments):
            if target_label != pred_label:
                continue
            intersection = max(0, min(pred_end, target_end) - max(pred_start, target_start))
            if intersection <= 0:
                continue
            union = max(pred_end, target_end) - min(pred_start, target_start)
            iou = float(intersection) / float(max(1, union))
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= overlap_threshold and not hits[best_index]:
            hits[best_index] = True
            true_positive += 1
        else:
            false_positive += 1

    false_negative = int((~hits).sum())
    return true_positive, false_positive, false_negative


def _f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    precision = float(true_positive) / float(max(1, true_positive + false_positive))
    recall = float(true_positive) / float(max(1, true_positive + false_negative))
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_step_metrics(
    records: Sequence[Dict[str, object]],
    *,
    num_classes: int,
    pred_key: str,
    pred_label_key: Optional[str] = None,
    relaxed_tolerance_frames: int = 10,
    transition_tolerance_frames: int = 10,
) -> Dict[str, object]:
    valid_records = [record for record in records if 0 <= int(record["step_target"]) < num_classes]
    label_names = LABELS12[:num_classes] if num_classes == len(LABELS12) else list(range(num_classes))
    if not valid_records:
        zeros = [0.0 for _ in range(num_classes)]
        return {
            "acc": 0.0,
            "video_acc_mean": 0.0,
            "video_acc_std": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_jaccard": 0.0,
            "balanced_accuracy": 0.0,
            "edit": 0.0,
            "step_metric": 0.0,
            "precision": zeros,
            "recall": zeros,
            "f1": zeros,
            "jaccard": zeros,
            "support": [0 for _ in range(num_classes)],
            "confusion_matrix": torch.zeros((num_classes, num_classes), dtype=torch.int64),
            "label_names": label_names,
            "per_class_video_stats": {
                "precision_mean": zeros,
                "precision_std": zeros,
                "recall_mean": zeros,
                "recall_std": zeros,
                "f1_mean": zeros,
                "f1_std": zeros,
                "jaccard_mean": zeros,
                "jaccard_std": zeros,
                "valid_video_count": [0 for _ in range(num_classes)],
            },
            "per_video": [],
            "f1_at_10": 0.0,
            "f1_at_25": 0.0,
            "f1_at_50": 0.0,
            "frame_macro_precision_global": 0.0,
            "frame_macro_recall_global": 0.0,
            "frame_macro_f1_global": 0.0,
            "frame_macro_jaccard_global": 0.0,
            "weighted_precision_global": 0.0,
            "weighted_recall_global": 0.0,
            "weighted_f1_global": 0.0,
            "weighted_jaccard_global": 0.0,
            "kappa_global": 0.0,
            "mcc_global": 0.0,
            f"relaxed_acc_at_{relaxed_tolerance_frames}": 0.0,
            f"relaxed_macro_f1_at_{relaxed_tolerance_frames}": 0.0,
            f"relaxed_macro_jaccard_at_{relaxed_tolerance_frames}": 0.0,
            f"transition_precision_at_{transition_tolerance_frames}": 0.0,
            f"transition_recall_at_{transition_tolerance_frames}": 0.0,
            f"transition_f1_at_{transition_tolerance_frames}": 0.0,
            f"boundary_mae_frames_at_{transition_tolerance_frames}": 0.0,
            f"boundary_median_frames_at_{transition_tolerance_frames}": 0.0,
            f"boundary_p90_frames_at_{transition_tolerance_frames}": 0.0,
            "valid_transition_rate": 0.0,
            "invalid_transition_count": 0.0,
            "duration_mae_frames": 0.0,
            "duration_mape": 0.0,
            "bootstrap_ci": {},
        }

    ordered_records = sorted(valid_records, key=lambda item: (int(item["video_id"]), int(item["int_time"])))
    y_true = [int(item["step_target"]) for item in ordered_records]
    y_pred = [int(item[pred_key]) for item in ordered_records]
    pred_label_key = pred_label_key or f"{pred_key}_label"
    y_true_raw = [int(item["step_target_raw"]) for item in ordered_records]
    y_pred_raw = [
        int(item.get(pred_label_key, step_index_to_raw_label(int(item[pred_key]))))
        for item in ordered_records
    ]
    confusion_np = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy().astype(np.int64)
    global_stats = compute_global_classification_statistics(confusion_np, y_true=y_true, y_pred=y_pred)
    confusion = torch.as_tensor(confusion_np, dtype=torch.int64)
    acc = float(confusion.diag().sum().item() / max(1, confusion.sum().item()))

    records_by_video: Dict[int, List[Dict[str, object]]] = {}
    for item in ordered_records:
        records_by_video.setdefault(int(item["video_id"]), []).append(item)

    per_video = []
    per_class_video_values = {
        "precision": [[] for _ in range(num_classes)],
        "recall": [[] for _ in range(num_classes)],
        "f1": [[] for _ in range(num_classes)],
        "jaccard": [[] for _ in range(num_classes)],
    }
    video_acc = []
    video_macro_precision = []
    video_macro_recall = []
    video_macro_f1 = []
    video_macro_jaccard = []
    video_edits = []
    video_step_metric = []
    video_f1_at_10 = []
    video_f1_at_25 = []
    video_f1_at_50 = []
    relaxed_video_acc = []
    relaxed_video_macro_f1 = []
    relaxed_video_macro_jaccard = []
    transition_precision_values = []
    transition_recall_values = []
    transition_f1_values = []
    boundary_mae_values = []
    boundary_median_values = []
    boundary_p90_values = []
    valid_transition_rate_values = []
    invalid_transition_count_values = []
    duration_mae_values = []
    duration_mape_values = []
    segment_counts = {
        10: [0, 0, 0],
        25: [0, 0, 0],
        50: [0, 0, 0],
    }
    for video_id in sorted(records_by_video):
        video_records = records_by_video[video_id]
        video_true = [int(item["step_target"]) for item in video_records]
        video_pred = [int(item[pred_key]) for item in video_records]
        video_true_raw = [int(item["step_target_raw"]) for item in video_records]
        video_pred_raw = [
            int(item.get(pred_label_key, step_index_to_raw_label(int(item[pred_key]))))
            for item in video_records
        ]
        video_stats = compute_videowise_phase_metrics(video_true, video_pred, num_classes=num_classes)
        macro_precision = float(video_stats["macro_precision"])
        macro_recall = float(video_stats["macro_recall"])
        macro_f1 = float(video_stats["macro_f1"])
        macro_jaccard = float(video_stats["macro_jaccard"])
        balanced_accuracy = float(video_stats["balanced_accuracy"])
        video_accuracy = float(video_stats["acc"])
        edit = compute_edit_score(video_true, video_pred)
        step_metric = float((macro_f1 + edit) / 2.0)
        relaxed_pred = apply_transition_tolerant_correction(
            video_true,
            video_pred,
            tolerance_frames=relaxed_tolerance_frames,
        )
        relaxed_stats = compute_videowise_phase_metrics(video_true, relaxed_pred, num_classes=num_classes)
        temporal_stats = compute_transition_metrics(
            video_true_raw,
            video_pred_raw,
            allowed_transitions_raw=ALLOWED_TRANSITIONS_RAW,
            tolerance_frames=transition_tolerance_frames,
            label_set=label_names,
        )
        video_tp_10, video_fp_10, video_fn_10 = compute_segment_f1_counts(
            video_true,
            video_pred,
            overlap_threshold=10,
        )
        video_tp_25, video_fp_25, video_fn_25 = compute_segment_f1_counts(
            video_true,
            video_pred,
            overlap_threshold=25,
        )
        video_tp_50, video_fp_50, video_fn_50 = compute_segment_f1_counts(
            video_true,
            video_pred,
            overlap_threshold=50,
        )
        video_segment_f1_10 = _f1_from_counts(video_tp_10, video_fp_10, video_fn_10)
        video_segment_f1_25 = _f1_from_counts(video_tp_25, video_fp_25, video_fn_25)
        video_segment_f1_50 = _f1_from_counts(video_tp_50, video_fp_50, video_fn_50)
        per_video.append(
            {
                "video_id": int(video_id),
                "acc": video_accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "macro_jaccard": macro_jaccard,
                "balanced_accuracy": balanced_accuracy,
                "edit": edit,
                "step_metric": step_metric,
                "f1_at_10": video_segment_f1_10,
                "f1_at_25": video_segment_f1_25,
                "f1_at_50": video_segment_f1_50,
                f"relaxed_acc_at_{relaxed_tolerance_frames}": float(relaxed_stats["acc"]),
                f"relaxed_macro_f1_at_{relaxed_tolerance_frames}": float(relaxed_stats["macro_f1"]),
                f"relaxed_macro_jaccard_at_{relaxed_tolerance_frames}": float(relaxed_stats["macro_jaccard"]),
                f"transition_precision_at_{transition_tolerance_frames}": float(temporal_stats["transition_precision_at_tol"]),
                f"transition_recall_at_{transition_tolerance_frames}": float(temporal_stats["transition_recall_at_tol"]),
                f"transition_f1_at_{transition_tolerance_frames}": float(temporal_stats["transition_f1_at_tol"]),
                f"boundary_mae_frames_at_{transition_tolerance_frames}": float(temporal_stats["boundary_mae_frames_at_tol"]),
                "valid_transition_rate": float(temporal_stats["valid_transition_rate"]),
                "duration_mae_frames": float(temporal_stats["duration_mae_frames"]),
                "duration_mape": float(temporal_stats["duration_mape"]),
            }
        )
        video_acc.append(video_accuracy)
        video_macro_precision.append(macro_precision)
        video_macro_recall.append(macro_recall)
        video_macro_f1.append(macro_f1)
        video_macro_jaccard.append(macro_jaccard)
        video_edits.append(edit)
        video_step_metric.append(step_metric)
        video_f1_at_10.append(video_segment_f1_10)
        video_f1_at_25.append(video_segment_f1_25)
        video_f1_at_50.append(video_segment_f1_50)
        relaxed_video_acc.append(float(relaxed_stats["acc"]))
        relaxed_video_macro_f1.append(float(relaxed_stats["macro_f1"]))
        relaxed_video_macro_jaccard.append(float(relaxed_stats["macro_jaccard"]))
        transition_precision_values.append(float(temporal_stats["transition_precision_at_tol"]))
        transition_recall_values.append(float(temporal_stats["transition_recall_at_tol"]))
        transition_f1_values.append(float(temporal_stats["transition_f1_at_tol"]))
        boundary_mae_values.append(float(temporal_stats["boundary_mae_frames_at_tol"]))
        boundary_median_values.append(float(temporal_stats["boundary_median_frames_at_tol"]))
        boundary_p90_values.append(float(temporal_stats["boundary_p90_frames_at_tol"]))
        valid_transition_rate_values.append(float(temporal_stats["valid_transition_rate"]))
        invalid_transition_count_values.append(float(temporal_stats["invalid_transition_count"]))
        duration_mae_values.append(float(temporal_stats["duration_mae_frames"]))
        duration_mape_values.append(float(temporal_stats["duration_mape"]))
        for metric_name, target_lists in per_class_video_values.items():
            for class_idx, value in enumerate(video_stats["per_class_valid"][metric_name]):
                if value is not None:
                    target_lists[class_idx].append(float(value))
        for key, threshold in ((10, 0.10), (25, 0.25), (50, 0.50)):
            true_positive, false_positive, false_negative = compute_segment_f1_counts(
                video_true,
                video_pred,
                overlap_threshold=threshold,
            )
            segment_counts[key][0] += true_positive
            segment_counts[key][1] += false_positive
            segment_counts[key][2] += false_negative

    macro_f1 = float(sum(video_macro_f1) / len(video_macro_f1)) if video_macro_f1 else 0.0
    macro_precision = float(sum(video_macro_precision) / len(video_macro_precision)) if video_macro_precision else 0.0
    macro_recall = float(sum(video_macro_recall) / len(video_macro_recall)) if video_macro_recall else 0.0
    macro_jaccard = float(sum(video_macro_jaccard) / len(video_macro_jaccard)) if video_macro_jaccard else 0.0
    edit = float(sum(video_edits) / len(video_edits)) if video_edits else 0.0
    step_metric = float((macro_f1 + edit) / 2.0)
    f1_at_10 = _f1_from_counts(*segment_counts[10])
    f1_at_25 = _f1_from_counts(*segment_counts[25])
    f1_at_50 = _f1_from_counts(*segment_counts[50])
    per_class_video_stats = {
        "precision_mean": [float(np.mean(values)) if values else 0.0 for values in per_class_video_values["precision"]],
        "precision_std": [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in per_class_video_values["precision"]],
        "recall_mean": [float(np.mean(values)) if values else 0.0 for values in per_class_video_values["recall"]],
        "recall_std": [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in per_class_video_values["recall"]],
        "f1_mean": [float(np.mean(values)) if values else 0.0 for values in per_class_video_values["f1"]],
        "f1_std": [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in per_class_video_values["f1"]],
        "jaccard_mean": [float(np.mean(values)) if values else 0.0 for values in per_class_video_values["jaccard"]],
        "jaccard_std": [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in per_class_video_values["jaccard"]],
        "valid_video_count": [len(values) for values in per_class_video_values["f1"]],
    }
    bootstrap_ci = {
        "acc": bootstrap_confidence_interval(video_acc),
        "macro_precision": bootstrap_confidence_interval(video_macro_precision),
        "macro_recall": bootstrap_confidence_interval(video_macro_recall),
        "macro_f1": bootstrap_confidence_interval(video_macro_f1),
        "macro_jaccard": bootstrap_confidence_interval(video_macro_jaccard),
        "edit": bootstrap_confidence_interval(video_edits),
        "step_metric": bootstrap_confidence_interval([row["step_metric"] for row in per_video]),
    }
    return {
        "acc": acc,
        "video_acc_mean": float(np.mean(video_acc)) if video_acc else 0.0,
        "video_acc_std": float(np.std(video_acc, ddof=1)) if len(video_acc) > 1 else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_f1_std": float(np.std(video_macro_f1, ddof=1)) if len(video_macro_f1) > 1 else 0.0,
        "macro_jaccard": macro_jaccard,
        "macro_jaccard_std": float(np.std(video_macro_jaccard, ddof=1)) if len(video_macro_jaccard) > 1 else 0.0,
        "balanced_accuracy": macro_recall,
        "edit": edit,
        "edit_std": float(np.std(video_edits, ddof=1)) if len(video_edits) > 1 else 0.0,
        "step_metric": step_metric,
        "step_metric_std": float(np.std(video_step_metric, ddof=1)) if len(video_step_metric) > 1 else 0.0,
        "f1_at_10": f1_at_10,
        "f1_at_10_std": float(np.std(video_f1_at_10, ddof=1)) if len(video_f1_at_10) > 1 else 0.0,
        "f1_at_25": f1_at_25,
        "f1_at_25_std": float(np.std(video_f1_at_25, ddof=1)) if len(video_f1_at_25) > 1 else 0.0,
        "f1_at_50": f1_at_50,
        "f1_at_50_std": float(np.std(video_f1_at_50, ddof=1)) if len(video_f1_at_50) > 1 else 0.0,
        "precision": global_stats["precision"],
        "recall": global_stats["recall"],
        "f1": global_stats["f1"],
        "jaccard": global_stats["jaccard"],
        "support": global_stats["support"],
        "confusion_matrix": confusion,
        "label_names": label_names,
        "frame_macro_precision_global": global_stats["macro_precision_global"],
        "frame_macro_recall_global": global_stats["macro_recall_global"],
        "frame_macro_f1_global": global_stats["macro_f1_global"],
        "frame_macro_jaccard_global": global_stats["macro_jaccard_global"],
        "weighted_precision_global": global_stats["weighted_precision_global"],
        "weighted_recall_global": global_stats["weighted_recall_global"],
        "weighted_f1_global": global_stats["weighted_f1_global"],
        "weighted_jaccard_global": global_stats["weighted_jaccard_global"],
        "kappa_global": global_stats["kappa_global"],
        "mcc_global": global_stats["mcc_global"],
        "per_class_video_stats": per_class_video_stats,
        "per_video": per_video,
        f"relaxed_acc_at_{relaxed_tolerance_frames}": float(np.mean(relaxed_video_acc)) if relaxed_video_acc else 0.0,
        f"relaxed_acc_at_{relaxed_tolerance_frames}_std": float(np.std(relaxed_video_acc, ddof=1)) if len(relaxed_video_acc) > 1 else 0.0,
        f"relaxed_macro_f1_at_{relaxed_tolerance_frames}": float(np.mean(relaxed_video_macro_f1)) if relaxed_video_macro_f1 else 0.0,
        f"relaxed_macro_jaccard_at_{relaxed_tolerance_frames}": float(np.mean(relaxed_video_macro_jaccard)) if relaxed_video_macro_jaccard else 0.0,
        f"transition_precision_at_{transition_tolerance_frames}": float(np.mean(transition_precision_values)) if transition_precision_values else 0.0,
        f"transition_recall_at_{transition_tolerance_frames}": float(np.mean(transition_recall_values)) if transition_recall_values else 0.0,
        f"transition_f1_at_{transition_tolerance_frames}": float(np.mean(transition_f1_values)) if transition_f1_values else 0.0,
        f"transition_f1_at_{transition_tolerance_frames}_std": float(np.std(transition_f1_values, ddof=1)) if len(transition_f1_values) > 1 else 0.0,
        f"boundary_mae_frames_at_{transition_tolerance_frames}": float(np.mean(boundary_mae_values)) if boundary_mae_values else 0.0,
        f"boundary_mae_frames_at_{transition_tolerance_frames}_std": float(np.std(boundary_mae_values, ddof=1)) if len(boundary_mae_values) > 1 else 0.0,
        f"boundary_median_frames_at_{transition_tolerance_frames}": float(np.mean(boundary_median_values)) if boundary_median_values else 0.0,
        f"boundary_p90_frames_at_{transition_tolerance_frames}": float(np.mean(boundary_p90_values)) if boundary_p90_values else 0.0,
        "valid_transition_rate": float(np.mean(valid_transition_rate_values)) if valid_transition_rate_values else 0.0,
        "invalid_transition_count": float(np.mean(invalid_transition_count_values)) if invalid_transition_count_values else 0.0,
        "duration_mae_frames": float(np.mean(duration_mae_values)) if duration_mae_values else 0.0,
        "duration_mae_frames_std": float(np.std(duration_mae_values, ddof=1)) if len(duration_mae_values) > 1 else 0.0,
        "duration_mape": float(np.mean(duration_mape_values)) if duration_mape_values else 0.0,
        "bootstrap_ci": bootstrap_ci,
    }


def compute_instrument_metrics(
    records: Sequence[Dict[str, object]],
    *,
    instrument_class_ids: Sequence[int],
    threshold: float = 0.5,
) -> Dict[str, object]:
    valid_records = [record for record in records if bool(record.get("instrument_valid_mask", False))]
    num_classes = len(instrument_class_ids)
    if not valid_records or num_classes == 0:
        zeros = [0.0 for _ in range(num_classes)]
        return {
            "instrument_micro_f1": 0.0,
            "instrument_macro_f1": 0.0,
            "instrument_precision": zeros,
            "instrument_recall": zeros,
            "instrument_f1": zeros,
            "instrument_support": [0 for _ in range(num_classes)],
        }

    targets = torch.stack([record["instrument_target"] for record in valid_records]).float()
    probs = torch.stack([record["instrument_pred_prob"] for record in valid_records]).float()
    preds = probs.ge(float(threshold)).float()

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - preds) * targets).sum(dim=0)
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2.0 * precision * recall / (precision + recall), torch.zeros_like(tp))

    tp_micro = float(tp.sum().item())
    fp_micro = float(fp.sum().item())
    fn_micro = float(fn.sum().item())
    micro_precision = tp_micro / max(1e-8, tp_micro + fp_micro)
    micro_recall = tp_micro / max(1e-8, tp_micro + fn_micro)
    micro_f1 = (
        0.0
        if (micro_precision + micro_recall) <= 0
        else 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )

    return {
        "instrument_micro_f1": float(micro_f1),
        "instrument_macro_f1": float(f1.mean().item()),
        "instrument_precision": [float(value) for value in precision.tolist()],
        "instrument_recall": [float(value) for value in recall.tolist()],
        "instrument_f1": [float(value) for value in f1.tolist()],
        "instrument_support": [int(value) for value in targets.sum(dim=0).tolist()],
    }


def accumulate_frame_predictions(
    prediction_store: Dict[Tuple[int, int], Dict[str, object]],
    *,
    video_ids: torch.Tensor,
    frame_indices: torch.Tensor,
    step_logits: torch.Tensor,
    step_targets: torch.Tensor,
    instrument_logits: torch.Tensor,
    instrument_targets: torch.Tensor,
    instrument_valid_mask: torch.Tensor,
    boundary_logits: torch.Tensor,
    boundary_targets: torch.Tensor,
    padding_mask: torch.Tensor,
) -> None:
    step_logits_cpu = step_logits.detach().cpu().float()
    instrument_logits_cpu = instrument_logits.detach().cpu().float()
    boundary_logits_cpu = boundary_logits.detach().cpu().float()
    step_targets_cpu = step_targets.detach().cpu()
    instrument_targets_cpu = instrument_targets.detach().cpu().float()
    instrument_valid_mask_cpu = instrument_valid_mask.detach().cpu()
    boundary_targets_cpu = boundary_targets.detach().cpu().float()
    padding_mask_cpu = padding_mask.detach().cpu()
    frame_indices_cpu = frame_indices.detach().cpu()
    video_ids_cpu = video_ids.detach().cpu()

    batch_size, seq_len = frame_indices_cpu.shape
    for batch_idx in range(batch_size):
        video_id = int(video_ids_cpu[batch_idx].item())
        for time_idx in range(seq_len):
            if bool(padding_mask_cpu[batch_idx, time_idx].item()):
                continue
            frame_index = int(frame_indices_cpu[batch_idx, time_idx].item())
            key = (video_id, frame_index)
            boundary_logit_value = boundary_logits_cpu[batch_idx, time_idx].clone()
            boundary_target_value = boundary_targets_cpu[batch_idx, time_idx].clone()
            if boundary_logit_value.dim() == 0:
                boundary_logit_value = boundary_logit_value.unsqueeze(0)
            if boundary_target_value.dim() == 0:
                boundary_target_value = boundary_target_value.unsqueeze(0)
            if key not in prediction_store:
                prediction_store[key] = {
                    "video_id": video_id,
                    "int_time": frame_index,
                    "step_target": int(step_targets_cpu[batch_idx, time_idx].item()),
                    "step_logits_sum": step_logits_cpu[batch_idx, time_idx].clone(),
                    "instrument_logits_sum": instrument_logits_cpu[batch_idx, time_idx].clone(),
                    "instrument_target": instrument_targets_cpu[batch_idx, time_idx].clone(),
                    "instrument_valid_mask": bool(instrument_valid_mask_cpu[batch_idx, time_idx].item()),
                    "boundary_logit_sum": boundary_logit_value,
                    "boundary_target": boundary_target_value,
                    "count": 1,
                }
            else:
                prediction_store[key]["step_logits_sum"] += step_logits_cpu[batch_idx, time_idx]
                prediction_store[key]["instrument_logits_sum"] += instrument_logits_cpu[batch_idx, time_idx]
                prediction_store[key]["boundary_logit_sum"] += boundary_logit_value
                prediction_store[key]["count"] += 1


def finalize_frame_predictions(
    prediction_store: Dict[Tuple[int, int], Dict[str, object]]
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for _, entry in sorted(prediction_store.items(), key=lambda item: item[0]):
        count = max(1, int(entry["count"]))
        avg_step_logits = entry["step_logits_sum"] / count
        avg_instrument_logits = entry["instrument_logits_sum"] / count
        avg_boundary_logits = entry["boundary_logit_sum"] / count
        boundary_probs = torch.sigmoid(avg_boundary_logits)
        record = {
            "video_id": int(entry["video_id"]),
            "int_time": int(entry["int_time"]),
            "step_target": int(entry["step_target"]),
            "step_target_raw": step_index_to_raw_label(int(entry["step_target"])),
            "step_logits_raw": avg_step_logits,
            "step_prob_raw": torch.softmax(avg_step_logits, dim=-1),
            "step_pred_raw": int(avg_step_logits.argmax(dim=-1).item()),
            "step_pred_raw_label": step_index_to_raw_label(int(avg_step_logits.argmax(dim=-1).item())),
            "instrument_target": entry["instrument_target"].clone(),
            "instrument_valid_mask": bool(entry["instrument_valid_mask"]),
            "instrument_pred_prob": torch.sigmoid(avg_instrument_logits),
            "instrument_pred_binary": torch.sigmoid(avg_instrument_logits).ge(0.5).float(),
            "boundary_target": float(entry["boundary_target"].max().item()),
            "boundary_pred_prob": float(boundary_probs.max().item()),
        }
        records.append(record)
    return records


def apply_light_postprocess(records: Sequence[Dict[str, object]], *, kernel_size: int = 3) -> List[Dict[str, object]]:
    if int(kernel_size) <= 1:
        for record in records:
            record["step_pred_light"] = int(record["step_pred_raw"])
        return list(records)

    by_video: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        by_video.setdefault(int(record["video_id"]), []).append(record)
    final_records: List[Dict[str, object]] = []
    radius = int(kernel_size) // 2
    for video_id in sorted(by_video):
        video_records = sorted(by_video[video_id], key=lambda item: int(item["int_time"]))
        raw_preds = [int(item["step_pred_raw"]) for item in video_records]
        smoothed = []
        for idx in range(len(raw_preds)):
            left = max(0, idx - radius)
            right = min(len(raw_preds), idx + radius + 1)
            window = raw_preds[left:right]
            best = max(set(window), key=window.count)
            smoothed.append(int(best))
        for record, pred in zip(video_records, smoothed):
            record["step_pred_light"] = int(pred)
            final_records.append(record)
    return sorted(final_records, key=lambda item: (int(item["video_id"]), int(item["int_time"])))


def apply_min_duration_postprocess(records: Sequence[Dict[str, object]], *, min_duration: int = 30) -> List[Dict[str, object]]:
    """Merge phase segments shorter than min_duration frames into the longer neighbour."""
    if int(min_duration) <= 1:
        for record in records:
            record["step_pred_mindur"] = int(record["step_pred_light"])
        return list(records)

    by_video: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        by_video.setdefault(int(record["video_id"]), []).append(record)
    final_records: List[Dict[str, object]] = []
    for video_id in sorted(by_video):
        video_records = sorted(by_video[video_id], key=lambda item: int(item["int_time"]))
        preds = [int(item["step_pred_light"]) for item in video_records]
        changed = True
        while changed:
            changed = False
            # build segments
            segments: List[Tuple[int, int, int]] = []
            start = 0
            for i in range(1, len(preds)):
                if preds[i] != preds[start]:
                    segments.append((start, i, preds[start]))
                    start = i
            segments.append((start, len(preds), preds[start]))
            # find shortest segment below threshold
            for seg_idx, (seg_start, seg_end, seg_label) in enumerate(segments):
                seg_len = seg_end - seg_start
                if seg_len < int(min_duration) and len(segments) > 1:
                    prev_len = (segments[seg_idx - 1][1] - segments[seg_idx - 1][0]) if seg_idx > 0 else 0
                    next_len = (segments[seg_idx + 1][1] - segments[seg_idx + 1][0]) if seg_idx < len(segments) - 1 else 0
                    if prev_len >= next_len and prev_len > 0:
                        merge_label = segments[seg_idx - 1][2]
                    elif next_len > 0:
                        merge_label = segments[seg_idx + 1][2]
                    else:
                        continue
                    for i in range(seg_start, seg_end):
                        preds[i] = merge_label
                    changed = True
                    break
        for record, pred in zip(video_records, preds):
            record["step_pred_mindur"] = int(pred)
            final_records.append(record)
    return sorted(final_records, key=lambda item: (int(item["video_id"]), int(item["int_time"])))


# Cache the log-transition matrix (built once, reused every epoch)
_LOG_TRANS_CACHE: Optional[np.ndarray] = None

# Emission prior: multiplicative adjustment to softmax probs before Viterbi.
# Suppresses over-predicted classes, boosts under-predicted ones.
# Key = class index (0-based), value = multiplier on probability.
_EMISSION_PRIOR: Dict[int, float] = build_emission_prior_by_index(_L2I)


def apply_viterbi_postprocess(
    records: Sequence[Dict[str, object]],
    *,
    log_transition_matrix: Optional[np.ndarray] = None,
    log_position_prior_matrix: Optional[np.ndarray] = None,
    position_prior_weight: float = 0.0,
    late_chain_decoder_guard: float = 0.0,
) -> List[Dict[str, object]]:
    """Apply Viterbi decoding using frame-level softmax probabilities + transition prior.

    Reads ``step_prob_raw`` (softmax tensor) from each record and writes ``step_pred_viterbi``.
    Falls back to ``step_pred_mindur`` if ``step_prob_raw`` is unavailable.
    """
    global _LOG_TRANS_CACHE
    num_classes = len(LABELS12)
    if log_transition_matrix is not None:
        log_trans = log_transition_matrix
    else:
        if _LOG_TRANS_CACHE is None:
            _LOG_TRANS_CACHE = _build_log_transition_matrix(num_classes)
        log_trans = _LOG_TRANS_CACHE

    # Build log emission prior vector
    emission_log_prior = np.zeros(num_classes, dtype=np.float64)
    for cls_idx, multiplier in _EMISSION_PRIOR.items():
        emission_log_prior[cls_idx] = math.log(max(1e-6, multiplier))

    by_video: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        by_video.setdefault(int(record["video_id"]), []).append(record)

    final_records: List[Dict[str, object]] = []
    for video_id in sorted(by_video):
        video_records = sorted(by_video[video_id], key=lambda item: int(item["int_time"]))
        # Build log-probability matrix (T, C)
        has_probs = "step_prob_raw" in video_records[0] and torch.is_tensor(video_records[0]["step_prob_raw"])
        if has_probs:
            probs = torch.stack([r["step_prob_raw"] for r in video_records], dim=0).numpy().astype(np.float64)
            probs = np.clip(probs, 1e-12, 1.0)
            log_probs = np.log(probs) + emission_log_prior[np.newaxis, :]  # apply prior
            log_probs = apply_position_log_prior_to_log_probs(
                log_probs,
                log_position_prior_matrix=log_position_prior_matrix,
                position_prior_weight=position_prior_weight,
            )
            if "late_chain_logits" in video_records[0]:
                late_chain_logits = np.stack(
                    [np.asarray(r["late_chain_logits"], dtype=np.float64) for r in video_records],
                    axis=0,
                )
                late_chain_gate = np.asarray(
                    [float(r.get("late_chain_gate", 1.0)) for r in video_records],
                    dtype=np.float64,
                )
                log_probs = apply_late_chain_decoder_guard(
                    log_probs,
                    late_chain_logits=late_chain_logits,
                    late_chain_gate=late_chain_gate,
                    late_chain_decoder_guard=late_chain_decoder_guard,
                )
        else:
            # Fallback: copy mindur predictions
            for record in video_records:
                record["step_pred_viterbi"] = int(record.get("step_pred_mindur", record.get("step_pred_raw", 0)))
                record["step_pred_viterbi_label"] = int(
                    step_index_to_raw_label(int(record["step_pred_viterbi"]))
                )
                final_records.append(record)
            continue
        decoded = _viterbi_decode(log_probs, log_trans)
        for record, pred in zip(video_records, decoded):
            record["step_pred_viterbi"] = int(pred)
            record["step_pred_viterbi_label"] = int(step_index_to_raw_label(int(pred)))
            final_records.append(record)
    return sorted(final_records, key=lambda item: (int(item["video_id"]), int(item["int_time"])))


def _format_instrument_ids(vector: torch.Tensor, instrument_class_ids: Sequence[int]) -> str:
    active_ids = [int(instrument_class_ids[idx]) for idx, value in enumerate(vector.tolist()) if float(value) > 0.5]
    return json.dumps(active_ids, ensure_ascii=False)


def export_all_frames_csv(records: Sequence[Dict[str, object]], path: Path, instrument_class_ids: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "int_time",
        "step_target",
        "step_target_raw",
        "step_pred_raw",
        "step_pred_raw_label",
        "step_pred_light",
        "step_pred_light_label",
        "step_pred_mindur",
        "step_pred_mindur_label",
        "step_pred_viterbi",
        "step_pred_viterbi_label",
        "instrument_valid_mask",
        "instrument_target",
        "instrument_pred_prob",
        "instrument_pred_binary",
        "boundary_target",
        "boundary_pred_prob",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "video_id": int(record["video_id"]),
                    "int_time": int(record["int_time"]),
                    "step_target": int(record["step_target"]),
                    "step_target_raw": int(record["step_target_raw"]),
                    "step_pred_raw": int(record["step_pred_raw"]),
                    "step_pred_raw_label": int(step_index_to_raw_label(int(record["step_pred_raw"]))),
                    "step_pred_light": int(record.get("step_pred_light", record["step_pred_raw"])),
                    "step_pred_light_label": int(step_index_to_raw_label(int(record.get("step_pred_light", record["step_pred_raw"])))),
                    "step_pred_mindur": int(record.get("step_pred_mindur", record["step_pred_raw"])),
                    "step_pred_mindur_label": int(
                        step_index_to_raw_label(int(record.get("step_pred_mindur", record["step_pred_raw"])))
                    ),
                    "step_pred_viterbi": int(record.get("step_pred_viterbi", record["step_pred_raw"])),
                    "step_pred_viterbi_label": int(
                        step_index_to_raw_label(int(record.get("step_pred_viterbi", record["step_pred_raw"])))
                    ),
                    "instrument_valid_mask": bool(record["instrument_valid_mask"]),
                    "instrument_target": _format_instrument_ids(record["instrument_target"], instrument_class_ids),
                    "instrument_pred_prob": json.dumps([round(float(v), 6) for v in record["instrument_pred_prob"].tolist()], ensure_ascii=False),
                    "instrument_pred_binary": _format_instrument_ids(record["instrument_pred_binary"], instrument_class_ids),
                    "boundary_target": round(float(record["boundary_target"]), 6),
                    "boundary_pred_prob": round(float(record["boundary_pred_prob"]), 6),
                }
            )


def export_eval_frames_csv(records: Sequence[Dict[str, object]], path: Path, instrument_class_ids: Sequence[int]) -> None:
    eval_records = [record for record in records if 0 <= int(record["step_target"]) < len(LABELS12)]
    export_all_frames_csv(eval_records, path, instrument_class_ids)


def update_resource_peaks(resource_peaks: Dict[str, float], *, device: torch.device, gpu_limit: float, ram_limit: float) -> None:
    gpu_status = check_gpu_memory(device, limit_ratio=gpu_limit)
    ram_status = check_ram_usage(limit_ratio=ram_limit)
    resource_peaks["max_gpu_used_gb"] = max(float(resource_peaks.get("max_gpu_used_gb", 0.0)), float(gpu_status["used_gb"]))
    resource_peaks["max_gpu_used_ratio"] = max(float(resource_peaks.get("max_gpu_used_ratio", 0.0)), float(gpu_status["used_ratio"]))
    resource_peaks["max_ram_used_gb"] = max(float(resource_peaks.get("max_ram_used_gb", 0.0)), float(ram_status["used_gb"]))
    resource_peaks["max_ram_used_ratio"] = max(float(resource_peaks.get("max_ram_used_ratio", 0.0)), float(ram_status["used_ratio"]))


def build_resource_summary(
    *,
    model: torch.nn.Module,
    total_frames: int,
    elapsed_seconds: float,
    resource_peaks: Mapping[str, float],
) -> Dict[str, float]:
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    total_params = sum(parameter.numel() for parameter in base_model.parameters())
    trainable_params = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    elapsed = max(1e-8, float(elapsed_seconds))
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "eval_total_frames": int(total_frames),
        "eval_elapsed_seconds": float(elapsed_seconds),
        "eval_frames_per_second": float(total_frames / elapsed),
        "eval_milliseconds_per_frame": float(1000.0 * elapsed / max(1, total_frames)),
        "peak_gpu_used_gb": float(resource_peaks.get("max_gpu_used_gb", 0.0)),
        "peak_gpu_used_ratio": float(resource_peaks.get("max_gpu_used_ratio", 0.0)),
        "peak_ram_used_gb": float(resource_peaks.get("max_ram_used_gb", 0.0)),
        "peak_ram_used_ratio": float(resource_peaks.get("max_ram_used_ratio", 0.0)),
    }


def format_confusion_matrix(confusion: torch.Tensor, label_names: Sequence[int]) -> str:
    header = "true\\pred " + " ".join(f"{label:>5}" for label in label_names)
    lines = [header]
    for row_idx, label in enumerate(label_names):
        row_values = " ".join(f"{int(value):>5}" for value in confusion[row_idx].tolist())
        lines.append(f"{label:>8} {row_values}")
    return "\n".join(lines)


def print_validation_details(metrics: Dict[str, object]) -> None:
    label_names = metrics["label_names"]
    if "macro_jaccard" in metrics:
        print(
            "val_summary_extended: "
            f"video_acc_mean={metrics.get('video_acc_mean', 0.0):.4f} "
            f"macro_precision={metrics.get('macro_precision', 0.0):.4f} "
            f"macro_recall={metrics.get('macro_recall', 0.0):.4f} "
            f"macro_jaccard={metrics.get('macro_jaccard', 0.0):.4f} "
            f"kappa={metrics.get('kappa_global', 0.0):.4f} "
            f"mcc={metrics.get('mcc_global', 0.0):.4f}"
        )
        print(
            "val_surgical_extended: "
            f"relaxed_acc@10={metrics.get('relaxed_acc_at_10', 0.0):.4f} "
            f"relaxed_macro_f1@10={metrics.get('relaxed_macro_f1_at_10', 0.0):.4f} "
            f"transition_f1@10={metrics.get('transition_f1_at_10', 0.0):.4f} "
            f"valid_transition_rate={metrics.get('valid_transition_rate', 0.0):.4f} "
            f"duration_mae_frames={metrics.get('duration_mae_frames', 0.0):.2f}"
        )
    if "resource_summary" in metrics:
        resource_summary = metrics["resource_summary"]
        print(
            "val_resource_summary: "
            f"fps={float(resource_summary.get('eval_frames_per_second', 0.0)):.2f} "
            f"ms_per_frame={float(resource_summary.get('eval_milliseconds_per_frame', 0.0)):.2f} "
            f"peak_gpu_gb={float(resource_summary.get('peak_gpu_used_gb', 0.0)):.2f} "
            f"peak_ram_gb={float(resource_summary.get('peak_ram_used_gb', 0.0)):.2f}"
        )
    print(
        "val_segmental_global: "
        f"f1@10={metrics.get('f1_at_10', 0.0):.4f} "
        f"f1@25={metrics.get('f1_at_25', 0.0):.4f} "
        f"f1@50={metrics.get('f1_at_50', 0.0):.4f}"
    )
    print("val_per_class_global:")
    print("raw_step  precision  recall  f1  support")
    for label, p, r, f, s in zip(label_names, metrics["precision"], metrics["recall"], metrics["f1"], metrics["support"]):
        print(f"{label:>8}  {p:>9.4f}  {r:>6.4f}  {f:>4.4f}  {s:>7}")
    print("val_confusion_matrix_global:")
    print(format_confusion_matrix(metrics["confusion_matrix"], label_names))


@torch.no_grad()
def run_validation_epoch(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    instrument_pos_weight: Optional[torch.Tensor],
    lambda_bd: float,
    lambda_inst: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    progress_bar: bool,
    log_interval: int,
    non_blocking: bool,
    light_post_kernel_size: int,
    min_duration_frames: int,
    instrument_class_ids: Sequence[int],
    export_prefix: Optional[Path],
    gpu_mem_limit: float,
    ram_limit: float,
    log_transition_matrix: Optional[np.ndarray] = None,
    log_position_prior_matrix: Optional[np.ndarray] = None,
    position_prior_weight: float = 0.0,
    late_chain_decoder_guard: float = 0.0,
    hard_class_profile_raw_labels: Sequence[int] = HARD_CLASS_RAW_LABELS,
) -> Dict[str, object]:
    model.eval()
    base_frame_weights = build_center_frame_weights(loader.dataset.window_size).to(device)
    prediction_store: Dict[Tuple[int, int], Dict[str, object]] = {}
    resource_peaks = {
        "max_gpu_used_gb": 0.0,
        "max_gpu_used_ratio": 0.0,
        "max_ram_used_gb": 0.0,
        "max_ram_used_ratio": 0.0,
    }
    aggregate = {
        "step_weighted_sum": 0.0,
        "step_weight_sum": 0.0,
        "step_unweighted_sum": 0.0,
        "step_valid_count": 0.0,
        "boundary_loss_sum": 0.0,
        "boundary_weight_sum": 0.0,
        "instrument_loss_sum": 0.0,
        "instrument_valid_count": 0.0,
    }
    iterator = maybe_make_progress_bar(loader, total=len(loader), desc="val", enabled=progress_bar)
    eval_start = perf_counter()

    for step, batch in enumerate(iterator, start=1):
        check_resources_and_warn(device, step, gpu_limit=gpu_mem_limit, ram_limit=ram_limit)
        update_resource_peaks(resource_peaks, device=device, gpu_limit=gpu_mem_limit, ram_limit=ram_limit)
        inputs = batch["features"].to(device, non_blocking=non_blocking) if "features" in batch else batch["images"].to(device, non_blocking=non_blocking)
        device_batch = {
            key: value.to(device, non_blocking=non_blocking)
            for key, value in batch.items()
            if torch.is_tensor(value) and key not in {"features", "images"}
        }
        frame_weights = base_frame_weights.unsqueeze(0).expand(device_batch["step_targets"].shape[0], -1)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
            if use_amp
            else torch.no_grad()
        )
        with autocast_context:
            outputs = model(
                inputs,
                padding_mask=device_batch["padding_mask"],
                stage="stage_b",
            )
            losses = compute_multitask_loss(
                outputs,
                device_batch,
                class_weights=class_weights,
                frame_weights=frame_weights,
                instrument_pos_weight=instrument_pos_weight,
                lambda_bd=lambda_bd,
                lambda_inst=lambda_inst,
            )

        aggregate["step_weighted_sum"] += float(losses["step_ce_weighted_sum"].item())
        aggregate["step_weight_sum"] += float(losses["step_frame_weight_sum"].item())
        aggregate["step_unweighted_sum"] += float(losses["step_ce_unweighted_sum"].item())
        aggregate["step_valid_count"] += float(losses["step_valid_count"].item())
        aggregate["boundary_loss_sum"] += float(losses["boundary_loss_sum"].item())
        aggregate["boundary_weight_sum"] += float(losses["boundary_weight_sum"].item())
        aggregate["instrument_loss_sum"] += float(losses["instrument_loss_sum"].item())
        aggregate["instrument_valid_count"] += float(losses["instrument_valid_count"].item())

        accumulate_frame_predictions(
            prediction_store,
            video_ids=device_batch["video_id"],
            frame_indices=device_batch["frame_indices"],
            step_logits=outputs["step_logits"],
            step_targets=device_batch["step_targets"],
            instrument_logits=outputs["instrument_logits"],
            instrument_targets=device_batch["instrument_targets"],
            instrument_valid_mask=device_batch["instrument_valid_mask"],
            boundary_logits=outputs["boundary_logits"],
            boundary_targets=device_batch["boundary_targets"],
            padding_mask=device_batch["padding_mask"],
        )
        if progress_bar and tqdm is not None and (step % max(1, log_interval) == 0 or step == len(loader)):
            running_step_loss = aggregate["step_weighted_sum"] / max(1e-8, aggregate["step_weight_sum"])
            running_boundary_loss = aggregate["boundary_loss_sum"] / max(1e-8, aggregate["boundary_weight_sum"])
            iterator.set_postfix_str(
                f"step={running_step_loss:.4f} bd={running_boundary_loss:.4f}"
            )

    if progress_bar and tqdm is not None and hasattr(iterator, "close"):
        iterator.close()

    raw_records = finalize_frame_predictions(prediction_store)
    records = apply_light_postprocess(raw_records, kernel_size=light_post_kernel_size)
    records = apply_min_duration_postprocess(records, min_duration=min_duration_frames)
    records = apply_viterbi_postprocess(
        records,
        log_transition_matrix=log_transition_matrix,
        log_position_prior_matrix=log_position_prior_matrix,
        position_prior_weight=position_prior_weight,
        late_chain_decoder_guard=late_chain_decoder_guard,
    )
    raw_metrics = compute_step_metrics(records, num_classes=len(LABELS12), pred_key="step_pred_raw", pred_label_key="step_pred_raw_label")
    light_metrics = compute_step_metrics(records, num_classes=len(LABELS12), pred_key="step_pred_light")
    mindur_metrics = compute_step_metrics(records, num_classes=len(LABELS12), pred_key="step_pred_mindur")
    viterbi_metrics = compute_step_metrics(records, num_classes=len(LABELS12), pred_key="step_pred_viterbi", pred_label_key="step_pred_viterbi_label")
    instrument_metrics = compute_instrument_metrics(records, instrument_class_ids=instrument_class_ids)
    total_eval_frames = sum(1 for record in records if 0 <= int(record["step_target"]) < len(LABELS12))
    resource_summary = build_resource_summary(
        model=model,
        total_frames=total_eval_frames,
        elapsed_seconds=perf_counter() - eval_start,
        resource_peaks=resource_peaks,
    )
    variant_metrics = {
        "raw": raw_metrics,
        "light": light_metrics,
        "mindur": mindur_metrics,
        "viterbi": viterbi_metrics,
    }
    statistical_tests = compute_variant_statistical_tests(
        {name: metrics["per_video"] for name, metrics in variant_metrics.items()},
        metric_names=("acc", "macro_f1", "macro_jaccard", "edit", "step_metric"),
    )

    if export_prefix is not None:
        export_all_frames_csv(records, export_prefix.parent / f"{export_prefix.name}_all_frames.csv", instrument_class_ids)
        export_eval_frames_csv(records, export_prefix.parent / f"{export_prefix.name}_eval_frames.csv", instrument_class_ids)
        export_hard_class_error_profile(
            records,
            export_prefix.parent / f"{export_prefix.name}_hardclass_profile.json",
            hard_raw_labels=hard_class_profile_raw_labels,
            instrument_class_ids=instrument_class_ids,
        )
        export_hard_class_error_profile(
            records,
            export_prefix.parent / f"{export_prefix.name}_hardclass_profile_viterbi.json",
            hard_raw_labels=hard_class_profile_raw_labels,
            instrument_class_ids=instrument_class_ids,
            pred_label_key="step_pred_viterbi_label",
        )
        export_metric_bundle(
            export_prefix=export_prefix,
            label_names=LABELS12,
            variant_metrics=variant_metrics,
            resource_summary=resource_summary,
            statistical_tests=statistical_tests,
        )

    step_loss = aggregate["step_weighted_sum"] / max(1e-8, aggregate["step_weight_sum"])
    boundary_loss = aggregate["boundary_loss_sum"] / max(1e-8, aggregate["boundary_weight_sum"])
    instrument_denominator = max(1.0, aggregate["instrument_valid_count"] * max(1, len(instrument_class_ids)))
    instrument_loss = aggregate["instrument_loss_sum"] / instrument_denominator
    total_loss = step_loss + float(lambda_bd) * boundary_loss + float(lambda_inst) * instrument_loss
    metrics = {
        "loss": float(total_loss),
        "total_loss": float(total_loss),
        "step_loss": float(step_loss),
        "boundary_loss": float(boundary_loss),
        "instrument_loss": float(instrument_loss),
        "acc": raw_metrics["acc"],
        "video_acc_mean": raw_metrics["video_acc_mean"],
        "video_acc_std": raw_metrics["video_acc_std"],
        "macro_precision": raw_metrics["macro_precision"],
        "macro_recall": raw_metrics["macro_recall"],
        "macro_f1": raw_metrics["macro_f1"],
        "macro_f1_std": raw_metrics["macro_f1_std"],
        "macro_jaccard": raw_metrics["macro_jaccard"],
        "macro_jaccard_std": raw_metrics["macro_jaccard_std"],
        "balanced_accuracy": raw_metrics["balanced_accuracy"],
        "edit": raw_metrics["edit"],
        "edit_std": raw_metrics["edit_std"],
        "step_metric": raw_metrics["step_metric"],
        "step_metric_std": raw_metrics["step_metric_std"],
        "f1_at_10": raw_metrics["f1_at_10"],
        "f1_at_10_std": raw_metrics["f1_at_10_std"],
        "f1_at_25": raw_metrics["f1_at_25"],
        "f1_at_25_std": raw_metrics["f1_at_25_std"],
        "f1_at_50": raw_metrics["f1_at_50"],
        "f1_at_50_std": raw_metrics["f1_at_50_std"],
        "precision": raw_metrics["precision"],
        "recall": raw_metrics["recall"],
        "f1": raw_metrics["f1"],
        "jaccard": raw_metrics["jaccard"],
        "support": raw_metrics["support"],
        "confusion_matrix": raw_metrics["confusion_matrix"],
        "label_names": raw_metrics["label_names"],
        "per_video": raw_metrics["per_video"],
        "kappa_global": raw_metrics["kappa_global"],
        "mcc_global": raw_metrics["mcc_global"],
        "relaxed_acc_at_10": raw_metrics["relaxed_acc_at_10"],
        "relaxed_acc_at_10_std": raw_metrics["relaxed_acc_at_10_std"],
        "relaxed_macro_f1_at_10": raw_metrics["relaxed_macro_f1_at_10"],
        "relaxed_macro_jaccard_at_10": raw_metrics["relaxed_macro_jaccard_at_10"],
        "transition_precision_at_10": raw_metrics["transition_precision_at_10"],
        "transition_recall_at_10": raw_metrics["transition_recall_at_10"],
        "transition_f1_at_10": raw_metrics["transition_f1_at_10"],
        "transition_f1_at_10_std": raw_metrics["transition_f1_at_10_std"],
        "boundary_mae_frames_at_10": raw_metrics["boundary_mae_frames_at_10"],
        "boundary_mae_frames_at_10_std": raw_metrics["boundary_mae_frames_at_10_std"],
        "valid_transition_rate": raw_metrics["valid_transition_rate"],
        "duration_mae_frames": raw_metrics["duration_mae_frames"],
        "duration_mae_frames_std": raw_metrics["duration_mae_frames_std"],
        "duration_mape": raw_metrics["duration_mape"],
        "bootstrap_ci": raw_metrics["bootstrap_ci"],
        "light_post_macro_f1": light_metrics["macro_f1"],
        "light_post_macro_jaccard": light_metrics["macro_jaccard"],
        "light_post_edit": light_metrics["edit"],
        "light_post_step_metric": light_metrics["step_metric"],
        "light_post_f1_at_10": light_metrics["f1_at_10"],
        "light_post_f1_at_25": light_metrics["f1_at_25"],
        "light_post_f1_at_50": light_metrics["f1_at_50"],
        "mindur_post_macro_f1": mindur_metrics["macro_f1"],
        "mindur_post_macro_jaccard": mindur_metrics["macro_jaccard"],
        "mindur_post_edit": mindur_metrics["edit"],
        "mindur_post_step_metric": mindur_metrics["step_metric"],
        "mindur_post_f1_at_10": mindur_metrics["f1_at_10"],
        "mindur_post_f1_at_25": mindur_metrics["f1_at_25"],
        "mindur_post_f1_at_50": mindur_metrics["f1_at_50"],
        "viterbi_macro_f1": viterbi_metrics["macro_f1"],
        "viterbi_macro_f1_std": viterbi_metrics["macro_f1_std"],
        "viterbi_macro_jaccard": viterbi_metrics["macro_jaccard"],
        "viterbi_macro_jaccard_std": viterbi_metrics["macro_jaccard_std"],
        "viterbi_edit": viterbi_metrics["edit"],
        "viterbi_edit_std": viterbi_metrics["edit_std"],
        "viterbi_step_metric": viterbi_metrics["step_metric"],
        "viterbi_step_metric_std": viterbi_metrics["step_metric_std"],
        "viterbi_f1_at_10": viterbi_metrics["f1_at_10"],
        "viterbi_f1_at_10_std": viterbi_metrics["f1_at_10_std"],
        "viterbi_f1_at_25": viterbi_metrics["f1_at_25"],
        "viterbi_f1_at_25_std": viterbi_metrics["f1_at_25_std"],
        "viterbi_f1_at_50": viterbi_metrics["f1_at_50"],
        "viterbi_f1_at_50_std": viterbi_metrics["f1_at_50_std"],
        "viterbi_relaxed_acc_at_10": viterbi_metrics["relaxed_acc_at_10"],
        "viterbi_relaxed_acc_at_10_std": viterbi_metrics["relaxed_acc_at_10_std"],
        "viterbi_relaxed_macro_f1_at_10": viterbi_metrics["relaxed_macro_f1_at_10"],
        "viterbi_relaxed_macro_jaccard_at_10": viterbi_metrics["relaxed_macro_jaccard_at_10"],
        "viterbi_transition_precision_at_10": viterbi_metrics["transition_precision_at_10"],
        "viterbi_transition_recall_at_10": viterbi_metrics["transition_recall_at_10"],
        "viterbi_transition_f1_at_10": viterbi_metrics["transition_f1_at_10"],
        "viterbi_transition_f1_at_10_std": viterbi_metrics["transition_f1_at_10_std"],
        "viterbi_boundary_mae_frames_at_10": viterbi_metrics["boundary_mae_frames_at_10"],
        "viterbi_boundary_mae_frames_at_10_std": viterbi_metrics["boundary_mae_frames_at_10_std"],
        "viterbi_valid_transition_rate": viterbi_metrics["valid_transition_rate"],
        "viterbi_duration_mae_frames": viterbi_metrics["duration_mae_frames"],
        "viterbi_duration_mae_frames_std": viterbi_metrics["duration_mae_frames_std"],
        "viterbi_duration_mape": viterbi_metrics["duration_mape"],
        "resource_summary": resource_summary,
        "statistical_tests": statistical_tests,
        **instrument_metrics,
    }
    return metrics


@torch.no_grad()
def run_mstcn_validation_epoch(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    instrument_pos_weight: Optional[torch.Tensor],
    lambda_bd: float,
    lambda_inst: float,
    lambda_tmse: float,
    lambda_late_chain: float = 0.0,
    lambda_coarse: float = 0.0,
    use_amp: bool,
    amp_dtype: torch.dtype,
    progress_bar: bool,
    log_interval: int,
    non_blocking: bool,
    light_post_kernel_size: int,
    min_duration_frames: int,
    instrument_class_ids: Sequence[int],
    export_prefix: Optional[Path],
    gpu_mem_limit: float,
    ram_limit: float,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    log_transition_matrix: Optional[np.ndarray] = None,
    log_position_prior_matrix: Optional[np.ndarray] = None,
    position_prior_weight: float = 0.0,
    late_chain_decoder_guard: float = 0.0,
    online_eval: bool = False,
    online_transition_alpha: float = 1.0,
    ema_decay: float = 0.0,
    hard_class_profile_raw_labels: Sequence[int] = HARD_CLASS_RAW_LABELS,
) -> Dict[str, object]:
    """Validation for the full-video MS-TCN model; no window aggregation is needed."""
    model.eval()
    all_records: List[Dict[str, object]] = []
    loss_sum = 0.0
    late_chain_loss_sum = 0.0
    coarse_loss_sum = 0.0
    num_batches = 0
    resource_peaks = {
        "max_gpu_used_gb": 0.0,
        "max_gpu_used_ratio": 0.0,
        "max_ram_used_gb": 0.0,
        "max_ram_used_ratio": 0.0,
    }

    iterator = maybe_make_progress_bar(loader, total=len(loader), desc="val", enabled=progress_bar)
    eval_start = perf_counter()

    for step, batch in enumerate(iterator, start=1):
        check_resources_and_warn(device, step, gpu_limit=gpu_mem_limit, ram_limit=ram_limit)
        update_resource_peaks(resource_peaks, device=device, gpu_limit=gpu_mem_limit, ram_limit=ram_limit)
        features = batch["features"].to(device, non_blocking=non_blocking)
        device_batch = {
            key: value.to(device, non_blocking=non_blocking)
            for key, value in batch.items()
            if torch.is_tensor(value) and key != "features"
        }
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
            if use_amp else torch.no_grad()
        )
        with autocast_context:
            outputs = model(features, mask=device_batch["mask"])
            losses = compute_mstcn_loss(
                outputs, device_batch,
                class_weights=class_weights,
                instrument_pos_weight=instrument_pos_weight,
                lambda_bd=lambda_bd, lambda_inst=lambda_inst, lambda_tmse=lambda_tmse,
                lambda_late_chain=lambda_late_chain,
                lambda_coarse=lambda_coarse,
                label_smoothing=label_smoothing, focal_gamma=focal_gamma,
            )

        loss_sum += float(losses["loss"].item())
        late_chain_loss_sum += float(losses.get("late_chain_loss", losses["loss"] * 0.0).item())
        coarse_loss_sum += float(losses.get("coarse_loss", losses["loss"] * 0.0).item())
        num_batches += 1

        # Build per-frame records
        step_logits = outputs["step_logits"].detach().cpu().float()
        instrument_logits = outputs["instrument_logits"].detach().cpu().float()
        boundary_logits = outputs["boundary_logits"].detach().cpu().float()
        coarse_logits = outputs.get("coarse_logits")
        coarse_probs = outputs.get("coarse_probs")
        late_chain_logits = outputs.get("late_chain_logits")
        late_chain_gate = outputs.get("late_chain_gate")
        coarse_logits_cpu = coarse_logits.detach().cpu().float() if torch.is_tensor(coarse_logits) else None
        coarse_probs_cpu = coarse_probs.detach().cpu().float() if torch.is_tensor(coarse_probs) else None
        late_chain_logits_cpu = late_chain_logits.detach().cpu().float() if torch.is_tensor(late_chain_logits) else None
        late_chain_gate_cpu = late_chain_gate.detach().cpu().float() if torch.is_tensor(late_chain_gate) else None
        step_targets_cpu = batch["step_targets"]
        lengths = batch["length"]
        video_ids = batch["video_id"]

        B = step_logits.shape[0]
        for b_idx in range(B):
            vid = int(video_ids[b_idx].item())
            L = int(lengths[b_idx].item())
            for t in range(L):
                logits_t = step_logits[b_idx, t]
                prob_t = torch.softmax(logits_t, dim=-1)
                inst_logits_t = instrument_logits[b_idx, t]
                inst_prob = torch.sigmoid(inst_logits_t)
                record = {
                    "video_id": vid,
                    "int_time": t,
                    "step_target": int(step_targets_cpu[b_idx, t].item()),
                    "step_target_raw": step_index_to_raw_label(int(step_targets_cpu[b_idx, t].item())),
                    "step_prob_raw": prob_t,
                    "step_pred_raw": int(logits_t.argmax().item()),
                    "step_pred_raw_label": step_index_to_raw_label(int(logits_t.argmax().item())),
                    "instrument_target": batch["instrument_targets"][b_idx, t].float(),
                    "instrument_valid_mask": bool(batch["instrument_valid_mask"][b_idx, t].item()),
                    "instrument_pred_prob": inst_prob,
                    "instrument_pred_binary": inst_prob.ge(0.5).float(),
                    "boundary_target": float(batch["boundary_targets"][b_idx, t].max().item()),
                    "boundary_pred_prob": float(torch.sigmoid(boundary_logits[b_idx, t]).max().item()),
                }
                if coarse_logits_cpu is not None:
                    record["coarse_logits"] = coarse_logits_cpu[b_idx, t].tolist()
                    record["coarse_pred"] = int(coarse_logits_cpu[b_idx, t].argmax().item())
                if coarse_probs_cpu is not None:
                    record["coarse_prob_raw"] = coarse_probs_cpu[b_idx, t].tolist()
                if late_chain_logits_cpu is not None:
                    record["late_chain_logits"] = late_chain_logits_cpu[b_idx, t].tolist()
                if late_chain_gate_cpu is not None:
                    record["late_chain_gate"] = float(late_chain_gate_cpu[b_idx, t].max().item())
                all_records.append(record)

        if progress_bar and tqdm is not None and (step % max(1, log_interval) == 0 or step == len(loader)):
            iterator.set_postfix_str(f"loss={loss_sum / max(1, num_batches):.4f}")

    if progress_bar and tqdm is not None and hasattr(iterator, "close"):
        iterator.close()

    if online_eval:
        records = sorted(all_records, key=lambda item: (int(item["video_id"]), int(item["int_time"])))
        records = apply_online_transition_postprocess(
            records,
            log_transition_matrix=log_transition_matrix,
            transition_alpha=online_transition_alpha,
            ema_decay=ema_decay,
        )
        for record in records:
            record["step_pred_light"] = int(record["step_pred_raw"])
            record["step_pred_mindur"] = int(record["step_pred_raw"])
            record["step_pred_viterbi"] = int(record["step_pred_raw"])
            record["step_pred_viterbi_label"] = int(record["step_pred_raw_label"])
    else:
        # Post-processing and metrics (same pipeline as window-based)
        records = apply_light_postprocess(all_records, kernel_size=light_post_kernel_size)
        records = apply_min_duration_postprocess(records, min_duration=min_duration_frames)
        records = apply_viterbi_postprocess(
            records,
            log_transition_matrix=log_transition_matrix,
            log_position_prior_matrix=log_position_prior_matrix,
            position_prior_weight=position_prior_weight,
            late_chain_decoder_guard=late_chain_decoder_guard,
        )

    raw_metrics = compute_step_metrics(records, num_classes=len(LABELS12), pred_key="step_pred_raw", pred_label_key="step_pred_raw_label")
    online_metrics = compute_step_metrics(
        records,
        num_classes=len(LABELS12),
        pred_key="step_pred_online" if online_eval else "step_pred_raw",
        pred_label_key="step_pred_online_label" if online_eval else "step_pred_raw_label",
    )
    light_metrics = compute_step_metrics(records, num_classes=len(LABELS12), pred_key="step_pred_light")
    mindur_metrics = compute_step_metrics(records, num_classes=len(LABELS12), pred_key="step_pred_mindur")
    viterbi_metrics = compute_step_metrics(records, num_classes=len(LABELS12), pred_key="step_pred_viterbi", pred_label_key="step_pred_viterbi_label")
    instrument_metrics = compute_instrument_metrics(records, instrument_class_ids=instrument_class_ids)
    total_eval_frames = sum(1 for record in records if 0 <= int(record["step_target"]) < len(LABELS12))
    resource_summary = build_resource_summary(
        model=model,
        total_frames=total_eval_frames,
        elapsed_seconds=perf_counter() - eval_start,
        resource_peaks=resource_peaks,
    )
    variant_metrics = {
        "raw": raw_metrics,
        "online": online_metrics,
        "light": light_metrics,
        "mindur": mindur_metrics,
        "viterbi": viterbi_metrics,
    }
    statistical_tests = compute_variant_statistical_tests(
        {name: metrics["per_video"] for name, metrics in variant_metrics.items()},
        metric_names=("acc", "macro_f1", "macro_jaccard", "edit", "step_metric"),
    )

    if export_prefix is not None:
        export_all_frames_csv(
            records, export_prefix.parent / f"{export_prefix.name}_all_frames.csv", instrument_class_ids)
        export_eval_frames_csv(records, export_prefix.parent / f"{export_prefix.name}_eval_frames.csv", instrument_class_ids)
        export_hard_class_error_profile(
            records,
            export_prefix.parent / f"{export_prefix.name}_hardclass_profile.json",
            hard_raw_labels=hard_class_profile_raw_labels,
            instrument_class_ids=instrument_class_ids,
        )
        export_hard_class_error_profile(
            records,
            export_prefix.parent / f"{export_prefix.name}_hardclass_profile_viterbi.json",
            hard_raw_labels=hard_class_profile_raw_labels,
            instrument_class_ids=instrument_class_ids,
            pred_label_key="step_pred_viterbi_label",
        )
        export_metric_bundle(
            export_prefix=export_prefix,
            label_names=LABELS12,
            variant_metrics=variant_metrics,
            resource_summary=resource_summary,
            statistical_tests=statistical_tests,
        )

    avg_loss = loss_sum / max(1, num_batches)
    avg_late_chain_loss = late_chain_loss_sum / max(1, num_batches)
    avg_coarse_loss = coarse_loss_sum / max(1, num_batches)
    primary_metrics = online_metrics if online_eval else raw_metrics
    metrics = {
        "loss": avg_loss,
        "total_loss": avg_loss,
        "step_loss": avg_loss,
        "boundary_loss": 0.0,
        "instrument_loss": 0.0,
        "late_chain_loss": avg_late_chain_loss,
        "coarse_loss": avg_coarse_loss,
        "acc": primary_metrics["acc"],
        "video_acc_mean": primary_metrics["video_acc_mean"],
        "video_acc_std": primary_metrics["video_acc_std"],
        "macro_precision": primary_metrics["macro_precision"],
        "macro_recall": primary_metrics["macro_recall"],
        "macro_f1": primary_metrics["macro_f1"],
        "macro_f1_std": primary_metrics["macro_f1_std"],
        "macro_jaccard": primary_metrics["macro_jaccard"],
        "macro_jaccard_std": primary_metrics["macro_jaccard_std"],
        "balanced_accuracy": primary_metrics["balanced_accuracy"],
        "edit": primary_metrics["edit"],
        "edit_std": primary_metrics["edit_std"],
        "step_metric": primary_metrics["step_metric"],
        "step_metric_std": primary_metrics["step_metric_std"],
        "online_macro_f1": online_metrics["macro_f1"],
        "online_macro_f1_std": online_metrics["macro_f1_std"],
        "online_macro_jaccard": online_metrics["macro_jaccard"],
        "online_macro_jaccard_std": online_metrics["macro_jaccard_std"],
        "online_edit": online_metrics["edit"],
        "online_edit_std": online_metrics["edit_std"],
        "online_step_metric": online_metrics["step_metric"],
        "online_step_metric_std": online_metrics["step_metric_std"],
        "online_f1_at_10": online_metrics["f1_at_10"],
        "online_f1_at_10_std": online_metrics["f1_at_10_std"],
        "online_f1_at_25": online_metrics["f1_at_25"],
        "online_f1_at_25_std": online_metrics["f1_at_25_std"],
        "online_f1_at_50": online_metrics["f1_at_50"],
        "online_f1_at_50_std": online_metrics["f1_at_50_std"],
        "f1_at_10": primary_metrics["f1_at_10"],
        "f1_at_10_std": primary_metrics["f1_at_10_std"],
        "f1_at_25": primary_metrics["f1_at_25"],
        "f1_at_25_std": primary_metrics["f1_at_25_std"],
        "f1_at_50": primary_metrics["f1_at_50"],
        "f1_at_50_std": primary_metrics["f1_at_50_std"],
        "precision": primary_metrics["precision"],
        "recall": primary_metrics["recall"],
        "f1": primary_metrics["f1"],
        "jaccard": primary_metrics["jaccard"],
        "support": primary_metrics["support"],
        "confusion_matrix": primary_metrics["confusion_matrix"],
        "label_names": primary_metrics["label_names"],
        "per_video": primary_metrics["per_video"],
        "kappa_global": primary_metrics["kappa_global"],
        "mcc_global": primary_metrics["mcc_global"],
        "relaxed_acc_at_10": primary_metrics["relaxed_acc_at_10"],
        "relaxed_acc_at_10_std": primary_metrics["relaxed_acc_at_10_std"],
        "relaxed_macro_f1_at_10": primary_metrics["relaxed_macro_f1_at_10"],
        "relaxed_macro_jaccard_at_10": primary_metrics["relaxed_macro_jaccard_at_10"],
        "transition_precision_at_10": primary_metrics["transition_precision_at_10"],
        "transition_recall_at_10": primary_metrics["transition_recall_at_10"],
        "transition_f1_at_10": primary_metrics["transition_f1_at_10"],
        "transition_f1_at_10_std": primary_metrics["transition_f1_at_10_std"],
        "boundary_mae_frames_at_10": primary_metrics["boundary_mae_frames_at_10"],
        "boundary_mae_frames_at_10_std": primary_metrics["boundary_mae_frames_at_10_std"],
        "valid_transition_rate": primary_metrics["valid_transition_rate"],
        "duration_mae_frames": primary_metrics["duration_mae_frames"],
        "duration_mae_frames_std": primary_metrics["duration_mae_frames_std"],
        "duration_mape": primary_metrics["duration_mape"],
        "bootstrap_ci": primary_metrics["bootstrap_ci"],
        "raw_step_metric": raw_metrics["step_metric"],
        "raw_macro_f1": raw_metrics["macro_f1"],
        "light_post_macro_f1": light_metrics["macro_f1"],
        "light_post_macro_jaccard": light_metrics["macro_jaccard"],
        "light_post_edit": light_metrics["edit"],
        "light_post_step_metric": light_metrics["step_metric"],
        "light_post_f1_at_10": light_metrics["f1_at_10"],
        "light_post_f1_at_25": light_metrics["f1_at_25"],
        "light_post_f1_at_50": light_metrics["f1_at_50"],
        "mindur_post_macro_f1": mindur_metrics["macro_f1"],
        "mindur_post_macro_jaccard": mindur_metrics["macro_jaccard"],
        "mindur_post_edit": mindur_metrics["edit"],
        "mindur_post_step_metric": mindur_metrics["step_metric"],
        "mindur_post_f1_at_10": mindur_metrics["f1_at_10"],
        "mindur_post_f1_at_25": mindur_metrics["f1_at_25"],
        "mindur_post_f1_at_50": mindur_metrics["f1_at_50"],
        "viterbi_macro_f1": viterbi_metrics["macro_f1"],
        "viterbi_macro_f1_std": viterbi_metrics["macro_f1_std"],
        "viterbi_macro_jaccard": viterbi_metrics["macro_jaccard"],
        "viterbi_macro_jaccard_std": viterbi_metrics["macro_jaccard_std"],
        "viterbi_edit": viterbi_metrics["edit"],
        "viterbi_edit_std": viterbi_metrics["edit_std"],
        "viterbi_step_metric": viterbi_metrics["step_metric"],
        "viterbi_step_metric_std": viterbi_metrics["step_metric_std"],
        "viterbi_f1_at_10": viterbi_metrics["f1_at_10"],
        "viterbi_f1_at_10_std": viterbi_metrics["f1_at_10_std"],
        "viterbi_f1_at_25": viterbi_metrics["f1_at_25"],
        "viterbi_f1_at_25_std": viterbi_metrics["f1_at_25_std"],
        "viterbi_f1_at_50": viterbi_metrics["f1_at_50"],
        "viterbi_f1_at_50_std": viterbi_metrics["f1_at_50_std"],
        "viterbi_relaxed_acc_at_10": viterbi_metrics["relaxed_acc_at_10"],
        "viterbi_relaxed_acc_at_10_std": viterbi_metrics["relaxed_acc_at_10_std"],
        "viterbi_relaxed_macro_f1_at_10": viterbi_metrics["relaxed_macro_f1_at_10"],
        "viterbi_relaxed_macro_jaccard_at_10": viterbi_metrics["relaxed_macro_jaccard_at_10"],
        "viterbi_transition_precision_at_10": viterbi_metrics["transition_precision_at_10"],
        "viterbi_transition_recall_at_10": viterbi_metrics["transition_recall_at_10"],
        "viterbi_transition_f1_at_10": viterbi_metrics["transition_f1_at_10"],
        "viterbi_transition_f1_at_10_std": viterbi_metrics["transition_f1_at_10_std"],
        "viterbi_boundary_mae_frames_at_10": viterbi_metrics["boundary_mae_frames_at_10"],
        "viterbi_boundary_mae_frames_at_10_std": viterbi_metrics["boundary_mae_frames_at_10_std"],
        "viterbi_valid_transition_rate": viterbi_metrics["valid_transition_rate"],
        "viterbi_duration_mae_frames": viterbi_metrics["duration_mae_frames"],
        "viterbi_duration_mae_frames_std": viterbi_metrics["duration_mae_frames_std"],
        "viterbi_duration_mape": viterbi_metrics["duration_mape"],
        "resource_summary": resource_summary,
        "statistical_tests": statistical_tests,
        **instrument_metrics,
    }
    # Free large intermediate structures to reclaim memory
    del all_records, records, raw_metrics, online_metrics, light_metrics, mindur_metrics, viterbi_metrics, instrument_metrics
    gc.collect()
    return metrics


__all__ = [
    "accumulate_frame_predictions",
    "apply_light_postprocess",
    "apply_online_transition_postprocess",
    "apply_viterbi_postprocess",
    "compute_edit_score",
    "compute_instrument_metrics",
    "compute_step_metrics",
    "export_all_frames_csv",
    "export_eval_frames_csv",
    "finalize_frame_predictions",
    "print_validation_details",
    "run_mstcn_validation_epoch",
    "run_validation_epoch",
]
