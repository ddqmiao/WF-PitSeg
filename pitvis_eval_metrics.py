import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_std(values: Sequence[float], *, ddof: int = 1) -> float:
    if len(values) <= ddof:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=ddof))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


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


def _extract_transition_events(sequence: Sequence[int]) -> List[Tuple[int, int, int]]:
    segments = _extract_segments(sequence)
    events: List[Tuple[int, int, int]] = []
    for seg_idx in range(len(segments) - 1):
        current_label, _, seg_end = segments[seg_idx]
        next_label, _, _ = segments[seg_idx + 1]
        events.append((int(current_label), int(next_label), int(seg_end)))
    return events


def compute_global_classification_statistics(
    confusion: np.ndarray,
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, object]:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    support = confusion.sum(axis=1).astype(np.int64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    jaccard = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)

    weights = support.astype(np.float64)
    weight_sum = float(weights.sum())
    weighted_precision = float(np.dot(precision, weights) / weight_sum) if weight_sum > 0 else 0.0
    weighted_recall = float(np.dot(recall, weights) / weight_sum) if weight_sum > 0 else 0.0
    weighted_f1 = float(np.dot(f1, weights) / weight_sum) if weight_sum > 0 else 0.0
    weighted_jaccard = float(np.dot(jaccard, weights) / weight_sum) if weight_sum > 0 else 0.0

    try:
        kappa = float(cohen_kappa_score(list(y_true), list(y_pred)))
    except Exception:
        kappa = 0.0
    try:
        mcc = float(matthews_corrcoef(list(y_true), list(y_pred)))
    except Exception:
        mcc = 0.0

    return {
        "precision": [float(value) for value in precision.tolist()],
        "recall": [float(value) for value in recall.tolist()],
        "f1": [float(value) for value in f1.tolist()],
        "jaccard": [float(value) for value in jaccard.tolist()],
        "support": [int(value) for value in support.tolist()],
        "macro_precision_global": float(np.mean(precision)) if len(precision) > 0 else 0.0,
        "macro_recall_global": float(np.mean(recall)) if len(recall) > 0 else 0.0,
        "macro_f1_global": float(np.mean(f1)) if len(f1) > 0 else 0.0,
        "macro_jaccard_global": float(np.mean(jaccard)) if len(jaccard) > 0 else 0.0,
        "weighted_precision_global": weighted_precision,
        "weighted_recall_global": weighted_recall,
        "weighted_f1_global": weighted_f1,
        "weighted_jaccard_global": weighted_jaccard,
        "balanced_accuracy_global": float(np.mean(recall)) if len(recall) > 0 else 0.0,
        "kappa_global": kappa,
        "mcc_global": mcc,
    }


def compute_videowise_phase_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    num_classes: int,
) -> Dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        confusion[int(target), int(pred)] += 1

    acc = _safe_div(float(np.trace(confusion)), float(confusion.sum()))
    per_class_values = {
        "precision": [],
        "recall": [],
        "f1": [],
        "jaccard": [],
    }
    per_class_valid: Dict[str, List[Optional[float]]] = {
        "precision": [],
        "recall": [],
        "f1": [],
        "jaccard": [],
    }

    for class_idx in range(num_classes):
        tp = float(confusion[class_idx, class_idx])
        fn = float(confusion[class_idx].sum() - confusion[class_idx, class_idx])
        fp = float(confusion[:, class_idx].sum() - confusion[class_idx, class_idx])
        support_true = float(confusion[class_idx].sum())
        support_pred = float(confusion[:, class_idx].sum())

        # Strategy B from "Metrics Matter in Surgical Phase Recognition":
        # exclude all metrics if the class is absent from the annotation, regardless of false positives.
        if support_true <= 0.0:
            for metric_name in per_class_valid:
                per_class_valid[metric_name].append(None)
            continue

        recall_value = _safe_div(tp, tp + fn)
        f1_value = 0.0
        jaccard_value = 0.0
        precision_value: Optional[float] = None

        if support_pred > 0.0:
            precision_value = _safe_div(tp, tp + fp)
            if (precision_value + recall_value) > 0.0:
                f1_value = float(2.0 * precision_value * recall_value / (precision_value + recall_value))
            jaccard_value = _safe_div(tp, tp + fp + fn)

        if precision_value is not None:
            per_class_values["precision"].append(float(precision_value))
        per_class_values["recall"].append(float(recall_value))
        per_class_values["f1"].append(float(f1_value))
        per_class_values["jaccard"].append(float(jaccard_value))

        per_class_valid["precision"].append(None if precision_value is None else float(precision_value))
        per_class_valid["recall"].append(float(recall_value))
        per_class_valid["f1"].append(float(f1_value))
        per_class_valid["jaccard"].append(float(jaccard_value))

    macro_precision = _safe_mean(per_class_values["precision"])
    macro_recall = _safe_mean(per_class_values["recall"])
    macro_f1 = _safe_mean(per_class_values["f1"])
    macro_jaccard = _safe_mean(per_class_values["jaccard"])

    return {
        "confusion": confusion,
        "acc": float(acc),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_jaccard": macro_jaccard,
        "balanced_accuracy": macro_recall,
        "per_class_valid": per_class_valid,
    }


def apply_transition_tolerant_correction(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    tolerance_frames: int,
) -> List[int]:
    corrected = [int(value) for value in y_pred]
    if int(tolerance_frames) <= 0:
        return corrected

    segments = _extract_segments(y_true)
    for seg_idx, (label, start, end) in enumerate(segments):
        accepted_labels = {int(label)}
        prev_label = int(segments[seg_idx - 1][0]) if seg_idx > 0 else None
        next_label = int(segments[seg_idx + 1][0]) if seg_idx + 1 < len(segments) else None
        if prev_label is not None:
            for time_idx in range(start, min(end, start + int(tolerance_frames))):
                if corrected[time_idx] == prev_label:
                    corrected[time_idx] = int(label)
        if next_label is not None:
            for time_idx in range(max(start, end - int(tolerance_frames)), end):
                if corrected[time_idx] == next_label:
                    corrected[time_idx] = int(label)
    return corrected


def compute_transition_metrics(
    y_true_raw: Sequence[int],
    y_pred_raw: Sequence[int],
    *,
    allowed_transitions_raw: Mapping[int, Sequence[int]],
    tolerance_frames: int,
    label_set: Sequence[int],
) -> Dict[str, float]:
    pred_events = _extract_transition_events(y_pred_raw)
    true_events = _extract_transition_events(y_true_raw)
    allowed_map = {int(src): {int(dst) for dst in dsts} for src, dsts in allowed_transitions_raw.items()}

    valid_transition_count = 0
    for src, dst, _ in pred_events:
        if dst in allowed_map.get(src, set()):
            valid_transition_count += 1

    matched_errors: List[int] = []
    matched_true = set()
    true_positive = 0
    for pred_idx, (pred_src, pred_dst, pred_time) in enumerate(pred_events):
        best_match = None
        best_error = None
        for true_idx, (true_src, true_dst, true_time) in enumerate(true_events):
            if true_idx in matched_true:
                continue
            if pred_src != true_src or pred_dst != true_dst:
                continue
            error = abs(int(pred_time) - int(true_time))
            if error > int(tolerance_frames):
                continue
            if best_error is None or error < best_error:
                best_match = true_idx
                best_error = error
        if best_match is not None and best_error is not None:
            matched_true.add(best_match)
            matched_errors.append(int(best_error))
            true_positive += 1

    false_positive = max(0, len(pred_events) - true_positive)
    false_negative = max(0, len(true_events) - true_positive)
    transition_precision = _safe_div(true_positive, true_positive + false_positive)
    transition_recall = _safe_div(true_positive, true_positive + false_negative)
    transition_f1 = (
        0.0
        if (transition_precision + transition_recall) <= 0.0
        else float(2.0 * transition_precision * transition_recall / (transition_precision + transition_recall))
    )

    duration_errors = []
    duration_relative_errors = []
    for raw_label in label_set:
        gt_duration = sum(1 for value in y_true_raw if int(value) == int(raw_label))
        pred_duration = sum(1 for value in y_pred_raw if int(value) == int(raw_label))
        duration_errors.append(abs(int(pred_duration) - int(gt_duration)))
        if gt_duration > 0:
            duration_relative_errors.append(abs(float(pred_duration) - float(gt_duration)) / float(gt_duration))

    return {
        "transition_count_pred": float(len(pred_events)),
        "transition_count_true": float(len(true_events)),
        "valid_transition_rate": _safe_div(valid_transition_count, max(1, len(pred_events))),
        "invalid_transition_count": float(max(0, len(pred_events) - valid_transition_count)),
        "transition_precision_at_tol": float(transition_precision),
        "transition_recall_at_tol": float(transition_recall),
        "transition_f1_at_tol": float(transition_f1),
        "boundary_mae_frames_at_tol": _safe_mean(matched_errors),
        "boundary_median_frames_at_tol": float(np.median(np.asarray(matched_errors, dtype=np.float64))) if matched_errors else 0.0,
        "boundary_p90_frames_at_tol": float(np.percentile(np.asarray(matched_errors, dtype=np.float64), 90)) if matched_errors else 0.0,
        "duration_mae_frames": _safe_mean(duration_errors),
        "duration_mape": _safe_mean(duration_relative_errors),
    }


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    num_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.95,
) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if array.size == 1:
        scalar = float(array[0])
        return {"mean": scalar, "std": 0.0, "ci_low": scalar, "ci_high": scalar}

    rng = np.random.default_rng(int(seed))
    samples = rng.choice(array, size=(int(num_resamples), array.size), replace=True).mean(axis=1)
    q_low = float((1.0 - alpha) / 2.0)
    q_high = float(1.0 - q_low)
    return {
        "mean": float(np.mean(array)),
        "std": _safe_std(array.tolist()),
        "ci_low": float(np.quantile(samples, q_low)),
        "ci_high": float(np.quantile(samples, q_high)),
    }


def compute_variant_statistical_tests(
    variant_to_per_video: Mapping[str, Sequence[Mapping[str, float]]],
    *,
    metric_names: Sequence[str],
    baseline_variant: str = "raw",
    compare_variants: Sequence[str] = ("light", "mindur", "viterbi"),
) -> Dict[str, Dict[str, object]]:
    tests: Dict[str, Dict[str, object]] = {}
    baseline_rows = list(variant_to_per_video.get(baseline_variant, []))
    baseline_by_video = {int(row["video_id"]): row for row in baseline_rows}

    for compare_variant in compare_variants:
        compare_rows = list(variant_to_per_video.get(compare_variant, []))
        compare_by_video = {int(row["video_id"]): row for row in compare_rows}
        common_ids = sorted(set(baseline_by_video) & set(compare_by_video))
        if not common_ids:
            continue
        pair_key = f"{compare_variant}_vs_{baseline_variant}"
        tests[pair_key] = {}
        for metric_name in metric_names:
            baseline_values = np.asarray([float(baseline_by_video[video_id][metric_name]) for video_id in common_ids], dtype=np.float64)
            compare_values = np.asarray([float(compare_by_video[video_id][metric_name]) for video_id in common_ids], dtype=np.float64)
            diff = compare_values - baseline_values
            if np.allclose(diff, 0.0):
                tests[pair_key][metric_name] = {
                    "n": int(len(common_ids)),
                    "delta_mean": 0.0,
                    "wilcoxon_p": None,
                }
                continue
            try:
                wilcoxon_result = wilcoxon(compare_values, baseline_values, zero_method="wilcox", alternative="two-sided")
                p_value = float(wilcoxon_result.pvalue)
            except Exception:
                p_value = None
            tests[pair_key][metric_name] = {
                "n": int(len(common_ids)),
                "delta_mean": float(np.mean(diff)),
                "wilcoxon_p": p_value,
            }
    return tests


def export_metric_bundle(
    *,
    export_prefix: Path,
    label_names: Sequence[int],
    variant_metrics: Mapping[str, Mapping[str, object]],
    resource_summary: Mapping[str, object],
    statistical_tests: Mapping[str, object],
) -> Dict[str, Path]:
    export_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = export_prefix.parent / f"{export_prefix.name}_metrics_summary.json"
    per_video_path = export_prefix.parent / f"{export_prefix.name}_metrics_per_video.csv"
    per_class_path = export_prefix.parent / f"{export_prefix.name}_metrics_per_class.csv"

    serializable_summary = {
        "label_names": [int(value) for value in label_names],
        "resource_summary": dict(resource_summary),
        "statistical_tests": dict(statistical_tests),
        "variants": {},
    }
    per_video_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []

    for variant_name, metrics in variant_metrics.items():
        summary_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in {"per_video", "confusion_matrix", "per_class_video_stats"}
        }
        if "confusion_matrix" in metrics:
            summary_metrics["confusion_matrix"] = np.asarray(metrics["confusion_matrix"], dtype=np.int64).tolist()
        serializable_summary["variants"][variant_name] = summary_metrics

        for row in metrics.get("per_video", []):
            per_video_rows.append({"variant": variant_name, **row})

        per_class_video_stats = metrics.get("per_class_video_stats", {})
        global_precision = metrics.get("precision", [])
        global_recall = metrics.get("recall", [])
        global_f1 = metrics.get("f1", [])
        global_jaccard = metrics.get("jaccard", [])
        global_support = metrics.get("support", [])
        for class_idx, raw_label in enumerate(label_names):
            per_class_rows.append(
                {
                    "variant": variant_name,
                    "raw_step": int(raw_label),
                    "global_precision": float(global_precision[class_idx]) if class_idx < len(global_precision) else 0.0,
                    "global_recall": float(global_recall[class_idx]) if class_idx < len(global_recall) else 0.0,
                    "global_f1": float(global_f1[class_idx]) if class_idx < len(global_f1) else 0.0,
                    "global_jaccard": float(global_jaccard[class_idx]) if class_idx < len(global_jaccard) else 0.0,
                    "support": int(global_support[class_idx]) if class_idx < len(global_support) else 0,
                    "videowise_precision_mean": float(per_class_video_stats.get("precision_mean", [0.0] * len(label_names))[class_idx]),
                    "videowise_precision_std": float(per_class_video_stats.get("precision_std", [0.0] * len(label_names))[class_idx]),
                    "videowise_recall_mean": float(per_class_video_stats.get("recall_mean", [0.0] * len(label_names))[class_idx]),
                    "videowise_recall_std": float(per_class_video_stats.get("recall_std", [0.0] * len(label_names))[class_idx]),
                    "videowise_f1_mean": float(per_class_video_stats.get("f1_mean", [0.0] * len(label_names))[class_idx]),
                    "videowise_f1_std": float(per_class_video_stats.get("f1_std", [0.0] * len(label_names))[class_idx]),
                    "videowise_jaccard_mean": float(per_class_video_stats.get("jaccard_mean", [0.0] * len(label_names))[class_idx]),
                    "videowise_jaccard_std": float(per_class_video_stats.get("jaccard_std", [0.0] * len(label_names))[class_idx]),
                    "valid_video_count": int(per_class_video_stats.get("valid_video_count", [0] * len(label_names))[class_idx]),
                }
            )

    summary_path.write_text(json.dumps(serializable_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if per_video_rows:
        fieldnames = sorted({key for row in per_video_rows for key in row.keys()})
        with per_video_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_video_rows)

    if per_class_rows:
        fieldnames = [
            "variant",
            "raw_step",
            "global_precision",
            "global_recall",
            "global_f1",
            "global_jaccard",
            "support",
            "videowise_precision_mean",
            "videowise_precision_std",
            "videowise_recall_mean",
            "videowise_recall_std",
            "videowise_f1_mean",
            "videowise_f1_std",
            "videowise_jaccard_mean",
            "videowise_jaccard_std",
            "valid_video_count",
        ]
        with per_class_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_class_rows)

    return {
        "summary_json": summary_path,
        "per_video_csv": per_video_path,
        "per_class_csv": per_class_path,
    }


__all__ = [
    "apply_transition_tolerant_correction",
    "bootstrap_confidence_interval",
    "compute_global_classification_statistics",
    "compute_transition_metrics",
    "compute_variant_statistical_tests",
    "compute_videowise_phase_metrics",
    "export_metric_bundle",
]
