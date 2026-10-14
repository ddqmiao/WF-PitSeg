import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _normalize_instrument_signature(value, instrument_class_ids: Optional[Sequence[int]] = None) -> str:
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return json.dumps(sorted(int(v) for v in value), ensure_ascii=False)
    if hasattr(value, "tolist"):
        flattened = value.tolist()
        if isinstance(flattened, list):
            active_ids = []
            for idx, item in enumerate(flattened):
                try:
                    if float(item) > 0.5:
                        if instrument_class_ids is not None and idx < len(instrument_class_ids):
                            active_ids.append(int(instrument_class_ids[idx]))
                        else:
                            active_ids.append(idx)
                except (TypeError, ValueError):
                    continue
            return json.dumps(active_ids, ensure_ascii=False)
    return "[]"


def _segment_length_bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length <= 5:
        return "2-5"
    if length <= 15:
        return "6-15"
    return "16+"


def _iter_video_segments(video_records: Sequence[Mapping[str, object]]) -> Iterable[Tuple[int, int, int, int]]:
    if not video_records:
        return
    start = 0
    current_label = int(video_records[0]["step_target_raw"])
    for idx in range(1, len(video_records)):
        label = int(video_records[idx]["step_target_raw"])
        if label != current_label:
            yield start, idx, current_label, idx - start
            start = idx
            current_label = label
    yield start, len(video_records), current_label, len(video_records) - start


def build_hard_class_error_profile(
    records: Sequence[Mapping[str, object]],
    *,
    hard_raw_labels: Sequence[int],
    instrument_class_ids: Optional[Sequence[int]] = None,
    pred_label_key: str = "step_pred_raw_label",
) -> Dict[str, object]:
    by_video: MutableMapping[int, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        by_video[int(record["video_id"])].append(record)

    per_class: Dict[str, Dict[str, object]] = {
        str(int(raw_label)): {
            "support_frames": 0,
            "correct_frames": 0,
            "confusion_counts": Counter(),
            "per_video": {},
            "segment_length_bins": Counter(),
            "transition_context_counts": Counter(),
            "misclassified_instrument_signatures": Counter(),
        }
        for raw_label in hard_raw_labels
    }

    for video_id, raw_video_records in by_video.items():
        video_records = sorted(raw_video_records, key=lambda item: int(item["int_time"]))

        for record in video_records:
            target_raw = int(record["step_target_raw"])
            if target_raw not in hard_raw_labels:
                continue
            pred_value = record.get(pred_label_key)
            if pred_value is None and pred_label_key != "step_pred_raw_label":
                pred_value = record.get("step_pred_raw_label")
            if pred_value is None:
                pred_value = record.get("step_pred_raw", -100)
            pred_raw = int(pred_value)
            cls_entry = per_class[str(target_raw)]
            cls_entry["support_frames"] += 1
            cls_entry["confusion_counts"][str(pred_raw)] += 1
            if pred_raw == target_raw:
                cls_entry["correct_frames"] += 1
            else:
                signature = _normalize_instrument_signature(
                    record.get("instrument_target"),
                    instrument_class_ids=instrument_class_ids,
                )
                cls_entry["misclassified_instrument_signatures"][signature] += 1

            video_key = f"video_{int(video_id):02d}"
            video_entry = cls_entry["per_video"].setdefault(
                video_key,
                {"support_frames": 0, "correct_frames": 0},
            )
            video_entry["support_frames"] += 1
            if pred_raw == target_raw:
                video_entry["correct_frames"] += 1

        segments = list(_iter_video_segments(video_records))
        for seg_idx, (seg_start, seg_end, seg_label, seg_len) in enumerate(segments):
            if seg_label not in hard_raw_labels:
                continue
            cls_entry = per_class[str(seg_label)]
            cls_entry["segment_length_bins"][_segment_length_bucket(seg_len)] += 1

            prev_label = segments[seg_idx - 1][2] if seg_idx > 0 else -1
            next_label = segments[seg_idx + 1][2] if seg_idx < len(segments) - 1 else -1
            context_key = f"{prev_label}->{seg_label}->{next_label}"
            cls_entry["transition_context_counts"][context_key] += 1

    serializable: Dict[str, object] = {"hard_raw_labels": [int(v) for v in hard_raw_labels], "per_class": {}}
    for raw_label, cls_entry in per_class.items():
        support = int(cls_entry["support_frames"])
        correct = int(cls_entry["correct_frames"])
        serializable["per_class"][raw_label] = {
            "support_frames": support,
            "correct_frames": correct,
            "frame_recall": (correct / support) if support > 0 else 0.0,
            "confusion_counts": dict(sorted(cls_entry["confusion_counts"].items(), key=lambda item: int(item[0]))),
            "per_video": cls_entry["per_video"],
            "segment_length_bins": dict(sorted(cls_entry["segment_length_bins"].items())),
            "transition_context_counts": dict(sorted(cls_entry["transition_context_counts"].items())),
            "misclassified_instrument_signatures": dict(
                sorted(cls_entry["misclassified_instrument_signatures"].items(), key=lambda item: item[0])
            ),
        }
    return serializable


def export_hard_class_error_profile(
    records: Sequence[Mapping[str, object]],
    path: Path,
    *,
    hard_raw_labels: Sequence[int],
    instrument_class_ids: Optional[Sequence[int]] = None,
    pred_label_key: str = "step_pred_raw_label",
) -> Dict[str, object]:
    profile = build_hard_class_error_profile(
        records,
        hard_raw_labels=hard_raw_labels,
        instrument_class_ids=instrument_class_ids,
        pred_label_key=pred_label_key,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile
