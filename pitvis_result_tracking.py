import csv
from pathlib import Path
from typing import Optional


BEST_METRIC_FIELDNAMES = [
    "output_dir",
    "run_name",
    "run_group",
    "architecture",
    "config",
    "checkpoint_path",
    "result_source",
    "best_epoch",
    "selection_metric_name",
    "selection_metric_value",
    "val_online_step_metric",
    "val_online_step_metric_std",
    "val_online_macro_f1",
    "val_online_macro_f1_std",
    "val_online_macro_jaccard",
    "val_online_macro_jaccard_std",
    "val_online_edit",
    "val_online_edit_std",
    "val_online_f1_at_10",
    "val_online_f1_at_10_std",
    "val_online_f1_at_25",
    "val_online_f1_at_25_std",
    "val_online_f1_at_50",
    "val_online_f1_at_50_std",
    "val_viterbi_step_metric",
    "val_viterbi_step_metric_std",
    "val_viterbi_macro_f1",
    "val_viterbi_macro_f1_std",
    "val_viterbi_macro_jaccard",
    "val_viterbi_macro_jaccard_std",
    "val_viterbi_edit",
    "val_viterbi_edit_std",
    "val_viterbi_f1_at_10",
    "val_viterbi_f1_at_10_std",
    "val_viterbi_f1_at_25",
    "val_viterbi_f1_at_25_std",
    "val_viterbi_f1_at_50",
    "val_viterbi_f1_at_50_std",
    "val_viterbi_relaxed_acc_at_10",
    "val_viterbi_transition_f1_at_10",
    "val_viterbi_boundary_mae_frames_at_10",
    "val_viterbi_valid_transition_rate",
    "val_viterbi_duration_mae_frames",
    "val_viterbi_duration_mape",
    "val_acc",
    "val_video_acc_std",
    "val_macro_f1",
    "val_macro_f1_std",
    "val_macro_jaccard",
    "val_macro_jaccard_std",
    "val_edit",
    "val_edit_std",
    "val_step_metric",
    "val_step_metric_std",
    "val_instrument_micro_f1",
    "val_instrument_macro_f1",
]


def build_best_metric_row(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    epoch: int,
    metrics: dict,
    args,
    result_source: str = "best_epoch",
    selection_metric_name: str = "val_viterbi_step_metric",
    selection_metric_value: Optional[float] = None,
) -> dict:
    resolved_output_dir = output_dir.resolve()
    run_name = resolved_output_dir.parent.name if resolved_output_dir.parent != resolved_output_dir else resolved_output_dir.name
    run_group = (
        resolved_output_dir.parent.parent.name
        if resolved_output_dir.parent != resolved_output_dir and resolved_output_dir.parent.parent != resolved_output_dir.parent
        else ""
    )
    config_path = getattr(args, "config", None)
    if selection_metric_value is None:
        selection_metric_value = float(metrics.get("viterbi_step_metric", 0.0))
    return {
        "output_dir": str(resolved_output_dir),
        "run_name": run_name,
        "run_group": run_group,
        "architecture": str(getattr(args, "architecture", "unknown")),
        "config": str(config_path.resolve()) if config_path else "",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "result_source": str(result_source),
        "best_epoch": int(epoch),
        "selection_metric_name": str(selection_metric_name),
        "selection_metric_value": float(selection_metric_value),
        "val_online_step_metric": float(metrics.get("online_step_metric", metrics.get("step_metric", 0.0))),
        "val_online_step_metric_std": float(metrics.get("online_step_metric_std", metrics.get("step_metric_std", 0.0))),
        "val_online_macro_f1": float(metrics.get("online_macro_f1", metrics.get("macro_f1", 0.0))),
        "val_online_macro_f1_std": float(metrics.get("online_macro_f1_std", metrics.get("macro_f1_std", 0.0))),
        "val_online_macro_jaccard": float(metrics.get("online_macro_jaccard", metrics.get("macro_jaccard", 0.0))),
        "val_online_macro_jaccard_std": float(metrics.get("online_macro_jaccard_std", metrics.get("macro_jaccard_std", 0.0))),
        "val_online_edit": float(metrics.get("online_edit", metrics.get("edit", 0.0))),
        "val_online_edit_std": float(metrics.get("online_edit_std", metrics.get("edit_std", 0.0))),
        "val_online_f1_at_10": float(metrics.get("online_f1_at_10", metrics.get("f1_at_10", 0.0))),
        "val_online_f1_at_10_std": float(metrics.get("online_f1_at_10_std", metrics.get("f1_at_10_std", 0.0))),
        "val_online_f1_at_25": float(metrics.get("online_f1_at_25", metrics.get("f1_at_25", 0.0))),
        "val_online_f1_at_25_std": float(metrics.get("online_f1_at_25_std", metrics.get("f1_at_25_std", 0.0))),
        "val_online_f1_at_50": float(metrics.get("online_f1_at_50", metrics.get("f1_at_50", 0.0))),
        "val_online_f1_at_50_std": float(metrics.get("online_f1_at_50_std", metrics.get("f1_at_50_std", 0.0))),
        "val_viterbi_step_metric": float(metrics.get("viterbi_step_metric", 0.0)),
        "val_viterbi_step_metric_std": float(metrics.get("viterbi_step_metric_std", 0.0)),
        "val_viterbi_macro_f1": float(metrics.get("viterbi_macro_f1", 0.0)),
        "val_viterbi_macro_f1_std": float(metrics.get("viterbi_macro_f1_std", 0.0)),
        "val_viterbi_macro_jaccard": float(metrics.get("viterbi_macro_jaccard", 0.0)),
        "val_viterbi_macro_jaccard_std": float(metrics.get("viterbi_macro_jaccard_std", 0.0)),
        "val_viterbi_edit": float(metrics.get("viterbi_edit", 0.0)),
        "val_viterbi_edit_std": float(metrics.get("viterbi_edit_std", 0.0)),
        "val_viterbi_f1_at_10": float(metrics.get("viterbi_f1_at_10", 0.0)),
        "val_viterbi_f1_at_10_std": float(metrics.get("viterbi_f1_at_10_std", 0.0)),
        "val_viterbi_f1_at_25": float(metrics.get("viterbi_f1_at_25", 0.0)),
        "val_viterbi_f1_at_25_std": float(metrics.get("viterbi_f1_at_25_std", 0.0)),
        "val_viterbi_f1_at_50": float(metrics.get("viterbi_f1_at_50", 0.0)),
        "val_viterbi_f1_at_50_std": float(metrics.get("viterbi_f1_at_50_std", 0.0)),
        "val_viterbi_relaxed_acc_at_10": float(metrics.get("viterbi_relaxed_acc_at_10", 0.0)),
        "val_viterbi_transition_f1_at_10": float(metrics.get("viterbi_transition_f1_at_10", 0.0)),
        "val_viterbi_boundary_mae_frames_at_10": float(metrics.get("viterbi_boundary_mae_frames_at_10", 0.0)),
        "val_viterbi_valid_transition_rate": float(metrics.get("viterbi_valid_transition_rate", 0.0)),
        "val_viterbi_duration_mae_frames": float(metrics.get("viterbi_duration_mae_frames", 0.0)),
        "val_viterbi_duration_mape": float(metrics.get("viterbi_duration_mape", 0.0)),
        "val_acc": float(metrics.get("acc", 0.0)),
        "val_video_acc_std": float(metrics.get("video_acc_std", 0.0)),
        "val_macro_f1": float(metrics.get("macro_f1", 0.0)),
        "val_macro_f1_std": float(metrics.get("macro_f1_std", 0.0)),
        "val_macro_jaccard": float(metrics.get("macro_jaccard", 0.0)),
        "val_macro_jaccard_std": float(metrics.get("macro_jaccard_std", 0.0)),
        "val_edit": float(metrics.get("edit", 0.0)),
        "val_edit_std": float(metrics.get("edit_std", 0.0)),
        "val_step_metric": float(metrics.get("step_metric", 0.0)),
        "val_step_metric_std": float(metrics.get("step_metric_std", 0.0)),
        "val_instrument_micro_f1": float(metrics.get("instrument_micro_f1", 0.0)),
        "val_instrument_macro_f1": float(metrics.get("instrument_macro_f1", 0.0)),
    }


def write_best_metric_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEST_METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in BEST_METRIC_FIELDNAMES})


def upsert_best_metric_summary_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    key = str(row["output_dir"])
    updated = False
    normalized_row = {field: row.get(field, "") for field in BEST_METRIC_FIELDNAMES}
    for idx, existing in enumerate(rows):
        if existing.get("output_dir") == key:
            rows[idx] = normalized_row
            updated = True
            break
    if not updated:
        rows.append(normalized_row)
    rows = sorted(rows, key=lambda item: (item.get("run_group", ""), item.get("run_name", ""), item.get("output_dir", "")))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEST_METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "BEST_METRIC_FIELDNAMES",
    "build_best_metric_row",
    "write_best_metric_csv",
    "upsert_best_metric_summary_csv",
]
