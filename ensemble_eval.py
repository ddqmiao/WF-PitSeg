#!/usr/bin/env python3
"""Evaluate one or more compatible temporal checkpoints.

The paper recipe passes one MS-TCN checkpoint with weight 1.0. Multi-checkpoint
support is retained only for compatibility with earlier development runs.
Structured outputs include raw predictions and workflow-aware decoding.
"""
import argparse
import csv
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pitvis_mainline_common import (
    FEATURE_DIM,
    IGNORE_INDEX,
    LABELS12,
    LABEL_TO_INDEX,
    resolve_video_split,
    set_seed,
    step_index_to_raw_label,
)
from pitvis_priors import (
    DEFAULT_ENSEMBLE_PRESET_NAME,
    ENSEMBLE_WEIGHT_PRESETS,
    build_emission_prior_by_index,
    build_ensemble_weight_preset,
)
from pitvis_mainline_data import (
    collate_full_videos,
    discover_instrument_class_ids,
    discover_video_ids_from_features,
    build_video_items,
    PitVisFullVideoDataset,
)
# Internal helpers imported for direct logit-level ensembling
from pitvis_mainline_eval import (
    _build_log_transition_matrix,
    _viterbi_decode,
    apply_light_postprocess,
    apply_min_duration_postprocess,
    compute_step_metrics,
)
from pitvis_mainline_model import PitVisASFormerModel, PitVisMSTCNModel
from pitvis_position_prior import (
    apply_position_log_prior_to_log_probs,
    build_log_position_prior_numpy,
    collect_position_bin_counts_from_csv_root,
)


ENSEMBLE_RESULT_FIELDNAMES = [
    "result_mode",
    "features_root",
    "csv_root",
    "checkpoint_paths",
    "checkpoint_architectures",
    "preset",
    "initial_weights",
    "best_weights",
    "best_priors",
    "light_post_kernel_size",
    "min_duration_frames",
    "grid_step",
    "raw_step_metric",
    "raw_macro_f1",
    "raw_macro_jaccard",
    "raw_edit",
    "raw_f1_at_10",
    "raw_f1_at_25",
    "raw_f1_at_50",
    "light_step_metric",
    "mindur_step_metric",
    "viterbi_step_metric",
    "viterbi_step_metric_std",
    "viterbi_macro_f1",
    "viterbi_macro_f1_std",
    "viterbi_macro_jaccard",
    "viterbi_macro_jaccard_std",
    "viterbi_edit",
    "viterbi_edit_std",
    "viterbi_f1_at_10",
    "viterbi_f1_at_10_std",
    "viterbi_f1_at_25",
    "viterbi_f1_at_25_std",
    "viterbi_f1_at_50",
    "viterbi_f1_at_50_std",
    "viterbi_relaxed_acc_at_10",
    "viterbi_transition_f1_at_10",
    "viterbi_transition_f1_at_10_std",
    "viterbi_boundary_mae_frames_at_10",
    "viterbi_boundary_mae_frames_at_10_std",
    "viterbi_valid_transition_rate",
    "viterbi_duration_mae_frames",
    "viterbi_duration_mae_frames_std",
    "viterbi_duration_mape",
    "viterbi_acc",
    "viterbi_video_acc_std",
]


def _format_step_prob(vector: torch.Tensor) -> str:
    return json.dumps([round(float(v), 6) for v in vector.tolist()], ensure_ascii=False)


def export_ensemble_all_frames_csv(records: Sequence[Dict[str, object]], path: Path) -> None:
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
        "step_prob_raw",
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
                    "step_pred_light_label": int(
                        step_index_to_raw_label(int(record.get("step_pred_light", record["step_pred_raw"])))
                    ),
                    "step_pred_mindur": int(record.get("step_pred_mindur", record["step_pred_raw"])),
                    "step_pred_mindur_label": int(
                        step_index_to_raw_label(int(record.get("step_pred_mindur", record["step_pred_raw"])))
                    ),
                    "step_pred_viterbi": int(record.get("step_pred_viterbi", record["step_pred_raw"])),
                    "step_pred_viterbi_label": int(
                        step_index_to_raw_label(int(record.get("step_pred_viterbi", record["step_pred_raw"])))
                    ),
                    "step_prob_raw": _format_step_prob(record["step_prob_raw"]),
                }
            )


def export_ensemble_eval_frames_csv(records: Sequence[Dict[str, object]], path: Path) -> None:
    eval_records = [record for record in records if 0 <= int(record["step_target"]) < len(LABELS12)]
    export_ensemble_all_frames_csv(eval_records, path)


def _jsonify(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _weights_to_string(weights: Optional[Sequence[float]]) -> str:
    if weights is None:
        return ""
    return json.dumps([float(weight) for weight in weights], ensure_ascii=False)


def _raw_label_prior_dict(prior_by_index: Optional[Dict[int, float]]) -> Dict[int, float]:
    if not prior_by_index:
        return {}
    inverse = {index: raw_label for raw_label, index in LABEL_TO_INDEX.items()}
    return {
        int(inverse.get(class_index, class_index)): float(value)
        for class_index, value in sorted(prior_by_index.items())
    }


def _build_result_row(metadata: dict, metrics: dict) -> dict:
    return {
        "result_mode": metadata.get("result_mode", ""),
        "features_root": metadata.get("features_root", ""),
        "csv_root": metadata.get("csv_root", ""),
        "checkpoint_paths": json.dumps(metadata.get("checkpoint_paths", []), ensure_ascii=False),
        "checkpoint_architectures": json.dumps(metadata.get("checkpoint_architectures", []), ensure_ascii=False),
        "preset": metadata.get("preset", ""),
        "initial_weights": _weights_to_string(metadata.get("initial_weights")),
        "best_weights": _weights_to_string(metadata.get("best_weights")),
        "best_priors": json.dumps(metadata.get("best_priors_by_raw_label", {}), ensure_ascii=False),
        "light_post_kernel_size": metadata.get("light_post_kernel_size", ""),
        "min_duration_frames": metadata.get("min_duration_frames", ""),
        "grid_step": metadata.get("grid_step", ""),
        "raw_step_metric": metrics.get("raw_step_metric", ""),
        "raw_macro_f1": metrics.get("raw_macro_f1", ""),
        "raw_macro_jaccard": metrics.get("raw_macro_jaccard", ""),
        "raw_edit": metrics.get("raw_edit", ""),
        "raw_f1_at_10": metrics.get("raw_f1_at_10", ""),
        "raw_f1_at_25": metrics.get("raw_f1_at_25", ""),
        "raw_f1_at_50": metrics.get("raw_f1_at_50", ""),
        "light_step_metric": metrics.get("light_step_metric", ""),
        "mindur_step_metric": metrics.get("mindur_step_metric", ""),
        "viterbi_step_metric": metrics.get("viterbi_step_metric", ""),
        "viterbi_step_metric_std": metrics.get("viterbi_step_metric_std", ""),
        "viterbi_macro_f1": metrics.get("viterbi_macro_f1", ""),
        "viterbi_macro_f1_std": metrics.get("viterbi_macro_f1_std", ""),
        "viterbi_macro_jaccard": metrics.get("viterbi_macro_jaccard", ""),
        "viterbi_macro_jaccard_std": metrics.get("viterbi_macro_jaccard_std", ""),
        "viterbi_edit": metrics.get("viterbi_edit", ""),
        "viterbi_edit_std": metrics.get("viterbi_edit_std", ""),
        "viterbi_f1_at_10": metrics.get("viterbi_f1_at_10", ""),
        "viterbi_f1_at_10_std": metrics.get("viterbi_f1_at_10_std", ""),
        "viterbi_f1_at_25": metrics.get("viterbi_f1_at_25", ""),
        "viterbi_f1_at_25_std": metrics.get("viterbi_f1_at_25_std", ""),
        "viterbi_f1_at_50": metrics.get("viterbi_f1_at_50", ""),
        "viterbi_f1_at_50_std": metrics.get("viterbi_f1_at_50_std", ""),
        "viterbi_relaxed_acc_at_10": metrics.get("viterbi_relaxed_acc_at_10", ""),
        "viterbi_transition_f1_at_10": metrics.get("viterbi_transition_f1_at_10", ""),
        "viterbi_transition_f1_at_10_std": metrics.get("viterbi_transition_f1_at_10_std", ""),
        "viterbi_boundary_mae_frames_at_10": metrics.get("viterbi_boundary_mae_frames_at_10", ""),
        "viterbi_boundary_mae_frames_at_10_std": metrics.get("viterbi_boundary_mae_frames_at_10_std", ""),
        "viterbi_valid_transition_rate": metrics.get("viterbi_valid_transition_rate", ""),
        "viterbi_duration_mae_frames": metrics.get("viterbi_duration_mae_frames", ""),
        "viterbi_duration_mae_frames_std": metrics.get("viterbi_duration_mae_frames_std", ""),
        "viterbi_duration_mape": metrics.get("viterbi_duration_mape", ""),
        "viterbi_acc": metrics.get("viterbi_acc", ""),
        "viterbi_video_acc_std": metrics.get("viterbi_video_acc_std", ""),
    }


def _write_result_artifacts(
    *,
    json_path: Optional[Path],
    csv_path: Optional[Path],
    all_frames_csv_path: Optional[Path],
    eval_frames_csv_path: Optional[Path],
    metadata: dict,
    metrics: dict,
    frame_records: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    sanitized_metadata = _jsonify(metadata)
    sanitized_metrics = _jsonify(metrics)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "metadata": sanitized_metadata,
                    "metrics": sanitized_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = _build_result_row(sanitized_metadata, sanitized_metrics)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ENSEMBLE_RESULT_FIELDNAMES)
            writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in ENSEMBLE_RESULT_FIELDNAMES})
    if frame_records is not None:
        if all_frames_csv_path is not None:
            export_ensemble_all_frames_csv(frame_records, all_frames_csv_path)
        if eval_frames_csv_path is not None:
            export_ensemble_eval_frames_csv(frame_records, eval_frames_csv_path)


# ---------------------------------------------------------------------------
# Model factory from checkpoint args
# ---------------------------------------------------------------------------

def build_model_from_ckpt_args(ckpt_args: dict, *, num_instrument_classes: int,
                               model_state: Optional[dict] = None) -> torch.nn.Module:
    """Reconstruct a model from saved args dict.
    
    If model_state is provided, infer num_boundary_outputs from the checkpoint
    weights to handle older checkpoints with different boundary head configs.
    """
    architecture = ckpt_args.get("architecture", "mstcn")

    # Infer num_boundary_outputs from checkpoint weights if available
    num_boundary_outputs = ckpt_args.get("num_boundary_outputs", 1)
    if model_state is not None and "boundary_proj.3.bias" in model_state:
        num_boundary_outputs = model_state["boundary_proj.3.bias"].shape[0]

    if architecture == "asformer":
        return PitVisASFormerModel(
            backbone_wrapper=None,
            in_channels=FEATURE_DIM,
            num_step_classes=len(LABELS12),
            num_instrument_classes=num_instrument_classes,
            num_channels=ckpt_args.get("asformer_channels", 64),
            num_layers=ckpt_args.get("asformer_layers", 10),
            num_decoders=ckpt_args.get("asformer_decoders", 3),
            r=ckpt_args.get("asformer_r", 1),
            dropout=ckpt_args.get("dropout", 0.5),
            input_dropout=ckpt_args.get("input_dropout", 0.0),
            channel_masking_rate=ckpt_args.get("channel_masking_rate", 0.3),
            instrument_fusion=ckpt_args.get("instrument_fusion", False),
            num_boundary_outputs=num_boundary_outputs,
            coarse_to_fine=ckpt_args.get("coarse_to_fine", False),
            coarse_gate_scale=ckpt_args.get("coarse_gate_scale", 1.0),
        )
    else:
        return PitVisMSTCNModel(
            backbone_wrapper=None,
            in_channels=FEATURE_DIM,
            num_step_classes=len(LABELS12),
            num_instrument_classes=num_instrument_classes,
            num_channels=ckpt_args.get("mstcn_channels", 128),
            num_layers=ckpt_args.get("mstcn_layers", 10),
            num_stages=ckpt_args.get("mstcn_stages", 4),
            dropout=ckpt_args.get("dropout", 0.5),
            input_dropout=ckpt_args.get("input_dropout", 0.0),
            instrument_fusion=ckpt_args.get("instrument_fusion", False),
            num_boundary_outputs=num_boundary_outputs,
            coarse_to_fine=ckpt_args.get("coarse_to_fine", False),
            coarse_gate_scale=ckpt_args.get("coarse_gate_scale", 1.0),
        )


# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[int, torch.Tensor]:
    """Run model on val loader, return {video_id: (T, num_classes)} logits."""
    model.eval()
    video_logits: Dict[int, torch.Tensor] = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(features, mask=mask)
        else:
            outputs = model(features, mask=mask)

        step_logits = outputs["step_logits"].detach().cpu().float()
        lengths = batch["length"]
        video_ids = batch["video_id"]

        B = step_logits.shape[0]
        for b in range(B):
            vid = int(video_ids[b].item())
            L = int(lengths[b].item())
            video_logits[vid] = step_logits[b, :L]  # (T, C)

    return video_logits


def build_step_targets_by_video(dataset: PitVisFullVideoDataset) -> Dict[int, torch.Tensor]:
    """Materialize per-video step targets once to avoid repeated dataset scans."""
    targets_by_video: Dict[int, torch.Tensor] = {}
    for item_idx in range(len(dataset)):
        item = dataset[item_idx]
        video_id = int(item["video_id"].item())
        step_targets = item["step_targets"].detach().cpu().long()
        targets_by_video[video_id] = step_targets
    return targets_by_video


LOCAL_RERANKER_SUBSETS_BY_NAME: Dict[str, Tuple[int, ...]] = {
    "triplet_234": (2, 3, 4),
    "late_chain": (7, 8, 9, 10, 12, 14),
}

TARGET_CORRECTOR_SPECS_BY_RAW_LABEL: Dict[int, Dict[str, object]] = {
    3: {
        "subset_raw_labels": (2, 3, 4),
    },
    9: {
        "subset_raw_labels": (7, 8, 9, 10),
    },
}


def _build_static_fused_logits(
    per_model_logits: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    clip_length = min(int(logits.shape[0]) for logits in per_model_logits)
    fused_logits = torch.zeros(
        clip_length,
        int(per_model_logits[0].shape[1]),
        dtype=per_model_logits[0].dtype,
    )
    for model_idx, logits in enumerate(per_model_logits):
        fused_logits += float(weights[model_idx]) * logits[:clip_length]
    return fused_logits


def _stack_logits_and_targets(
    all_model_logits: List[Dict[int, torch.Tensor]],
    targets_by_video: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    shared_video_ids = sorted(targets_by_video.keys())
    for model_logits in all_model_logits:
        shared_video_ids = [video_id for video_id in shared_video_ids if video_id in model_logits]
    if not shared_video_ids:
        raise ValueError("No shared videos found between logits and targets for learned fusion.")

    per_video_logits: List[torch.Tensor] = []
    per_video_targets: List[torch.Tensor] = []
    for video_id in shared_video_ids:
        target = targets_by_video[video_id]
        lengths = [int(model_logits[video_id].shape[0]) for model_logits in all_model_logits]
        target_length = int(target.shape[0])
        clip_length = min(lengths + [target_length])
        stacked_logits = torch.stack(
            [model_logits[video_id][:clip_length].float() for model_logits in all_model_logits],
            dim=1,
        )
        per_video_logits.append(stacked_logits)
        per_video_targets.append(target[:clip_length].long())

    flat_logits = torch.cat(per_video_logits, dim=0)
    flat_targets = torch.cat(per_video_targets, dim=0)
    valid_mask = flat_targets.ne(IGNORE_INDEX)
    if not bool(valid_mask.any()):
        raise ValueError("No valid targets available for learned fusion training.")
    return flat_logits[valid_mask], flat_targets[valid_mask], shared_video_ids


def fit_target_specific_correctors(
    all_model_logits: List[Dict[int, torch.Tensor]],
    train_targets_by_video: Dict[int, torch.Tensor],
    *,
    initial_weights: List[float],
    device: torch.device,
    num_epochs: int = 120,
    lr: float = 5e-2,
    weight_decay: float = 0.0,
    log_interval: int = 25,
) -> Dict[str, object]:
    """Fit one-vs-rest correctors for raw labels 3 and 9 inside narrow confusion sets."""
    normalized_weights = [float(weight) for weight in initial_weights]
    total_weight = sum(normalized_weights)
    normalized_weights = [weight / max(1e-8, total_weight) for weight in normalized_weights]

    correctors_by_raw_label: Dict[int, Dict[str, object]] = {}
    target_corrector_summary_by_raw_label: Dict[int, Dict[str, object]] = {}
    shared_video_ids = sorted(train_targets_by_video.keys())
    for model_logits in all_model_logits:
        shared_video_ids = [video_id for video_id in shared_video_ids if video_id in model_logits]
    if not shared_video_ids:
        raise ValueError("No shared training videos found for target corrector.")

    for target_raw_label, spec in TARGET_CORRECTOR_SPECS_BY_RAW_LABEL.items():
        subset_raw_labels = tuple(int(raw_label) for raw_label in spec["subset_raw_labels"])
        subset_class_indices = [int(LABEL_TO_INDEX[raw_label]) for raw_label in subset_raw_labels]
        target_class_index = int(LABEL_TO_INDEX[target_raw_label])
        target_local_index = subset_class_indices.index(target_class_index)

        features_per_video: List[torch.Tensor] = []
        labels_per_video: List[torch.Tensor] = []
        for video_id in shared_video_ids:
            per_model_logits = [model_logits[video_id].float() for model_logits in all_model_logits]
            fused_logits = _build_static_fused_logits(per_model_logits, normalized_weights)
            target = train_targets_by_video[video_id].long()
            clip_length = min(int(target.shape[0]), int(fused_logits.shape[0]))
            target = target[:clip_length]
            valid_mask = target.ne(IGNORE_INDEX)
            subset_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            for class_index in subset_class_indices:
                subset_mask |= target.eq(class_index)
            selected_mask = valid_mask & subset_mask
            if not bool(selected_mask.any()):
                continue

            per_model_subset_logits = [logits[:clip_length, subset_class_indices] for logits in per_model_logits]
            fused_subset_logits = fused_logits[:clip_length, subset_class_indices]
            feature_tensor = torch.cat(per_model_subset_logits + [fused_subset_logits], dim=-1)
            binary_target = target[selected_mask].eq(target_class_index).float()
            features_per_video.append(feature_tensor[selected_mask])
            labels_per_video.append(binary_target)

        if not features_per_video:
            print(f"target_corrector_skip: raw_label={target_raw_label} no training frames")
            continue

        train_features = torch.cat(features_per_video, dim=0).to(device=device, dtype=torch.float32)
        train_labels = torch.cat(labels_per_video, dim=0).to(device=device, dtype=torch.float32)
        linear = torch.nn.Linear(int(train_features.shape[1]), 1, bias=True).to(device)
        optimizer = torch.optim.Adam(
            linear.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        best_loss = float("inf")
        best_acc = 0.0
        best_weight = None
        best_bias = None
        for epoch in range(1, int(num_epochs) + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = linear(train_features).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, train_labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = probs.ge(0.5).float()
                acc = float(preds.eq(train_labels).float().mean().item())
                loss_value = float(loss.item())
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_acc = acc
                    best_weight = linear.weight.detach().cpu().float().clone()
                    best_bias = linear.bias.detach().cpu().float().clone()
            if epoch == 1 or epoch % max(1, int(log_interval)) == 0 or epoch == int(num_epochs):
                print(
                    f"target_corrector_train: raw_label={target_raw_label} epoch={epoch}/{num_epochs} "
                    f"loss={loss_value:.4f} acc={acc:.4f}"
                )

        if best_weight is None or best_bias is None:
            raise RuntimeError(f"Failed to fit target corrector for raw_label={target_raw_label}.")

        correctors_by_raw_label[int(target_raw_label)] = {
            "target_raw_label": int(target_raw_label),
            "target_class_index": target_class_index,
            "target_local_index": target_local_index,
            "subset_raw_labels": subset_raw_labels,
            "subset_class_indices": tuple(subset_class_indices),
            "linear_weight": best_weight,
            "linear_bias": best_bias,
            "train_loss": float(best_loss),
            "train_acc": float(best_acc),
            "train_frames": int(train_labels.shape[0]),
        }
        target_corrector_summary_by_raw_label[int(target_raw_label)] = {
            "subset_raw_labels": subset_raw_labels,
            "train_loss": float(best_loss),
            "train_acc": float(best_acc),
            "train_frames": int(train_labels.shape[0]),
        }

    return {
        "correctors_by_raw_label": correctors_by_raw_label,
        "target_corrector_summary_by_raw_label": target_corrector_summary_by_raw_label,
    }


def apply_target_specific_correctors(
    per_model_logits: Sequence[torch.Tensor],
    fused_logits: torch.Tensor,
    target_corrector: Dict[str, object],
    *,
    subset_mass_threshold: float = 0.60,
    topk: int = 3,
    logit_scale: float = 0.75,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Apply narrow target-only positive residuals for hard classes 3 and 9."""
    updated_logits = fused_logits.clone()
    probs = torch.softmax(fused_logits.float(), dim=-1)
    topk_indices = torch.topk(
        fused_logits,
        k=min(int(topk), int(fused_logits.shape[-1])),
        dim=-1,
    ).indices
    applied_counts_by_raw_label: Dict[int, int] = {}

    for target_raw_label, corrector in target_corrector["correctors_by_raw_label"].items():
        subset_class_indices = list(corrector["subset_class_indices"])
        target_class_index = int(corrector["target_class_index"])
        subset_index_tensor = torch.tensor(subset_class_indices, dtype=torch.long, device=updated_logits.device)
        topk_in_subset = (topk_indices.unsqueeze(-1) == subset_index_tensor.view(1, -1)).any(dim=-1).sum(dim=-1)
        subset_mass = probs[:, subset_class_indices].sum(dim=-1)
        trigger_mask = (
            topk_in_subset.ge(2)
            & subset_mass.ge(float(subset_mass_threshold))
            & fused_logits.argmax(dim=-1).ne(target_class_index)
        )
        applied_counts_by_raw_label[int(target_raw_label)] = int(trigger_mask.sum().item())
        if not bool(trigger_mask.any()):
            continue

        per_model_subset_logits = [logits[: updated_logits.shape[0], subset_class_indices].to(device=updated_logits.device, dtype=updated_logits.dtype)
                                   for logits in per_model_logits]
        fused_subset_logits = updated_logits[:, subset_class_indices]
        feature_tensor = torch.cat(per_model_subset_logits + [fused_subset_logits], dim=-1)
        linear_weight = corrector["linear_weight"].to(device=updated_logits.device, dtype=updated_logits.dtype)
        linear_bias = corrector["linear_bias"].to(device=updated_logits.device, dtype=updated_logits.dtype)
        binary_logits = feature_tensor.matmul(linear_weight.transpose(0, 1)).squeeze(-1) + linear_bias.squeeze(0)
        positive_delta = float(logit_scale) * torch.clamp(2.0 * torch.sigmoid(binary_logits) - 1.0, min=0.0)
        trigger_rows = torch.nonzero(trigger_mask, as_tuple=False).squeeze(-1)
        updated_logits[trigger_rows, target_class_index] = (
            updated_logits[trigger_rows, target_class_index] + positive_delta[trigger_rows]
        )

    return updated_logits, applied_counts_by_raw_label


def fit_local_subset_rerankers(
    all_model_logits: List[Dict[int, torch.Tensor]],
    train_targets_by_video: Dict[int, torch.Tensor],
    *,
    initial_weights: List[float],
    device: torch.device,
    num_epochs: int = 150,
    lr: float = 5e-2,
    weight_decay: float = 0.0,
    log_interval: int = 25,
) -> Dict[str, object]:
    """Train small subset specialists on hard-class groups only."""
    normalized_weights = [float(weight) for weight in initial_weights]
    total_weight = sum(normalized_weights)
    normalized_weights = [weight / max(1e-8, total_weight) for weight in normalized_weights]

    subset_models: Dict[str, Dict[str, object]] = {}
    local_reranker_summary_by_subset: Dict[str, Dict[str, object]] = {}
    shared_video_ids = sorted(train_targets_by_video.keys())
    for model_logits in all_model_logits:
        shared_video_ids = [video_id for video_id in shared_video_ids if video_id in model_logits]
    if not shared_video_ids:
        raise ValueError("No shared training videos found for local reranker.")

    for subset_name, raw_labels in LOCAL_RERANKER_SUBSETS_BY_NAME.items():
        subset_class_indices = [int(LABEL_TO_INDEX[raw_label]) for raw_label in raw_labels]
        inputs_per_video: List[torch.Tensor] = []
        targets_per_video: List[torch.Tensor] = []

        for video_id in shared_video_ids:
            per_model_logits = [model_logits[video_id].float() for model_logits in all_model_logits]
            target = train_targets_by_video[video_id].long()
            clip_length = min([int(target.shape[0])] + [int(logits.shape[0]) for logits in per_model_logits])
            target = target[:clip_length]
            valid_mask = target.ne(IGNORE_INDEX)
            subset_target_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            for class_index in subset_class_indices:
                subset_target_mask |= target.eq(class_index)
            selected_mask = valid_mask & subset_target_mask
            if not bool(selected_mask.any()):
                continue
            subset_logits = torch.stack(
                [logits[:clip_length, subset_class_indices] for logits in per_model_logits],
                dim=1,
            )
            local_targets = torch.tensor(
                [subset_class_indices.index(int(class_index)) for class_index in target[selected_mask].tolist()],
                dtype=torch.long,
            )
            inputs_per_video.append(subset_logits[selected_mask])
            targets_per_video.append(local_targets)

        if not inputs_per_video:
            print(f"local_reranker_skip: subset={subset_name} no training frames")
            continue

        train_inputs = torch.cat(inputs_per_video, dim=0).to(device=device, dtype=torch.float32)
        train_targets = torch.cat(targets_per_video, dim=0).to(device=device, dtype=torch.long)
        subset_size = len(subset_class_indices)
        num_models = int(train_inputs.shape[1])

        init = torch.tensor(normalized_weights, dtype=torch.float32, device=device)
        weight_logits = torch.nn.Parameter(
            torch.log(init.clamp_min(1e-6)).view(num_models, 1).expand(num_models, subset_size).clone()
        )
        class_bias = torch.nn.Parameter(torch.zeros(subset_size, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam(
            [weight_logits, class_bias],
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        best_loss = float("inf")
        best_acc = 0.0
        best_weights = None
        best_bias = None
        for epoch in range(1, int(num_epochs) + 1):
            optimizer.zero_grad(set_to_none=True)
            classwise_weights = torch.softmax(weight_logits, dim=0)
            reranked_logits = (train_inputs * classwise_weights.unsqueeze(0)).sum(dim=1) + class_bias.unsqueeze(0)
            loss = F.cross_entropy(reranked_logits, train_targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = reranked_logits.argmax(dim=-1)
                acc = float(preds.eq(train_targets).float().mean().item())
                loss_value = float(loss.item())
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_acc = acc
                    best_weights = classwise_weights.detach().cpu().float().clone()
                    best_bias = class_bias.detach().cpu().float().clone()
            if epoch == 1 or epoch % max(1, int(log_interval)) == 0 or epoch == int(num_epochs):
                print(
                    f"local_reranker_train: subset={subset_name} epoch={epoch}/{num_epochs} "
                    f"loss={loss_value:.4f} acc={acc:.4f}"
                )

        if best_weights is None or best_bias is None:
            raise RuntimeError(f"Failed to fit local reranker for subset={subset_name}.")

        subset_models[subset_name] = {
            "subset_name": subset_name,
            "raw_labels": tuple(int(raw_label) for raw_label in raw_labels),
            "class_indices": tuple(subset_class_indices),
            "classwise_weights": best_weights,
            "classwise_bias": best_bias,
            "train_loss": float(best_loss),
            "train_acc": float(best_acc),
            "train_frames": int(train_targets.shape[0]),
        }
        local_reranker_summary_by_subset[subset_name] = {
            "raw_labels": tuple(int(raw_label) for raw_label in raw_labels),
            "train_loss": float(best_loss),
            "train_acc": float(best_acc),
            "train_frames": int(train_targets.shape[0]),
            "weights_by_raw_label": {
                int(raw_label): [float(best_weights[model_idx, class_idx].item()) for model_idx in range(num_models)]
                for class_idx, raw_label in enumerate(raw_labels)
            },
            "bias_by_raw_label": {
                int(raw_label): float(best_bias[class_idx].item())
                for class_idx, raw_label in enumerate(raw_labels)
            },
        }

    return {
        "subsets": subset_models,
        "local_reranker_summary_by_subset": local_reranker_summary_by_subset,
    }


def apply_local_subset_rerankers(
    per_model_logits: Sequence[torch.Tensor],
    fused_logits: torch.Tensor,
    local_reranker: Dict[str, object],
    *,
    subset_mass_threshold: float = 0.60,
    topk: int = 3,
    blend: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Apply subset specialists only on ambiguous frames tied to each hard subset."""
    updated_logits = fused_logits.clone()
    probs = torch.softmax(fused_logits.float(), dim=-1)
    topk_indices = torch.topk(
        fused_logits,
        k=min(int(topk), int(fused_logits.shape[-1])),
        dim=-1,
    ).indices
    applied_counts_by_subset: Dict[str, int] = {}

    for subset_name, subset_model in local_reranker["subsets"].items():
        subset_class_indices = list(subset_model["class_indices"])
        subset_index_tensor = torch.tensor(
            subset_class_indices,
            dtype=torch.long,
            device=topk_indices.device,
        )
        topk_in_subset = (topk_indices.unsqueeze(-1) == subset_index_tensor.view(1, -1)).any(dim=-1).sum(dim=-1)
        subset_mass = probs[:, subset_class_indices].sum(dim=-1)
        trigger_mask = topk_in_subset.ge(2) | subset_mass.ge(float(subset_mass_threshold))
        applied_counts_by_subset[subset_name] = int(trigger_mask.sum().item())
        if not bool(trigger_mask.any()):
            continue

        classwise_weights = subset_model["classwise_weights"].to(device=updated_logits.device, dtype=updated_logits.dtype)
        classwise_bias = subset_model["classwise_bias"].to(device=updated_logits.device, dtype=updated_logits.dtype)
        subset_logits = torch.stack(
            [logits[: updated_logits.shape[0], subset_class_indices].to(device=updated_logits.device, dtype=updated_logits.dtype)
             for logits in per_model_logits],
            dim=1,
        )
        reranked_subset_logits = (subset_logits * classwise_weights.unsqueeze(0)).sum(dim=1) + classwise_bias.unsqueeze(0)
        blended_subset_logits = (
            (1.0 - float(blend)) * updated_logits[:, subset_class_indices]
            + float(blend) * reranked_subset_logits
        )
        trigger_rows = torch.nonzero(trigger_mask, as_tuple=False).squeeze(-1)
        updated_logits[trigger_rows[:, None], subset_index_tensor.view(1, -1)] = blended_subset_logits[trigger_rows]

    return updated_logits, applied_counts_by_subset


def fit_classwise_learned_fusion(
    all_model_logits: List[Dict[int, torch.Tensor]],
    train_targets_by_video: Dict[int, torch.Tensor],
    *,
    initial_weights: List[float],
    device: torch.device,
    num_epochs: int = 200,
    lr: float = 5e-2,
    weight_decay: float = 0.0,
    log_interval: int = 25,
) -> Dict[str, object]:
    """Fit a tiny class-wise fusion head on training logits only."""
    if len(all_model_logits) < 2:
        raise ValueError("Learned fusion requires at least two models.")

    train_logits, train_targets, train_video_ids = _stack_logits_and_targets(
        all_model_logits,
        train_targets_by_video,
    )
    train_logits = train_logits.to(device=device, dtype=torch.float32)
    train_targets = train_targets.to(device=device, dtype=torch.long)

    num_models = int(train_logits.shape[1])
    num_classes = int(train_logits.shape[2])
    init = torch.tensor(initial_weights, dtype=torch.float32, device=device)
    if init.numel() != num_models:
        raise ValueError(f"Expected {num_models} initial weights, got {init.numel()}")
    init = init / init.sum().clamp_min(1e-8)

    weight_logits = torch.nn.Parameter(
        torch.log(init.clamp_min(1e-6)).view(num_models, 1).expand(num_models, num_classes).clone()
    )
    class_bias = torch.nn.Parameter(torch.zeros(num_classes, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        [weight_logits, class_bias],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    best_loss = float("inf")
    best_acc = 0.0
    best_weights = None
    best_bias = None

    for epoch in range(1, int(num_epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        classwise_weights = torch.softmax(weight_logits, dim=0)
        fused_logits = (train_logits * classwise_weights.unsqueeze(0)).sum(dim=1) + class_bias.unsqueeze(0)
        loss = F.cross_entropy(fused_logits, train_targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = fused_logits.argmax(dim=-1)
            acc = float(preds.eq(train_targets).float().mean().item())
            loss_value = float(loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_acc = acc
                best_weights = classwise_weights.detach().cpu().float().clone()
                best_bias = class_bias.detach().cpu().float().clone()
        if epoch == 1 or epoch % max(1, int(log_interval)) == 0 or epoch == int(num_epochs):
            print(
                f"learned_fusion_train: epoch={epoch}/{num_epochs} "
                f"loss={loss_value:.4f} acc={acc:.4f}"
            )

    if best_weights is None or best_bias is None:
        raise RuntimeError("Failed to fit learned fusion weights.")

    learned_fusion_weights_by_raw_label = {
        int(raw_label): [float(best_weights[model_idx, class_idx].item()) for model_idx in range(num_models)]
        for class_idx, raw_label in enumerate(LABELS12)
    }
    learned_fusion_bias_by_raw_label = {
        int(raw_label): float(best_bias[class_idx].item())
        for class_idx, raw_label in enumerate(LABELS12)
    }
    return {
        "classwise_weights": best_weights,
        "classwise_bias": best_bias,
        "train_loss": float(best_loss),
        "train_acc": float(best_acc),
        "train_frames": int(train_targets.shape[0]),
        "train_video_ids": train_video_ids,
        "learned_fusion_weights_by_raw_label": learned_fusion_weights_by_raw_label,
        "learned_fusion_bias_by_raw_label": learned_fusion_bias_by_raw_label,
    }


def apply_classwise_learned_fusion(
    per_model_logits: Sequence[torch.Tensor],
    learned_fusion: Dict[str, object],
) -> torch.Tensor:
    """Fuse per-model logits using class-wise learned weights and class bias."""
    classwise_weights = learned_fusion["classwise_weights"]
    classwise_bias = learned_fusion["classwise_bias"]
    clip_length = min(int(logits.shape[0]) for logits in per_model_logits)
    fused_logits = torch.zeros(
        clip_length,
        int(classwise_bias.shape[0]),
        dtype=per_model_logits[0].dtype,
    )
    for model_idx, logits in enumerate(per_model_logits):
        fused_logits += logits[:clip_length] * classwise_weights[model_idx].view(1, -1)
    fused_logits += classwise_bias.view(1, -1)
    return fused_logits


# ---------------------------------------------------------------------------
# Ensemble + Viterbi evaluation
# ---------------------------------------------------------------------------

_EMISSION_PRIOR: Dict[int, float] = build_emission_prior_by_index(LABEL_TO_INDEX)


def build_position_priors_from_training_csv(
    csv_root: Path,
    *,
    train_video_ids: Sequence[int],
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    position_counts = collect_position_bin_counts_from_csv_root(
        csv_root,
        video_ids=train_video_ids,
        labels12=LABELS12,
        num_bins=num_bins,
        normalize_per_video_bin=True,
    )
    log_position_prior = build_log_position_prior_numpy(position_counts)
    nonempty_bins = int((position_counts.sum(axis=1) > 0).sum())
    print(
        f"position_priors: train_videos={list(train_video_ids)} "
        f"num_bins={int(num_bins)} nonempty_bins={nonempty_bins}"
    )
    return position_counts, log_position_prior


def resolve_ensemble_weights(
    *,
    weights: Optional[List[float]],
    preset: Optional[str],
    n_models: int,
) -> List[float]:
    if weights is not None:
        resolved = [float(weight) for weight in weights]
    elif preset is not None:
        resolved = build_ensemble_weight_preset(preset)
    else:
        resolved = [1.0 / n_models] * n_models

    if len(resolved) != n_models:
        raise ValueError(f"Expected {n_models} weights, got {len(resolved)}")
    return resolved


def evaluate_ensemble(
    all_model_logits: List[Dict[int, torch.Tensor]],
    weights: Optional[List[float]],
    val_dataset: PitVisFullVideoDataset,
    *,
    light_post_kernel_size: int = 11,
    min_duration_frames: int = 30,
    emission_prior: Optional[Dict[int, float]] = None,
    learned_fusion: Optional[Dict[str, object]] = None,
    local_reranker: Optional[Dict[str, object]] = None,
    target_corrector: Optional[Dict[str, object]] = None,
    targets_by_video: Optional[Dict[int, torch.Tensor]] = None,
    local_reranker_subset_mass_threshold: float = 0.60,
    local_reranker_topk: int = 3,
    local_reranker_blend: float = 0.5,
    target_corrector_subset_mass_threshold: float = 0.60,
    target_corrector_topk: int = 3,
    target_corrector_logit_scale: float = 0.75,
    log_position_prior_matrix: Optional[np.ndarray] = None,
    position_prior_weight: float = 0.0,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Fuse logits, apply Viterbi, and compute metrics."""
    if emission_prior is None:
        emission_prior = _EMISSION_PRIOR

    num_classes = len(LABELS12)
    log_trans = _build_log_transition_matrix(num_classes)

    # Build emission log prior
    emission_log_prior = np.zeros(num_classes, dtype=np.float64)
    for cls_idx, multiplier in emission_prior.items():
        emission_log_prior[cls_idx] = math.log(max(1e-6, multiplier))

    w = None
    if learned_fusion is None:
        if weights is None:
            raise ValueError("Static ensemble evaluation requires explicit weights.")
        total_w = sum(weights)
        w = [wt / total_w for wt in weights]

    # Get all video IDs from first model's logits
    video_ids = sorted(all_model_logits[0].keys())

    all_records: List[Dict[str, object]] = []
    local_reranker_applied_by_subset: Dict[str, int] = {}
    target_corrector_applied_by_raw_label: Dict[int, int] = {}
    for vid in video_ids:
        per_model_logits = [model_logits[vid].float() for model_logits in all_model_logits]
        if learned_fusion is not None:
            fused_logits = apply_classwise_learned_fusion(per_model_logits, learned_fusion)
        else:
            fused_logits = _build_static_fused_logits(per_model_logits, w)
        if local_reranker is not None:
            fused_logits, applied_counts = apply_local_subset_rerankers(
                per_model_logits,
                fused_logits,
                local_reranker,
                subset_mass_threshold=local_reranker_subset_mass_threshold,
                topk=local_reranker_topk,
                blend=local_reranker_blend,
            )
            for subset_name, count in applied_counts.items():
                local_reranker_applied_by_subset[subset_name] = local_reranker_applied_by_subset.get(subset_name, 0) + int(count)
        if target_corrector is not None:
            fused_logits, applied_counts = apply_target_specific_correctors(
                per_model_logits,
                fused_logits,
                target_corrector,
                subset_mass_threshold=target_corrector_subset_mass_threshold,
                topk=target_corrector_topk,
                logit_scale=target_corrector_logit_scale,
            )
            for raw_label, count in applied_counts.items():
                target_corrector_applied_by_raw_label[int(raw_label)] = target_corrector_applied_by_raw_label.get(int(raw_label), 0) + int(count)

        gt_targets = None
        if targets_by_video is not None:
            gt_targets = targets_by_video.get(vid)
            if gt_targets is not None:
                gt_targets = gt_targets[: fused_logits.shape[0]]
        else:
            for item_idx in range(len(val_dataset)):
                item = val_dataset[item_idx]
                if int(item["video_id"].item()) == vid:
                    gt_targets = item["step_targets"][: fused_logits.shape[0]]
                    break

        if gt_targets is None:
            print(f"WARNING: video {vid} not found in val_dataset, skipping")
            continue

        # Softmax -> Viterbi
        probs = torch.softmax(fused_logits, dim=-1).numpy().astype(np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        log_probs = np.log(probs) + emission_log_prior[np.newaxis, :]
        log_probs = apply_position_log_prior_to_log_probs(
            log_probs,
            log_position_prior_matrix=log_position_prior_matrix,
            position_prior_weight=position_prior_weight,
        )
        viterbi_preds = _viterbi_decode(log_probs, log_trans)

        # Build per-frame records
        for t in range(int(fused_logits.shape[0])):
            target = int(gt_targets[t].item())
            record = {
                "video_id": vid,
                "int_time": t,
                "step_target": target,
                "step_target_raw": step_index_to_raw_label(target),
                "step_prob_raw": torch.softmax(fused_logits[t], dim=-1),
                "step_pred_raw": int(fused_logits[t].argmax().item()),
                "step_pred_viterbi": int(viterbi_preds[t]),
            }
            all_records.append(record)

    # Light post + min duration on raw predictions
    records = apply_light_postprocess(all_records, kernel_size=light_post_kernel_size)
    records = apply_min_duration_postprocess(records, min_duration=min_duration_frames)

    # Re-apply viterbi from ensemble probs (already done above, copy over)
    for r in records:
        pass  # step_pred_viterbi already set

    raw_metrics = compute_step_metrics(records, num_classes=num_classes, pred_key="step_pred_raw")
    light_metrics = compute_step_metrics(records, num_classes=num_classes, pred_key="step_pred_light")
    mindur_metrics = compute_step_metrics(records, num_classes=num_classes, pred_key="step_pred_mindur")
    viterbi_metrics = compute_step_metrics(records, num_classes=num_classes, pred_key="step_pred_viterbi")

    metrics = {
        "raw_metrics": raw_metrics,
        "light_metrics": light_metrics,
        "mindur_metrics": mindur_metrics,
        "viterbi_metrics": viterbi_metrics,
        "raw_step_metric": raw_metrics["step_metric"],
        "raw_macro_f1": raw_metrics["macro_f1"],
        "raw_macro_jaccard": raw_metrics["macro_jaccard"],
        "raw_edit": raw_metrics["edit"],
        "raw_f1_at_10": raw_metrics["f1_at_10"],
        "raw_f1_at_25": raw_metrics["f1_at_25"],
        "raw_f1_at_50": raw_metrics["f1_at_50"],
        "light_step_metric": light_metrics["step_metric"],
        "mindur_step_metric": mindur_metrics["step_metric"],
        "viterbi_step_metric": viterbi_metrics["step_metric"],
        "viterbi_step_metric_std": viterbi_metrics["step_metric_std"],
        "viterbi_macro_f1": viterbi_metrics["macro_f1"],
        "viterbi_macro_f1_std": viterbi_metrics["macro_f1_std"],
        "viterbi_macro_jaccard": viterbi_metrics["macro_jaccard"],
        "viterbi_macro_jaccard_std": viterbi_metrics["macro_jaccard_std"],
        "viterbi_edit": viterbi_metrics["edit"],
        "viterbi_edit_std": viterbi_metrics["edit_std"],
        "viterbi_f1_at_10": viterbi_metrics["f1_at_10"],
        "viterbi_f1_at_10_std": viterbi_metrics["f1_at_10_std"],
        "viterbi_f1_at_25": viterbi_metrics["f1_at_25"],
        "viterbi_f1_at_25_std": viterbi_metrics["f1_at_25_std"],
        "viterbi_f1_at_50": viterbi_metrics["f1_at_50"],
        "viterbi_f1_at_50_std": viterbi_metrics["f1_at_50_std"],
        "viterbi_relaxed_acc_at_10": viterbi_metrics["relaxed_acc_at_10"],
        "viterbi_transition_f1_at_10": viterbi_metrics["transition_f1_at_10"],
        "viterbi_transition_f1_at_10_std": viterbi_metrics["transition_f1_at_10_std"],
        "viterbi_boundary_mae_frames_at_10": viterbi_metrics["boundary_mae_frames_at_10"],
        "viterbi_boundary_mae_frames_at_10_std": viterbi_metrics["boundary_mae_frames_at_10_std"],
        "viterbi_valid_transition_rate": viterbi_metrics["valid_transition_rate"],
        "viterbi_duration_mae_frames": viterbi_metrics["duration_mae_frames"],
        "viterbi_duration_mae_frames_std": viterbi_metrics["duration_mae_frames_std"],
        "viterbi_duration_mape": viterbi_metrics["duration_mape"],
        "viterbi_acc": viterbi_metrics["acc"],
        "viterbi_video_acc_std": viterbi_metrics["video_acc_std"],
        "viterbi_per_class_f1": viterbi_metrics["f1"],
        "viterbi_per_class_recall": viterbi_metrics["recall"],
        "viterbi_per_class_precision": viterbi_metrics["precision"],
        "viterbi_support": viterbi_metrics["support"],
        "viterbi_per_video": viterbi_metrics["per_video"],
        "viterbi_confusion": viterbi_metrics["confusion_matrix"],
    }
    if learned_fusion is not None:
        metrics["learned_fusion_weights_by_raw_label"] = learned_fusion["learned_fusion_weights_by_raw_label"]
        metrics["learned_fusion_bias_by_raw_label"] = learned_fusion["learned_fusion_bias_by_raw_label"]
        metrics["learned_fusion_train_loss"] = learned_fusion["train_loss"]
        metrics["learned_fusion_train_acc"] = learned_fusion["train_acc"]
        metrics["learned_fusion_train_frames"] = learned_fusion["train_frames"]
    if local_reranker is not None:
        metrics["local_reranker_summary_by_subset"] = local_reranker["local_reranker_summary_by_subset"]
        metrics["local_reranker_applied_frames_by_subset"] = local_reranker_applied_by_subset
    if target_corrector is not None:
        metrics["target_corrector_summary_by_raw_label"] = target_corrector["target_corrector_summary_by_raw_label"]
        metrics["target_corrector_applied_frames_by_raw_label"] = target_corrector_applied_by_raw_label
    return metrics, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ensemble evaluation")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True,
                        help="Paths to model checkpoint .pt files")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Ensemble weights for each checkpoint (default: equal)")
    parser.add_argument(
        "--preset",
        choices=sorted(ENSEMBLE_WEIGHT_PRESETS.keys()),
        default=None,
        help=f"Use a named ensemble recipe (current best: {DEFAULT_ENSEMBLE_PRESET_NAME})",
    )
    parser.add_argument("--grid-search", action="store_true",
                        help="Grid search over ensemble weights (2 models only)")
    parser.add_argument("--grid-step", type=float, default=0.1,
                        help="Step size for weight grid search")
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=None,
                        help="Explicit train/validation split JSON; otherwise use the official split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--light-post-kernel-size", type=int, default=11)
    parser.add_argument("--min-duration-frames", type=int, default=30)
    parser.add_argument("--position-prior-bins", type=int, default=8)
    parser.add_argument("--position-prior-weight", type=float, default=0.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-priors", action="store_true",
                        help="Grid search emission priors with fixed ensemble weights")
    parser.add_argument("--learned-fusion", action="store_true",
                        help="Fit a class-wise fusion head on train logits before evaluating val")
    parser.add_argument("--learned-fusion-epochs", type=int, default=200)
    parser.add_argument("--learned-fusion-lr", type=float, default=5e-2)
    parser.add_argument("--learned-fusion-weight-decay", type=float, default=0.0)
    parser.add_argument("--local-reranker", action="store_true",
                        help="Fit subset specialists for hard-class groups on top of static ensemble logits")
    parser.add_argument("--local-reranker-epochs", type=int, default=150)
    parser.add_argument("--local-reranker-lr", type=float, default=5e-2)
    parser.add_argument("--local-reranker-weight-decay", type=float, default=0.0)
    parser.add_argument("--local-reranker-subset-mass-threshold", type=float, default=0.60)
    parser.add_argument("--local-reranker-topk", type=int, default=3)
    parser.add_argument("--local-reranker-blend", type=float, default=0.5)
    parser.add_argument("--target-corrector", action="store_true",
                        help="Fit target-specific correctors for hard classes 3 and 9")
    parser.add_argument("--target-corrector-epochs", type=int, default=120)
    parser.add_argument("--target-corrector-lr", type=float, default=5e-2)
    parser.add_argument("--target-corrector-weight-decay", type=float, default=0.0)
    parser.add_argument("--target-corrector-subset-mass-threshold", type=float, default=0.60)
    parser.add_argument("--target-corrector-topk", type=int, default=3)
    parser.add_argument("--target-corrector-logit-scale", type=float, default=0.75)
    parser.add_argument("--result-json", type=Path, default=None,
                        help="Optional path to save structured ensemble results as JSON.")
    parser.add_argument("--result-csv", type=Path, default=None,
                        help="Optional path to save key ensemble metrics as CSV.")
    parser.add_argument("--all-frames-csv", type=Path, default=None,
                        help="Optional path to save all per-frame ensemble predictions as CSV.")
    parser.add_argument("--eval-frames-csv", type=Path, default=None,
                        help="Optional path to save eval-only per-frame ensemble predictions as CSV.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed, deterministic=bool(args.deterministic))
    if args.learned_fusion and args.grid_search:
        raise ValueError("--learned-fusion and --grid-search are mutually exclusive.")
    if args.local_reranker and args.grid_search:
        raise ValueError("--local-reranker and --grid-search are mutually exclusive.")
    if args.local_reranker and args.learned_fusion:
        raise ValueError("--local-reranker and --learned-fusion are mutually exclusive.")
    if args.target_corrector and args.grid_search:
        raise ValueError("--target-corrector and --grid-search are mutually exclusive.")
    if args.target_corrector and args.learned_fusion:
        raise ValueError("--target-corrector and --learned-fusion are mutually exclusive.")
    if args.target_corrector and args.local_reranker:
        raise ValueError("--target-corrector and --local-reranker are mutually exclusive.")

    device = torch.device(args.device)
    if device.type == "cuda":
        if bool(args.deterministic):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
    if args.split_json is not None and not Path(args.split_json).is_file():
        raise FileNotFoundError(f"split json not found: {args.split_json}")
    if int(args.position_prior_bins) <= 0:
        raise ValueError("--position-prior-bins must be positive.")
    if float(args.position_prior_weight) < 0.0:
        raise ValueError("--position-prior-weight must be non-negative.")

    # Build datasets
    video_ids = discover_video_ids_from_features(args.features_root, args.csv_root)
    split = resolve_video_split(video_ids, split_json=args.split_json)
    instrument_class_ids = discover_instrument_class_ids(args.csv_root)
    train_dataset = None
    train_loader = None
    train_targets_by_video = None
    if args.learned_fusion or args.local_reranker or args.target_corrector:
        train_items = build_video_items(args.features_root, args.csv_root, split["train_videos"], input_mode="features")
        train_dataset = PitVisFullVideoDataset(
            train_items, input_mode="features", instrument_class_ids=instrument_class_ids,
            boundary_radius=3, base_boundary_weight=2.0, focus_boundary_weight=3.0,
            boundary_kernel="gaussian",
        )
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_full_videos, num_workers=0)
        train_targets_by_video = build_step_targets_by_video(train_dataset)
        print(f"train_videos={len(train_dataset)}")
    val_items = build_video_items(args.features_root, args.csv_root, split["val_videos"], input_mode="features")
    val_dataset = PitVisFullVideoDataset(
        val_items, input_mode="features", instrument_class_ids=instrument_class_ids,
        boundary_radius=3, base_boundary_weight=2.0, focus_boundary_weight=3.0,
        boundary_kernel="gaussian",
    )
    val_targets_by_video = build_step_targets_by_video(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_full_videos, num_workers=0)
    print(f"val_videos={len(val_dataset)} instrument_classes={len(instrument_class_ids)}")
    position_log_prior_np = None
    if float(args.position_prior_weight) > 0.0:
        _, position_log_prior_np = build_position_priors_from_training_csv(
            args.csv_root,
            train_video_ids=split["train_videos"],
            num_bins=int(args.position_prior_bins),
        )

    # Load models and extract logits
    all_model_logits: List[Dict[int, torch.Tensor]] = []
    all_model_train_logits: List[Dict[int, torch.Tensor]] = []
    checkpoint_summaries: List[dict] = []
    for ckpt_path in args.checkpoints:
        print(f"loading {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = ckpt["args"]
        architecture = ckpt_args.get("architecture", "mstcn")
        epoch = ckpt.get("epoch", "?")
        saved_metrics = ckpt.get("metrics", {})
        print(f"  architecture={architecture} epoch={epoch} "
              f"saved_metrics={json.dumps(saved_metrics, indent=None)}")

        model = build_model_from_ckpt_args(ckpt_args, num_instrument_classes=len(instrument_class_ids),
                                           model_state=ckpt["model_state"])
        # Filter out weights with shape mismatch (e.g. old boundary head layout)
        state = ckpt["model_state"]
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state if k not in filtered]
        if skipped:
            print(f"  [warn] skipped keys (shape mismatch): {skipped}")
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  [warn] missing keys (using random init): {missing}")
        model = model.to(device)

        if (args.learned_fusion or args.local_reranker or args.target_corrector) and train_loader is not None:
            train_logits = extract_logits(model, train_loader, device, use_amp=args.amp)
            all_model_train_logits.append(train_logits)
            print(f"  extracted train logits for {len(train_logits)} videos")
        logits = extract_logits(model, val_loader, device, use_amp=args.amp)
        all_model_logits.append(logits)
        print(f"  extracted val logits for {len(logits)} videos")
        checkpoint_summaries.append(
            {
                "checkpoint_path": str(ckpt_path.resolve()),
                "architecture": architecture,
                "epoch": epoch,
                "config": str(ckpt_args.get("config", "")),
                "saved_metrics": saved_metrics,
            }
        )

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_models = len(all_model_logits)
    initial_weights = resolve_ensemble_weights(weights=args.weights, preset=args.preset, n_models=n_models)
    emission_prior = (
        build_emission_prior_by_index(LABEL_TO_INDEX, preset=args.preset)
        if args.preset is not None
        else dict(_EMISSION_PRIOR)
    )
    if args.preset is not None:
        preset_weights = build_ensemble_weight_preset(args.preset)
        print(f"preset={args.preset} preset_weights={preset_weights} preset_priors_by_index={emission_prior}")

    base_metadata = {
        "features_root": str(args.features_root.resolve()),
        "csv_root": str(args.csv_root.resolve()),
        "split_json": str(Path(args.split_json).resolve()) if args.split_json is not None else "",
        "train_videos": [int(video_id) for video_id in split["train_videos"]],
        "val_videos": [int(video_id) for video_id in split["val_videos"]],
        "device": str(device),
        "gpu_ids": args.gpu_ids,
        "seed": int(args.seed),
        "preset": args.preset or "",
        "grid_step": float(args.grid_step),
        "light_post_kernel_size": int(args.light_post_kernel_size),
        "min_duration_frames": int(args.min_duration_frames),
        "position_prior_bins": int(args.position_prior_bins),
        "position_prior_weight": float(args.position_prior_weight),
        "deterministic": bool(args.deterministic),
        "checkpoint_paths": [summary["checkpoint_path"] for summary in checkpoint_summaries],
        "checkpoint_architectures": [summary["architecture"] for summary in checkpoint_summaries],
        "checkpoints": checkpoint_summaries,
        "initial_weights": [float(weight) for weight in initial_weights],
        "initial_emission_prior_by_raw_label": _raw_label_prior_dict(emission_prior),
    }
    learned_fusion = None
    if args.learned_fusion:
        learned_fusion = fit_classwise_learned_fusion(
            all_model_train_logits,
            train_targets_by_video,
            initial_weights=initial_weights,
            device=device,
            num_epochs=args.learned_fusion_epochs,
            lr=args.learned_fusion_lr,
            weight_decay=args.learned_fusion_weight_decay,
        )
        print(
            "learned_fusion_ready: "
            f"train_loss={learned_fusion['train_loss']:.4f} "
            f"train_acc={learned_fusion['train_acc']:.4f} "
            f"train_frames={learned_fusion['train_frames']}"
        )
    local_reranker = None
    if args.local_reranker:
        local_reranker = fit_local_subset_rerankers(
            all_model_train_logits,
            train_targets_by_video,
            initial_weights=initial_weights,
            device=device,
            num_epochs=args.local_reranker_epochs,
            lr=args.local_reranker_lr,
            weight_decay=args.local_reranker_weight_decay,
        )
        print("local_reranker_ready:")
        for subset_name, summary in local_reranker["local_reranker_summary_by_subset"].items():
            print(
                f"  subset={subset_name} raw_labels={summary['raw_labels']} "
                f"train_loss={summary['train_loss']:.4f} "
                f"train_acc={summary['train_acc']:.4f} "
                f"train_frames={summary['train_frames']}"
            )
    target_corrector = None
    if args.target_corrector:
        target_corrector = fit_target_specific_correctors(
            all_model_train_logits,
            train_targets_by_video,
            initial_weights=initial_weights,
            device=device,
            num_epochs=args.target_corrector_epochs,
            lr=args.target_corrector_lr,
            weight_decay=args.target_corrector_weight_decay,
        )
        print("target_corrector_ready:")
        for raw_label, summary in target_corrector["target_corrector_summary_by_raw_label"].items():
            print(
                f"  raw_label={raw_label} subset_raw_labels={summary['subset_raw_labels']} "
                f"train_loss={summary['train_loss']:.4f} "
                f"train_acc={summary['train_acc']:.4f} "
                f"train_frames={summary['train_frames']}"
            )

    # ---- Emission prior search mode ----
    if args.search_priors:
        weights = initial_weights
        print(f"\n===== Emission Prior Search (weights={weights}) =====")

        # Classes to tune and their candidate multipliers
        # Key = raw label, Values = list of multiplier candidates
        prior_search_space: Dict[int, List[float]] = {
            7:  [0.10, 0.15, 0.20, 0.30, 0.40],     # dominant class: suppress
            5:  [2.0, 3.0, 3.5, 4.5, 6.0],           # confused with 4/7/8
            6:  [1.5, 2.0, 2.2, 3.0, 4.0],           # confused with 7
            9:  [2.5, 3.5, 5.0, 7.0, 10.0],          # confused with 7
            3:  [5.0, 8.0, 12.0, 16.0, 20.0],        # heavily confused with 4
            8:  [0.5, 0.9, 1.2, 1.5, 2.0],           # confused with 5
            14: [1.0, 1.5, 2.5, 4.0, 6.0],           # confused with 10
            12: [1.0, 2.0, 3.0, 5.0, 8.0],           # confused with 14
            4:  [0.3, 0.5, 0.7, 1.0],                # over-predicted, absorbs 3/5
            10: [0.5, 0.7, 1.0, 1.5],                # over-predicted, absorbs 14
        }

        # Greedy coordinate-wise search: tune one class at a time, repeat
        current_prior = dict(emission_prior)
        best_global = -1.0

        # First evaluate baseline
        baseline, _ = evaluate_ensemble(
            all_model_logits, weights, val_dataset,
            light_post_kernel_size=args.light_post_kernel_size,
            min_duration_frames=args.min_duration_frames,
            emission_prior=current_prior,
            learned_fusion=learned_fusion,
            local_reranker=local_reranker,
            target_corrector=target_corrector,
            targets_by_video=val_targets_by_video,
            log_position_prior_matrix=position_log_prior_np,
            position_prior_weight=float(args.position_prior_weight),
            local_reranker_subset_mass_threshold=args.local_reranker_subset_mass_threshold,
            local_reranker_topk=args.local_reranker_topk,
            local_reranker_blend=args.local_reranker_blend,
            target_corrector_subset_mass_threshold=args.target_corrector_subset_mass_threshold,
            target_corrector_topk=args.target_corrector_topk,
            target_corrector_logit_scale=args.target_corrector_logit_scale,
        )
        best_global = baseline["viterbi_step_metric"]
        print(f"  baseline priors -> viterbi_step_metric={best_global:.4f}")

        for round_idx in range(3):  # repeat coordinate descent up to 3 rounds
            improved_this_round = False
            for raw_label, candidates in prior_search_space.items():
                cls_idx = LABEL_TO_INDEX[raw_label]
                old_val = current_prior.get(cls_idx, 1.0)
                best_val = old_val
                best_metric_for_cls = best_global

                for candidate in candidates:
                    trial_prior = dict(current_prior)
                    trial_prior[cls_idx] = candidate
                    metrics, _ = evaluate_ensemble(
                        all_model_logits, weights, val_dataset,
                        light_post_kernel_size=args.light_post_kernel_size,
                        min_duration_frames=args.min_duration_frames,
                        emission_prior=trial_prior,
                        learned_fusion=learned_fusion,
                        local_reranker=local_reranker,
                        target_corrector=target_corrector,
                        targets_by_video=val_targets_by_video,
                        log_position_prior_matrix=position_log_prior_np,
                        position_prior_weight=float(args.position_prior_weight),
                        local_reranker_subset_mass_threshold=args.local_reranker_subset_mass_threshold,
                        local_reranker_topk=args.local_reranker_topk,
                        local_reranker_blend=args.local_reranker_blend,
                        target_corrector_subset_mass_threshold=args.target_corrector_subset_mass_threshold,
                        target_corrector_topk=args.target_corrector_topk,
                        target_corrector_logit_scale=args.target_corrector_logit_scale,
                    )
                    vm = metrics["viterbi_step_metric"]
                    if vm > best_metric_for_cls:
                        best_metric_for_cls = vm
                        best_val = candidate
                        best_metrics = metrics

                if best_val != old_val:
                    current_prior[cls_idx] = best_val
                    best_global = best_metric_for_cls
                    improved_this_round = True
                    print(f"  round {round_idx+1}: class {raw_label} prior {old_val:.2f} -> {best_val:.2f} "
                          f"viterbi_step_metric={best_global:.4f} ***")
                else:
                    print(f"  round {round_idx+1}: class {raw_label} prior {old_val:.2f} (unchanged)")

            if not improved_this_round:
                print(f"  round {round_idx+1}: no improvement, stopping")
                break

        # Print final result
        prior_readable = {raw_label: current_prior.get(LABEL_TO_INDEX[raw_label], 1.0)
                          for raw_label in sorted(prior_search_space.keys())}
        print(f"\nbest_priors (raw_label -> multiplier): {prior_readable}")
        print(f"best_viterbi_step_metric={best_global:.4f}")

        # Final detailed eval
        final, final_records = evaluate_ensemble(
            all_model_logits, weights, val_dataset,
            light_post_kernel_size=args.light_post_kernel_size,
            min_duration_frames=args.min_duration_frames,
            emission_prior=current_prior,
            learned_fusion=learned_fusion,
            local_reranker=local_reranker,
            target_corrector=target_corrector,
            targets_by_video=val_targets_by_video,
            log_position_prior_matrix=position_log_prior_np,
            position_prior_weight=float(args.position_prior_weight),
            local_reranker_subset_mass_threshold=args.local_reranker_subset_mass_threshold,
            local_reranker_topk=args.local_reranker_topk,
            local_reranker_blend=args.local_reranker_blend,
            target_corrector_subset_mass_threshold=args.target_corrector_subset_mass_threshold,
            target_corrector_topk=args.target_corrector_topk,
            target_corrector_logit_scale=args.target_corrector_logit_scale,
        )
        _print_detailed_results(final)

        # Print code-ready prior dict
        print("\n--- Copy-paste prior dict ---")
        print("_EMISSION_PRIOR = {")
        for raw_label in sorted(prior_search_space.keys()):
            cls_idx = LABEL_TO_INDEX[raw_label]
            val = current_prior.get(cls_idx, 1.0)
            print(f"    LABEL_TO_INDEX[{raw_label}]: {val},")
        print("}")
        _write_result_artifacts(
            json_path=args.result_json.resolve() if args.result_json is not None else None,
            csv_path=args.result_csv.resolve() if args.result_csv is not None else None,
            all_frames_csv_path=args.all_frames_csv.resolve() if args.all_frames_csv is not None else None,
            eval_frames_csv_path=args.eval_frames_csv.resolve() if args.eval_frames_csv is not None else None,
            metadata={
                **base_metadata,
                "result_mode": "prior_search",
                "baseline_viterbi_step_metric": float(baseline["viterbi_step_metric"]),
                "best_weights": [float(weight) for weight in weights],
                "best_priors_by_raw_label": prior_readable,
            },
            metrics=final,
            frame_records=final_records,
        )
        return

    # Grid search or single evaluation
    if args.grid_search and n_models == 2:
        print("\n===== Grid Search =====")
        step = args.grid_step
        best_metric = -1.0
        best_weights = None
        results = []

        alpha_values = np.arange(0.0, 1.0 + step / 2, step)
        for alpha in alpha_values:
            w = [alpha, 1.0 - alpha]
            metrics, records = evaluate_ensemble(
                all_model_logits, w, val_dataset,
                light_post_kernel_size=args.light_post_kernel_size,
                min_duration_frames=args.min_duration_frames,
                emission_prior=emission_prior,
                targets_by_video=val_targets_by_video,
                log_position_prior_matrix=position_log_prior_np,
                position_prior_weight=float(args.position_prior_weight),
                local_reranker_subset_mass_threshold=args.local_reranker_subset_mass_threshold,
                local_reranker_topk=args.local_reranker_topk,
                local_reranker_blend=args.local_reranker_blend,
                target_corrector_subset_mass_threshold=args.target_corrector_subset_mass_threshold,
                target_corrector_topk=args.target_corrector_topk,
                target_corrector_logit_scale=args.target_corrector_logit_scale,
            )
            vm = metrics["viterbi_step_metric"]
            results.append((alpha, vm, metrics))
            marker = " ***" if vm > best_metric else ""
            print(f"  w=[{alpha:.2f}, {1-alpha:.2f}] "
                  f"viterbi_step_metric={vm:.4f} "
                  f"viterbi_macro_f1={metrics['viterbi_macro_f1']:.4f} "
                f"viterbi_edit={metrics['viterbi_edit']:.4f} "
                f"viterbi_f1@50={metrics['viterbi_f1_at_50']:.4f}{marker}")
            if vm > best_metric:
                best_metric = vm
                best_weights = [float(x) for x in w]
                best_result = metrics
                best_records = records

        print(f"\nbest_weights={best_weights} best_viterbi_step_metric={best_metric:.4f}")
        _print_detailed_results(best_result)
        _write_result_artifacts(
            json_path=args.result_json.resolve() if args.result_json is not None else None,
            csv_path=args.result_csv.resolve() if args.result_csv is not None else None,
            all_frames_csv_path=args.all_frames_csv.resolve() if args.all_frames_csv is not None else None,
            eval_frames_csv_path=args.eval_frames_csv.resolve() if args.eval_frames_csv is not None else None,
            metadata={
                **base_metadata,
                "result_mode": "grid_search",
                "best_weights": best_weights,
            },
            metrics=best_result,
            frame_records=best_records,
        )

    elif args.grid_search and n_models == 3:
        print("\n===== Grid Search (3 models) =====")
        step = args.grid_step
        best_metric = -1.0
        best_weights = None

        alpha_values = np.arange(0.0, 1.0 + step / 2, step)
        for a1 in alpha_values:
            for a2 in alpha_values:
                a3 = 1.0 - a1 - a2
                if a3 < -1e-6:
                    continue
                a3 = max(0.0, a3)
                w = [a1, a2, a3]
                metrics, records = evaluate_ensemble(
                    all_model_logits, w, val_dataset,
                    light_post_kernel_size=args.light_post_kernel_size,
                    min_duration_frames=args.min_duration_frames,
                    emission_prior=emission_prior,
                    targets_by_video=val_targets_by_video,
                    log_position_prior_matrix=position_log_prior_np,
                    position_prior_weight=float(args.position_prior_weight),
                    local_reranker_subset_mass_threshold=args.local_reranker_subset_mass_threshold,
                    local_reranker_topk=args.local_reranker_topk,
                    local_reranker_blend=args.local_reranker_blend,
                    target_corrector_subset_mass_threshold=args.target_corrector_subset_mass_threshold,
                    target_corrector_topk=args.target_corrector_topk,
                    target_corrector_logit_scale=args.target_corrector_logit_scale,
                )
                vm = metrics["viterbi_step_metric"]
                if vm > best_metric:
                    best_metric = vm
                    best_weights = [float(x) for x in w]
                    best_result = metrics
                    best_records = records
                    print(f"  NEW BEST w=[{a1:.2f}, {a2:.2f}, {a3:.2f}] "
                          f"viterbi_step_metric={vm:.4f}")

        print(f"\nbest_weights={best_weights} best_viterbi_step_metric={best_metric:.4f}")
        _print_detailed_results(best_result)
        _write_result_artifacts(
            json_path=args.result_json.resolve() if args.result_json is not None else None,
            csv_path=args.result_csv.resolve() if args.result_csv is not None else None,
            all_frames_csv_path=args.all_frames_csv.resolve() if args.all_frames_csv is not None else None,
            eval_frames_csv_path=args.eval_frames_csv.resolve() if args.eval_frames_csv is not None else None,
            metadata={
                **base_metadata,
                "result_mode": "grid_search",
                "best_weights": best_weights,
            },
            metrics=best_result,
            frame_records=best_records,
        )

    else:
        # Single evaluation
        weights = initial_weights
        if learned_fusion is not None:
            print(f"\nensemble_init_weights={weights}")
            print("fusion_mode=classwise_learned")
        elif local_reranker is not None:
            print(f"\nensemble_init_weights={weights}")
            print("fusion_mode=local_reranker")
        elif target_corrector is not None:
            print(f"\nensemble_init_weights={weights}")
            print("fusion_mode=target_corrector")
        else:
            print(f"\nensemble_weights={weights}")
        metrics, records = evaluate_ensemble(
            all_model_logits, weights, val_dataset,
            light_post_kernel_size=args.light_post_kernel_size,
            min_duration_frames=args.min_duration_frames,
            emission_prior=emission_prior,
            learned_fusion=learned_fusion,
            local_reranker=local_reranker,
            target_corrector=target_corrector,
            targets_by_video=val_targets_by_video,
            log_position_prior_matrix=position_log_prior_np,
            position_prior_weight=float(args.position_prior_weight),
            local_reranker_subset_mass_threshold=args.local_reranker_subset_mass_threshold,
            local_reranker_topk=args.local_reranker_topk,
            local_reranker_blend=args.local_reranker_blend,
            target_corrector_subset_mass_threshold=args.target_corrector_subset_mass_threshold,
            target_corrector_topk=args.target_corrector_topk,
            target_corrector_logit_scale=args.target_corrector_logit_scale,
        )
        print(f"viterbi_step_metric={metrics['viterbi_step_metric']:.4f} "
              f"viterbi_macro_f1={metrics['viterbi_macro_f1']:.4f} "
              f"viterbi_edit={metrics['viterbi_edit']:.4f} "
              f"viterbi_f1@10={metrics['viterbi_f1_at_10']:.4f} "
              f"viterbi_f1@25={metrics['viterbi_f1_at_25']:.4f} "
              f"viterbi_f1@50={metrics['viterbi_f1_at_50']:.4f}")
        _print_detailed_results(metrics)
        _write_result_artifacts(
            json_path=args.result_json.resolve() if args.result_json is not None else None,
            csv_path=args.result_csv.resolve() if args.result_csv is not None else None,
            all_frames_csv_path=args.all_frames_csv.resolve() if args.all_frames_csv is not None else None,
            eval_frames_csv_path=args.eval_frames_csv.resolve() if args.eval_frames_csv is not None else None,
            metadata={
                **base_metadata,
                "result_mode": "single_eval",
                "best_weights": [float(weight) for weight in weights],
            },
            metrics=metrics,
            frame_records=records,
        )


def _print_detailed_results(metrics: dict):
    """Print per-class and per-video detailed results."""
    if "learned_fusion_weights_by_raw_label" in metrics:
        print("\n--- Learned Fusion ---")
        print(
            f"  train_loss={metrics['learned_fusion_train_loss']:.4f} "
            f"train_acc={metrics['learned_fusion_train_acc']:.4f} "
            f"train_frames={metrics['learned_fusion_train_frames']}"
        )
        print("  learned_fusion_weights_by_raw_label=")
        for raw_label in LABELS12:
            print(
                f"    {raw_label}: "
                f"{metrics['learned_fusion_weights_by_raw_label'][raw_label]}"
            )
        print("  learned_fusion_bias_by_raw_label=")
        for raw_label in LABELS12:
            print(
                f"    {raw_label}: "
                f"{metrics['learned_fusion_bias_by_raw_label'][raw_label]:.4f}"
            )
    if "local_reranker_summary_by_subset" in metrics:
        print("\n--- Local Reranker ---")
        applied_frames = metrics.get("local_reranker_applied_frames_by_subset", {})
        for subset_name, summary in metrics["local_reranker_summary_by_subset"].items():
            print(
                f"  subset={subset_name} raw_labels={summary['raw_labels']} "
                f"train_loss={summary['train_loss']:.4f} "
                f"train_acc={summary['train_acc']:.4f} "
                f"train_frames={summary['train_frames']} "
                f"applied_frames={applied_frames.get(subset_name, 0)}"
            )
            print("    weights_by_raw_label=")
            for raw_label in summary["raw_labels"]:
                print(f"      {raw_label}: {summary['weights_by_raw_label'][raw_label]}")
            print("    bias_by_raw_label=")
            for raw_label in summary["raw_labels"]:
                print(f"      {raw_label}: {summary['bias_by_raw_label'][raw_label]:.4f}")
    if "target_corrector_summary_by_raw_label" in metrics:
        print("\n--- Target Corrector ---")
        applied_frames = metrics.get("target_corrector_applied_frames_by_raw_label", {})
        for raw_label, summary in metrics["target_corrector_summary_by_raw_label"].items():
            print(
                f"  raw_label={raw_label} subset_raw_labels={summary['subset_raw_labels']} "
                f"train_loss={summary['train_loss']:.4f} "
                f"train_acc={summary['train_acc']:.4f} "
                f"train_frames={summary['train_frames']} "
                f"applied_frames={applied_frames.get(raw_label, 0)}"
            )

    print("\n--- Segmental F1 (viterbi) ---")
    print(f"  F1@10={metrics['viterbi_f1_at_10']:.4f} F1@25={metrics['viterbi_f1_at_25']:.4f} F1@50={metrics['viterbi_f1_at_50']:.4f}")

    print("\n--- Per-class results (viterbi) ---")
    print(f"{'raw_step':>10s}  {'precision':>10s}  {'recall':>10s}  {'f1':>10s}  {'support':>10s}")
    for i, raw_label in enumerate(LABELS12):
        p = metrics["viterbi_per_class_precision"][i]
        r = metrics["viterbi_per_class_recall"][i]
        f = metrics["viterbi_per_class_f1"][i]
        s = metrics["viterbi_support"][i]
        print(f"{raw_label:>10d}  {p:>10.4f}  {r:>10.4f}  {f:>10.4f}  {s:>10d}")

    print("\n--- Per-video results (viterbi) ---")
    for pv in metrics.get("viterbi_per_video", []):
        print(f"  video_{pv['video_id']:02d}: macro_f1={pv['macro_f1']:.4f} "
              f"edit={pv['edit']:.4f} step_metric={pv['step_metric']:.4f}")

    cm = metrics.get("viterbi_confusion")
    if cm is not None:
        print("\n--- Confusion Matrix (viterbi) ---")
        header = "true\\pred " + " ".join(f"{l:>5d}" for l in LABELS12)
        print(header)
        for i, raw_from in enumerate(LABELS12):
            row = " ".join(f"{int(cm[i, j]):>5d}" for j in range(len(LABELS12)))
            print(f"{raw_from:>9d} {row}")


if __name__ == "__main__":
    main()
