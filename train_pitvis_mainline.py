import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from pitvis_mainline_common import (
    ALLOWED_TRANSITIONS_RAW,
    FEATURE_DIM,
    GPU_MEM_LIMIT_RATIO,
    LABELS12,
    RAM_LIMIT_RATIO,
    build_log_transition_matrix_numpy,
    build_log_transition_tensor,
    collect_segment_transition_counts_from_csv_root,
    load_json_config,
    maybe_apply_gpu_memory_fraction,
    parse_transition_pairs_spec,
    print_resource_snapshot,
    resolve_device_and_gpu_ids,
    set_seed,
    unwrap_model,
    wrap_model_data_parallel,
)
from pitvis_mainline_data import build_pitvis_datasets, build_mstcn_datasets, build_sample_weights_for_balanced_sampling, build_step_class_weights, collate_full_videos
from pitvis_mainline_eval import print_validation_details, run_validation_epoch, run_mstcn_validation_epoch
from pitvis_mainline_model import (
    ASRMFeatureAdapter,
    BoundaryHead,
    InstrumentHead,
    LemonFMBackboneWrapper,
    LocalTransformerEncoder,
    ParallelStepHead,
    PitVisASFormerModel,
    PitVisMSTCNModel,
    PitVisMultitaskModel,
    TemporalRefinementTCN,
)
from pitvis_position_prior import (
    build_log_position_prior_numpy,
    collect_position_bin_counts_from_csv_root,
)
from pitvis_result_tracking import (
    build_best_metric_row,
    upsert_best_metric_summary_csv,
    write_best_metric_csv,
)
try:
    from pitvis_result_tracking import write_best_metric_json
except ImportError:
    def write_best_metric_json(json_path: Path, row: dict) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(dict(row), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
from pitvis_mainline_train import (
    MSTCNTrainer,
    StageATrainer,
    StageBTrainer,
    build_amp_components,
    build_instrument_pos_weight,
    build_optimizer_and_scheduler,
    current_lr_summary,
)


def build_transition_priors_from_training_csv(
    csv_root: Path,
    *,
    train_video_ids: Sequence[int],
) -> Tuple[dict, torch.Tensor, object]:
    transition_counts = collect_segment_transition_counts_from_csv_root(
        csv_root,
        video_ids=train_video_ids,
        allowed_transitions_raw=ALLOWED_TRANSITIONS_RAW,
    )
    log_trans_torch = build_log_transition_tensor(
        transition_counts=transition_counts,
        allowed_transitions_raw=ALLOWED_TRANSITIONS_RAW,
    )
    log_trans_numpy = build_log_transition_matrix_numpy(
        transition_counts=transition_counts,
        allowed_transitions_raw=ALLOWED_TRANSITIONS_RAW,
    )
    nonzero_edges = sum(len(targets) for targets in transition_counts.values())
    print(f"segment_transition_priors: train_videos={list(train_video_ids)} nonzero_edges={nonzero_edges}")
    return transition_counts, log_trans_torch, log_trans_numpy


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PitVis mainline training script")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON configuration path; explicitly supplied CLI values take precedence",
    )
    parser.add_argument("--input-mode", choices=["features", "images"], default="features")
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--csv-root", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pitvis_mainline"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Explicit train/validation split JSON; otherwise use the official split",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--gpu-mem-limit", type=float, default=GPU_MEM_LIMIT_RATIO)
    parser.add_argument("--ram-limit", type=float, default=RAM_LIMIT_RATIO)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-val-videos", type=int, default=None)

    parser.add_argument("--stage-a-epochs", type=int, default=0)
    parser.add_argument("--stage-b-epochs", type=int, default=20)
    parser.add_argument("--stage-a-window-size", type=int, default=32)
    parser.add_argument("--stage-a-stride", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--temporal-adapter-kernel-size", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=1)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos-encoding-type", choices=["learned", "sinusoidal", "none"], default="learned")
    parser.add_argument("--max-positional-length", type=int, default=None)

    parser.add_argument("--lambda-bd", type=float, default=0.2)
    parser.add_argument("--lambda-inst", type=float, default=0.1)
    parser.add_argument("--lambda-tmse", type=float, default=0.15)
    parser.add_argument("--lambda-trans", type=float, default=0.0,
                        help="Weight for transition-aware loss (expected log-transition probability)")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--tcn-channels", type=int, default=64)
    parser.add_argument("--tcn-layers", type=int, default=4)
    parser.add_argument("--lambda-refine-init", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--use-instrument-pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-balance-beta", type=float, default=0.0,
                        help="CB loss beta (0=sqrt inverse freq, >0=CB formula, e.g. 0.999)")
    parser.add_argument("--speed-perturb-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Speed perturbation range for training, e.g. 0.8 1.2")

    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--boundary-base-weight", type=float, default=2.0)
    parser.add_argument("--focus-boundary-weight", type=float, default=3.0)
    parser.add_argument("--boundary-mode", choices=["scalar", "pairwise"], default="scalar")
    parser.add_argument("--boundary-kernel", choices=["gaussian", "triangle", "asymmetric_gaussian", "pairwise_asymmetric_gaussian"], default="gaussian")
    parser.add_argument("--boundary-left-sigma", type=float, default=None)
    parser.add_argument("--boundary-right-sigma", type=float, default=None)
    parser.add_argument("--transition-pairs", nargs="*", default=None,
                        help="Directed transition pairs like 2-3 3-4 8-9; config may also provide [[2,3], ...]")

    parser.add_argument("--center-frame-size", type=int, default=32)
    parser.add_argument("--center-min-weight", type=float, default=0.5)

    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--norm", choices=["lemon_video", "imagenet"], default="lemon_video")
    parser.add_argument(
        "--pretrain-source",
        choices=["lemonfm", "imagenet", "none"],
        default="lemonfm",
        help="Backbone initialization source for image-mode finetuning",
    )
    parser.add_argument(
        "--unfreeze-strategy",
        choices=["last_stage", "last_two_stages", "full_finetune"],
        default="last_stage",
        help="Backbone unfreeze strategy for image-mode finetuning",
    )

    parser.add_argument("--light-post-kernel-size", type=int, default=3)
    parser.add_argument("--min-duration-frames", type=int, default=0)
    parser.add_argument("--position-prior-bins", type=int, default=8)
    parser.add_argument("--position-prior-weight", type=float, default=0.0)
    parser.add_argument("--prior-position-prior-weight", type=float, default=None)
    parser.add_argument("--late-chain-refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--late-chain-loss-weight", type=float, default=0.0)
    parser.add_argument("--late-chain-gate-power", type=float, default=1.0)
    parser.add_argument("--late-chain-decoder-guard", type=float, default=0.0)
    parser.add_argument("--coarse-to-fine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.0)
    parser.add_argument("--coarse-gate-scale", type=float, default=1.0)

    # MS-TCN architecture args
    parser.add_argument("--architecture", choices=["window", "mstcn", "mstcn_online", "asformer"], default="window",
                        help="Model architecture. The paper recipe uses offline full-video MS-TCN; other choices are retained for legacy checkpoint compatibility.")
    # ASFormer architecture args
    parser.add_argument("--asformer-channels", type=int, default=64, help="Hidden channels in ASFormer")
    parser.add_argument("--asformer-layers", type=int, default=10, help="Dilated attention layers per encoder/decoder")
    parser.add_argument("--asformer-decoders", type=int, default=3, help="Number of ASFormer decoder stages")
    parser.add_argument("--asformer-r", type=int, default=1, help="Attention dimension reduction factor")
    parser.add_argument("--asformer-attention-type", choices=["local", "full"], default="local",
                        help="Attention pattern used by ASFormer layers.")
    parser.add_argument("--asformer-attention-dropout", type=float, default=0.1,
                        help="Dropout applied on ASFormer attention weights.")
    parser.add_argument("--asformer-decoder-key-source", choices=["decoder", "encoder"], default="decoder",
                        help="Whether ASFormer decoder keys come from decoder states or encoder states.")
    parser.add_argument("--asformer-feature-layernorm", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply LayerNorm to per-frame features before ASFormer encoding.")
    parser.add_argument("--channel-masking-rate", type=float, default=0.3, help="Channel dropout rate on encoder input")
    parser.add_argument("--mstcn-channels", type=int, default=64, help="Hidden channels in MS-TCN stages")
    parser.add_argument("--mstcn-layers", type=int, default=10, help="Dilated conv layers per TCN stage")
    parser.add_argument("--mstcn-stages", type=int, default=4, help="Number of TCN stages (1 prediction + N-1 refinement)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps (for MS-TCN with batch=1)")
    parser.add_argument("--instrument-fusion", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable instrument-conditioned fusion stage in MS-TCN")
    parser.add_argument("--input-dropout", type=float, default=0.0,
                        help="Dropout applied to input features before MS-TCN (regularization)")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=False,
                        help="Use causal temporal convolutions for strict-online MSTCN execution.")
    parser.add_argument("--online-transition-alpha", type=float, default=1.0,
                        help="Weight for transition priors in strict-online causal decoding.")
    parser.add_argument("--online-ema-decay", type=float, default=0.0,
                        help="EMA decay for strict-online probability smoothing before causal decoding.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint (.pt) to resume from or evaluate")
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Only run validation (requires --resume)")
    parser.add_argument(
        "--export-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to export validation prediction files for every epoch (default: only export best epoch).",
    )
    parser.add_argument(
        "--best-metrics-summary-csv",
        type=Path,
        default=Path("outputs/best_epoch_metrics_summary.csv"),
        help="Global CSV file that stores one best-epoch row per output directory.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=["viterbi_step_metric", "step_metric", "online_step_metric"],
        default="viterbi_step_metric",
        help="Validation metric used for early stopping and best-checkpoint selection.",
    )

    preliminary, _ = parser.parse_known_args(argv)
    config_data = load_json_config(preliminary.config)
    if config_data:
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in config_data.items()})
        print(f"config_loaded: {preliminary.config} ({len(config_data)} keys)")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.split_json is not None and not Path(args.split_json).is_file():
        raise FileNotFoundError(f"split json not found: {args.split_json}")
    if args.csv_root is None:
        raise ValueError("--csv-root is required.")
    if args.input_mode == "features":
        if args.features_root is None:
            raise ValueError("--features-root is required in features mode.")
    else:
        required_fields = [("images-root", args.images_root)]
        if getattr(args, "pretrain_source", "lemonfm") == "lemonfm":
            required_fields.extend(
                [
                    ("repo-root", args.repo_root),
                    ("weights", args.weights),
                ]
            )
        missing = [name for name, value in required_fields if value is None]
        if missing:
            raise ValueError(f"Missing required argument(s) in images mode: {', '.join(missing)}")
    if args.stage_a_epochs < 0 or args.stage_b_epochs < 0:
        raise ValueError("Stage epoch counts cannot be negative.")
    if args.stage_a_epochs == 0 and args.stage_b_epochs == 0:
        raise ValueError("stage-a-epochs and stage-b-epochs cannot both be zero.")
    if args.d_model % args.nhead != 0:
        raise ValueError("d-model must be divisible by nhead.")
    if int(getattr(args, "position_prior_bins", 0)) <= 0:
        raise ValueError("position-prior-bins must be positive.")
    if float(getattr(args, "position_prior_weight", 0.0)) < 0.0:
        raise ValueError("position-prior-weight cannot be negative.")
    if getattr(args, "prior_position_prior_weight", None) is not None and float(args.prior_position_prior_weight) < 0.0:
        raise ValueError("prior-position-prior-weight cannot be negative.")
    if float(getattr(args, "late_chain_loss_weight", 0.0)) < 0.0:
        raise ValueError("late-chain-loss-weight cannot be negative.")
    if float(getattr(args, "late_chain_gate_power", 1.0)) <= 0.0:
        raise ValueError("late-chain-gate-power must be positive.")
    if float(getattr(args, "late_chain_decoder_guard", 0.0)) < 0.0:
        raise ValueError("late-chain-decoder-guard cannot be negative.")
    if float(getattr(args, "coarse_loss_weight", 0.0)) < 0.0:
        raise ValueError("coarse-loss-weight cannot be negative.")
    if float(getattr(args, "coarse_gate_scale", 1.0)) < 0.0:
        raise ValueError("coarse-gate-scale cannot be negative.")
    if float(getattr(args, "online_transition_alpha", 1.0)) < 0.0:
        raise ValueError("online-transition-alpha cannot be negative.")
    ema_decay = float(getattr(args, "online_ema_decay", 0.0))
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError("online-ema-decay must satisfy 0 <= decay < 1.")
    args.transition_pairs = parse_transition_pairs_spec(getattr(args, "transition_pairs", None))


def build_data_loader_kwargs(args: argparse.Namespace) -> dict:
    use_pin_memory = bool(args.pin_memory and str(args.device).startswith("cuda"))
    effective_workers = min(int(args.num_workers), 8) if args.input_mode == "images" else int(args.num_workers)
    effective_prefetch = int(args.prefetch_factor)
    kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": effective_workers,
        "pin_memory": use_pin_memory,
    }
    if effective_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = effective_prefetch
    return kwargs


def build_model(args: argparse.Namespace, *, num_instrument_classes: int) -> PitVisMultitaskModel:
    max_positional_length = int(
        args.max_positional_length
        if args.max_positional_length is not None
        else max(int(args.window_size), int(args.stage_a_window_size))
    )
    backbone_wrapper = None
    if args.input_mode == "images":
        backbone_wrapper = LemonFMBackboneWrapper.from_pretrain_source(
            repo_root=args.repo_root,
            weights_path=args.weights,
            pretrain_source=args.pretrain_source,
            channels_last=args.channels_last,
        )
        backbone_wrapper.freeze_for_peft(args.unfreeze_strategy)

    feature_adapter = ASRMFeatureAdapter(
        input_dim=FEATURE_DIM,
        adapter_dim=args.adapter_dim,
        temporal_kernel_size=args.temporal_adapter_kernel_size,
        dropout=args.dropout,
    )
    encoder = LocalTransformerEncoder(
        input_dim=FEATURE_DIM,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_encoder_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        pos_encoding_type=args.pos_encoding_type,
        max_positional_length=max_positional_length,
    )
    step_head = ParallelStepHead(
        d_model=args.d_model,
        num_classes=len(LABELS12),
        dropout=args.dropout,
    )
    step_refine_tcn = None
    if args.tcn_layers > 0:
        step_refine_tcn = TemporalRefinementTCN(
            num_classes=len(LABELS12),
            num_channels=args.tcn_channels,
            num_layers=args.tcn_layers,
            dropout=args.dropout,
        )
    instrument_head = InstrumentHead(d_model=args.d_model, num_instrument_classes=num_instrument_classes)
    num_boundary_outputs = len(args.transition_pairs) if getattr(args, "boundary_mode", "scalar") == "pairwise" else 1
    boundary_head = BoundaryHead(d_model=args.d_model, num_outputs=num_boundary_outputs)
    model = PitVisMultitaskModel(
        backbone_wrapper=backbone_wrapper,
        feature_adapter=feature_adapter,
        encoder=encoder,
        step_head=step_head,
        instrument_head=instrument_head,
        boundary_head=boundary_head,
        num_step_classes=len(LABELS12),
        step_refine_tcn=step_refine_tcn,
    )
    return model


def print_model_summary(args: argparse.Namespace, model: torch.nn.Module, instrument_class_ids: Sequence[int]) -> None:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"total_params={total_params}")
    print(f"trainable_params={trainable_params}")
    architecture = getattr(args, "architecture", "window")
    if architecture in ("mstcn", "mstcn_online", "asformer"):
        if architecture == "asformer":
            print(
                f"asformer_model_config: channels={getattr(args, 'asformer_channels', 64)} "
                f"layers={getattr(args, 'asformer_layers', 10)} "
                f"decoders={getattr(args, 'asformer_decoders', 3)} "
                f"r={getattr(args, 'asformer_r', 1)} "
                f"channel_masking_rate={getattr(args, 'channel_masking_rate', 0.3)} "
                f"dropout={args.dropout} "
                f"input_dropout={getattr(args, 'input_dropout', 0.0)} "
                f"boundary_mode={getattr(args, 'boundary_mode', 'scalar')} "
                f"grad_accum_steps={args.grad_accum_steps} "
                f"instrument_fusion={getattr(args, 'instrument_fusion', False)} "
                f"coarse_to_fine={getattr(args, 'coarse_to_fine', False)} "
                f"coarse_loss_weight={getattr(args, 'coarse_loss_weight', 0.0)} "
                f"coarse_gate_scale={getattr(args, 'coarse_gate_scale', 1.0)}"
            )
        else:
            print(
                f"mstcn_model_config: channels={args.mstcn_channels} layers={args.mstcn_layers} "
                f"stages={args.mstcn_stages} dropout={args.dropout} "
                f"input_dropout={getattr(args, 'input_dropout', 0.0)} "
                f"causal={bool(getattr(args, 'causal', False) or architecture == 'mstcn_online')} "
                f"boundary_mode={getattr(args, 'boundary_mode', 'scalar')} "
                f"grad_accum_steps={args.grad_accum_steps} "
                f"instrument_fusion={getattr(args, 'instrument_fusion', False)} "
                f"late_chain_refine={getattr(args, 'late_chain_refine', False)} "
                f"late_chain_loss_weight={getattr(args, 'late_chain_loss_weight', 0.0)} "
                f"late_chain_gate_power={getattr(args, 'late_chain_gate_power', 1.0)} "
                f"late_chain_decoder_guard={getattr(args, 'late_chain_decoder_guard', 0.0)} "
                f"coarse_to_fine={getattr(args, 'coarse_to_fine', False)} "
                f"coarse_loss_weight={getattr(args, 'coarse_loss_weight', 0.0)} "
                f"coarse_gate_scale={getattr(args, 'coarse_gate_scale', 1.0)}"
            )
            rf = 2 * sum(2 ** i for i in range(args.mstcn_layers)) + 1
            print(f"mstcn_receptive_field_per_stage={rf}")
    else:
        print(
            f"mainline_model_config: input_mode={args.input_mode} adapter_dim={args.adapter_dim} "
            f"d_model={args.d_model} nhead={args.nhead} num_encoder_layers={args.num_encoder_layers} "
            f"ffn_dim={args.ffn_dim} dropout={args.dropout} "
            f"pos_encoding_type={args.pos_encoding_type} label_smoothing={args.label_smoothing} "
            f"pretrain_source={getattr(args, 'pretrain_source', 'lemonfm')} "
            f"unfreeze_strategy={getattr(args, 'unfreeze_strategy', 'last_stage')} "
            f"boundary_mode={getattr(args, 'boundary_mode', 'scalar')}"
        )
        if hasattr(model, "backbone_wrapper") and model.backbone_wrapper is not None:
            backbone = model.backbone_wrapper if not isinstance(model, torch.nn.DataParallel) else model.module.backbone_wrapper
            for stage_idx, module in enumerate(backbone.backbone.features):
                is_trainable = any(param.requires_grad for param in module.parameters())
                print(f"features[{stage_idx}]_trainable={is_trainable}")
            print(f"classifier[0]_trainable={any(param.requires_grad for param in backbone.backbone.classifier[0].parameters())}")
    print(
        f"task_config: num_steps={len(LABELS12)} num_instruments={len(instrument_class_ids)} "
        f"lambda_bd={args.lambda_bd} lambda_inst={args.lambda_inst} "
        f"lambda_trans={getattr(args, 'lambda_trans', 0.0)} "
        f"lambda_late_chain={getattr(args, 'late_chain_loss_weight', 0.0)} "
        f"lambda_coarse={getattr(args, 'coarse_loss_weight', 0.0)} "
        f"position_prior_bins={getattr(args, 'position_prior_bins', 8)} "
        f"position_prior_weight={getattr(args, 'position_prior_weight', 0.0)} "
        f"prior_position_prior_weight={getattr(args, 'prior_position_prior_weight', None)}"
    )


def save_best_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_step_metric.pt"
    selection_metric_name = f"val_{args.early_stopping_metric}"
    selection_metric_value = resolve_selection_metric(metrics, args.early_stopping_metric)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": {
                "best_epoch": int(epoch),
                "selection_metric_name": selection_metric_name,
                "selection_metric_value": float(selection_metric_value),
                "val_acc": float(metrics["acc"]),
                "val_video_acc_std": float(metrics.get("video_acc_std", 0.0)),
                "val_macro_f1": float(metrics["macro_f1"]),
                "val_macro_f1_std": float(metrics.get("macro_f1_std", 0.0)),
                "val_macro_jaccard": float(metrics.get("macro_jaccard", 0.0)),
                "val_macro_jaccard_std": float(metrics.get("macro_jaccard_std", 0.0)),
                "val_edit": float(metrics["edit"]),
                "val_edit_std": float(metrics.get("edit_std", 0.0)),
                "val_step_metric": float(metrics["step_metric"]),
                "val_step_metric_std": float(metrics.get("step_metric_std", 0.0)),
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
                "val_online_step_metric": float(metrics.get("online_step_metric", metrics.get("step_metric", 0.0))),
                "val_online_step_metric_std": float(metrics.get("online_step_metric_std", metrics.get("step_metric_std", 0.0))),
                "val_viterbi_macro_f1": float(metrics["viterbi_macro_f1"]),
                "val_viterbi_macro_f1_std": float(metrics.get("viterbi_macro_f1_std", 0.0)),
                "val_viterbi_macro_jaccard": float(metrics.get("viterbi_macro_jaccard", 0.0)),
                "val_viterbi_macro_jaccard_std": float(metrics.get("viterbi_macro_jaccard_std", 0.0)),
                "val_viterbi_edit": float(metrics["viterbi_edit"]),
                "val_viterbi_edit_std": float(metrics.get("viterbi_edit_std", 0.0)),
                "val_viterbi_f1_at_10": float(metrics["viterbi_f1_at_10"]),
                "val_viterbi_f1_at_10_std": float(metrics.get("viterbi_f1_at_10_std", 0.0)),
                "val_viterbi_f1_at_25": float(metrics["viterbi_f1_at_25"]),
                "val_viterbi_f1_at_25_std": float(metrics.get("viterbi_f1_at_25_std", 0.0)),
                "val_viterbi_f1_at_50": float(metrics["viterbi_f1_at_50"]),
                "val_viterbi_f1_at_50_std": float(metrics.get("viterbi_f1_at_50_std", 0.0)),
                "val_viterbi_step_metric": float(metrics["viterbi_step_metric"]),
                "val_viterbi_step_metric_std": float(metrics.get("viterbi_step_metric_std", 0.0)),
            },
            "args": vars(args),
        },
        checkpoint_path,
    )
    return checkpoint_path


def resolve_selection_metric(metrics: dict, metric_name: str) -> float:
    value = metrics.get(str(metric_name))
    if value is None:
        raise KeyError(f"selection metric '{metric_name}' not found in validation metrics")
    return float(value)


def resolve_checkpoint_epoch(checkpoint: Optional[dict]) -> int:
    if not isinstance(checkpoint, dict):
        return 0
    checkpoint_metrics = checkpoint.get("metrics")
    if isinstance(checkpoint_metrics, dict) and checkpoint_metrics.get("best_epoch") is not None:
        return int(checkpoint_metrics["best_epoch"])
    if checkpoint.get("epoch") is not None:
        return int(checkpoint["epoch"])
    return 0


def persist_result_tracking(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    result_source: str,
) -> Path:
    selection_metric_name = f"val_{args.early_stopping_metric}"
    selection_metric_value = resolve_selection_metric(metrics, args.early_stopping_metric)
    best_metric_row = build_best_metric_row(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        epoch=epoch,
        metrics=metrics,
        args=args,
        result_source=result_source,
        selection_metric_name=selection_metric_name,
        selection_metric_value=selection_metric_value,
    )
    local_best_csv = output_dir / "best_metrics.csv"
    local_best_json = output_dir / "result_brief.json"
    write_best_metric_csv(local_best_csv, best_metric_row)
    write_best_metric_json(local_best_json, best_metric_row)
    upsert_best_metric_summary_csv(args.best_metrics_summary_csv, best_metric_row)
    return local_best_csv


def build_asformer_model(args: argparse.Namespace, *, num_instrument_classes: int) -> PitVisASFormerModel:
    """Build a full-video ASFormer model."""
    return PitVisASFormerModel(
        backbone_wrapper=None,
        in_channels=FEATURE_DIM,
        num_step_classes=len(LABELS12),
        num_instrument_classes=num_instrument_classes,
        num_channels=getattr(args, "asformer_channels", 64),
        num_layers=getattr(args, "asformer_layers", 10),
        num_decoders=getattr(args, "asformer_decoders", 3),
        r=getattr(args, "asformer_r", 1),
        dropout=args.dropout,
        attention_dropout=getattr(args, "asformer_attention_dropout", 0.1),
        attention_type=getattr(args, "asformer_attention_type", "local"),
        decoder_key_source=getattr(args, "asformer_decoder_key_source", "decoder"),
        input_dropout=getattr(args, "input_dropout", 0.0),
        feature_layernorm=getattr(args, "asformer_feature_layernorm", True),
        channel_masking_rate=getattr(args, "channel_masking_rate", 0.3),
        instrument_fusion=getattr(args, "instrument_fusion", False),
        num_boundary_outputs=(len(args.transition_pairs) if getattr(args, "boundary_mode", "scalar") == "pairwise" else 1),
        coarse_to_fine=getattr(args, "coarse_to_fine", False),
        coarse_gate_scale=getattr(args, "coarse_gate_scale", 1.0),
    )


def build_mstcn_model(args: argparse.Namespace, *, num_instrument_classes: int) -> PitVisMSTCNModel:
    """Build a full-video MS-TCN model."""
    return PitVisMSTCNModel(
        backbone_wrapper=None,
        in_channels=FEATURE_DIM,
        num_step_classes=len(LABELS12),
        num_instrument_classes=num_instrument_classes,
        num_channels=args.mstcn_channels,
        num_layers=args.mstcn_layers,
        num_stages=args.mstcn_stages,
        dropout=args.dropout,
        input_dropout=getattr(args, "input_dropout", 0.0),
        instrument_fusion=getattr(args, "instrument_fusion", False),
        num_boundary_outputs=(len(args.transition_pairs) if getattr(args, "boundary_mode", "scalar") == "pairwise" else 1),
        late_chain_refine=getattr(args, "late_chain_refine", False),
        late_chain_gate_power=getattr(args, "late_chain_gate_power", 1.0),
        coarse_to_fine=getattr(args, "coarse_to_fine", False),
        coarse_gate_scale=getattr(args, "coarse_gate_scale", 1.0),
        causal=bool(getattr(args, "causal", False) or getattr(args, "architecture", "") == "mstcn_online"),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    set_seed(args.seed, deterministic=bool(getattr(args, "deterministic", False)))

    device, gpu_ids = resolve_device_and_gpu_ids(args)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "cuda":
        if bool(getattr(args, "deterministic", False)):
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
        maybe_apply_gpu_memory_fraction(gpu_ids, args.gpu_mem_limit)

    architecture = getattr(args, "architecture", "window")
    print(f"architecture={architecture}")

    if architecture in ("mstcn", "mstcn_online", "asformer"):
        # ===== MS-TCN / ASFormer full-video path =====
        train_dataset, val_dataset, instrument_class_ids = build_mstcn_datasets(args)
        print(f"mstcn_data: train_videos={len(train_dataset)} val_videos={len(val_dataset)}")
        for i in range(len(train_dataset)):
            print(f"  train video_{train_dataset.video_ids[i]:02d}: {train_dataset.video_lengths[i]} frames")

        class_weights = build_step_class_weights(
            train_dataset, class_balance_beta=getattr(args, "class_balance_beta", 0.0))
        instrument_pos_weight, instrument_positive_counts, instrument_valid_frames = build_instrument_pos_weight(train_dataset)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        if getattr(args, "use_instrument_pos_weight", True) and instrument_pos_weight is not None:
            instrument_pos_weight = instrument_pos_weight.to(device)
        else:
            instrument_pos_weight = None
        _, transition_log_trans, transition_log_trans_np = build_transition_priors_from_training_csv(
            args.csv_root,
            train_video_ids=train_dataset.video_ids,
        )
        position_log_prior_np = None
        if float(getattr(args, "position_prior_weight", 0.0)) > 0.0:
            _, position_log_prior_np = build_position_priors_from_training_csv(
                args.csv_root,
                train_video_ids=train_dataset.video_ids,
                num_bins=int(getattr(args, "position_prior_bins", 8)),
            )

        print(f"instrument_class_ids={instrument_class_ids}")
        print(f"instrument_valid_frames={instrument_valid_frames}")

        loader_kwargs = build_data_loader_kwargs(args)
        # For MS-TCN, override batch_size (typically 1 for full videos)
        loader_kwargs["batch_size"] = int(args.batch_size)
        train_loader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_full_videos, **loader_kwargs)
        val_loader = DataLoader(
            val_dataset, shuffle=False, collate_fn=collate_full_videos, **loader_kwargs)
        print(
            f"dataloader_config: batch_size={args.batch_size} num_workers={loader_kwargs['num_workers']} "
            f"collate=collate_full_videos grad_accum_steps={args.grad_accum_steps}"
        )

        if architecture == "asformer":
            model = build_asformer_model(args, num_instrument_classes=len(instrument_class_ids)).to(device)
        else:
            model = build_mstcn_model(args, num_instrument_classes=len(instrument_class_ids)).to(device)
        online_eval_mode = architecture == "mstcn_online"
        resume_checkpoint = None
        if getattr(args, "resume", None) is not None:
            resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(resume_checkpoint["model_state"])
            print(f"resumed_checkpoint={args.resume} saved_epoch={resume_checkpoint.get('epoch', '?')}")
        print_model_summary(args, model, instrument_class_ids)
        model = wrap_model_data_parallel(model, gpu_ids)

        use_amp, amp_dtype, scaler = build_amp_components(args, device)
        print(
            f"runtime_config: device={device.type} gpu_ids={gpu_ids} amp={use_amp} amp_dtype={args.amp_dtype} "
            f"early_stopping_metric=val_{args.early_stopping_metric}"
        )
        print_resource_snapshot("initial_resources", device, args.gpu_mem_limit, args.ram_limit)
        non_blocking = bool(args.pin_memory and device.type == "cuda")

        # --- Eval-only mode: run one validation pass and exit ---
        if getattr(args, "eval_only", False):
            if getattr(args, "resume", None) is None:
                raise ValueError("--eval-only requires a checkpoint path supplied with --resume")
            model.eval()
            val_metrics = run_mstcn_validation_epoch(
                model=model, loader=val_loader,
                device=device,
                class_weights=class_weights,
                instrument_pos_weight=instrument_pos_weight,
                lambda_bd=args.lambda_bd, lambda_inst=args.lambda_inst,
                lambda_tmse=args.lambda_tmse,
                lambda_late_chain=getattr(args, "late_chain_loss_weight", 0.0),
                lambda_coarse=getattr(args, "coarse_loss_weight", 0.0),
                use_amp=use_amp, amp_dtype=amp_dtype,
                progress_bar=args.progress_bar, log_interval=args.log_interval,
                non_blocking=non_blocking,
                light_post_kernel_size=args.light_post_kernel_size,
                min_duration_frames=args.min_duration_frames,
                instrument_class_ids=instrument_class_ids,
                export_prefix=args.output_dir / "eval_only",
                gpu_mem_limit=args.gpu_mem_limit, ram_limit=args.ram_limit,
                label_smoothing=args.label_smoothing, focal_gamma=args.focal_gamma,
                log_transition_matrix=transition_log_trans_np,
                log_position_prior_matrix=position_log_prior_np,
                position_prior_weight=float(getattr(args, "position_prior_weight", 0.0)),
                late_chain_decoder_guard=float(getattr(args, "late_chain_decoder_guard", 0.0)),
                online_eval=online_eval_mode,
                online_transition_alpha=float(getattr(args, "online_transition_alpha", 1.0)),
                ema_decay=float(getattr(args, "online_ema_decay", 0.0)),
            )
            print(
                f"eval_only: "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} val_edit={val_metrics['edit']:.4f} "
                f"val_macro_jaccard={val_metrics.get('macro_jaccard', 0.0):.4f} "
                f"val_relaxed_acc@10={val_metrics.get('relaxed_acc_at_10', 0.0):.4f} "
                f"val_transition_f1@10={val_metrics.get('transition_f1_at_10', 0.0):.4f} "
                f"val_step_metric={val_metrics['step_metric']:.4f} "
                f"val_online_step_metric={val_metrics.get('online_step_metric', val_metrics['step_metric']):.4f} "
                f"val_viterbi_macro_f1={val_metrics['viterbi_macro_f1']:.4f} "
                f"val_viterbi_macro_jaccard={val_metrics.get('viterbi_macro_jaccard', 0.0):.4f} "
                f"val_viterbi_edit={val_metrics['viterbi_edit']:.4f} "
                f"val_viterbi_transition_f1@10={val_metrics.get('viterbi_transition_f1_at_10', 0.0):.4f} "
                f"val_viterbi_step_metric={val_metrics['viterbi_step_metric']:.4f} "
                f"val_eval_fps={float(val_metrics.get('resource_summary', {}).get('eval_frames_per_second', 0.0)):.2f}"
            )
            print_validation_details(val_metrics)
            local_best_csv = persist_result_tracking(
                output_dir=args.output_dir,
                checkpoint_path=args.resume.resolve(),
                epoch=resolve_checkpoint_epoch(resume_checkpoint),
                metrics=val_metrics,
                args=args,
                result_source="eval_only",
            )
            print(f"eval_result_saved={local_best_csv} best_metrics_summary_csv={args.best_metrics_summary_csv}")
            return

        optimizer, scheduler = build_optimizer_and_scheduler(
            args, unwrap_model(model),
            steps_per_epoch=len(train_loader),
            total_epochs=args.stage_b_epochs,
        )
        trainer = MSTCNTrainer(
            class_weights=class_weights,
            instrument_pos_weight=instrument_pos_weight,
            lambda_bd=args.lambda_bd,
            lambda_inst=args.lambda_inst,
            lambda_tmse=args.lambda_tmse,
            lambda_trans=getattr(args, "lambda_trans", 0.0),
            lambda_late_chain=getattr(args, "late_chain_loss_weight", 0.0),
            lambda_coarse=getattr(args, "coarse_loss_weight", 0.0),
            log_trans=transition_log_trans if getattr(args, "lambda_trans", 0.0) > 0 else None,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            gpu_mem_limit=args.gpu_mem_limit,
            ram_limit=args.ram_limit,
            grad_accum_steps=args.grad_accum_steps,
        )

        best_step_metric = float("-inf")
        epochs_without_improvement = 0
        for epoch in range(1, args.stage_b_epochs + 1):
            epoch_start = perf_counter()
            train_metrics = trainer.run_epoch(
                model=model, loader=train_loader,
                optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, num_epochs=args.stage_b_epochs,
                progress_bar=args.progress_bar, log_interval=args.log_interval,
                non_blocking=non_blocking,
            )
            train_time = perf_counter() - epoch_start
            if device.type == "cuda":
                torch.cuda.empty_cache()

            val_start = perf_counter()
            epoch_export_prefix = (
                args.output_dir / "val_predictions" / f"epoch_{epoch:03d}"
                if args.export_every_epoch else None
            )
            val_metrics = run_mstcn_validation_epoch(
                model=model, loader=val_loader,
                device=device,
                class_weights=class_weights,
                instrument_pos_weight=instrument_pos_weight,
                lambda_bd=args.lambda_bd, lambda_inst=args.lambda_inst,
                lambda_tmse=args.lambda_tmse,
                lambda_late_chain=getattr(args, "late_chain_loss_weight", 0.0),
                lambda_coarse=getattr(args, "coarse_loss_weight", 0.0),
                use_amp=use_amp, amp_dtype=amp_dtype,
                progress_bar=args.progress_bar, log_interval=args.log_interval,
                non_blocking=non_blocking,
                light_post_kernel_size=args.light_post_kernel_size,
                min_duration_frames=args.min_duration_frames,
                instrument_class_ids=instrument_class_ids,
                export_prefix=epoch_export_prefix,
                gpu_mem_limit=args.gpu_mem_limit, ram_limit=args.ram_limit,
                label_smoothing=args.label_smoothing, focal_gamma=args.focal_gamma,
                log_transition_matrix=transition_log_trans_np,
                log_position_prior_matrix=position_log_prior_np,
                position_prior_weight=float(getattr(args, "position_prior_weight", 0.0)),
                late_chain_decoder_guard=float(getattr(args, "late_chain_decoder_guard", 0.0)),
                online_eval=online_eval_mode,
                online_transition_alpha=float(getattr(args, "online_transition_alpha", 1.0)),
                ema_decay=float(getattr(args, "online_ema_decay", 0.0)),
            )
            val_time = perf_counter() - val_start
            epoch_time = perf_counter() - epoch_start
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print_resource_snapshot(f"epoch_{epoch}_resources", device, args.gpu_mem_limit, args.ram_limit)
            print(
                f"epoch={epoch}/{args.stage_b_epochs} epoch_time_min={epoch_time/60.0:.2f} "
                f"train_time_min={train_time/60.0:.2f} val_time_min={val_time/60.0:.2f} "
                f"{current_lr_summary(optimizer)} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
                f"train_step_loss={train_metrics['step_loss']:.4f} "
                f"train_coarse_loss={train_metrics['coarse_loss']:.4f} "
                f"train_late_chain_loss={train_metrics['late_chain_loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"val_coarse_loss={val_metrics['coarse_loss']:.4f} "
                f"val_late_chain_loss={val_metrics['late_chain_loss']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} val_edit={val_metrics['edit']:.4f} "
                f"val_macro_jaccard={val_metrics.get('macro_jaccard', 0.0):.4f} "
                f"val_relaxed_acc@10={val_metrics.get('relaxed_acc_at_10', 0.0):.4f} "
                f"val_transition_f1@10={val_metrics.get('transition_f1_at_10', 0.0):.4f} "
                f"val_f1@10={val_metrics['f1_at_10']:.4f} val_f1@25={val_metrics['f1_at_25']:.4f} "
                f"val_f1@50={val_metrics['f1_at_50']:.4f} "
                f"val_step_metric={val_metrics['step_metric']:.4f} "
                f"val_online_step_metric={val_metrics.get('online_step_metric', val_metrics['step_metric']):.4f} "
                f"val_light_post_step_metric={val_metrics['light_post_step_metric']:.4f} "
                f"val_mindur_post_step_metric={val_metrics['mindur_post_step_metric']:.4f} "
                f"val_viterbi_macro_f1={val_metrics['viterbi_macro_f1']:.4f} "
                f"val_viterbi_macro_jaccard={val_metrics.get('viterbi_macro_jaccard', 0.0):.4f} "
                f"val_viterbi_edit={val_metrics['viterbi_edit']:.4f} "
                f"val_viterbi_f1@10={val_metrics['viterbi_f1_at_10']:.4f} "
                f"val_viterbi_f1@25={val_metrics['viterbi_f1_at_25']:.4f} "
                f"val_viterbi_f1@50={val_metrics['viterbi_f1_at_50']:.4f} "
                f"val_viterbi_transition_f1@10={val_metrics.get('viterbi_transition_f1_at_10', 0.0):.4f} "
                f"val_viterbi_step_metric={val_metrics['viterbi_step_metric']:.4f} "
                f"val_eval_fps={float(val_metrics.get('resource_summary', {}).get('eval_frames_per_second', 0.0)):.2f} "
                f"val_instrument_micro_f1={val_metrics['instrument_micro_f1']:.4f} "
                f"val_instrument_macro_f1={val_metrics['instrument_macro_f1']:.4f}"
            )
            print_validation_details(val_metrics)

            es_metric = resolve_selection_metric(val_metrics, args.early_stopping_metric)
            if es_metric > best_step_metric:
                best_step_metric = es_metric
                epochs_without_improvement = 0
                checkpoint_path = save_best_checkpoint(
                    output_dir=args.output_dir, model=model, optimizer=optimizer,
                    epoch=epoch, metrics=val_metrics, args=args,
                )
                best_export_prefix = args.output_dir / "best_epoch"
                best_metrics = run_mstcn_validation_epoch(
                    model=model, loader=val_loader,
                    device=device,
                    class_weights=class_weights,
                    instrument_pos_weight=instrument_pos_weight,
                    lambda_bd=args.lambda_bd, lambda_inst=args.lambda_inst,
                    lambda_tmse=args.lambda_tmse,
                    lambda_late_chain=getattr(args, "late_chain_loss_weight", 0.0),
                    lambda_coarse=getattr(args, "coarse_loss_weight", 0.0),
                    use_amp=use_amp, amp_dtype=amp_dtype,
                    progress_bar=False, log_interval=args.log_interval,
                    non_blocking=non_blocking,
                    light_post_kernel_size=args.light_post_kernel_size,
                    min_duration_frames=args.min_duration_frames,
                    instrument_class_ids=instrument_class_ids,
                    export_prefix=best_export_prefix,
                    gpu_mem_limit=args.gpu_mem_limit, ram_limit=args.ram_limit,
                    label_smoothing=args.label_smoothing, focal_gamma=args.focal_gamma,
                    log_transition_matrix=transition_log_trans_np,
                    log_position_prior_matrix=position_log_prior_np,
                    position_prior_weight=float(getattr(args, "position_prior_weight", 0.0)),
                    late_chain_decoder_guard=float(getattr(args, "late_chain_decoder_guard", 0.0)),
                    online_eval=online_eval_mode,
                    online_transition_alpha=float(getattr(args, "online_transition_alpha", 1.0)),
                    ema_decay=float(getattr(args, "online_ema_decay", 0.0)),
                )
                local_best_csv = persist_result_tracking(
                    output_dir=args.output_dir,
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    metrics=best_metrics,
                    args=args,
                    result_source="best_epoch",
                )
                print(
                    f"best_checkpoint_saved={checkpoint_path} "
                    f"best_val_{args.early_stopping_metric}={best_step_metric:.4f}"
                )
                print(f"best_metrics_saved={local_best_csv} best_metrics_summary_csv={args.best_metrics_summary_csv}")
            else:
                epochs_without_improvement += 1
                if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                    print(
                        f"early_stopping_triggered epoch={epoch} "
                        f"best_val_{args.early_stopping_metric}={best_step_metric:.4f}"
                    )
                    break
        return

    # ===== Original window-based path =====
    stage_b_train_dataset, val_dataset, instrument_class_ids = build_pitvis_datasets(args, window_size=args.window_size, stride=args.stride)
    stage_a_train_dataset = None
    if args.stage_a_epochs > 0:
        stage_a_train_dataset, _, _ = build_pitvis_datasets(
            args,
            window_size=args.stage_a_window_size,
            stride=args.stage_a_stride,
        )

    class_weights = build_step_class_weights(
        stage_b_train_dataset, class_balance_beta=getattr(args, "class_balance_beta", 0.0))
    instrument_pos_weight, instrument_positive_counts, instrument_valid_frames = build_instrument_pos_weight(stage_b_train_dataset)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if args.use_instrument_pos_weight and instrument_pos_weight is not None:
        instrument_pos_weight = instrument_pos_weight.to(device)
    else:
        instrument_pos_weight = None
    _, _, transition_log_trans_np = build_transition_priors_from_training_csv(
        args.csv_root,
        train_video_ids=stage_b_train_dataset.video_ids,
    )
    position_log_prior_np = None
    if float(getattr(args, "position_prior_weight", 0.0)) > 0.0:
        _, position_log_prior_np = build_position_priors_from_training_csv(
            args.csv_root,
            train_video_ids=stage_b_train_dataset.video_ids,
            num_bins=int(getattr(args, "position_prior_bins", 8)),
        )

    print(f"instrument_class_ids={instrument_class_ids}")
    print(f"instrument_positive_counts={[float(v) for v in instrument_positive_counts.tolist()]}")
    if instrument_pos_weight is not None:
        print(f"instrument_pos_weight={[float(v) for v in instrument_pos_weight.detach().cpu().tolist()]}")
    print(f"instrument_valid_frames={instrument_valid_frames}")

    loader_kwargs = build_data_loader_kwargs(args)
    print(
        "dataloader_config: "
        f"batch_size={args.batch_size} num_workers={loader_kwargs['num_workers']} "
        f"pin_memory={loader_kwargs['pin_memory']} "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')} "
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)}"
    )
    stage_b_sample_weights = build_sample_weights_for_balanced_sampling(stage_b_train_dataset)
    stage_b_sampler = WeightedRandomSampler(stage_b_sample_weights, num_samples=len(stage_b_sample_weights), replacement=True)
    print(f"balanced_sampling: min_w={stage_b_sample_weights.min():.3f} max_w={stage_b_sample_weights.max():.3f} mean_w={stage_b_sample_weights.mean():.3f}")
    stage_b_train_loader = DataLoader(stage_b_train_dataset, sampler=stage_b_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    stage_a_train_loader = DataLoader(stage_a_train_dataset, shuffle=True, **loader_kwargs) if stage_a_train_dataset is not None else None

    model = build_model(args, num_instrument_classes=len(instrument_class_ids)).to(device)
    if args.input_mode == "images" and model.backbone_wrapper is not None and args.channels_last:
        model.backbone_wrapper.backbone.to(memory_format=torch.channels_last)
    print_model_summary(args, model, instrument_class_ids)
    model = wrap_model_data_parallel(model, gpu_ids)

    use_amp, amp_dtype, scaler = build_amp_components(args, device)
    print(
        f"runtime_config: device={device.type} gpu_ids={gpu_ids} amp={use_amp} amp_dtype={args.amp_dtype} "
        f"channels_last={args.channels_last} progress_bar={args.progress_bar} "
        f"gpu_mem_limit={args.gpu_mem_limit*100:.0f}% ram_limit={args.ram_limit*100:.0f}% "
        f"early_stopping_metric=val_{args.early_stopping_metric}"
    )
    print_resource_snapshot("initial_resources", device, args.gpu_mem_limit, args.ram_limit)

    non_blocking = bool(args.pin_memory and device.type == "cuda")
    if stage_a_train_loader is not None:
        stage_a_optimizer, stage_a_scheduler = build_optimizer_and_scheduler(
            args,
            unwrap_model(model),
            steps_per_epoch=len(stage_a_train_loader),
            total_epochs=args.stage_a_epochs,
        )
        stage_a_trainer = StageATrainer(
            class_weights=class_weights,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            gpu_mem_limit=args.gpu_mem_limit,
            ram_limit=args.ram_limit,
        )
        for epoch in range(1, args.stage_a_epochs + 1):
            metrics = stage_a_trainer.run_epoch(
                model=model,
                loader=stage_a_train_loader,
                optimizer=stage_a_optimizer,
                scheduler=stage_a_scheduler,
                epoch=epoch,
                num_epochs=args.stage_a_epochs,
                progress_bar=args.progress_bar,
                log_interval=args.log_interval,
                non_blocking=non_blocking,
            )
            print(
                f"stageA_epoch={epoch}/{args.stage_a_epochs} {current_lr_summary(stage_a_optimizer)} "
                f"loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stage_b_optimizer, stage_b_scheduler = build_optimizer_and_scheduler(
        args,
        unwrap_model(model),
        steps_per_epoch=len(stage_b_train_loader),
        total_epochs=args.stage_b_epochs,
    )
    stage_b_trainer = StageBTrainer(
        class_weights=class_weights,
        instrument_pos_weight=instrument_pos_weight,
        lambda_bd=args.lambda_bd,
        lambda_inst=args.lambda_inst,
        lambda_tmse=args.lambda_tmse,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        lambda_refine_init=args.lambda_refine_init,
        mixup_alpha=args.mixup_alpha,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=scaler,
        gpu_mem_limit=args.gpu_mem_limit,
        ram_limit=args.ram_limit,
        grad_accum_steps=getattr(args, "grad_accum_steps", 1),
    )

    best_step_metric = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.stage_b_epochs + 1):
        epoch_start = perf_counter()
        train_start = perf_counter()
        train_metrics = stage_b_trainer.run_epoch(
            model=model,
            loader=stage_b_train_loader,
            optimizer=stage_b_optimizer,
            scheduler=stage_b_scheduler,
            epoch=epoch,
            num_epochs=args.stage_b_epochs,
            progress_bar=args.progress_bar,
            log_interval=args.log_interval,
            non_blocking=non_blocking,
        )
        train_time = perf_counter() - train_start
        if device.type == "cuda":
            torch.cuda.empty_cache()

        val_start = perf_counter()
        epoch_export_prefix = (
            args.output_dir / "val_predictions" / f"epoch_{epoch:03d}"
            if args.export_every_epoch else None
        )
        val_metrics = run_validation_epoch(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            instrument_pos_weight=instrument_pos_weight,
            lambda_bd=args.lambda_bd,
            lambda_inst=args.lambda_inst,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            progress_bar=args.progress_bar,
            log_interval=args.log_interval,
            non_blocking=non_blocking,
            light_post_kernel_size=args.light_post_kernel_size,
            min_duration_frames=args.min_duration_frames,
            instrument_class_ids=instrument_class_ids,
            export_prefix=epoch_export_prefix,
            gpu_mem_limit=args.gpu_mem_limit,
            ram_limit=args.ram_limit,
            log_transition_matrix=transition_log_trans_np,
            log_position_prior_matrix=position_log_prior_np,
            position_prior_weight=float(getattr(args, "position_prior_weight", 0.0)),
        )
        val_time = perf_counter() - val_start
        epoch_time = perf_counter() - epoch_start
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print_resource_snapshot(f"epoch_{epoch}_resources", device, args.gpu_mem_limit, args.ram_limit)
        print(
            f"epoch={epoch}/{args.stage_b_epochs} epoch_time_min={epoch_time/60.0:.2f} "
            f"train_time_min={train_time/60.0:.2f} val_time_min={val_time/60.0:.2f} "
            f"{current_lr_summary(stage_b_optimizer)} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_step_loss={train_metrics['step_loss']:.4f} train_boundary_loss={train_metrics['boundary_loss']:.4f} "
            f"train_instrument_loss={train_metrics['instrument_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_step_loss={val_metrics['step_loss']:.4f} val_boundary_loss={val_metrics['boundary_loss']:.4f} "
            f"val_instrument_loss={val_metrics['instrument_loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_edit={val_metrics['edit']:.4f} "
            f"val_macro_jaccard={val_metrics.get('macro_jaccard', 0.0):.4f} "
            f"val_relaxed_acc@10={val_metrics.get('relaxed_acc_at_10', 0.0):.4f} "
            f"val_transition_f1@10={val_metrics.get('transition_f1_at_10', 0.0):.4f} "
            f"val_f1@10={val_metrics['f1_at_10']:.4f} val_f1@25={val_metrics['f1_at_25']:.4f} "
            f"val_f1@50={val_metrics['f1_at_50']:.4f} "
            f"val_step_metric={val_metrics['step_metric']:.4f} "
            f"val_light_post_step_metric={val_metrics['light_post_step_metric']:.4f} "
            f"val_mindur_post_step_metric={val_metrics['mindur_post_step_metric']:.4f} "
            f"val_viterbi_macro_f1={val_metrics['viterbi_macro_f1']:.4f} "
            f"val_viterbi_macro_jaccard={val_metrics.get('viterbi_macro_jaccard', 0.0):.4f} "
            f"val_viterbi_edit={val_metrics['viterbi_edit']:.4f} "
            f"val_viterbi_f1@10={val_metrics['viterbi_f1_at_10']:.4f} "
            f"val_viterbi_f1@25={val_metrics['viterbi_f1_at_25']:.4f} "
            f"val_viterbi_f1@50={val_metrics['viterbi_f1_at_50']:.4f} "
            f"val_viterbi_transition_f1@10={val_metrics.get('viterbi_transition_f1_at_10', 0.0):.4f} "
            f"val_viterbi_step_metric={val_metrics['viterbi_step_metric']:.4f} "
            f"val_eval_fps={float(val_metrics.get('resource_summary', {}).get('eval_frames_per_second', 0.0)):.2f} "
            f"val_instrument_micro_f1={val_metrics['instrument_micro_f1']:.4f} "
            f"val_instrument_macro_f1={val_metrics['instrument_macro_f1']:.4f}"
        )
        print_validation_details(val_metrics)

        es_metric = resolve_selection_metric(val_metrics, args.early_stopping_metric)
        if es_metric > best_step_metric:
            best_step_metric = es_metric
            epochs_without_improvement = 0
            checkpoint_path = save_best_checkpoint(
                output_dir=args.output_dir,
                model=model,
                optimizer=stage_b_optimizer,
                epoch=epoch,
                metrics=val_metrics,
                args=args,
            )
            best_export_prefix = args.output_dir / "best_epoch"
            best_metrics = run_validation_epoch(
                model=model,
                loader=val_loader,
                device=device,
                class_weights=class_weights,
                instrument_pos_weight=instrument_pos_weight,
                lambda_bd=args.lambda_bd,
                lambda_inst=args.lambda_inst,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                progress_bar=False,
                log_interval=args.log_interval,
                non_blocking=non_blocking,
                light_post_kernel_size=args.light_post_kernel_size,
                min_duration_frames=args.min_duration_frames,
                instrument_class_ids=instrument_class_ids,
                export_prefix=best_export_prefix,
                gpu_mem_limit=args.gpu_mem_limit,
                ram_limit=args.ram_limit,
                log_transition_matrix=transition_log_trans_np,
                log_position_prior_matrix=position_log_prior_np,
                position_prior_weight=float(getattr(args, "position_prior_weight", 0.0)),
            )
            local_best_csv = persist_result_tracking(
                output_dir=args.output_dir,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                metrics=best_metrics,
                args=args,
                result_source="best_epoch",
            )
            print(
                f"best_checkpoint_saved={checkpoint_path} "
                f"best_val_{args.early_stopping_metric}={best_step_metric:.4f}"
            )
            print(f"best_metrics_saved={local_best_csv} best_metrics_summary_csv={args.best_metrics_summary_csv}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"early_stopping_triggered epoch={epoch} "
                    f"best_val_{args.early_stopping_metric}={best_step_metric:.4f}"
                )
                break


if __name__ == "__main__":
    main()
