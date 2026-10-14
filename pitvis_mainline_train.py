import math
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data import DataLoader

from pitvis_mainline_common import GRAD_CLIP_MAX_NORM, IGNORE_INDEX, LABELS12, check_resources_and_warn
from pitvis_mainline_data import build_center_frame_weights
from pitvis_mainline_model import LATE_CHAIN_CLASS_INDICES, LATE_CHAIN_RAW_LABELS
from pitvis_coarse_to_fine import build_fine_to_coarse_index

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


COARSE_STAGE_INDEX_BY_FINE_CLASS = tuple(build_fine_to_coarse_index(LABELS12))


def maybe_make_progress_bar(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def current_lr_summary(optimizer: torch.optim.Optimizer) -> str:
    lrs = [group["lr"] for group in optimizer.param_groups]
    if len(lrs) == 1:
        return f"lr={lrs[0]:.2e}"
    return " ".join(f"lr{idx}={value:.2e}" for idx, value in enumerate(lrs))


def build_instrument_pos_weight(dataset, clip_max: float = 20.0) -> Tuple[Optional[torch.Tensor], torch.Tensor, int]:
    if not getattr(dataset, "video_instrument_targets", None):
        empty = torch.empty(0, dtype=torch.float32)
        return None, empty, 0
    positive_counts = torch.zeros(dataset.video_instrument_targets[0].shape[1], dtype=torch.float32)
    valid_frame_count = 0
    for targets, valid_mask in zip(dataset.video_instrument_targets, dataset.video_instrument_valid_masks):
        valid_targets = targets[valid_mask]
        if valid_targets.numel() == 0:
            continue
        positive_counts += valid_targets.sum(dim=0)
        valid_frame_count += int(valid_targets.shape[0])
    if valid_frame_count == 0:
        return None, positive_counts, valid_frame_count
    negative_counts = float(valid_frame_count) - positive_counts
    pos_weight = negative_counts / positive_counts.clamp_min(1.0)
    pos_weight = pos_weight.clamp(min=1.0, max=float(clip_max))
    return pos_weight, positive_counts, valid_frame_count


def build_amp_components(args, device: torch.device):
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp and amp_dtype == torch.float16 else None
    return use_amp, amp_dtype, scaler


def build_optimizer_and_scheduler(
    args,
    model: torch.nn.Module,
    *,
    steps_per_epoch: int,
    total_epochs: int,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LambdaLR]]:
    if hasattr(model, "backbone_wrapper") and model.backbone_wrapper is not None:
        backbone_params = [param for param in model.backbone_wrapper.parameters() if param.requires_grad]
        head_params = [
            param
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith("backbone_wrapper.")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.backbone_lr},
                {"params": head_params, "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )
        print(
            f"optimizer_lr_groups: backbone_lr={args.backbone_lr:.2e} "
            f"head_lr={args.head_lr:.2e} weight_decay={args.weight_decay:.2e}"
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"optimizer_lr_groups: lr={args.lr:.2e} weight_decay={args.weight_decay:.2e}")

    total_steps = max(1, int(total_epochs) * max(1, steps_per_epoch))
    warmup_steps = int(float(args.warmup_ratio) * total_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def _masked_mean_loss(values: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    if weights is None:
        return values.mean()
    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


def compute_step_ar_loss(
    step_logits: torch.Tensor,
    step_targets: torch.Tensor,
    *,
    class_weights: Optional[torch.Tensor] = None,
    frame_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> Dict[str, torch.Tensor]:
    valid_mask = step_targets.ne(IGNORE_INDEX)
    zero = step_logits.sum() * 0.0
    if not bool(valid_mask.any()):
        return {
            "step_loss": zero,
            "step_ce_unweighted": zero,
            "step_valid_count": zero,
            "step_frame_weight_sum": zero,
            "step_ce_unweighted_sum": zero,
            "step_ce_weighted_sum": zero,
        }
    valid_logits = step_logits[valid_mask].float()
    valid_targets = step_targets[valid_mask]
    class_weight_tensor = class_weights.float() if class_weights is not None else None
    per_frame_ce = F.cross_entropy(valid_logits, valid_targets, weight=class_weight_tensor, reduction="none", label_smoothing=float(label_smoothing))
    if focal_gamma > 0:
        with torch.no_grad():
            p_t = torch.exp(-F.cross_entropy(valid_logits, valid_targets, reduction="none"))
        focal_weight = (1.0 - p_t) ** focal_gamma
        per_frame_ce = focal_weight * per_frame_ce
    if frame_weights is None:
        expanded_weights = torch.ones(step_targets.shape, dtype=torch.float32, device=step_logits.device)
    elif frame_weights.dim() == 1:
        expanded_weights = frame_weights.unsqueeze(0).expand(step_targets.shape[0], -1)
    else:
        expanded_weights = frame_weights
    valid_frame_weights = expanded_weights[valid_mask].to(dtype=per_frame_ce.dtype, device=per_frame_ce.device)
    ce_unweighted_sum = per_frame_ce.sum()
    ce_weighted_sum = (per_frame_ce * valid_frame_weights).sum()
    step_loss = ce_weighted_sum / valid_frame_weights.sum().clamp_min(1e-8)
    step_ce_unweighted = ce_unweighted_sum / valid_mask.sum().to(dtype=per_frame_ce.dtype).clamp_min(1.0)
    return {
        "step_loss": step_loss,
        "step_ce_unweighted": step_ce_unweighted,
        "step_valid_count": valid_mask.sum().to(dtype=per_frame_ce.dtype),
        "step_frame_weight_sum": valid_frame_weights.sum(),
        "step_ce_unweighted_sum": ce_unweighted_sum,
        "step_ce_weighted_sum": ce_weighted_sum,
    }


def compute_boundary_loss(
    boundary_logits: torch.Tensor,
    boundary_targets: torch.Tensor,
    *,
    boundary_loss_weights: torch.Tensor,
    padding_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if boundary_logits.dim() == 2:
        boundary_logits = boundary_logits.unsqueeze(-1)
    if boundary_targets.dim() == 2:
        boundary_targets = boundary_targets.unsqueeze(-1)
    if boundary_loss_weights.dim() == 2:
        boundary_loss_weights = boundary_loss_weights.unsqueeze(-1)
    valid_mask = ~padding_mask
    zero = boundary_logits.sum() * 0.0
    if not bool(valid_mask.any()):
        return {
            "boundary_loss": zero,
            "boundary_valid_count": zero,
            "boundary_weight_sum": zero,
            "boundary_loss_sum": zero,
        }
    expanded_mask = valid_mask.unsqueeze(-1).expand_as(boundary_logits)
    valid_logits = boundary_logits[expanded_mask].float()
    valid_targets = boundary_targets[expanded_mask].float()
    valid_weights = boundary_loss_weights[expanded_mask].float().to(device=valid_logits.device)
    per_frame_bce = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction="none")
    loss_sum = (per_frame_bce * valid_weights).sum()
    loss = loss_sum / valid_weights.sum().clamp_min(1e-8)
    return {
        "boundary_loss": loss,
        "boundary_valid_count": valid_mask.sum().to(dtype=valid_logits.dtype),
        "boundary_weight_sum": valid_weights.sum(),
        "boundary_loss_sum": loss_sum,
    }


def compute_instrument_loss(
    instrument_logits: torch.Tensor,
    instrument_targets: torch.Tensor,
    *,
    instrument_valid_mask: torch.Tensor,
    instrument_pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    zero = instrument_logits.sum() * 0.0
    if not bool(instrument_valid_mask.any()):
        return {
            "instrument_loss": zero,
            "instrument_valid_count": zero,
            "instrument_loss_sum": zero,
        }
    valid_logits = instrument_logits[instrument_valid_mask].float()
    valid_targets = instrument_targets[instrument_valid_mask].float()
    pos_weight = instrument_pos_weight.float().to(valid_logits.device) if instrument_pos_weight is not None else None
    per_class_bce = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    loss_sum = per_class_bce.sum()
    loss = loss_sum / valid_logits.shape[0] / max(1, valid_logits.shape[1])
    return {
        "instrument_loss": loss,
        "instrument_valid_count": torch.tensor(float(valid_logits.shape[0]), device=valid_logits.device),
        "instrument_loss_sum": loss_sum,
    }


def compute_tmse_loss(
    step_logits: torch.Tensor,
    *,
    padding_mask: torch.Tensor,
    clamp_max: float = 16.0,
) -> torch.Tensor:
    """Truncated Mean Squared Error on log-probabilities between adjacent frames (MS-TCN++)."""
    log_probs = F.log_softmax(step_logits.float(), dim=-1)  # [B, L, C]
    diff = (log_probs[:, 1:, :] - log_probs[:, :-1, :]) ** 2  # [B, L-1, C]
    diff = torch.clamp(diff, min=0.0, max=float(clamp_max))
    # mask out diffs involving padded frames
    valid_mask = (~padding_mask[:, 1:]) & (~padding_mask[:, :-1])  # [B, L-1]
    if not bool(valid_mask.any()):
        return step_logits.sum() * 0.0
    masked_diff = diff[valid_mask]  # [N, C]
    return masked_diff.mean()


def compute_transition_loss(
    step_logits: torch.Tensor,
    *,
    step_targets: torch.Tensor,
    padding_mask: torch.Tensor,
    log_trans: torch.Tensor,
) -> torch.Tensor:
    """Boundary-only transition-aware loss.

    Only fires at GT transition frames (where the ground-truth label changes).
    At each boundary t where GT transitions from class a to class b:
        q(j) = sum_i  p_{t-1}(i) * T_change[i, j]
        loss_t = -log q(b)

    T_change is the transition matrix with self-loops removed and renormalized,
    so it only models the *direction* of transitions, not whether to transition.

    This encourages p_{t-1} to peak on classes that legally lead to y_t,
    teaching the model to "prepare" for upcoming phase changes.
    """
    C = step_logits.shape[-1]
    probs = F.softmax(step_logits.float(), dim=-1)  # [B, L, C]

    # Build change-only transition matrix (remove self-transitions, renormalize)
    trans = log_trans.to(device=step_logits.device, dtype=torch.float32).exp()  # [C, C]
    change_trans = trans.clone()
    change_trans.fill_diagonal_(0.0)
    change_trans = change_trans / change_trans.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # Find GT boundary positions
    prev_targets = step_targets[:, :-1]   # [B, L-1]
    curr_targets = step_targets[:, 1:]    # [B, L-1]
    valid = (~padding_mask[:, 1:]) & (~padding_mask[:, :-1])
    valid = valid & prev_targets.ne(IGNORE_INDEX) & curr_targets.ne(IGNORE_INDEX)
    is_boundary = valid & (prev_targets != curr_targets)

    if not bool(is_boundary.any()):
        return step_logits.sum() * 0.0

    # p_{t-1} at boundary frames -> expected successor distribution
    p_prev = probs[:, :-1, :][is_boundary]     # [N, C]
    q = p_prev @ change_trans                   # [N, C]

    # Target = GT class at current frame
    target_idx = curr_targets[is_boundary]      # [N]

    # -log q(target)
    q_target = q[torch.arange(q.shape[0], device=q.device), target_idx]
    loss = -torch.log(q_target.clamp_min(1e-8)).mean()
    return loss


def build_coarse_targets(step_targets: torch.Tensor) -> torch.Tensor:
    coarse_targets = torch.full_like(step_targets, IGNORE_INDEX)
    valid_mask = step_targets.ne(IGNORE_INDEX)
    if not bool(valid_mask.any()):
        return coarse_targets
    mapping = torch.tensor(
        COARSE_STAGE_INDEX_BY_FINE_CLASS,
        dtype=torch.long,
        device=step_targets.device,
    )
    coarse_targets[valid_mask] = mapping[step_targets[valid_mask]]
    return coarse_targets


def compute_coarse_loss(
    coarse_logits: Optional[torch.Tensor],
    *,
    step_targets: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    zero = step_targets.sum(dtype=torch.float32) * 0.0
    if coarse_logits is None:
        return zero
    coarse_targets = build_coarse_targets(step_targets)
    valid_mask = (~padding_mask) & coarse_targets.ne(IGNORE_INDEX)
    if not bool(valid_mask.any()):
        return zero.to(device=coarse_logits.device, dtype=coarse_logits.dtype)
    valid_logits = coarse_logits[valid_mask].float()
    valid_targets = coarse_targets[valid_mask]
    return F.cross_entropy(valid_logits, valid_targets)


def compute_late_chain_loss(
    outputs: Dict[str, torch.Tensor],
    step_targets: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    late_chain_raw_labels=LATE_CHAIN_RAW_LABELS,
) -> torch.Tensor:
    late_chain_logits = outputs.get("late_chain_logits")
    step_logits = outputs.get("step_logits")
    if torch.is_tensor(step_logits):
        zero = step_logits.sum() * 0.0
    elif torch.is_tensor(late_chain_logits):
        zero = late_chain_logits.sum() * 0.0
    else:
        zero = step_targets.sum(dtype=torch.float32) * 0.0
    if late_chain_logits is None:
        return zero

    late_chain_class_indices = tuple(int(idx) for idx in LATE_CHAIN_CLASS_INDICES)
    late_chain_raw_labels = tuple(int(label) for label in late_chain_raw_labels)
    class_index_lookup = {class_idx: subset_idx for subset_idx, class_idx in enumerate(late_chain_class_indices)}
    raw_to_subset_index = {
        int(raw_label): class_index_lookup[class_idx]
        for raw_label, class_idx in zip(late_chain_raw_labels, late_chain_class_indices)
    }
    subset_targets = torch.full_like(step_targets, IGNORE_INDEX)
    for raw_label in late_chain_raw_labels:
        class_idx = late_chain_class_indices[late_chain_raw_labels.index(raw_label)]
        subset_idx = raw_to_subset_index[raw_label]
        subset_targets = torch.where(step_targets.eq(class_idx), torch.full_like(step_targets, subset_idx), subset_targets)

    late_valid_mask = valid_mask & subset_targets.ne(IGNORE_INDEX)
    if not bool(late_valid_mask.any()):
        return zero
    valid_logits = late_chain_logits[late_valid_mask].float()
    valid_targets = subset_targets[late_valid_mask]
    return F.cross_entropy(valid_logits, valid_targets)


def _compute_step_soft_loss(
    step_logits: torch.Tensor,
    step_targets: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    frame_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """KL-divergence loss for mixup soft targets."""
    valid_mask = step_targets.ne(IGNORE_INDEX)
    zero = step_logits.sum() * 0.0
    if not bool(valid_mask.any()):
        return {
            "step_loss": zero,
            "step_ce_unweighted": zero,
            "step_valid_count": zero,
            "step_frame_weight_sum": zero,
            "step_ce_unweighted_sum": zero,
            "step_ce_weighted_sum": zero,
        }
    valid_logits = step_logits[valid_mask].float()
    valid_soft = soft_targets[valid_mask].float()
    # Normalize soft targets to sum to 1
    valid_soft = valid_soft / valid_soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(valid_logits, dim=-1)
    per_frame_kl = F.kl_div(log_probs, valid_soft, reduction="none").sum(dim=-1)
    if frame_weights is None:
        expanded_weights = torch.ones(step_targets.shape, dtype=torch.float32, device=step_logits.device)
    elif frame_weights.dim() == 1:
        expanded_weights = frame_weights.unsqueeze(0).expand(step_targets.shape[0], -1)
    else:
        expanded_weights = frame_weights
    valid_frame_weights = expanded_weights[valid_mask].to(dtype=per_frame_kl.dtype, device=per_frame_kl.device)
    ce_unweighted_sum = per_frame_kl.sum()
    ce_weighted_sum = (per_frame_kl * valid_frame_weights).sum()
    step_loss = ce_weighted_sum / valid_frame_weights.sum().clamp_min(1e-8)
    step_ce_unweighted = ce_unweighted_sum / valid_mask.sum().to(dtype=per_frame_kl.dtype).clamp_min(1.0)
    return {
        "step_loss": step_loss,
        "step_ce_unweighted": step_ce_unweighted,
        "step_valid_count": valid_mask.sum().to(dtype=per_frame_kl.dtype),
        "step_frame_weight_sum": valid_frame_weights.sum(),
        "step_ce_unweighted_sum": ce_unweighted_sum,
        "step_ce_weighted_sum": ce_weighted_sum,
    }


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    class_weights: Optional[torch.Tensor] = None,
    frame_weights: Optional[torch.Tensor] = None,
    instrument_pos_weight: Optional[torch.Tensor] = None,
    lambda_bd: float = 0.2,
    lambda_inst: float = 0.1,
    lambda_tmse: float = 0.15,
    lambda_refine_init: float = 0.3,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> Dict[str, torch.Tensor]:
    # Use soft-target KL loss if mixup was applied, otherwise standard CE
    if "step_targets_soft" in batch:
        step_losses = _compute_step_soft_loss(
            outputs["step_logits"],
            batch["step_targets"],
            batch["step_targets_soft"],
            frame_weights=frame_weights,
        )
    else:
        step_losses = compute_step_ar_loss(
            outputs["step_logits"],
            batch["step_targets"],
            class_weights=class_weights,
            frame_weights=frame_weights,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
    boundary_losses = compute_boundary_loss(
        outputs["boundary_logits"],
        batch["boundary_targets"],
        boundary_loss_weights=batch["boundary_loss_weights"],
        padding_mask=batch["padding_mask"],
    )
    instrument_losses = compute_instrument_loss(
        outputs["instrument_logits"],
        batch["instrument_targets"],
        instrument_valid_mask=batch["instrument_valid_mask"],
        instrument_pos_weight=instrument_pos_weight,
    )
    tmse_loss = compute_tmse_loss(
        outputs["step_logits"],
        padding_mask=batch["padding_mask"],
    )
    total_loss = (
        step_losses["step_loss"]
        + float(lambda_bd) * boundary_losses["boundary_loss"]
        + float(lambda_inst) * instrument_losses["instrument_loss"]
        + float(lambda_tmse) * tmse_loss
    )
    # Auxiliary loss on initial (pre-TCN) logits to keep the parallel head trained
    init_loss = outputs["step_logits"].sum() * 0.0  # zero by default
    if "step_logits_init" in outputs and float(lambda_refine_init) > 0:
        if "step_targets_soft" in batch:
            init_losses = _compute_step_soft_loss(
                outputs["step_logits_init"],
                batch["step_targets"],
                batch["step_targets_soft"],
                frame_weights=frame_weights,
            )
        else:
            init_losses = compute_step_ar_loss(
                outputs["step_logits_init"],
                batch["step_targets"],
                class_weights=class_weights,
                frame_weights=frame_weights,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
            )
        init_loss = init_losses["step_loss"]
        total_loss = total_loss + float(lambda_refine_init) * init_loss
    return {
        "loss": total_loss,
        "total_loss": total_loss,
        "tmse_loss": tmse_loss,
        "init_loss": init_loss,
        **step_losses,
        **boundary_losses,
        **instrument_losses,
    }


def compute_step_accuracy(step_logits: torch.Tensor, step_targets: torch.Tensor) -> float:
    valid_mask = step_targets.ne(IGNORE_INDEX)
    if not bool(valid_mask.any()):
        return 0.0
    predictions = step_logits.argmax(dim=-1)
    correct = ((predictions == step_targets) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return float(correct / max(1, total))


def select_center_supervision_targets(step_targets: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = step_targets.shape
    center_idx = seq_len // 2
    center_targets = torch.full((batch_size,), IGNORE_INDEX, dtype=torch.long, device=step_targets.device)
    valid_samples = torch.zeros((batch_size,), dtype=torch.bool, device=step_targets.device)
    for batch_idx in range(batch_size):
        candidate_positions = list(range(seq_len))
        candidate_positions.sort(key=lambda pos: (abs(pos - center_idx), pos))
        for pos in candidate_positions:
            if bool(padding_mask[batch_idx, pos].item()):
                continue
            target = int(step_targets[batch_idx, pos].item())
            if 0 <= target < 12:
                center_targets[batch_idx] = target
                valid_samples[batch_idx] = True
                break
    return center_targets, valid_samples


class StageATrainer:
    def __init__(
        self,
        *,
        class_weights: Optional[torch.Tensor],
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        scaler: Optional[torch.cuda.amp.GradScaler],
        gpu_mem_limit: float,
        ram_limit: float,
    ):
        self.class_weights = class_weights
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.scaler = scaler
        self.gpu_mem_limit = gpu_mem_limit
        self.ram_limit = ram_limit

    def run_epoch(
        self,
        *,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        num_epochs: int,
        progress_bar: bool,
        log_interval: int,
        non_blocking: bool,
    ) -> Dict[str, float]:
        model.train()
        loss_sum = 0.0
        correct_sum = 0
        valid_sum = 0
        iterator = maybe_make_progress_bar(loader, total=len(loader), desc=f"stageA {epoch}/{num_epochs}", enabled=progress_bar)

        for step, batch in enumerate(iterator, start=1):
            check_resources_and_warn(self.device, step, gpu_limit=self.gpu_mem_limit, ram_limit=self.ram_limit)
            inputs = batch["features"].to(self.device, non_blocking=non_blocking) if "features" in batch else batch["images"].to(self.device, non_blocking=non_blocking)
            step_targets = batch["step_targets"].to(self.device, non_blocking=non_blocking)
            padding_mask = batch["padding_mask"].to(self.device, non_blocking=non_blocking)
            center_targets, valid_center_mask = select_center_supervision_targets(step_targets, padding_mask)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp)
                if self.use_amp
                else nullcontext()
            )
            with autocast_context:
                outputs = model(inputs, padding_mask=padding_mask, stage="stage_a")
                center_logits = outputs["center_step_logits"].float()
                if bool(valid_center_mask.any()):
                    loss = F.cross_entropy(
                        center_logits[valid_center_mask],
                        center_targets[valid_center_mask],
                        weight=self.class_weights.float() if self.class_weights is not None else None,
                    )
                else:
                    loss = center_logits.sum() * 0.0

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_sum += float(loss.item())
            if bool(valid_center_mask.any()):
                predictions = center_logits.argmax(dim=-1)
                correct_sum += int((predictions[valid_center_mask] == center_targets[valid_center_mask]).sum().item())
                valid_sum += int(valid_center_mask.sum().item())
            if progress_bar and tqdm is not None and (step % max(1, log_interval) == 0 or step == len(loader)):
                running_acc = float(correct_sum / max(1, valid_sum))
                iterator.set_postfix_str(f"loss={loss_sum / max(1, step):.4f} acc={running_acc:.4f} {current_lr_summary(optimizer)}")

        if progress_bar and tqdm is not None and hasattr(iterator, "close"):
            iterator.close()
        return {
            "loss": float(loss_sum / max(1, len(loader))),
            "acc": float(correct_sum / max(1, valid_sum)),
        }


class StageBTrainer:
    def __init__(
        self,
        *,
        class_weights: Optional[torch.Tensor],
        instrument_pos_weight: Optional[torch.Tensor],
        lambda_bd: float,
        lambda_inst: float,
        lambda_tmse: float = 0.15,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        lambda_refine_init: float = 0.3,
        mixup_alpha: float = 0.0,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        scaler: Optional[torch.cuda.amp.GradScaler],
        gpu_mem_limit: float,
        ram_limit: float,
        grad_accum_steps: int = 1,
    ):
        self.class_weights = class_weights
        self.instrument_pos_weight = instrument_pos_weight
        self.lambda_bd = float(lambda_bd)
        self.lambda_inst = float(lambda_inst)
        self.lambda_tmse = float(lambda_tmse)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.lambda_refine_init = float(lambda_refine_init)
        self.mixup_alpha = float(mixup_alpha)
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.scaler = scaler
        self.gpu_mem_limit = gpu_mem_limit
        self.ram_limit = ram_limit
        self.grad_accum_steps = max(1, int(grad_accum_steps))

    def run_epoch(
        self,
        *,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        num_epochs: int,
        progress_bar: bool,
        log_interval: int,
        non_blocking: bool,
    ) -> Dict[str, float]:
        model.train()
        base_frame_weights = build_center_frame_weights(loader.dataset.window_size).to(self.device)
        aggregate = {
            "step_weighted_sum": 0.0,
            "step_weight_sum": 0.0,
            "step_unweighted_sum": 0.0,
            "step_valid_count": 0.0,
            "boundary_loss_sum": 0.0,
            "boundary_weight_sum": 0.0,
            "instrument_loss_sum": 0.0,
            "instrument_valid_count": 0.0,
            "total_loss_sum": 0.0,
            "num_steps": 0,
            "correct": 0,
            "valid": 0,
        }
        if self.grad_accum_steps > 1:
            optimizer.zero_grad(set_to_none=True)
        iterator = maybe_make_progress_bar(loader, total=len(loader), desc=f"train {epoch}/{num_epochs}", enabled=progress_bar)

        for step, batch in enumerate(iterator, start=1):
            check_resources_and_warn(self.device, step, gpu_limit=self.gpu_mem_limit, ram_limit=self.ram_limit)
            inputs = batch["features"].to(self.device, non_blocking=non_blocking) if "features" in batch else batch["images"].to(self.device, non_blocking=non_blocking)
            device_batch = {
                key: value.to(self.device, non_blocking=non_blocking)
                for key, value in batch.items()
                if torch.is_tensor(value) and key not in {"features", "images"}
            }
            # --- Feature-space Mixup ---
            if self.mixup_alpha > 0 and inputs.dim() == 3 and inputs.shape[0] > 1:
                lam = float(Beta(self.mixup_alpha, self.mixup_alpha).sample().item())
                lam = max(lam, 1.0 - lam)  # bias toward the primary sample
                perm = torch.randperm(inputs.shape[0], device=inputs.device)
                inputs = lam * inputs + (1.0 - lam) * inputs[perm]
                # Mix targets: convert hard step targets to soft one-hot then blend
                num_classes = 12
                st = device_batch["step_targets"]  # [B, L]
                valid_a = st.ne(IGNORE_INDEX)
                valid_b = st[perm].ne(IGNORE_INDEX)
                soft_a = F.one_hot(st.clamp(min=0), num_classes).float() * valid_a.unsqueeze(-1).float()
                soft_b = F.one_hot(st[perm].clamp(min=0), num_classes).float() * valid_b.unsqueeze(-1).float()
                device_batch["step_targets_soft"] = lam * soft_a + (1.0 - lam) * soft_b
                # Both must be valid for mixed frame to be valid
                device_batch["step_targets"] = torch.where(
                    valid_a & valid_b, st, torch.full_like(st, IGNORE_INDEX)
                )
                # Mix other targets
                for key in ("boundary_targets", "boundary_loss_weights"):
                    device_batch[key] = lam * device_batch[key] + (1.0 - lam) * device_batch[key][perm]
                device_batch["instrument_targets"] = lam * device_batch["instrument_targets"] + (1.0 - lam) * device_batch["instrument_targets"][perm]
                device_batch["instrument_valid_mask"] = device_batch["instrument_valid_mask"] & device_batch["instrument_valid_mask"][perm]
                device_batch["padding_mask"] = device_batch["padding_mask"] | device_batch["padding_mask"][perm]
            frame_weights = base_frame_weights.unsqueeze(0).expand(device_batch["step_targets"].shape[0], -1)
            if self.grad_accum_steps <= 1:
                optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp)
                if self.use_amp
                else nullcontext()
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
                    class_weights=self.class_weights,
                    frame_weights=frame_weights,
                    instrument_pos_weight=self.instrument_pos_weight,
                    lambda_bd=self.lambda_bd,
                    lambda_inst=self.lambda_inst,
                    lambda_tmse=self.lambda_tmse,
                    lambda_refine_init=self.lambda_refine_init,
                    label_smoothing=self.label_smoothing,
                    focal_gamma=self.focal_gamma,
                )
                loss = losses["loss"]
                if self.grad_accum_steps > 1:
                    loss = loss / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.grad_accum_steps <= 1 or step % self.grad_accum_steps == 0 or step == len(loader):
                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            aggregate["step_weighted_sum"] += float(losses["step_ce_weighted_sum"].item())
            aggregate["step_weight_sum"] += float(losses["step_frame_weight_sum"].item())
            aggregate["step_unweighted_sum"] += float(losses["step_ce_unweighted_sum"].item())
            aggregate["step_valid_count"] += float(losses["step_valid_count"].item())
            aggregate["boundary_loss_sum"] += float(losses["boundary_loss_sum"].item())
            aggregate["boundary_weight_sum"] += float(losses["boundary_weight_sum"].item())
            aggregate["instrument_loss_sum"] += float(losses["instrument_loss_sum"].item())
            aggregate["instrument_valid_count"] += float(losses["instrument_valid_count"].item())
            aggregate["total_loss_sum"] += float(loss.item())
            aggregate["num_steps"] += 1

            valid_mask = device_batch["step_targets"].ne(IGNORE_INDEX)
            if bool(valid_mask.any()):
                predictions = outputs["step_logits"].argmax(dim=-1)
                aggregate["correct"] += int(((predictions == device_batch["step_targets"]) & valid_mask).sum().item())
                aggregate["valid"] += int(valid_mask.sum().item())
            if progress_bar and tqdm is not None and (step % max(1, log_interval) == 0 or step == len(loader)):
                running_step_loss = aggregate["step_weighted_sum"] / max(1e-8, aggregate["step_weight_sum"])
                running_boundary_loss = aggregate["boundary_loss_sum"] / max(1e-8, aggregate["boundary_weight_sum"])
                running_instrument_loss = (
                    aggregate["instrument_loss_sum"] / max(1.0, aggregate["instrument_valid_count"] * max(1, outputs["instrument_logits"].shape[-1]))
                )
                running_acc = float(aggregate["correct"] / max(1, aggregate["valid"]))
                iterator.set_postfix_str(
                    f"total={aggregate['total_loss_sum']/max(1, aggregate['num_steps']):.4f} "
                    f"step={running_step_loss:.4f} bd={running_boundary_loss:.4f} inst={running_instrument_loss:.4f} "
                    f"acc={running_acc:.4f} {current_lr_summary(optimizer)}"
                )

        if progress_bar and tqdm is not None and hasattr(iterator, "close"):
            iterator.close()

        step_loss = aggregate["step_weighted_sum"] / max(1e-8, aggregate["step_weight_sum"])
        boundary_loss = aggregate["boundary_loss_sum"] / max(1e-8, aggregate["boundary_weight_sum"])
        instrument_denominator = max(1.0, aggregate["instrument_valid_count"] * max(1, loader.dataset.video_instrument_targets[0].shape[1]))
        instrument_loss = aggregate["instrument_loss_sum"] / instrument_denominator
        return {
            "loss": float(aggregate["total_loss_sum"] / max(1, aggregate["num_steps"])),
            "total_loss": float(aggregate["total_loss_sum"] / max(1, aggregate["num_steps"])),
            "step_loss": float(step_loss),
            "boundary_loss": float(boundary_loss),
            "instrument_loss": float(instrument_loss),
            "acc": float(aggregate["correct"] / max(1, aggregate["valid"])),
            "step_ce_unweighted": float(aggregate["step_unweighted_sum"] / max(1.0, aggregate["step_valid_count"])),
        }


# ========== MS-TCN Training ==========


def compute_mstcn_loss(
    outputs: Dict[str, object],
    batch: Dict[str, torch.Tensor],
    *,
    class_weights: Optional[torch.Tensor] = None,
    instrument_pos_weight: Optional[torch.Tensor] = None,
    lambda_bd: float = 0.1,
    lambda_inst: float = 0.1,
    lambda_tmse: float = 0.15,
    lambda_trans: float = 0.0,
    lambda_late_chain: float = 0.0,
    lambda_coarse: float = 0.0,
    log_trans: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Multi-stage loss for MS-TCN: average (CE + TMSE + trans) across all stages."""
    all_stage_logits = outputs["all_stage_logits"]
    step_targets = batch["step_targets"]
    padding_mask = batch["padding_mask"]
    mask = batch["mask"]

    valid = mask & step_targets.ne(IGNORE_INDEX)
    num_stages = len(all_stage_logits)
    zero = all_stage_logits[0].sum() * 0.0
    total_step_loss = zero
    total_tmse = zero
    total_trans = zero
    class_weight_tensor = class_weights.float() if class_weights is not None else None

    for stage_logits in all_stage_logits:
        valid_logits = stage_logits[valid].float()
        valid_targets = step_targets[valid]
        if valid_logits.numel() == 0:
            continue
        if focal_gamma > 0:
            per_frame_ce = F.cross_entropy(
                valid_logits, valid_targets, weight=class_weight_tensor,
                reduction="none", label_smoothing=float(label_smoothing),
            )
            with torch.no_grad():
                p_t = torch.exp(-F.cross_entropy(valid_logits, valid_targets, reduction="none"))
            focal_w = (1.0 - p_t) ** focal_gamma
            ce = (focal_w * per_frame_ce).mean()
        else:
            ce = F.cross_entropy(
                valid_logits, valid_targets, weight=class_weight_tensor,
                label_smoothing=float(label_smoothing),
            )
        tmse = compute_tmse_loss(stage_logits, padding_mask=padding_mask)
        total_step_loss = total_step_loss + ce
        total_tmse = total_tmse + tmse

        if float(lambda_trans) > 0 and log_trans is not None:
            trans_loss = compute_transition_loss(
                stage_logits, step_targets=step_targets,
                padding_mask=padding_mask, log_trans=log_trans)
            total_trans = total_trans + trans_loss

    avg_step_loss = total_step_loss / max(1, num_stages)
    avg_tmse = total_tmse / max(1, num_stages)
    avg_trans = total_trans / max(1, num_stages)

    boundary_losses = compute_boundary_loss(
        outputs["boundary_logits"], batch["boundary_targets"],
        boundary_loss_weights=batch["boundary_loss_weights"],
        padding_mask=padding_mask,
    )
    instrument_losses = compute_instrument_loss(
        outputs["instrument_logits"], batch["instrument_targets"],
        instrument_valid_mask=batch["instrument_valid_mask"],
        instrument_pos_weight=instrument_pos_weight,
    )
    late_chain_loss = compute_late_chain_loss(
        outputs,
        step_targets,
        valid_mask=valid,
        late_chain_raw_labels=LATE_CHAIN_RAW_LABELS,
    )
    coarse_loss = compute_coarse_loss(
        outputs.get("coarse_logits"),
        step_targets=step_targets,
        padding_mask=padding_mask,
    )

    total_loss = (
        avg_step_loss
        + float(lambda_tmse) * avg_tmse
        + float(lambda_trans) * avg_trans
        + float(lambda_bd) * boundary_losses["boundary_loss"]
        + float(lambda_inst) * instrument_losses["instrument_loss"]
        + float(lambda_late_chain) * late_chain_loss
        + float(lambda_coarse) * coarse_loss
    )

    return {
        "loss": total_loss,
        "total_loss": total_loss,
        "step_loss": avg_step_loss,
        "tmse_loss": avg_tmse,
        "trans_loss": avg_trans,
        "boundary_loss": boundary_losses["boundary_loss"],
        "instrument_loss": instrument_losses["instrument_loss"],
        "late_chain_loss": late_chain_loss,
        "coarse_loss": coarse_loss,
    }


class MSTCNTrainer:
    """Trainer for full-video MS-TCN architecture."""

    def __init__(
        self,
        *,
        class_weights: Optional[torch.Tensor],
        instrument_pos_weight: Optional[torch.Tensor],
        lambda_bd: float,
        lambda_inst: float,
        lambda_tmse: float = 0.15,
        lambda_trans: float = 0.0,
        lambda_late_chain: float = 0.0,
        lambda_coarse: float = 0.0,
        log_trans: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        scaler: Optional[torch.cuda.amp.GradScaler],
        gpu_mem_limit: float,
        ram_limit: float,
        grad_accum_steps: int = 1,
    ):
        self.class_weights = class_weights
        self.instrument_pos_weight = instrument_pos_weight
        self.lambda_bd = float(lambda_bd)
        self.lambda_inst = float(lambda_inst)
        self.lambda_tmse = float(lambda_tmse)
        self.lambda_trans = float(lambda_trans)
        self.lambda_late_chain = float(lambda_late_chain)
        self.lambda_coarse = float(lambda_coarse)
        self.log_trans = log_trans
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.scaler = scaler
        self.gpu_mem_limit = gpu_mem_limit
        self.ram_limit = ram_limit
        self.grad_accum_steps = max(1, int(grad_accum_steps))

    def run_epoch(
        self,
        *,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        num_epochs: int,
        progress_bar: bool,
        log_interval: int,
        non_blocking: bool,
    ) -> Dict[str, float]:
        model.train()
        aggregate = {
            "total_loss_sum": 0.0, "step_loss_sum": 0.0, "tmse_loss_sum": 0.0,
            "trans_loss_sum": 0.0, "boundary_loss_sum": 0.0, "instrument_loss_sum": 0.0,
            "late_chain_loss_sum": 0.0, "coarse_loss_sum": 0.0,
            "num_steps": 0, "correct": 0, "valid": 0,
        }
        optimizer.zero_grad(set_to_none=True)
        iterator = maybe_make_progress_bar(
            loader, total=len(loader), desc=f"train {epoch}/{num_epochs}", enabled=progress_bar)

        for step, batch in enumerate(iterator, start=1):
            check_resources_and_warn(self.device, step, gpu_limit=self.gpu_mem_limit, ram_limit=self.ram_limit)
            features = batch["features"].to(self.device, non_blocking=non_blocking)
            device_batch = {
                key: value.to(self.device, non_blocking=non_blocking)
                for key, value in batch.items()
                if torch.is_tensor(value) and key != "features"
            }
            autocast_context = (
                torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp)
                if self.use_amp else nullcontext()
            )
            with autocast_context:
                outputs = model(features, mask=device_batch["mask"])
                losses = compute_mstcn_loss(
                    outputs, device_batch,
                    class_weights=self.class_weights,
                    instrument_pos_weight=self.instrument_pos_weight,
                    lambda_bd=self.lambda_bd,
                    lambda_inst=self.lambda_inst,
                    lambda_tmse=self.lambda_tmse,
                    lambda_trans=self.lambda_trans,
                    lambda_late_chain=self.lambda_late_chain,
                    lambda_coarse=self.lambda_coarse,
                    log_trans=self.log_trans,
                    label_smoothing=self.label_smoothing,
                    focal_gamma=self.focal_gamma,
                )
                loss = losses["loss"] / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.grad_accum_steps == 0 or step == len(loader):
                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            aggregate["total_loss_sum"] += float(losses["loss"].item())
            aggregate["step_loss_sum"] += float(losses["step_loss"].item())
            aggregate["tmse_loss_sum"] += float(losses["tmse_loss"].item())
            aggregate["trans_loss_sum"] += float(losses["trans_loss"].item()) if "trans_loss" in losses else 0.0
            aggregate["boundary_loss_sum"] += float(losses["boundary_loss"].item())
            aggregate["instrument_loss_sum"] += float(losses["instrument_loss"].item())
            aggregate["late_chain_loss_sum"] += float(losses["late_chain_loss"].item()) if "late_chain_loss" in losses else 0.0
            aggregate["coarse_loss_sum"] += float(losses["coarse_loss"].item()) if "coarse_loss" in losses else 0.0
            aggregate["num_steps"] += 1

            bmask = device_batch["mask"]
            valid_mask = bmask & device_batch["step_targets"].ne(IGNORE_INDEX)
            if bool(valid_mask.any()):
                preds = outputs["step_logits"].argmax(dim=-1)
                aggregate["correct"] += int(((preds == device_batch["step_targets"]) & valid_mask).sum().item())
                aggregate["valid"] += int(valid_mask.sum().item())

            if progress_bar and tqdm is not None and (step % max(1, log_interval) == 0 or step == len(loader)):
                n = max(1, aggregate["num_steps"])
                running_acc = float(aggregate["correct"] / max(1, aggregate["valid"]))
                iterator.set_postfix_str(
                    f"total={aggregate['total_loss_sum']/n:.4f} "
                    f"step={aggregate['step_loss_sum']/n:.4f} "
                    f"coarse={aggregate['coarse_loss_sum']/n:.4f} "
                    f"late={aggregate['late_chain_loss_sum']/n:.4f} "
                    f"acc={running_acc:.4f} {current_lr_summary(optimizer)}"
                )

        if progress_bar and tqdm is not None and hasattr(iterator, "close"):
            iterator.close()

        n = max(1, aggregate["num_steps"])
        return {
            "loss": float(aggregate["total_loss_sum"] / n),
            "total_loss": float(aggregate["total_loss_sum"] / n),
            "step_loss": float(aggregate["step_loss_sum"] / n),
            "boundary_loss": float(aggregate["boundary_loss_sum"] / n),
            "instrument_loss": float(aggregate["instrument_loss_sum"] / n),
            "late_chain_loss": float(aggregate["late_chain_loss_sum"] / n),
            "coarse_loss": float(aggregate["coarse_loss_sum"] / n),
            "acc": float(aggregate["correct"] / max(1, aggregate["valid"])),
            "step_ce_unweighted": float(aggregate["step_loss_sum"] / n),
        }


__all__ = [
    "MSTCNTrainer",
    "StageATrainer",
    "StageBTrainer",
    "build_amp_components",
    "build_instrument_pos_weight",
    "build_optimizer_and_scheduler",
    "compute_boundary_loss",
    "compute_instrument_loss",
    "compute_mstcn_loss",
    "compute_multitask_loss",
    "compute_step_accuracy",
    "compute_step_ar_loss",
    "compute_transition_loss",
    "current_lr_summary",
    "maybe_make_progress_bar",
    "select_center_supervision_targets",
]
