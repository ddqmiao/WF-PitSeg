import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from pitvis_mainline_common import (
    CENTER_FRAME_SIZE,
    CLASS_WEIGHT_BOOSTS,
    DEFAULT_PAIR_TRANSITION_RAW_PAIRS,
    FEATURE_DIM,
    FOCUS_BOUNDARY_RAW_PAIRS,
    HARD_CLASS_RAW_LABELS,
    IGNORE_INDEX,
    LABELS12,
    LABEL_TO_INDEX,
    MIN_FRAME_WEIGHT,
    align_feature_and_rows,
    build_video_items,
    discover_video_ids_from_features,
    discover_video_ids_from_images,
    map_raw_steps_to_targets,
    read_annotation_rows,
    resolve_frame_paths,
    resolve_norm_stats,
    resolve_video_split,
    safe_torch_load,
    video_folder_name,
)


def discover_instrument_class_ids(csv_root: Path) -> List[int]:
    class_ids = set()
    for csv_path in Path(csv_root).glob("annotations_*.csv"):
        for row in read_annotation_rows(csv_path):
            inst1 = int(row.get("int_instrument1", "-1"))
            inst2 = int(row.get("int_instrument2", "-2"))
            if inst1 >= 0:
                class_ids.add(inst1)
            if inst2 >= 0:
                class_ids.add(inst2)
    return sorted(class_ids)


def build_instrument_targets(
    rows: Sequence[Dict[str, str]],
    instrument_class_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    class_to_index = {int(raw_id): idx for idx, raw_id in enumerate(instrument_class_ids)}
    targets = torch.zeros((len(rows), len(instrument_class_ids)), dtype=torch.float32)
    valid_mask = torch.zeros((len(rows),), dtype=torch.bool)

    for row_idx, row in enumerate(rows):
        inst1 = int(row.get("int_instrument1", "-1"))
        inst2 = int(row.get("int_instrument2", "-2"))
        if inst1 == -1:
            # out_of_frame: this frame does not participate in instrument supervision
            continue
        valid_mask[row_idx] = True
        if inst1 >= 0 and inst1 in class_to_index:
            targets[row_idx, class_to_index[inst1]] = 1.0
        if inst2 >= 0 and inst2 in class_to_index:
            targets[row_idx, class_to_index[inst2]] = 1.0
    return targets, valid_mask


def build_center_frame_weights(
    window_size: int,
    center_size: int = CENTER_FRAME_SIZE,
    min_weight: float = MIN_FRAME_WEIGHT,
) -> torch.Tensor:
    center_size = max(1, min(int(center_size), int(window_size)))
    min_weight = float(min(max(min_weight, 0.0), 1.0))
    weights = torch.full((int(window_size),), min_weight, dtype=torch.float32)
    center_start = max(0, (int(window_size) - center_size) // 2)
    center_end = min(int(window_size), center_start + center_size)
    weights[center_start:center_end] = 1.0
    return weights


def build_step_class_weights(video_step_targets, *, class_balance_beta: float = 0.0) -> Optional[torch.Tensor]:
    if hasattr(video_step_targets, "video_step_targets"):
        target_sequences = video_step_targets.video_step_targets
    else:
        target_sequences = video_step_targets
    counts = torch.zeros(len(LABELS12), dtype=torch.float32)
    for step_targets in target_sequences:
        valid_targets = step_targets[step_targets.ne(IGNORE_INDEX)]
        if valid_targets.numel() == 0:
            continue
        counts += torch.bincount(valid_targets, minlength=len(LABELS12)).float()
    if counts.sum().item() == 0:
        return None

    if class_balance_beta > 0:
        # Class-Balanced Loss (Cui et al., CVPR 2019)
        # effective_num = (1 - beta^n) / (1 - beta)
        beta = float(class_balance_beta)
        effective_num = (1.0 - beta ** counts.clamp_min(1.0)) / (1.0 - beta)
        weights = 1.0 / effective_num.clamp_min(1e-8)
    else:
        # Original sqrt-dampened inverse frequency
        weights = (counts.sum() / counts.clamp_min(1.0)).sqrt()

    for class_index, boost in CLASS_WEIGHT_BOOSTS.items():
        if 0 <= int(class_index) < len(weights):
            weights[int(class_index)] *= float(boost)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights


def make_window_starts(length: int, window_size: int, stride: int) -> List[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _mapped_focus_boundary_pairs() -> set:
    focus_pairs = set()
    for src_raw, dst_raw in FOCUS_BOUNDARY_RAW_PAIRS:
        src = LABEL_TO_INDEX.get(int(src_raw))
        dst = LABEL_TO_INDEX.get(int(dst_raw))
        if src is not None and dst is not None:
            focus_pairs.add((int(src), int(dst)))
    return focus_pairs


def _mapped_transition_pairs(raw_pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    mapped: List[Tuple[int, int]] = []
    for src_raw, dst_raw in raw_pairs:
        src = LABEL_TO_INDEX.get(int(src_raw))
        dst = LABEL_TO_INDEX.get(int(dst_raw))
        if src is not None and dst is not None:
            mapped.append((int(src), int(dst)))
    return mapped


def build_boundary_targets(
    step_targets: torch.Tensor,
    *,
    radius: int = 3,
    base_boundary_weight: float = 2.0,
    focus_boundary_weight: float = 3.0,
    kernel: str = "gaussian",
    left_sigma: Optional[float] = None,
    right_sigma: Optional[float] = None,
    transition_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a soft boundary target and a loss-weight map.

    Only transitions between adjacent valid 12-step labels create boundary supervision.
    Real invalid-step frames remain part of the sequence and are not treated as padding,
    but they also do not define boundaries by themselves.
    """

    seq_len = int(step_targets.shape[0])
    mapped_pairs = _mapped_transition_pairs(
        transition_pairs if transition_pairs is not None else DEFAULT_PAIR_TRANSITION_RAW_PAIRS
    )
    use_pair_channels = kernel == "pairwise_asymmetric_gaussian" and len(mapped_pairs) > 0
    num_channels = len(mapped_pairs) if use_pair_channels else 1
    boundary_targets = torch.zeros((seq_len, num_channels), dtype=torch.float32)
    boundary_weights = torch.ones((seq_len, num_channels), dtype=torch.float32)
    radius = max(0, int(radius))
    focus_pairs = _mapped_focus_boundary_pairs()
    sigma = max(float(radius) / 2.0, 1.0)
    left_sigma = max(float(left_sigma if left_sigma is not None else sigma), 1.0)
    right_sigma = max(float(right_sigma if right_sigma is not None else sigma), 1.0)
    left_radius = max(radius, int(math.ceil(3.0 * left_sigma)))
    right_radius = max(radius, int(math.ceil(3.0 * right_sigma)))

    for time_idx in range(1, seq_len):
        prev_step = int(step_targets[time_idx - 1].item())
        curr_step = int(step_targets[time_idx].item())
        if not (0 <= prev_step < len(LABELS12) and 0 <= curr_step < len(LABELS12)):
            continue
        if prev_step == curr_step:
            continue

        active_channel = 0
        if use_pair_channels:
            try:
                active_channel = mapped_pairs.index((prev_step, curr_step))
            except ValueError:
                continue

        is_focus = (prev_step, curr_step) in focus_pairs
        max_weight = float(focus_boundary_weight if is_focus else base_boundary_weight)
        if kernel in {"asymmetric_gaussian", "pairwise_asymmetric_gaussian"}:
            left = max(0, time_idx - left_radius)
            right = min(seq_len, time_idx + right_radius + 1)
        else:
            left = max(0, time_idx - radius)
            right = min(seq_len, time_idx + radius + 1)
        for pos in range(left, right):
            if kernel == "triangle":
                distance = abs(pos - time_idx)
                value = max(0.0, 1.0 - distance / float(max(radius, 1)))
            elif kernel in {"asymmetric_gaussian", "pairwise_asymmetric_gaussian"}:
                if pos < time_idx:
                    distance = float(time_idx - pos)
                    value = math.exp(-0.5 * (distance / left_sigma) ** 2)
                elif pos > time_idx:
                    distance = float(pos - time_idx)
                    value = math.exp(-0.5 * (distance / right_sigma) ** 2)
                else:
                    value = 1.0
            else:
                distance = abs(pos - time_idx)
                value = math.exp(-0.5 * (distance / sigma) ** 2)
            value = float(min(max(value, 0.0), 1.0))
            boundary_targets[pos, active_channel] = max(float(boundary_targets[pos, active_channel].item()), value)
            weighted = 1.0 + (max_weight - 1.0) * value
            boundary_weights[pos, active_channel] = max(float(boundary_weights[pos, active_channel].item()), float(weighted))
    if num_channels == 1:
        return boundary_targets.squeeze(-1), boundary_weights.squeeze(-1)
    return boundary_targets, boundary_weights


class ClipTransform:
    """Apply the same spatial/color parameters to every frame in a clip."""

    def __init__(self, train: bool, resize: int = 256, crop_size: int = 224, norm: str = "lemon_video"):
        self.train = bool(train)
        self.resize = int(resize)
        self.crop_size = int(crop_size)
        self.mean, self.std = resolve_norm_stats(norm)

    def __call__(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("ClipTransform requires at least one frame.")
        resized_first = TF.resize(images[0].convert("RGB"), self.resize)
        width, height = resized_first.size
        crop_h = min(self.crop_size, height)
        crop_w = min(self.crop_size, width)
        if self.train:
            top = 0 if height == crop_h else random.randint(0, height - crop_h)
            left = 0 if width == crop_w else random.randint(0, width - crop_w)
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            saturation = random.uniform(0.9, 1.1)
            apply_blur = random.random() < 0.25
            blur_sigma = random.uniform(0.1, 1.0) if apply_blur else None
        else:
            top = max(0, (height - crop_h) // 2)
            left = max(0, (width - crop_w) // 2)
            brightness = 1.0
            contrast = 1.0
            saturation = 1.0
            apply_blur = False
            blur_sigma = None

        tensors: List[torch.Tensor] = []
        for image in images:
            image = TF.resize(image.convert("RGB"), self.resize)
            image = TF.crop(image, top, left, crop_h, crop_w)
            tensor = TF.to_tensor(image)
            tensor = TF.adjust_brightness(tensor, brightness)
            tensor = TF.adjust_contrast(tensor, contrast)
            tensor = TF.adjust_saturation(tensor, saturation)
            if apply_blur and blur_sigma is not None:
                tensor = TF.gaussian_blur(tensor, kernel_size=3, sigma=blur_sigma)
            tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)


def build_clip_transform(train: bool, resize: int = 256, crop_size: int = 224, norm: str = "lemon_video") -> ClipTransform:
    return ClipTransform(train=train, resize=resize, crop_size=crop_size, norm=norm)


class PitVisWindowDataset(Dataset):
    """Unified dataset for feature mode and image mode.

    Returned `padding_mask=True` only for true padded frames whose `frame_indices == -1`.
    Real frames with invalid step labels remain sequence members and are not masked out from attention.
    """

    def __init__(
        self,
        video_items: Sequence[Dict[str, object]],
        *,
        input_mode: str,
        window_size: int = 64,
        stride: int = 16,
        images_root: Optional[Path] = None,
        transform: Optional[Callable[[Sequence[Image.Image]], torch.Tensor]] = None,
        instrument_class_ids: Sequence[int],
        boundary_radius: int = 3,
        base_boundary_weight: float = 2.0,
        focus_boundary_weight: float = 3.0,
        boundary_kernel: str = "gaussian",
        boundary_left_sigma: Optional[float] = None,
        boundary_right_sigma: Optional[float] = None,
        transition_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        self.input_mode = str(input_mode)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.images_root = Path(images_root) if images_root is not None else None
        self.transform = transform
        self.instrument_class_ids = list(instrument_class_ids)
        self.boundary_radius = int(boundary_radius)
        self.base_boundary_weight = float(base_boundary_weight)
        self.focus_boundary_weight = float(focus_boundary_weight)
        self.boundary_kernel = str(boundary_kernel)
        self.boundary_left_sigma = None if boundary_left_sigma is None else float(boundary_left_sigma)
        self.boundary_right_sigma = None if boundary_right_sigma is None else float(boundary_right_sigma)
        self.transition_pairs = None if transition_pairs is None else tuple((int(a), int(b)) for a, b in transition_pairs)

        self.video_ids: List[int] = []
        self.video_lengths: List[int] = []
        self.video_step_targets: List[torch.Tensor] = []
        self.video_instrument_targets: List[torch.Tensor] = []
        self.video_instrument_valid_masks: List[torch.Tensor] = []
        self.video_boundary_targets: List[torch.Tensor] = []
        self.video_boundary_loss_weights: List[torch.Tensor] = []
        self.video_features: List[torch.Tensor] = []
        self.video_frame_paths: List[List[Path]] = []
        self.samples: List[Tuple[int, int]] = []

        for item in video_items:
            video_id = int(item["video_id"])
            rows = read_annotation_rows(Path(item["label_path"]))
            raw_steps = [int(row["int_step"]) for row in rows]

            if self.input_mode == "features":
                features = safe_torch_load(Path(item["feature_path"])).float()
                features, rows = align_feature_and_rows(features, rows)
                raw_steps = raw_steps[: int(features.shape[0])]
                self.video_features.append(features)
                self.video_frame_paths.append([])
                video_length = int(features.shape[0])
            else:
                if self.images_root is None or self.transform is None:
                    raise ValueError("images mode requires images_root and transform.")
                video_dir = self.images_root / video_folder_name(video_id)
                frame_paths = resolve_frame_paths(video_dir, Path(item["label_path"]))
                rows = rows[: len(frame_paths)]
                raw_steps = raw_steps[: len(frame_paths)]
                self.video_features.append(torch.empty(0))
                self.video_frame_paths.append(frame_paths)
                video_length = len(frame_paths)

            step_targets = map_raw_steps_to_targets(raw_steps)
            instrument_targets, instrument_valid_mask = build_instrument_targets(rows, self.instrument_class_ids)
            boundary_targets, boundary_loss_weights = build_boundary_targets(
                step_targets,
                radius=self.boundary_radius,
                base_boundary_weight=self.base_boundary_weight,
                focus_boundary_weight=self.focus_boundary_weight,
                kernel=self.boundary_kernel,
                left_sigma=self.boundary_left_sigma,
                right_sigma=self.boundary_right_sigma,
                transition_pairs=self.transition_pairs,
            )

            video_index = len(self.video_ids)
            self.video_ids.append(video_id)
            self.video_lengths.append(video_length)
            self.video_step_targets.append(step_targets)
            self.video_instrument_targets.append(instrument_targets)
            self.video_instrument_valid_masks.append(instrument_valid_mask)
            self.video_boundary_targets.append(boundary_targets)
            self.video_boundary_loss_weights.append(boundary_loss_weights)

            for start in make_window_starts(video_length, self.window_size, self.stride):
                self.samples.append((video_index, int(start)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_index, start = self.samples[index]
        video_id = self.video_ids[video_index]
        video_length = self.video_lengths[video_index]
        end = min(start + self.window_size, video_length)
        actual_length = end - start

        step_targets = self.video_step_targets[video_index][start:end]
        instrument_targets = self.video_instrument_targets[video_index][start:end]
        instrument_valid_mask = self.video_instrument_valid_masks[video_index][start:end]
        boundary_targets = self.video_boundary_targets[video_index][start:end]
        boundary_loss_weights = self.video_boundary_loss_weights[video_index][start:end]
        frame_indices = torch.arange(start, end, dtype=torch.long)

        if self.input_mode == "features":
            features = self.video_features[video_index][start:end]
            sample: Dict[str, torch.Tensor] = {"features": features}
        else:
            clip_paths = self.video_frame_paths[video_index][start:end]
            clip_images: List[Image.Image] = []
            for frame_path in clip_paths:
                with Image.open(frame_path) as image:
                    clip_images.append(image.copy())
            sample = {"images": self.transform(clip_images)}

        pad_length = self.window_size - actual_length
        if pad_length > 0:
            if self.input_mode == "features":
                feature_pad = torch.zeros((pad_length, FEATURE_DIM), dtype=sample["features"].dtype)
                sample["features"] = torch.cat([sample["features"], feature_pad], dim=0)
            else:
                image_pad = torch.zeros((pad_length,) + tuple(sample["images"].shape[1:]), dtype=sample["images"].dtype)
                sample["images"] = torch.cat([sample["images"], image_pad], dim=0)
            step_targets = torch.cat([step_targets, torch.full((pad_length,), IGNORE_INDEX, dtype=torch.long)], dim=0)
            instrument_targets = torch.cat(
                [instrument_targets, torch.zeros((pad_length, instrument_targets.shape[1]), dtype=torch.float32)],
                dim=0,
            )
            instrument_valid_mask = torch.cat([instrument_valid_mask, torch.zeros((pad_length,), dtype=torch.bool)], dim=0)
            if boundary_targets.dim() == 2:
                boundary_targets = torch.cat(
                    [boundary_targets, torch.zeros((pad_length, boundary_targets.shape[1]), dtype=torch.float32)], dim=0
                )
                boundary_loss_weights = torch.cat(
                    [boundary_loss_weights, torch.ones((pad_length, boundary_loss_weights.shape[1]), dtype=torch.float32)], dim=0
                )
            else:
                boundary_targets = torch.cat([boundary_targets, torch.zeros((pad_length,), dtype=torch.float32)], dim=0)
                boundary_loss_weights = torch.cat([boundary_loss_weights, torch.ones((pad_length,), dtype=torch.float32)], dim=0)
            frame_indices = torch.cat([frame_indices, torch.full((pad_length,), -1, dtype=torch.long)], dim=0)

        padding_mask = frame_indices.eq(-1)
        sample.update(
            {
                "step_targets": step_targets,
                "instrument_targets": instrument_targets,
                "instrument_valid_mask": instrument_valid_mask,
                "boundary_targets": boundary_targets,
                "boundary_loss_weights": boundary_loss_weights,
                "frame_indices": frame_indices,
                "padding_mask": padding_mask,
                "video_id": torch.tensor(video_id, dtype=torch.long),
            }
        )
        return sample


def build_pitvis_datasets(
    args,
    *,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
) -> Tuple[PitVisWindowDataset, PitVisWindowDataset, List[int]]:
    if args.input_mode == "features":
        video_ids = discover_video_ids_from_features(args.features_root, args.csv_root)
    else:
        video_ids = discover_video_ids_from_images(args.images_root, args.csv_root)
    split = resolve_video_split(video_ids, split_json=getattr(args, "split_json", None))
    instrument_class_ids = discover_instrument_class_ids(args.csv_root)

    effective_window_size = int(window_size if window_size is not None else args.window_size)
    effective_stride = int(stride if stride is not None else args.stride)
    transition_pairs = args.transition_pairs if getattr(args, "boundary_mode", "scalar") == "pairwise" else None

    train_items = build_video_items(
        args.features_root,
        args.csv_root,
        split["train_videos"],
        input_mode=args.input_mode,
        max_videos=args.max_train_videos,
    )
    val_items = build_video_items(
        args.features_root,
        args.csv_root,
        split["val_videos"],
        input_mode=args.input_mode,
        max_videos=args.max_val_videos,
    )
    if not train_items:
        raise ValueError("The training set is empty.")
    if not val_items:
        raise ValueError("The validation set is empty.")

    train_transform = None
    val_transform = None
    if args.input_mode == "images":
        train_transform = build_clip_transform(True, resize=args.resize, crop_size=args.crop_size, norm=args.norm)
        val_transform = build_clip_transform(False, resize=args.resize, crop_size=args.crop_size, norm=args.norm)

    train_dataset = PitVisWindowDataset(
        train_items,
        input_mode=args.input_mode,
        window_size=effective_window_size,
        stride=effective_stride,
        images_root=args.images_root,
        transform=train_transform,
        instrument_class_ids=instrument_class_ids,
        boundary_radius=args.boundary_radius,
        base_boundary_weight=args.boundary_base_weight,
        focus_boundary_weight=args.focus_boundary_weight,
        boundary_kernel=args.boundary_kernel,
        boundary_left_sigma=getattr(args, "boundary_left_sigma", None),
        boundary_right_sigma=getattr(args, "boundary_right_sigma", None),
        transition_pairs=transition_pairs,
    )
    val_dataset = PitVisWindowDataset(
        val_items,
        input_mode=args.input_mode,
        window_size=effective_window_size,
        stride=effective_stride,
        images_root=args.images_root,
        transform=val_transform,
        instrument_class_ids=instrument_class_ids,
        boundary_radius=args.boundary_radius,
        base_boundary_weight=args.boundary_base_weight,
        focus_boundary_weight=args.focus_boundary_weight,
        boundary_kernel=args.boundary_kernel,
        boundary_left_sigma=getattr(args, "boundary_left_sigma", None),
        boundary_right_sigma=getattr(args, "boundary_right_sigma", None),
        transition_pairs=transition_pairs,
    )
    return train_dataset, val_dataset, instrument_class_ids


def build_sample_weights_for_balanced_sampling(dataset: PitVisWindowDataset) -> torch.Tensor:
    """Compute per-sample weights for WeightedRandomSampler.

    Base weight = max inverse-frequency weight of its valid step classes.
    Additional bonuses are applied to:
      1. windows whose center falls on hard classes (3/9/12/14)
      2. windows that cover focused transition boundaries such as 2<->3, 3<->4, 8<->9, 9<->10, ...
    """
    num_classes = len(LABELS12)
    hard_class_index_to_bonus = {
        LABEL_TO_INDEX[3]: 2.5,
        LABEL_TO_INDEX[9]: 2.2,
        LABEL_TO_INDEX[12]: 1.6,
        LABEL_TO_INDEX[14]: 1.6,
    }
    focus_pairs = _mapped_focus_boundary_pairs()
    center_radius = max(1, int(dataset.window_size) // 4)
    # Count total frames per class across all windows
    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    for video_index, start in dataset.samples:
        end = min(start + dataset.window_size, dataset.video_lengths[video_index])
        targets = dataset.video_step_targets[video_index][start:end]
        valid = targets[targets.ne(IGNORE_INDEX)]
        if valid.numel() > 0:
            class_counts += torch.bincount(valid.long(), minlength=num_classes).double()

    # Inverse-frequency weights per class (sqrt dampened)
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0)).sqrt()
    # Apply CLASS_WEIGHT_BOOSTS
    for class_index, boost in CLASS_WEIGHT_BOOSTS.items():
        if 0 <= int(class_index) < num_classes:
            class_weights[int(class_index)] *= float(boost)

    # Per-sample weight = max class weight among classes present in the window
    sample_weights = torch.ones(len(dataset.samples), dtype=torch.float64)
    for idx, (video_index, start) in enumerate(dataset.samples):
        end = min(start + dataset.window_size, dataset.video_lengths[video_index])
        targets = dataset.video_step_targets[video_index][start:end]
        valid = targets[targets.ne(IGNORE_INDEX)]
        sample_weight = 1.0
        if valid.numel() > 0:
            unique_classes = valid.unique()
            max_weight = max(float(class_weights[int(c)]) for c in unique_classes)
            sample_weight = max_weight

            center_local = min(len(targets) - 1, max(0, dataset.window_size // 2))
            center_target = int(targets[center_local].item())
            if center_target in hard_class_index_to_bonus:
                sample_weight *= hard_class_index_to_bonus[center_target]

            focus_bonus = 1.0
            for local_idx in range(1, len(targets)):
                prev_step = int(targets[local_idx - 1].item())
                curr_step = int(targets[local_idx].item())
                if prev_step == IGNORE_INDEX or curr_step == IGNORE_INDEX or prev_step == curr_step:
                    continue
                if (prev_step, curr_step) not in focus_pairs:
                    continue
                distance_to_center = abs(local_idx - center_local)
                focus_bonus = max(focus_bonus, 2.5 if distance_to_center <= center_radius else 1.8)
            sample_weight *= focus_bonus
        sample_weights[idx] = sample_weight

    return sample_weights


def _apply_speed_perturbation(
    features: torch.Tensor,
    step_targets: torch.Tensor,
    inst_targets: torch.Tensor,
    inst_mask: torch.Tensor,
    bd_targets: torch.Tensor,
    bd_weights: torch.Tensor,
    *,
    speed_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample a video temporally by speed_factor (>1 = compress, <1 = stretch)."""
    T = features.shape[0]
    new_T = max(1, round(T / speed_factor))

    # Features: linear interpolation (T, D) -> (new_T, D)
    feat_in = features.unsqueeze(0).permute(0, 2, 1)  # (1, D, T)
    feat_out = F.interpolate(feat_in, size=new_T, mode="linear", align_corners=False)
    features = feat_out.permute(0, 2, 1).squeeze(0)  # (new_T, D)

    # Discrete targets: nearest-neighbor resample
    idx = torch.linspace(0, T - 1, new_T).long().clamp(0, T - 1)
    step_targets = step_targets[idx]
    inst_targets = inst_targets[idx]
    inst_mask = inst_mask[idx]
    bd_targets = bd_targets[idx]
    bd_weights = bd_weights[idx]

    return features, step_targets, inst_targets, inst_mask, bd_targets, bd_weights


# ========== Full-Video Dataset for MS-TCN ==========


class PitVisFullVideoDataset(Dataset):
    """Full-video dataset for MS-TCN. One sample = one entire video, no windowing."""

    def __init__(
        self,
        video_items: Sequence[Dict[str, object]],
        *,
        input_mode: str,
        instrument_class_ids: Sequence[int],
        boundary_radius: int = 3,
        base_boundary_weight: float = 2.0,
        focus_boundary_weight: float = 3.0,
        boundary_kernel: str = "gaussian",
        boundary_left_sigma: Optional[float] = None,
        boundary_right_sigma: Optional[float] = None,
        transition_pairs: Optional[Sequence[Tuple[int, int]]] = None,
        speed_perturb_range: Optional[Tuple[float, float]] = None,
    ):
        self.input_mode = str(input_mode)
        self.instrument_class_ids = list(instrument_class_ids)
        self.speed_perturb_range = speed_perturb_range

        self.video_ids: List[int] = []
        self.video_lengths: List[int] = []
        self.video_step_targets: List[torch.Tensor] = []
        self.video_instrument_targets: List[torch.Tensor] = []
        self.video_instrument_valid_masks: List[torch.Tensor] = []
        self.video_boundary_targets: List[torch.Tensor] = []
        self.video_boundary_loss_weights: List[torch.Tensor] = []
        self.video_features: List[torch.Tensor] = []

        for item in video_items:
            video_id = int(item["video_id"])
            rows = read_annotation_rows(Path(item["label_path"]))
            raw_steps = [int(row["int_step"]) for row in rows]

            if input_mode == "features":
                features = safe_torch_load(Path(item["feature_path"])).float()
                features, rows = align_feature_and_rows(features, rows)
                raw_steps = raw_steps[: int(features.shape[0])]
                self.video_features.append(features)
                video_length = int(features.shape[0])
            else:
                raise NotImplementedError("MS-TCN full-video mode currently supports features only.")

            step_targets = map_raw_steps_to_targets(raw_steps)
            instrument_targets, instrument_valid_mask = build_instrument_targets(rows, self.instrument_class_ids)
            boundary_targets, boundary_loss_weights = build_boundary_targets(
                step_targets,
                radius=boundary_radius,
                base_boundary_weight=base_boundary_weight,
                focus_boundary_weight=focus_boundary_weight,
                kernel=boundary_kernel,
                left_sigma=boundary_left_sigma,
                right_sigma=boundary_right_sigma,
                transition_pairs=transition_pairs,
            )

            self.video_ids.append(video_id)
            self.video_lengths.append(video_length)
            self.video_step_targets.append(step_targets)
            self.video_instrument_targets.append(instrument_targets)
            self.video_instrument_valid_masks.append(instrument_valid_mask)
            self.video_boundary_targets.append(boundary_targets)
            self.video_boundary_loss_weights.append(boundary_loss_weights)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        features = self.video_features[index]
        step_targets = self.video_step_targets[index]
        inst_targets = self.video_instrument_targets[index]
        inst_mask = self.video_instrument_valid_masks[index]
        bd_targets = self.video_boundary_targets[index]
        bd_weights = self.video_boundary_loss_weights[index]

        if self.speed_perturb_range is not None:
            lo, hi = self.speed_perturb_range
            speed = random.uniform(lo, hi)
            if abs(speed - 1.0) > 1e-3:
                features, step_targets, inst_targets, inst_mask, bd_targets, bd_weights = \
                    _apply_speed_perturbation(
                        features, step_targets, inst_targets, inst_mask,
                        bd_targets, bd_weights, speed_factor=speed,
                    )

        T = features.shape[0]
        return {
            "features": features,
            "step_targets": step_targets,
            "instrument_targets": inst_targets,
            "instrument_valid_mask": inst_mask,
            "boundary_targets": bd_targets,
            "boundary_loss_weights": bd_weights,
            "video_id": torch.tensor(self.video_ids[index], dtype=torch.long),
            "length": torch.tensor(T, dtype=torch.long),
        }


def collate_full_videos(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate variable-length full videos with padding to max length in batch."""
    max_len = max(int(b["length"].item()) for b in batch)
    batch_size = len(batch)
    feature_dim = batch[0]["features"].shape[-1]
    num_inst = batch[0]["instrument_targets"].shape[-1]
    boundary_channels = batch[0]["boundary_targets"].shape[1] if batch[0]["boundary_targets"].dim() == 2 else 1

    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    padded_step_targets = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    padded_instrument_targets = torch.zeros(batch_size, max_len, num_inst)
    padded_instrument_valid = torch.zeros(batch_size, max_len, dtype=torch.bool)
    if boundary_channels > 1:
        padded_boundary_targets = torch.zeros(batch_size, max_len, boundary_channels)
        padded_boundary_weights = torch.ones(batch_size, max_len, boundary_channels)
    else:
        padded_boundary_targets = torch.zeros(batch_size, max_len)
        padded_boundary_weights = torch.ones(batch_size, max_len)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    video_ids = torch.stack([b["video_id"] for b in batch])
    lengths = torch.stack([b["length"] for b in batch])

    for i, b in enumerate(batch):
        L = int(b["length"].item())
        padded_features[i, :L] = b["features"]
        padded_step_targets[i, :L] = b["step_targets"]
        padded_instrument_targets[i, :L] = b["instrument_targets"]
        padded_instrument_valid[i, :L] = b["instrument_valid_mask"]
        padded_boundary_targets[i, :L] = b["boundary_targets"]
        padded_boundary_weights[i, :L] = b["boundary_loss_weights"]
        mask[i, :L] = True

    return {
        "features": padded_features,
        "step_targets": padded_step_targets,
        "instrument_targets": padded_instrument_targets,
        "instrument_valid_mask": padded_instrument_valid,
        "boundary_targets": padded_boundary_targets,
        "boundary_loss_weights": padded_boundary_weights,
        "mask": mask,
        "padding_mask": ~mask,
        "video_id": video_ids,
        "length": lengths,
    }


def build_mstcn_datasets(
    args,
) -> Tuple[PitVisFullVideoDataset, PitVisFullVideoDataset, List[int]]:
    """Build full-video datasets for MS-TCN architecture."""
    if args.input_mode != "features":
        raise ValueError("MS-TCN architecture currently requires input_mode='features'.")
    video_ids = discover_video_ids_from_features(args.features_root, args.csv_root)
    split = resolve_video_split(video_ids, split_json=getattr(args, "split_json", None))
    instrument_class_ids = discover_instrument_class_ids(args.csv_root)

    train_items = build_video_items(
        args.features_root, args.csv_root, split["train_videos"],
        input_mode="features", max_videos=args.max_train_videos,
    )
    val_items = build_video_items(
        args.features_root, args.csv_root, split["val_videos"],
        input_mode="features", max_videos=args.max_val_videos,
    )
    if not train_items:
        raise ValueError("The training set is empty.")
    if not val_items:
        raise ValueError("The validation set is empty.")

    train_dataset = PitVisFullVideoDataset(
        train_items,
        input_mode="features",
        instrument_class_ids=instrument_class_ids,
        boundary_radius=args.boundary_radius,
        base_boundary_weight=args.boundary_base_weight,
        focus_boundary_weight=args.focus_boundary_weight,
        boundary_kernel=args.boundary_kernel,
        boundary_left_sigma=getattr(args, "boundary_left_sigma", None),
        boundary_right_sigma=getattr(args, "boundary_right_sigma", None),
        transition_pairs=args.transition_pairs if getattr(args, "boundary_mode", "scalar") == "pairwise" else None,
        speed_perturb_range=getattr(args, "speed_perturb_range", None),
    )
    val_dataset = PitVisFullVideoDataset(
        val_items,
        input_mode="features",
        instrument_class_ids=instrument_class_ids,
        boundary_radius=args.boundary_radius,
        base_boundary_weight=args.boundary_base_weight,
        focus_boundary_weight=args.focus_boundary_weight,
        boundary_kernel=args.boundary_kernel,
        boundary_left_sigma=getattr(args, "boundary_left_sigma", None),
        boundary_right_sigma=getattr(args, "boundary_right_sigma", None),
        transition_pairs=args.transition_pairs if getattr(args, "boundary_mode", "scalar") == "pairwise" else None,
    )
    return train_dataset, val_dataset, instrument_class_ids


__all__ = [
    "ClipTransform",
    "PitVisFullVideoDataset",
    "PitVisWindowDataset",
    "build_center_frame_weights",
    "build_clip_transform",
    "build_instrument_targets",
    "build_mstcn_datasets",
    "build_pitvis_datasets",
    "build_sample_weights_for_balanced_sampling",
    "build_step_class_weights",
    "build_boundary_targets",
    "collate_full_videos",
    "discover_instrument_class_ids",
    "make_window_starts",
]
