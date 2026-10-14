import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    import psutil
except ImportError:
    psutil = None

from extract_lemonfm_features import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_TRAILING_MISSING_FRAMES,
    ensure_repo_importable,
)
from pitvis_transition_policy import (
    ALLOWED_TRANSITIONS_RAW,
    CLASS_WEIGHT_BOOSTS_RAW,
    DEFAULT_PAIR_TRANSITION_RAW_PAIRS,
    FOCUS_BOUNDARY_RAW_PAIRS,
    HARD_CLASS_RAW_LABELS,
    build_log_transition_matrix_numpy as build_log_transition_matrix_numpy_policy,
    count_segment_transition_counts,
    map_raw_boosts_to_index_boosts,
)


LABELS12 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS12)}
OFFICIAL_VAL_VIDEOS = [1, 12, 21, 24, 25]
IGNORE_INDEX = -100
FEATURE_DIM = 1536
BACKBONE_CHUNK_SIZE = 64
GPU_MEM_LIMIT_RATIO = 0.90
RAM_LIMIT_RATIO = 0.80
RESOURCE_CHECK_INTERVAL = 10
GRAD_CLIP_MAX_NORM = 1.0
CENTER_FRAME_SIZE = 32
MIN_FRAME_WEIGHT = 0.5
CLASS_WEIGHT_BOOSTS: Dict[int, float] = map_raw_boosts_to_index_boosts(LABELS12, CLASS_WEIGHT_BOOSTS_RAW)

# Raw-label transition counts from 25-video annotation statistics.
# Row = from (raw label), Col = to (raw label), Value = observed count.
RAW_TRANSITION_COUNTS: Dict[int, Dict[int, int]] = {
    1:  {2: 25},
    2:  {3: 23, 4: 7, 8: 5},
    3:  {2: 2, 4: 36},
    4:  {2: 5, 3: 14, 5: 38, 6: 1, 8: 13},
    5:  {4: 18, 6: 24, 7: 1, 8: 18},
    6:  {5: 10, 7: 23, 8: 11},
    7:  {5: 1, 6: 1, 8: 34, 9: 3, 10: 1},
    8:  {2: 4, 3: 1, 4: 9, 5: 12, 6: 18, 7: 16, 9: 15, 10: 11, 12: 1, 14: 1},
    9:  {7: 2, 8: 4, 10: 12, 12: 1},
    10: {8: 3, 9: 1, 12: 19, 14: 3},
    12: {7: 1, 14: 17},
    14: {8: 2, 12: 6},
}


def build_log_transition_tensor(
    num_classes: int = len(LABELS12),
    stay_prob: float = 0.98,
    smoothing: float = 1e-6,
    transition_counts: Optional[Dict[int, Dict[int, int]]] = None,
    allowed_transitions_raw: Optional[Dict[int, Sequence[int]]] = None,
) -> torch.Tensor:
    """Build a hard-masked log transition matrix for training losses."""
    if num_classes != len(LABELS12):
        raise ValueError(
            f"Only a {len(LABELS12)}-class transition matrix is supported; "
            f"received num_classes={num_classes}"
        )
    transition_counts = transition_counts if transition_counts is not None else RAW_TRANSITION_COUNTS
    allowed_map = (
        {int(src): set(int(dst) for dst in dsts) for src, dsts in allowed_transitions_raw.items()}
        if allowed_transitions_raw is not None
        else ALLOWED_TRANSITIONS_RAW
    )
    log_trans = build_log_transition_matrix_numpy_policy(
        labels12=LABELS12,
        transition_counts=transition_counts,
        allowed_transitions_raw=allowed_map,
        stay_prob=stay_prob,
        allowed_smoothing=smoothing,
    )
    return torch.from_numpy(log_trans).to(dtype=torch.float32)


def build_log_transition_matrix_numpy(
    *,
    transition_counts: Optional[Dict[int, Dict[int, int]]] = None,
    allowed_transitions_raw: Optional[Dict[int, Sequence[int]]] = None,
    stay_prob: float = 0.98,
    smoothing: float = 1e-6,
) -> np.ndarray:
    transition_counts = transition_counts if transition_counts is not None else RAW_TRANSITION_COUNTS
    allowed_map = (
        {int(src): set(int(dst) for dst in dsts) for src, dsts in allowed_transitions_raw.items()}
        if allowed_transitions_raw is not None
        else ALLOWED_TRANSITIONS_RAW
    )
    return build_log_transition_matrix_numpy_policy(
        labels12=LABELS12,
        transition_counts=transition_counts,
        allowed_transitions_raw=allowed_map,
        stay_prob=stay_prob,
        allowed_smoothing=smoothing,
    )


def parse_transition_pairs_spec(spec) -> Tuple[Tuple[int, int], ...]:
    if spec is None:
        return tuple(DEFAULT_PAIR_TRANSITION_RAW_PAIRS)
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return tuple(DEFAULT_PAIR_TRANSITION_RAW_PAIRS)
        if text.startswith("["):
            parsed = json.loads(text)
        else:
            parsed = []
            for item in text.split(","):
                item = item.strip()
                if not item:
                    continue
                if "->" in item:
                    left, right = item.split("->", 1)
                elif "-" in item:
                    left, right = item.split("-", 1)
                else:
                    raise ValueError(f"Could not parse transition pair: {item}")
                parsed.append([int(left), int(right)])
    else:
        parsed = spec

    pairs: List[Tuple[int, int]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"A transition pair must have length 2; received: {item}")
        pairs.append((int(item[0]), int(item[1])))
    if not pairs:
        raise ValueError("transition_pairs cannot be empty.")
    return tuple(pairs)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def resolve_device_and_gpu_ids(args: argparse.Namespace) -> Tuple[torch.device, Optional[List[int]]]:
    gpu_ids: Optional[List[int]] = None
    if getattr(args, "gpu_ids", None):
        gpu_ids = [int(x.strip()) for x in str(args.gpu_ids).split(",") if x.strip()]
        if not gpu_ids:
            gpu_ids = None
    if gpu_ids is not None and len(gpu_ids) >= 1:
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    return device, gpu_ids


def wrap_model_data_parallel(model: nn.Module, gpu_ids: Optional[List[int]]) -> nn.Module:
    if gpu_ids is not None and len(gpu_ids) > 1 and torch.cuda.is_available():
        print(f"multi_gpu: wrapping model with DataParallel on GPUs {gpu_ids}")
        return nn.DataParallel(model, device_ids=gpu_ids)
    return model


def check_gpu_memory(device: torch.device, limit_ratio: float = GPU_MEM_LIMIT_RATIO) -> Dict[str, float]:
    if device.type != "cuda":
        return {"used_ratio": 0.0, "used_gb": 0.0, "total_gb": 0.0, "exceeded": False}
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev_idx)
    total = getattr(props, "total_memory", 0)
    reserved = torch.cuda.memory_reserved(dev_idx)
    used_ratio = reserved / total if total > 0 else 0.0
    exceeded = used_ratio > limit_ratio
    if exceeded:
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved(dev_idx)
        used_ratio = reserved / total if total > 0 else 0.0
        exceeded = used_ratio > limit_ratio
    return {
        "used_ratio": float(used_ratio),
        "used_gb": float(reserved / (1024 ** 3)),
        "total_gb": float(total / (1024 ** 3)),
        "exceeded": bool(exceeded),
    }


def check_ram_usage(limit_ratio: float = RAM_LIMIT_RATIO) -> Dict[str, float]:
    if psutil is None:
        return {"used_ratio": 0.0, "used_gb": 0.0, "total_gb": 0.0, "exceeded": False}
    mem = psutil.virtual_memory()
    used_ratio = mem.percent / 100.0
    return {
        "used_ratio": float(used_ratio),
        "used_gb": float(mem.used / (1024 ** 3)),
        "total_gb": float(mem.total / (1024 ** 3)),
        "exceeded": bool(used_ratio > limit_ratio),
    }


def check_resources_and_warn(
    device: torch.device,
    step: int,
    gpu_limit: float = GPU_MEM_LIMIT_RATIO,
    ram_limit: float = RAM_LIMIT_RATIO,
    interval: int = RESOURCE_CHECK_INTERVAL,
) -> None:
    if step % max(1, int(interval)) != 0:
        return
    gpu_status = check_gpu_memory(device, limit_ratio=gpu_limit)
    ram_status = check_ram_usage(limit_ratio=ram_limit)
    if gpu_status["exceeded"]:
        print(
            f"WARNING: GPU memory {gpu_status['used_ratio']*100:.1f}% "
            f"({gpu_status['used_gb']:.1f}/{gpu_status['total_gb']:.1f}GB) exceeds limit "
            f"{gpu_limit*100:.0f}% at step {step}."
        )
    if ram_status["exceeded"]:
        print(
            f"WARNING: RAM usage {ram_status['used_ratio']*100:.1f}% "
            f"({ram_status['used_gb']:.1f}/{ram_status['total_gb']:.1f}GB) exceeds limit "
            f"{ram_limit*100:.0f}% at step {step}."
        )


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json_config(config_path: Optional[Path]) -> Dict[str, object]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def video_folder_name(video_id: int) -> str:
    return f"video_{int(video_id):02d}"


def video_feature_name(video_id: int) -> str:
    return f"video_{int(video_id):02d}_features.pt"


def annotation_name(video_id: int) -> str:
    return f"annotations_{int(video_id):02d}.csv"


def read_annotation_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_segment_transition_counts_from_csv_root(
    csv_root: Path,
    *,
    video_ids: Optional[Sequence[int]] = None,
    allowed_transitions_raw: Optional[Dict[int, Sequence[int]]] = None,
) -> Dict[int, Dict[int, int]]:
    selected_ids = None if video_ids is None else {int(video_id) for video_id in video_ids}
    raw_sequences: List[List[int]] = []
    for csv_path in sorted(Path(csv_root).glob("annotations_*.csv")):
        match = re.match(r"annotations_(\d+)\.csv$", csv_path.name)
        if match is None:
            continue
        video_id = int(match.group(1))
        if selected_ids is not None and video_id not in selected_ids:
            continue
        rows = read_annotation_rows(csv_path)
        raw_sequences.append([int(row["int_step"]) for row in rows])
    allowed_map = (
        {int(src): set(int(dst) for dst in dsts) for src, dsts in allowed_transitions_raw.items()}
        if allowed_transitions_raw is not None
        else ALLOWED_TRANSITIONS_RAW
    )
    return count_segment_transition_counts(
        raw_sequences,
        valid_labels=set(LABELS12),
        allowed_transitions_raw=allowed_map,
    )


def align_feature_and_rows(features: torch.Tensor, rows: Sequence[Dict[str, str]]) -> Tuple[torch.Tensor, List[Dict[str, str]]]:
    aligned_length = min(int(features.shape[0]), len(rows))
    return features[:aligned_length], list(rows[:aligned_length])


def map_raw_steps_to_targets(raw_steps: Sequence[int]) -> torch.Tensor:
    return torch.tensor([LABEL_TO_INDEX.get(int(step), IGNORE_INDEX) for step in raw_steps], dtype=torch.long)


def step_index_to_raw_label(step_index: int) -> int:
    if 0 <= int(step_index) < len(LABELS12):
        return int(LABELS12[int(step_index)])
    return IGNORE_INDEX


def discover_video_ids_from_features(features_root: Path, csv_root: Path) -> List[int]:
    feature_ids = set()
    for path in Path(features_root).glob("video_*_features.pt"):
        match = re.match(r"video_(\d+)_features\.pt$", path.name)
        if match:
            feature_ids.add(int(match.group(1)))
    csv_ids = set()
    for path in Path(csv_root).glob("annotations_*.csv"):
        match = re.match(r"annotations_(\d+)\.csv$", path.name)
        if match:
            csv_ids.add(int(match.group(1)))
    video_ids = sorted(feature_ids & csv_ids)
    if not video_ids:
        raise FileNotFoundError("No video has both a feature tensor and an annotation CSV.")
    return video_ids


def discover_video_ids_from_images(images_root: Path, csv_root: Path) -> List[int]:
    image_ids = set()
    for path in Path(images_root).glob("video_*"):
        match = re.match(r"video_(\d+)$", path.name)
        if match and path.is_dir():
            image_ids.add(int(match.group(1)))
    csv_ids = set()
    for path in Path(csv_root).glob("annotations_*.csv"):
        match = re.match(r"annotations_(\d+)\.csv$", path.name)
        if match:
            csv_ids.add(int(match.group(1)))
    video_ids = sorted(image_ids & csv_ids)
    if not video_ids:
        raise FileNotFoundError("No video has both an image directory and an annotation CSV.")
    return video_ids


def build_official_split(video_ids: Sequence[int]) -> Dict[str, List[int]]:
    train_videos = [int(v) for v in video_ids if int(v) not in OFFICIAL_VAL_VIDEOS]
    val_videos = [int(v) for v in video_ids if int(v) in OFFICIAL_VAL_VIDEOS]
    return {
        "train_videos": sorted(train_videos),
        "val_videos": sorted(val_videos),
    }


def load_split_json(split_json: Path) -> Dict[str, List[int]]:
    path = Path(split_json)
    if not path.is_file():
        raise FileNotFoundError(f"split json not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"split json must be an object: {path}")
    train_videos = payload.get("train_videos")
    val_videos = payload.get("val_videos")
    if not isinstance(train_videos, list) or not isinstance(val_videos, list):
        raise ValueError(f"split json must contain list fields train_videos and val_videos: {path}")
    return {
        "train_videos": [int(video_id) for video_id in train_videos],
        "val_videos": [int(video_id) for video_id in val_videos],
    }


def resolve_video_split(
    video_ids: Sequence[int],
    *,
    split_json: Optional[Path] = None,
) -> Dict[str, List[int]]:
    if split_json is None:
        return build_official_split(video_ids)

    discovered = {int(video_id) for video_id in video_ids}
    split = load_split_json(Path(split_json))
    train_videos = [int(video_id) for video_id in split["train_videos"]]
    val_videos = [int(video_id) for video_id in split["val_videos"]]
    train_set = set(train_videos)
    val_set = set(val_videos)
    if not train_videos:
        raise ValueError(f"split json has empty train_videos: {split_json}")
    if not val_videos:
        raise ValueError(f"split json has empty val_videos: {split_json}")
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(f"split json has overlapping train/val videos: {overlap}")
    unknown = sorted((train_set | val_set) - discovered)
    if unknown:
        raise ValueError(f"split json references unknown videos: {unknown}")
    unused = sorted(discovered - train_set - val_set)
    if unused:
        print(f"split_json_unused_videos: {unused}")
    return {
        "train_videos": sorted(train_set),
        "val_videos": sorted(val_set),
    }


def build_video_items(
    features_root: Optional[Path],
    csv_root: Path,
    video_ids: Sequence[int],
    *,
    input_mode: str,
    max_videos: Optional[int] = None,
) -> List[Dict[str, object]]:
    selected_ids = list(video_ids[: max_videos if max_videos is not None else None])
    items: List[Dict[str, object]] = []
    for video_id in selected_ids:
        label_path = Path(csv_root) / annotation_name(video_id)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {label_path}")
        item: Dict[str, object] = {"video_id": int(video_id), "label_path": label_path}
        if input_mode == "features":
            if features_root is None:
                raise ValueError("features_root is required in features mode.")
            feature_path = Path(features_root) / video_feature_name(video_id)
            if not feature_path.is_file():
                raise FileNotFoundError(f"Missing feature file: {feature_path}")
            item["feature_path"] = feature_path
        items.append(item)
    return items


def resolve_norm_stats(norm: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if norm == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    return DEFAULT_MEAN, DEFAULT_STD


def resolve_frame_paths(
    video_dir: Path,
    annotation_path: Path,
    *,
    max_trailing_missing: int = MAX_TRAILING_MISSING_FRAMES,
) -> List[Path]:
    rows = read_annotation_rows(annotation_path)
    frame_paths: List[Path] = []
    missing_paths: List[Path] = []
    for row in rows:
        frame_path = video_dir / f"{row['int_time']}.png"
        if not frame_path.is_file():
            missing_paths.append(frame_path)
            continue
        if missing_paths:
            raise FileNotFoundError(f"Missing interior frame cannot be ignored: {missing_paths[0]}")
        frame_paths.append(frame_path)
    if missing_paths and len(missing_paths) > max_trailing_missing:
        raise FileNotFoundError(
            f"Trailing missing frames exceed the limit {max_trailing_missing}: {missing_paths[0]}"
        )
    return frame_paths


def maybe_apply_gpu_memory_fraction(gpu_ids: Optional[Sequence[int]], limit: float) -> None:
    if not torch.cuda.is_available() or float(limit) >= 1.0:
        return
    fraction = min(max(float(limit), 0.1), 1.0)
    if gpu_ids:
        for gid in gpu_ids:
            torch.cuda.set_per_process_memory_fraction(fraction, gid)
            print(f"gpu_memory_fraction: device={gid} limit={fraction*100:.0f}%")
        return
    dev_idx = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(fraction, dev_idx)
    print(f"gpu_memory_fraction: device={dev_idx} limit={fraction*100:.0f}%")


def print_resource_snapshot(prefix: str, device: torch.device, gpu_limit: float, ram_limit: float) -> None:
    gpu_status = check_gpu_memory(device, limit_ratio=gpu_limit)
    ram_status = check_ram_usage(limit_ratio=ram_limit)
    print(
        f"{prefix}: GPU={gpu_status['used_gb']:.1f}/{gpu_status['total_gb']:.1f}GB "
        f"({gpu_status['used_ratio']*100:.1f}%) RAM={ram_status['used_gb']:.1f}/{ram_status['total_gb']:.1f}GB "
        f"({ram_status['used_ratio']*100:.1f}%)"
    )


__all__ = [
    "ALLOWED_TRANSITIONS_RAW",
    "BACKBONE_CHUNK_SIZE",
    "CENTER_FRAME_SIZE",
    "CLASS_WEIGHT_BOOSTS",
    "CLASS_WEIGHT_BOOSTS_RAW",
    "DEFAULT_PAIR_TRANSITION_RAW_PAIRS",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "FEATURE_DIM",
    "FOCUS_BOUNDARY_RAW_PAIRS",
    "GPU_MEM_LIMIT_RATIO",
    "GRAD_CLIP_MAX_NORM",
    "IGNORE_INDEX",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LABELS12",
    "LABEL_TO_INDEX",
    "MAX_TRAILING_MISSING_FRAMES",
    "MIN_FRAME_WEIGHT",
    "OFFICIAL_VAL_VIDEOS",
    "RAW_TRANSITION_COUNTS",
    "build_log_transition_tensor",
    "parse_transition_pairs_spec",
    "RAM_LIMIT_RATIO",
    "RESOURCE_CHECK_INTERVAL",
    "align_feature_and_rows",
    "annotation_name",
    "build_log_transition_matrix_numpy",
    "build_official_split",
    "build_video_items",
    "check_gpu_memory",
    "check_ram_usage",
    "check_resources_and_warn",
    "collect_segment_transition_counts_from_csv_root",
    "discover_video_ids_from_features",
    "discover_video_ids_from_images",
    "ensure_repo_importable",
    "HARD_CLASS_RAW_LABELS",
    "load_json_config",
    "load_split_json",
    "map_raw_steps_to_targets",
    "maybe_apply_gpu_memory_fraction",
    "print_resource_snapshot",
    "read_annotation_rows",
    "resolve_device_and_gpu_ids",
    "resolve_frame_paths",
    "resolve_norm_stats",
    "safe_torch_load",
    "resolve_video_split",
    "set_seed",
    "step_index_to_raw_label",
    "unwrap_model",
    "video_feature_name",
    "video_folder_name",
    "wrap_model_data_parallel",
]
