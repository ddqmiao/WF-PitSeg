import argparse
import csv
import re
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from pitvis_offline_preprocessing import (
    DEFAULT_OFFLINE_INTERPOLATION,
    OFFLINE_INTERPOLATION_CHOICES,
    OfflineCenterCropTransform,
)
from pitvis_progress_utils import build_extract_progress_postfix

FEATURE_DIM = 1536
DEFAULT_MEAN = (0.44476654, 0.35751768, 0.34613244)
DEFAULT_STD = (0.28885465, 0.25989645, 0.26113421)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FALLBACK_BATCH_SIZES = (128, 96, 64, 32)
MAX_TRAILING_MISSING_FRAMES = 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-frame LemonFM features and save one .pt tensor per video"
    )
    parser.add_argument("--repo-root", required=False, type=Path, default=None, help="Local LEMON source root")
    parser.add_argument("--weights", required=True, type=Path, help="LemonFM weights or a training checkpoint")
    parser.add_argument("--images-root", required=True, type=Path, help="Frame root containing video_xx directories")
    parser.add_argument("--csv-root", required=True, type=Path, help="Annotation directory or one annotations_xx.csv file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--video-id", type=int, help="Extract only one video, for example 25")
    parser.add_argument("--batch-size", type=int, default=64, help="Initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Execution device")
    parser.add_argument("--gpu-id", type=int, default=None, help="Optional single GPU index")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto", help="Inference precision")
    parser.add_argument("--norm", choices=("lemon_video", "imagenet"), default="lemon_video", help="Normalization preset")
    parser.add_argument("--resize", type=int, default=256, help="Resize short side")
    parser.add_argument("--crop-size", type=int, default=224, help="Center-crop size")
    parser.add_argument(
        "--interpolation",
        choices=OFFLINE_INTERPOLATION_CHOICES,
        default=DEFAULT_OFFLINE_INTERPOLATION,
        help="Offline resize interpolation",
    )
    parser.add_argument("--from-checkpoint", action="store_true",
                        help="Load a fine-tuned backbone from best_step_metric.pt")
    return parser.parse_args(argv)


def video_folder_name(video_id: int) -> str:
    return f"video_{int(video_id):02d}"


def annotation_file_name(video_id: int) -> str:
    return f"annotations_{int(video_id):02d}.csv"


def ensure_repo_importable(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    extra_paths = [repo_root, repo_root / "lemonfm", repo_root / "src"]
    for path in extra_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)


def load_lemonfm_backbone(repo_root: Path, weights_path: Path, device: torch.device) -> nn.Module:
    ensure_repo_importable(repo_root)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    if "teacher" not in checkpoint:
        raise KeyError("Weights file has no 'teacher' key for the LemonFM teacher backbone.")

    state_dict = checkpoint["teacher"]
    backbone_state = {
        key.replace("backbone.", ""): value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise KeyError("No backbone.* parameters were found in the weights file.")

    model = torchvision.models.convnext_large(weights=None)
    model.classifier[2] = nn.Identity()
    message = model.load_state_dict(backbone_state, strict=False)
    unexpected = list(message.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected weight keys: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


def load_backbone_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load backbone from a training checkpoint (best_step_metric.pt).

    The training checkpoint stores model_state with keys like:
      backbone_wrapper.backbone.features.0.* ...
    We extract the backbone.* portion and load into a ConvNeXt-Large.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint has no 'model_state' key.")

    state_dict = checkpoint["model_state"]
    # Extract backbone weights from backbone_wrapper.backbone.*
    prefix = "backbone_wrapper.backbone."
    backbone_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not backbone_state:
        raise KeyError(f"No {prefix}* parameters were found in the checkpoint.")

    model = torchvision.models.convnext_large(weights=None)
    model.classifier[2] = nn.Identity()
    message = model.load_state_dict(backbone_state, strict=False)
    unexpected = list(message.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected weight keys: {unexpected[:5]}")

    loaded_count = len(backbone_state)
    print(f"[INFO] Loaded {loaded_count} backbone parameters from the training checkpoint")
    model.to(device)
    model.eval()
    return model


def read_annotation_rows(annotation_path: Path) -> List[dict]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
    with annotation_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_frame_paths(
    video_dir: Path,
    annotation_path: Path,
    max_trailing_missing: int = MAX_TRAILING_MISSING_FRAMES,
) -> List[Path]:
    rows = read_annotation_rows(annotation_path)
    frame_paths: List[Path] = []
    missing_paths: List[Path] = []
    for row in rows:
        time_value = row["int_time"]
        frame_path = video_dir / f"{time_value}.png"
        if not frame_path.is_file():
            missing_paths.append(frame_path)
            continue
        if missing_paths:
            raise FileNotFoundError(f"Missing interior frame cannot be ignored: {missing_paths[0]}")
        frame_paths.append(frame_path)
    if missing_paths:
        if len(missing_paths) > max_trailing_missing:
            raise FileNotFoundError(
                f"Trailing missing frames exceed the limit {max_trailing_missing}: {missing_paths[0]}"
            )
        warnings.warn(
            f"Ignoring {len(missing_paths)} trailing missing frame(s): "
            f"{missing_paths[0].name} ... {missing_paths[-1].name}",
            stacklevel=2,
        )
    return frame_paths


def build_transform(norm: str, resize: int, crop_size: int, interpolation: str) -> transforms.Compose:
    if norm == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = DEFAULT_MEAN, DEFAULT_STD
    return transforms.Compose(
        [
            OfflineCenterCropTransform(resize=resize, crop_size=crop_size, interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class FrameDataset(Dataset):
    def __init__(self, frame_paths: Sequence[Path], transform: transforms.Compose):
        self.frame_paths = list(frame_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        frame_path = self.frame_paths[index]
        with Image.open(frame_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


def make_dataloader(
    frame_paths: Sequence[Path],
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = FrameDataset(frame_paths, transform)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(**loader_kwargs)


def resolve_precision(device: torch.device, requested: str) -> str:
    if requested == "fp32":
        return "fp32"
    if requested == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 is supported only on CUDA devices.")
        return "bf16"
    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def extract_single_video_features(
    model: nn.Module,
    video_id: int,
    annotation_path: Path,
    images_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    precision: str,
    norm: str = "lemon_video",
    resize: int = 256,
    crop_size: int = 224,
    interpolation: str = DEFAULT_OFFLINE_INTERPOLATION,
    global_frame_bar: Optional[tqdm] = None,
    completed_videos: int = 0,
    total_videos: int = 1,
) -> Path:
    video_dir = images_root / video_folder_name(video_id)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video frame directory does not exist: {video_dir}")

    frame_paths = resolve_frame_paths(video_dir, annotation_path)
    transform = build_transform(
        norm=norm,
        resize=resize,
        crop_size=crop_size,
        interpolation=interpolation,
    )
    data_loader = make_dataloader(
        frame_paths=frame_paths,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    features = torch.empty((len(frame_paths), FEATURE_DIM), dtype=torch.float32)
    precision = resolve_precision(device=device, requested=precision)

    offset = 0
    total_frames = len(frame_paths)
    processed_frames = 0
    video_name = video_folder_name(video_id)
    with torch.no_grad(), tqdm(
        total=total_frames,
        desc=f"{video_name} frames",
        unit="frame",
        leave=False,
        dynamic_ncols=True,
        position=2,
    ) as video_frame_bar:
        for images in data_loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            with autocast_context(device=device, precision=precision):
                batch_features = model(images)
            batch_features = batch_features.detach().to("cpu", dtype=torch.float32)
            if batch_features.ndim != 2 or batch_features.shape[1] != FEATURE_DIM:
                raise ValueError(
                    f"Invalid model output shape; expected [B, {FEATURE_DIM}], "
                    f"received {tuple(batch_features.shape)}"
                )
            next_offset = offset + batch_features.shape[0]
            features[offset:next_offset] = batch_features
            offset = next_offset
            processed = batch_features.shape[0]
            processed_frames += processed
            video_frame_bar.update(processed)
            video_frame_bar.set_postfix(
                build_extract_progress_postfix(
                    video_name=video_name,
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                    batch_size=images.shape[0],
                    completed_videos=completed_videos,
                    total_videos=total_videos,
                )
            )
            if global_frame_bar is not None:
                global_frame_bar.update(processed)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_folder_name(video_id)}_features.pt"
    torch.save(features, output_path)
    return output_path


def resolve_annotation_path(csv_root: Path, video_id: int) -> Path:
    if csv_root.is_file():
        return csv_root
    annotation_path = csv_root / annotation_file_name(video_id)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
    return annotation_path


def resolve_video_ids(csv_root: Path, video_id: Optional[int]) -> List[int]:
    if video_id is not None:
        return [int(video_id)]
    if csv_root.is_file():
        match = re.search(r"annotations_(\d+)\.csv$", csv_root.name)
        if not match:
            raise ValueError(f"Could not parse a video ID from the filename: {csv_root}")
        return [int(match.group(1))]
    video_ids = []
    for csv_path in sorted(csv_root.glob("annotations_*.csv")):
        match = re.search(r"annotations_(\d+)\.csv$", csv_path.name)
        if match:
            video_ids.append(int(match.group(1)))
    if not video_ids:
        raise FileNotFoundError(f"No annotations_*.csv files found in: {csv_root}")
    return video_ids


def run_with_batch_fallback(
    model: nn.Module,
    video_id: int,
    annotation_path: Path,
    images_root: Path,
    output_dir: Path,
    initial_batch_size: int,
    num_workers: int,
    device: torch.device,
    precision: str,
    norm: str,
    resize: int,
    crop_size: int,
    interpolation: str,
    global_frame_bar: Optional[tqdm] = None,
    completed_videos: int = 0,
    total_videos: int = 1,
) -> Path:
    candidates: List[int] = []
    for batch_size in (initial_batch_size,) + FALLBACK_BATCH_SIZES:
        if batch_size > 0 and batch_size not in candidates:
            candidates.append(batch_size)

    last_error: Optional[RuntimeError] = None
    for batch_size in candidates:
        try:
            return extract_single_video_features(
                model=model,
                video_id=video_id,
                annotation_path=annotation_path,
                images_root=images_root,
                output_dir=output_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                precision=precision,
                norm=norm,
                resize=resize,
                crop_size=crop_size,
                interpolation=interpolation,
                global_frame_bar=global_frame_bar,
                completed_videos=completed_videos,
                total_videos=total_videos,
            )
        except RuntimeError as error:
            if device.type != "cuda" or not is_oom_error(error):
                raise
            last_error = error
            torch.cuda.empty_cache()
            print(f"[WARN] batch_size={batch_size} ran out of GPU memory; retrying with a smaller batch.", flush=True)
    if last_error is not None:
        raise last_error
    raise RuntimeError("No viable batch-size candidate remains.")


def configure_runtime_threads() -> None:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_runtime_threads()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "cuda" and args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"[INFO] Using GPU {args.gpu_id}")

    if args.from_checkpoint:
        print(f"[INFO] Loading a fine-tuned backbone from checkpoint: {args.weights}")
        model = load_backbone_from_checkpoint(
            checkpoint_path=args.weights,
            device=device,
        )
    else:
        if args.repo_root is None:
            raise ValueError("--repo-root is required when loading original LemonFM weights.")
        model = load_lemonfm_backbone(
            repo_root=args.repo_root,
            weights_path=args.weights,
            device=device,
        )

    video_ids = resolve_video_ids(args.csv_root, args.video_id)
    jobs = []
    total_frames = 0
    for video_id in video_ids:
        annotation_path = resolve_annotation_path(args.csv_root, video_id)
        frame_count = len(resolve_frame_paths(args.images_root / video_folder_name(video_id), annotation_path))
        jobs.append((video_id, annotation_path, frame_count))
        total_frames += frame_count

    completed_videos = 0
    with tqdm(total=len(jobs), desc="Videos", unit="video", dynamic_ncols=True, position=0) as video_bar, \
         tqdm(total=total_frames, desc="Frames", unit="frame", dynamic_ncols=True, position=1) as frame_bar:
        for video_id, annotation_path, frame_count in jobs:
            video_name = video_folder_name(video_id)
            video_bar.set_postfix({"current": video_name, "frames": f"{frame_count}"})
            output_path = run_with_batch_fallback(
                model=model,
                video_id=video_id,
                annotation_path=annotation_path,
                images_root=args.images_root,
                output_dir=args.output_dir,
                initial_batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                precision=args.precision,
                norm=args.norm,
                resize=args.resize,
                crop_size=args.crop_size,
                interpolation=args.interpolation,
                global_frame_bar=frame_bar,
                completed_videos=completed_videos,
                total_videos=len(jobs),
            )
            completed_videos += 1
            video_bar.update(1)
            video_bar.set_postfix({"current": video_name, "done": f"{completed_videos}/{len(jobs)}"})
            print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    main()
