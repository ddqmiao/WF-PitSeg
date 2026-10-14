from collections import OrderedDict
from typing import Dict, Iterable, MutableMapping


def count_items_by_video(video_names: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = OrderedDict()
    for video_name in video_names:
        key = str(video_name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def note_video_item_completed(remaining_counts: MutableMapping[str, int], video_name: str) -> bool:
    key = str(video_name)
    if key not in remaining_counts:
        raise KeyError(f"Unknown video_name={key!r}")
    remaining = int(remaining_counts[key])
    if remaining <= 0:
        return False
    remaining -= 1
    remaining_counts[key] = remaining
    return remaining == 0


def build_resize_progress_postfix(
    *,
    processed: int,
    skipped: int,
    errors: int,
    completed_videos: int,
    total_videos: int,
) -> Dict[str, str]:
    return {
        "processed": str(int(processed)),
        "skipped": str(int(skipped)),
        "errors": str(int(errors)),
        "videos": f"{int(completed_videos)}/{int(total_videos)}",
    }


def build_extract_progress_postfix(
    *,
    video_name: str,
    processed_frames: int,
    total_frames: int,
    batch_size: int,
    completed_videos: int,
    total_videos: int,
) -> Dict[str, str]:
    return {
        "video": str(video_name),
        "frames": f"{int(processed_frames)}/{int(total_frames)}",
        "batch": str(int(batch_size)),
        "videos": f"{int(completed_videos)}/{int(total_videos)}",
    }
