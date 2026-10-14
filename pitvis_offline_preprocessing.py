from dataclasses import dataclass
from typing import Dict

from PIL import Image


DEFAULT_OFFLINE_INTERPOLATION = "bicubic"
OFFLINE_INTERPOLATION_CHOICES = ("bilinear", "bicubic", "lanczos")

_INTERPOLATION_TO_PIL: Dict[str, Image.Resampling] = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _resolve_resample(interpolation: str) -> Image.Resampling:
    key = str(interpolation).lower()
    if key not in _INTERPOLATION_TO_PIL:
        raise ValueError(
            f"Unknown interpolation={key!r}; choices: {', '.join(OFFLINE_INTERPOLATION_CHOICES)}"
        )
    return _INTERPOLATION_TO_PIL[key]


def validate_resize_and_crop(resize: int, crop_size: int) -> None:
    resize = int(resize)
    crop_size = int(crop_size)
    if resize <= 0 or crop_size <= 0:
        raise ValueError("resize and crop_size must be positive integers.")
    if crop_size > resize:
        raise ValueError(
            f"crop_size={crop_size} cannot exceed resize(short side)={resize}; "
            "the requested center crop would be undefined."
        )


def resize_short_side_preserve_aspect(
    image: Image.Image,
    short_side: int,
    interpolation: str = DEFAULT_OFFLINE_INTERPOLATION,
) -> Image.Image:
    short_side = int(short_side)
    if short_side <= 0:
        raise ValueError("short_side must be a positive integer.")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")
    if width <= height:
        new_width = short_side
        new_height = max(1, round(height * short_side / width))
    else:
        new_height = short_side
        new_width = max(1, round(width * short_side / height))
    return image.resize((new_width, new_height), _resolve_resample(interpolation))


def center_crop_square(image: Image.Image, crop_size: int) -> Image.Image:
    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError("crop_size must be a positive integer.")
    width, height = image.size
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return image.crop((left, top, left + crop_w, top + crop_h))


@dataclass(frozen=True)
class OfflineCenterCropTransform:
    resize: int
    crop_size: int
    interpolation: str = DEFAULT_OFFLINE_INTERPOLATION

    def __post_init__(self) -> None:
        validate_resize_and_crop(self.resize, self.crop_size)
        _resolve_resample(self.interpolation)

    def __call__(self, image: Image.Image) -> Image.Image:
        resized = resize_short_side_preserve_aspect(
            image,
            short_side=self.resize,
            interpolation=self.interpolation,
        )
        return center_crop_square(resized, crop_size=self.crop_size)
