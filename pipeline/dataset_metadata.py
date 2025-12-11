"""Utilities shared by dataset processors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class DatasetStats:
    label_counts: dict[str, int]
    total_images: int
    num_classes: int


def is_valid_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}

