"""Tomato Leaf Multiclass dataset processing."""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, ensure_dir, safe_rmtree

DATASET_ROOT_NAME = "Tomato Leaf Dataset  A dataset for multiclass disease detection and classification"
RAW_FOLDER = "TomatoLeafMulticlass (Raw Data)"
ANNOTATED_FOLDER = "TomatoLeafMulticlass (Annotated)"
YOLO_SPLITS = ["train", "test"]


@dataclass
class TomatoLeafRecord:
    filename: str
    label: str
    bbox: tuple[float, float, float, float] | None
    source: str


def process_tomato_leaf(raw_dir: Path, processed_dir: Path) -> list[dict[str, str]]:
    """Process tomato leaf dataset into YOLO-friendly structure."""
    raw_dir = Path(raw_dir)
    extracted_root = raw_dir
    processed_dir = Path(processed_dir)

    print("\n" + "=" * 60)
    print("PROCESSING TOMATO LEAF DATASET")
    print("=" * 60)

    safe_rmtree(processed_dir)

    dataset_root = raw_dir
    if (raw_dir / DATASET_ROOT_NAME).exists():
        dataset_root = raw_dir / DATASET_ROOT_NAME

    raw_images_root = dataset_root / RAW_FOLDER / "images"
    annotated_root = dataset_root / ANNOTATED_FOLDER

    if not raw_images_root.exists():
        print(f"ERROR: Raw images folder not found: {raw_images_root}")
        return []

    safe_rmtree(processed_dir)
    ensure_dir(processed_dir)

    image_out_dir = processed_dir / "images"
    label_out_dir = processed_dir / "labels"
    ensure_dir(image_out_dir)
    ensure_dir(label_out_dir)

    records: List[dict[str, str]] = []
    stats = Counter()

    # Copy raw images
    print("Copying raw images...")
    for image_path in sorted(raw_images_root.glob("*.jpg")):
        dst = image_out_dir / image_path.name
        copy_file(image_path, dst)
        records.append({
            "filename": image_path.name,
            "label": "unknown",
            "crop": "tomato",
            "disease": "unknown",
            "source": "tomato_leaf_raw",
            "path": f"TomatoLeaf/images/{image_path.name}"
        })
        stats["raw_images"] += 1

    # Load rename map
    rename_map_path = raw_dir.parent / "tomatoleaf_renamed_files.json"
    short_to_orig = {}
    orig_to_short = {}
    if rename_map_path.exists():
        with open(rename_map_path, "r", encoding="utf-8") as f:
            renamed_data = json.load(f)
            for item in renamed_data:
                # Normalize paths to handle separator differences
                orig = str(Path(item["original"]))
                short = item["short_name"]
                short_to_orig[short] = orig
                orig_to_short[orig] = short

    # Copy annotated YOLO structure if present
    for split in YOLO_SPLITS:
        split_images = annotated_root / split / "images"
        split_labels = annotated_root / split / "labels"
        if not split_images.exists():
            continue
        split_out_images = processed_dir / "annotated" / split / "images"
        split_out_labels = processed_dir / "annotated" / split / "labels"
        ensure_dir(split_out_images)
        ensure_dir(split_out_labels)
        print(f"Processing annotated split: {split}")
        for image_path in sorted(split_images.glob("*.jpg")):
            dst_img = split_out_images / image_path.name
            copy_file(image_path, dst_img)
            
            # Resolve label path using rename map
            label_found = False
            
            # 1. Determine original image path (relative to zip root)
            img_name = image_path.name
            if img_name in short_to_orig:
                orig_img_zip_path = short_to_orig[img_name]
            else:
                try:
                    # If not renamed, path relative to raw_dir is the zip path
                    orig_img_zip_path = str(image_path.relative_to(raw_dir))
                except ValueError:
                    orig_img_zip_path = ""

            if orig_img_zip_path:
                # 2. Derive expected original label path
                # Replace extension with .txt (handle .jpg, .JPG, etc)
                p = Path(orig_img_zip_path)
                
                # Handle images -> labels directory change
                parts = list(p.parts)
                # Find 'images' component and replace with 'labels'
                # Look from the end
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i].lower() == "images":
                        parts[i] = "labels"
                        break
                
                p = Path(*parts)
                orig_label_zip_path = str(p.with_suffix(".txt"))
                
                # 3. Find if label exists on disk (renamed or not)
                if orig_label_zip_path in orig_to_short:
                    label_name = orig_to_short[orig_label_zip_path]
                else:
                    label_name = Path(orig_label_zip_path).name
                
                label_path = split_labels / label_name
                if label_path.exists():
                    copy_file(label_path, split_out_labels / label_path.name)
                    label_found = True

            # Fallback to simple matching if complex logic failed
            if not label_found:
                label_path = split_labels / (image_path.stem + ".txt")
                if label_path.exists():
                    copy_file(label_path, split_out_labels / label_path.name)
                else:
                    (split_out_labels / (image_path.stem + ".txt")).write_text("", encoding="utf-8")
            
            stats[f"annotated_{split}"] += 1
            records.append({
                "filename": image_path.name,
                "label": "tomato_leaf",
                "crop": "tomato",
                "disease": "multiclass",
                "source": f"tomato_leaf_{split}",
                "path": f"TomatoLeaf/annotated/{split}/images/{image_path.name}"
            })

    # Save metadata
    metadata = {
        "dataset": "TomatoLeafMulticlass",
        "processed": list(stats.items()),
        "total_records": len(records)
    }
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    safe_rmtree(extracted_root)
    return records
