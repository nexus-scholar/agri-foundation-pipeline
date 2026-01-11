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


TOMATO_LEAF_CLASSES = {
    "0": "early_blight",
    "1": "black_spot",
    "2": "late_blight",
    "3": "mold",
    "4": "bacterial_spot",
    "5": "target_spot",
    "6": "healthy"
}


def _get_class_from_yolo(label_path: Path) -> str:
    """Read the first class index from a YOLO txt file."""
    if not label_path.exists():
        return "healthy"
    try:
        content = label_path.read_text().strip()
        if not content:
            return "healthy"
        # Take the first object's class
        first_line = content.split('\n')[0]
        class_idx = first_line.split()[0]
        return TOMATO_LEAF_CLASSES.get(class_idx, "healthy")
    except Exception:
        return "healthy"


def process_tomato_leaf(raw_dir: Path, processed_dir: Path) -> list[dict[str, str]]:
    """Process tomato leaf dataset into YOLO-friendly structure with specific labels."""
    raw_dir = Path(raw_dir)
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

    ensure_dir(processed_dir)

    image_out_dir = processed_dir / "images"
    label_out_dir = processed_dir / "labels"
    ensure_dir(image_out_dir)
    ensure_dir(label_out_dir)

    records: List[dict[str, str]] = []
    stats = Counter()

    # Copy raw images (Unannotated)
    print("Copying raw images...")
    for image_path in sorted(raw_images_root.glob("*.jpg")):
        dst = image_out_dir / image_path.name
        copy_file(image_path, dst)
        records.append({
            "filename": image_path.name,
            "label": "tomato_healthy",  # Assume healthy if in raw/unannotated? Or unknown.
            "crop": "tomato",
            "disease": "healthy",
            "source": "tomato_leaf_raw",
            "path": f"TomatoLeaf_processed/images/{image_path.name}"
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
                orig = str(Path(item["original"]))
                short = item["short_name"]
                short_to_orig[short] = orig
                orig_to_short[orig] = short

    # Copy annotated YOLO structure
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
            
            # Resolve label path
            label_found = False
            img_name = image_path.name
            
            if img_name in short_to_orig:
                orig_img_zip_path = short_to_orig[img_name]
            else:
                try:
                    orig_img_zip_path = str(image_path.relative_to(raw_dir))
                except ValueError:
                    orig_img_zip_path = ""

            label_name = None
            if orig_img_zip_path:
                parts = list(Path(orig_img_zip_path).parts)
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i].lower() == "images":
                        parts[i] = "labels"
                        break
                p = Path(*parts)
                orig_label_zip_path = str(p.with_suffix(".txt"))
                
                if orig_label_zip_path in orig_to_short:
                    label_name = orig_to_short[orig_label_zip_path]
                else:
                    label_name = Path(orig_label_zip_path).name
                
                label_path = split_labels / label_name
                if label_path.exists():
                    copy_file(label_path, split_out_labels / label_path.name)
                    label_found = True

            if not label_found:
                label_path = split_labels / (image_path.stem + ".txt")
                if label_path.exists():
                    copy_file(label_path, split_out_labels / label_path.name)
                    label_found = True
                else:
                    # Create empty label if missing
                    label_path = split_out_labels / (image_path.stem + ".txt")
                    label_path.write_text("", encoding="utf-8")
            
            # Get specific disease label from YOLO file
            disease_name = _get_class_from_yolo(split_labels / (label_name or (image_path.stem + ".txt")))
            full_label = f"tomato_{disease_name}"

            stats[f"annotated_{split}"] += 1
            records.append({
                "filename": image_path.name,
                "label": full_label,
                "crop": "tomato",
                "disease": disease_name,
                "source": f"tomato_leaf_{split}",
                "path": f"TomatoLeaf_processed/annotated/{split}/images/{image_path.name}"
            })

    # Save metadata
    metadata = {
        "dataset": "TomatoLeafMulticlass",
        "processed": list(stats.items()),
        "total_records": len(records),
        "class_mapping": TOMATO_LEAF_CLASSES
    }
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # Save labels.csv
    csv_path = processed_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "label", "crop", "disease", "source", "path"])
        writer.writeheader()
        writer.writerows(records)

    safe_rmtree(raw_dir)
    return records
