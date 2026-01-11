"""Cassava Leaf Disease dataset processing."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, safe_rmtree
from .label_utils import normalize_label


def process_cassava(source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Process Cassava dataset into normalized folder + CSV."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("PROCESSING CASSAVA LEAF DISEASE")
    print("=" * 60)

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return []

    # Check for required files
    train_csv_path = source_dir / "train.csv"
    label_map_path = source_dir / "label_num_to_disease_map.json"
    images_dir = source_dir / "train_images"

    if not all([train_csv_path.exists(), label_map_path.exists(), images_dir.exists()]):
        print(f"ERROR: Missing required files in {source_dir}")
        print(f"Expected: train.csv, label_num_to_disease_map.json, train_images/")
        return []

    # Load label map
    try:
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            # Ensure keys are integers (or matching CSV)
            label_map = {str(k): v for k, v in label_map.items()}
    except Exception as e:
        print(f"ERROR: Failed to load label map: {e}")
        return []

    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data: list[dict[str, str]] = []
    stats = defaultdict(int)
    label_counters = defaultdict(int)

    print(f"Reading {train_csv_path}...")
    
    # Read CSV
    try:
        with open(train_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        return []

    print(f"Found {len(rows)} entries in CSV")

    for row in rows:
        image_id = row['image_id']
        label_id = row['label']
        
        if label_id not in label_map:
            print(f"Warning: Unknown label id {label_id} for image {image_id}")
            continue

        original_disease_name = label_map[label_id]
        
        # Normalize: Crop is Cassava
        crop = "cassava"
        # Simplify disease name (remove parens, etc)
        # e.g. "Cassava Bacterial Blight (CBB)" -> "bacterial_blight"
        disease_part = original_disease_name.replace("Cassava", "").strip()
        disease_part = disease_part.split('(')[0].strip()
        disease_part = disease_part.lower().replace(" ", "_")
        
        if disease_part == "healthy":
            label = "cassava_healthy"
            disease = "healthy"
        else:
            label = f"cassava_{disease_part}"
            disease = disease_part

        src_file = images_dir / image_id
        if not src_file.exists():
             # Try checking if valid image
             continue
        
        if not is_valid_image(src_file):
            continue

        class_out = output_dir / label
        class_out.mkdir(exist_ok=True)

        idx = label_counters[label] + 1
        ext = src_file.suffix.lower()
        if not ext: ext = ".jpg" # Default
        
        new_name = f"image-{idx:05d}{ext}"
        dst = class_out / new_name
        
        copy_file(src_file, dst)
        
        csv_data.append({
            "filename": new_name,
            "label": label,
            "crop": crop,
            "disease": disease,
            "original_folder": "train_images",
            "path": f"Cassava_processed/{label}/{new_name}"
        })
        
        label_counters[label] += 1
        stats[label] += 1
        
        if len(csv_data) % 1000 == 0:
            print(f"Processed {len(csv_data)} images...", end='\r')

    print("\n" + "-" * 60)
    for label, count in stats.items():
        print(f"  {label:40} : {count:5} images")

    _write_metadata(output_dir, csv_data, stats)
    print("-" * 60)
    print(f"TOTAL: {len(csv_data)} images, {len(stats)} classes")
    print(f"CSV saved: {output_dir / 'labels.csv'}")

    safe_rmtree(source_dir)
    return csv_data


def _write_metadata(output_dir: Path, csv_data, stats):
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "filename", "label", "crop", "disease", "original_folder", "path"
        ])
        writer.writeheader()
        writer.writerows(csv_data)

    metadata = {
        "dataset": "Cassava",
        "processed_date": datetime.now().isoformat(),
        "total_images": len(csv_data),
        "num_classes": len(stats),
        "classes": dict(stats)
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
