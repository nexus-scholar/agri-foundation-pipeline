"""PlantSeg dataset processing."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, safe_rmtree
from .label_utils import normalize_label


def process_plantseg(source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Process PlantSeg dataset into normalized folder + CSV."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("PROCESSING PLANTSEG")
    print("=" * 60)

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return []

    # Find Metadata.csv
    metadata_files = list(source_dir.rglob("Metadata.csv"))
    if not metadata_files:
        print(f"ERROR: Metadata.csv not found in {source_dir}")
        return []
    
    metadata_path = metadata_files[0]
    print(f"Found metadata: {metadata_path}")
    
    # Assume images are in 'images/train' relative to metadata parent or source_dir
    # Based on zip: plantseg/Metadata.csv, plantseg/images/train/...
    # So metadata parent is 'plantseg'. images are in 'plantseg/images/train'.
    root_dir = metadata_path.parent
    images_dir = root_dir / "images" / "train"
    
    if not images_dir.exists():
        # Try checking just 'images' or recursive search for an image folder
        print(f"Warning: {images_dir} not found. Searching for images folder...")
        # Fallback: look for a folder with many images
        pass # For now, assume structure matches zip

    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data: list[dict[str, str]] = []
    stats = defaultdict(int)
    label_counters = defaultdict(int)

    print(f"Reading {metadata_path}...")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        return []

    print(f"Found {len(rows)} entries in Metadata")

    for row in rows:
        filename = row['Name']
        plant = row['Plant']
        disease_raw = row['Disease']
        
        # Construct label
        raw_label = f"{plant}_{disease_raw}"
        label, crop, disease = normalize_label(raw_label)
        
        # Find image file
        # Try direct path
        src_file = images_dir / filename
        if not src_file.exists():
            # Try recursive search if not found (slow)
            found = list(root_dir.rglob(filename))
            if found:
                src_file = found[0]
            else:
                # Skip if not found
                continue
        
        if not is_valid_image(src_file):
            continue

        class_out = output_dir / label
        class_out.mkdir(exist_ok=True)

        start_idx = label_counters[label]
        ext = src_file.suffix.lower() or ".jpg"
        new_name = f"image-{start_idx + 1:05d}{ext}"
        dst = class_out / new_name
        
        copy_file(src_file, dst)
        
        csv_data.append({
            "filename": new_name,
            "label": label,
            "crop": crop,
            "disease": disease,
            "original_folder": "plantseg",
            "path": f"PlantSeg_processed/{label}/{new_name}"
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
        "dataset": "PlantSeg",
        "processed_date": datetime.now().isoformat(),
        "total_images": len(csv_data),
        "num_classes": len(stats),
        "classes": dict(stats)
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
