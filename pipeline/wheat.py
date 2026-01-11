"""Wheat Disease dataset processing."""
from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, safe_rmtree
from .label_utils import normalize_label


def _find_class_folders_recursive(root: Path, max_depth: int = 4) -> List[Path]:
    """Recursively find folders containing images."""
    class_folders = []

    def scan(path: Path, depth: int):
        if depth > max_depth:
            return

        if not path.is_dir():
            return

        # Check if this folder contains images
        images = [f for f in path.iterdir() if f.is_file() and is_valid_image(f)]

        if images:
            # Assume it's a class folder if it has images
            class_folders.append(path)
        
        # Always recurse to find nested folders
        for subfolder in path.iterdir():
            if subfolder.is_dir() and not subfolder.name.startswith('.'):
                scan(subfolder, depth + 1)

    scan(root, 0)
    return class_folders


def process_wheat(source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Process Wheat dataset into normalized folder + CSV."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("PROCESSING WHEAT DISEASE")
    print("=" * 60)

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return []

    # Look for the inner zip file
    inner_zips = list(source_dir.rglob("*.zip"))
    if not inner_zips:
        print(f"ERROR: No inner zip file found in {source_dir}")
        # Fallback: maybe it was already extracted?
        search_root = source_dir
    else:
        zip_path = inner_zips[0]
        print(f"Found inner zip: {zip_path.name}")
        extract_dir = source_dir / "extracted_inner"
        if not extract_dir.exists():
            print(f"Extracting {zip_path} to {extract_dir}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_dir)
            except Exception as e:
                print(f"ERROR: Failed to extract zip: {e}")
                return []
        search_root = extract_dir

    print(f"Scanning {search_root} for class folders...")
    all_class_folders = _find_class_folders_recursive(search_root)

    if not all_class_folders:
        print(f"ERROR: No class folders found in {search_root}")
        return []

    print(f"Found {len(all_class_folders)} potential class folders")
    
    # Filter out non-class folders (optional)
    # For now, accept all folders with images as classes

    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data: list[dict[str, str]] = []
    stats = defaultdict(int)
    label_counters = defaultdict(int)

    for folder in all_class_folders:
        folder_name = folder.name
        # Normalize label: crop is wheat
        # Label usually "Septoria", "Healthy", "Stripe_Rust"
        
        # Check if folder name is generic like "train" or "images"
        if folder_name.lower() in {"train", "valid", "test", "images", "original", "augmented", "balanced"}:
             # If folder is "train", it might contain class folders. 
             # But _find_class_folders_recursive returns folders *containing* images.
             # If "train" has images directly, it's mixed?
             # Assuming this dataset is well structured with Class Name folders.
             # If folder name is "train", check if parent is meaningful?
             pass

        # Clean folder name
        clean_name = folder_name.lower().replace(" ", "_")
        
        if clean_name == "healthy":
            label = "wheat_healthy"
            disease = "healthy"
        else:
            label = f"wheat_{clean_name}"
            disease = clean_name

        images = [f for f in folder.glob("*") if is_valid_image(f)]
        if not images: continue

        class_out = output_dir / label
        class_out.mkdir(exist_ok=True)
        
        start_idx = label_counters[label]
        
        for idx, src in enumerate(images, start=start_idx + 1):
             ext = src.suffix.lower()
             new_name = f"image-{idx:05d}{ext}"
             dst = class_out / new_name
             copy_file(src, dst)
             
             csv_data.append({
                "filename": new_name,
                "label": label,
                "crop": "wheat",
                "disease": disease,
                "original_folder": folder_name,
                "path": f"Wheat_processed/{label}/{new_name}"
             })
             label_counters[label] += 1
             
        stats[label] += len(images)
        print(f"  {label:40} : {len(images):5} images")

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
        "dataset": "Wheat",
        "processed_date": datetime.now().isoformat(),
        "total_images": len(csv_data),
        "num_classes": len(stats),
        "classes": dict(stats)
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
