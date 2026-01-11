"""PlantWild dataset processing."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, safe_rmtree
from .label_utils import normalize_label


def _find_class_folders_recursive(root: Path, max_depth: int = 3) -> List[Path]:
    """Recursively find all class folders."""
    class_folders = []

    def scan(path: Path, depth: int):
        if depth > max_depth: return
        if not path.is_dir(): return
        images = [f for f in path.iterdir() if f.is_file() and is_valid_image(f)]
        # If it has images, it's a class folder
        if images:
            class_folders.append(path)
        
        for subfolder in path.iterdir():
            if subfolder.is_dir() and not subfolder.name.startswith('.'):
                scan(subfolder, depth + 1)
    scan(root, 0)
    return class_folders


def process_plantwild(source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Process PlantWild dataset into normalized folder + CSV."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("PROCESSING PLANTWILD")
    print("=" * 60)

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return []

    print(f"Scanning {source_dir} for class folders...")
    all_class_folders = _find_class_folders_recursive(source_dir)

    if not all_class_folders:
        print(f"ERROR: No class folders found in {source_dir}")
        return []

    print(f"Found {len(all_class_folders)} class folders")

    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data: list[dict[str, str]] = []
    stats = defaultdict(int)
    label_counters = defaultdict(int)

    for folder in all_class_folders:
        # Folder name: "apple black rot"
        raw_label = folder.name
        label, crop, disease = normalize_label(raw_label)

        images = [f for f in folder.glob("*") if is_valid_image(f)]
        if not images: continue
        
        # Deduplicate
        images = sorted(set(images), key=lambda f: f.name)

        class_out = output_dir / label
        class_out.mkdir(exist_ok=True)

        start_idx = label_counters[label]
        for idx, src in enumerate(images, start=start_idx + 1):
            ext = src.suffix.lower() or ".jpg"
            new_name = f"image-{idx:05d}{ext}"
            dst = class_out / new_name
            copy_file(src, dst)
            csv_data.append({
                "filename": new_name,
                "label": label,
                "crop": crop,
                "disease": disease,
                "original_folder": raw_label,
                "path": f"PlantWild_processed/{label}/{new_name}"
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
        "dataset": "PlantWild",
        "processed_date": datetime.now().isoformat(),
        "total_images": len(csv_data),
        "num_classes": len(stats),
        "classes": dict(stats)
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
