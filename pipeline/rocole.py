"""RoCoLe dataset processing."""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from .fs_utils import copy_file, safe_rmtree

def process_rocole(source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Process RoCoLe dataset."""
    print("\n" + "=" * 60)
    print("PROCESSING ROCOLE")
    print("=" * 60)

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return []

    # RoCoLe structure: Photos/ and Annotations/RoCoLE-csv.csv
    photos_dir = source_dir / "Photos"
    csv_file = source_dir / "Annotations" / "RoCoLE-csv.csv"

    if not photos_dir.exists() or not csv_file.exists():
        # Maybe nested?
        nested = list(source_dir.glob("**/RoCoLE-csv.csv"))
        if nested:
            csv_file = nested[0]
            photos_dir = csv_file.parent.parent / "Photos"
        else:
            print(f"ERROR: Structure not recognized in {source_dir}")
            return []

    print(f"Found CSV: {csv_file}")
    
    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data = []
    stats = defaultdict(int)
    
    # Read CSV mapping
    # Header: ID,DataRow ID,Labeled Data,Label,... ,External ID,...
    # External ID seems to be the filename
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('External ID')
            label_json_str = row.get('Label')
            
            if not filename or not label_json_str:
                continue
                
            src_path = photos_dir / filename
            if not src_path.exists():
                # Try finding it recursively? No, should be flat in Photos
                continue

            try:
                label_data = json.loads(label_json_str)
                # Structure: {"Leaf": [{"state": "healthy", ...}]}
                # Sometimes just {"Leaf": "url"} (mask exporter?)
                
                if "Leaf" in label_data and isinstance(label_data["Leaf"], list):
                    state = label_data["Leaf"][0].get("state", "unknown")
                else:
                    # Fallback or skip
                    continue
                    
                # Normalize state
                # RoCoLe classes: healthy, rust, red_spider_mite, etc.
                # Crop is Coffee.
                
                crop = "coffee"
                
                # Clean state name
                state = state.lower().replace(" ", "_")
                
                if state == "healthy":
                    label = "coffee_healthy"
                    disease = "healthy"
                else:
                    label = f"coffee_{state}"
                    disease = state

                # Save
                class_out = output_dir / label
                class_out.mkdir(exist_ok=True)
                
                new_idx = stats[label] + 1
                ext = src_path.suffix.lower()
                new_name = f"image-{new_idx:05d}{ext}"
                dst_path = class_out / new_name
                
                copy_file(src_path, dst_path)
                
                csv_data.append({
                    "filename": new_name,
                    "label": label,
                    "crop": crop,
                    "disease": disease,
                    "original_folder": "RoCoLe",
                    "path": f"RoCoLe_processed/{label}/{new_name}"
                })
                stats[label] += 1
                
            except json.JSONDecodeError:
                print(f"Warning: JSON error for {filename}")
                continue

    # Write Metadata
    _write_metadata(output_dir, csv_data, stats)
    
    print("-" * 60)
    for label, count in stats.items():
        print(f"  {label:40} : {count:5} images")
    print("-" * 60)
    print(f"TOTAL: {len(csv_data)} images")
    
    safe_rmtree(source_dir) # Cleanup raw extraction
    return csv_data

def _write_metadata(output_dir: Path, csv_data, stats):
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "filename", "label", "crop", "disease", "original_folder", "path"
        ])
        writer.writeheader()
        writer.writerows(csv_data)
