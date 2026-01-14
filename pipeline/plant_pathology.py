"Plant Pathology 2021 dataset processing."
from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

from .fs_utils import copy_file, safe_rmtree

def process_plant_pathology(source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Process Plant Pathology 2021 dataset."""
    print("\n" + "=" * 60)
    print("PROCESSING PLANT PATHOLOGY 2021")
    print("=" * 60)

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return []

    # Structure: train_images/ and train.csv
    train_images_dir = source_dir / "train_images"
    csv_file = source_dir / "train.csv"
    
    if not train_images_dir.exists() or not csv_file.exists():
        # Check nesting
        nested = list(source_dir.glob("**/train.csv"))
        if nested:
            csv_file = nested[0]
            train_images_dir = csv_file.parent / "train_images"
        else:
            print(f"ERROR: Structure not recognized in {source_dir}")
            return []

    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data = []
    stats = defaultdict(int)
    
    # Plant Pathology 2021 labels are space separated
    # Labels: healthy, scab, rust, powdery_mildew, complex, frog_eye_leaf_spot
    # We ignore "complex" as it's multi-disease usually? Or maybe keep it?
    # Strategy: Only keep single-label images to avoid confusion, OR create composite classes.
    # Given the goal is cleaning, simpler is better.
    # Actually, "scab frog_eye_leaf_spot" means BOTH.
    # For a robust dataset, we might skip multi-label images for now, or map them to "apple_complex".
    
    # We will map single labels to standard format: apple_{disease}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['image']
            labels_str = row['labels']
            
            src_path = train_images_dir / filename
            if not src_path.exists():
                continue
                
            labels = labels_str.split(' ')
            
            # Filter logic
            if len(labels) > 1:
                # Skip multi-label for now to ensure class purity
                # or map to "apple_multiple_diseases"
                continue
            
            disease = labels[0]
            
            # Map specific names
            # 'frog_eye_leaf_spot' -> 'frog_eye_spot' (matches PlantVillage 'apple_black_rot'?? No, distinct)
            # Actually PlantVillage has 'apple_black_rot', 'apple_scab', 'apple_rust', 'apple_cedar_rust'.
            # Frog Eye Leaf Spot is Botryosphaeria obtusa. Black Rot is also Botryosphaeria obtusa!
            # So 'frog_eye_leaf_spot' == 'black_rot' potentially?
            # Let's keep it as is for now and let the merge step handle semantic unification if needed.
            # But wait, we want to fix imbalance.
            
            crop = "apple"
            
            if disease == "healthy":
                label = "apple_healthy"
            elif disease == "scab":
                label = "apple_scab"
            elif disease == "rust":
                label = "apple_rust" # Likely Cedar Apple Rust
            elif disease == "powdery_mildew":
                label = "apple_powdery_mildew" # Not in PlantVillage apple list? PV has it for Cherry/Squash.
                # If PV doesn't have apple_powdery_mildew, this is a NEW class! Good.
            elif disease == "frog_eye_leaf_spot":
                label = "apple_frog_eye_leaf_spot"
            elif disease == "complex":
                continue # Skip generic complex
            else:
                label = f"apple_{disease}"

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
                "original_folder": "PlantPathology2021",
                "path": f"PlantPathology2021_processed/{label}/{new_name}"
            })
            stats[label] += 1

    # Write Metadata
    _write_metadata(output_dir, csv_data, stats)
    
    print("-" * 60)
    for label, count in stats.items():
        print(f"  {label:40} : {count:5} images")
    print("-" * 60)
    print(f"TOTAL: {len(csv_data)} images")
    
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
