import json
import os
import shutil
from pathlib import Path
from datetime import datetime

# Paths
MODEL_DIR = Path('../models/foundational')
EVAL_DIR = Path('evaluation_results')
DOCS_DIR = Path('docs')
OUTPUT_ZIP = 'experiment_v4_mobilenetv4_small.zip'

def create_metadata_files():
    # 1. Model Configuration
    config = {
        "model_name": "mobilenetv4_conv_small.e2400_r224_in1k",
        "input_size": 256,
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_function": "CrossEntropyLoss",
        "seed": 42,
        "device": "cuda",
        "dataset_version": "v4 (Semantic Clean)"
    }
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 2. Augmentation Parameters
    aug_params = {
        "train": {
            "RandomResizedCrop": {"size": 256},
            "RandomHorizontalFlip": {"p": 0.5},
            "RandomRotation": {"degrees": 15},
            "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
            "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        "val": {
            "Resize": {"size": 291}, # int(256 * 1.14)
            "CenterCrop": {"size": 256},
            "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        }
    }
    with open('augmentation_params.json', 'w') as f:
        json.dump(aug_params, f, indent=4)

    # 3. Dataset Versioning (Copy of merges and stats)
    dataset_info = {
        "dataset_name": "Agri-Foundation-V4",
        "source_path": "data/release/agri_foundation_v1/data",
        "total_classes": 164, # From eval log
        "cleaning_steps": ["Raw Process", "Verify", "Package", "Normalize", "Merge V3", "Merge V4"],
        "date_created": datetime.now().strftime("%Y-%m-%d")
    }
    with open('dataset_version.json', 'w') as f:
        json.dump(dataset_info, f, indent=4)

def package_artifacts():
    print("Gathering artifacts...")
    create_metadata_files()
    
    # Create a temporary folder for packing
    pkg_dir = Path('experiment_package')
    if pkg_dir.exists(): shutil.rmtree(pkg_dir)
    pkg_dir.mkdir()

    # 1. Model & Logs
    print(f"Copying model from {MODEL_DIR}")
    for file in MODEL_DIR.glob('*'):
        shutil.copy(file, pkg_dir / file.name)

    # 2. Evaluation Results
    print(f"Copying evaluation results from {EVAL_DIR}")
    eval_dest = pkg_dir / 'evaluation'
    eval_dest.mkdir()
    if EVAL_DIR.exists():
        for file in EVAL_DIR.glob('*'):
            shutil.copy(file, eval_dest / file.name)

    # 3. Documentation
    print(f"Copying documentation from {DOCS_DIR}")
    docs_dest = pkg_dir / 'docs'
    docs_dest.mkdir()
    if DOCS_DIR.exists():
        for file in DOCS_DIR.glob('*'):
            shutil.copy(file, docs_dest / file.name)
    
    # Copy metadata files created above
    shutil.move('model_config.json', pkg_dir / 'model_config.json')
    shutil.move('augmentation_params.json', pkg_dir / 'augmentation_params.json')
    shutil.move('dataset_version.json', pkg_dir / 'dataset_version.json')
    shutil.copy('merges.json', pkg_dir / 'merges_applied.json')

    # Zip it
    print(f"Zipping to {OUTPUT_ZIP}...")
    shutil.make_archive(OUTPUT_ZIP.replace('.zip', ''), 'zip', pkg_dir)
    
    # Cleanup
    shutil.rmtree(pkg_dir)
    print("Done!")

if __name__ == "__main__":
    package_artifacts()
