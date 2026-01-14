import json
import os
import shutil
from pathlib import Path
from datetime import datetime

# Paths
MODEL_PATH = Path('models/foundational/mobilenetv3_large_100_best.pth')
EVAL_DIR = Path('evaluation_results_v3_epoch3')
DOCS_DIR = Path('docs')
OUTPUT_ZIP = 'experiment_v4_mobilenetv3_large_epoch3.zip'

def create_metadata_files():
    # 1. Model Configuration
    config = {
        "model_name": "mobilenetv3_large_100.ra_in1k",
        "input_size": 256,
        "batch_size": 32,
        "epochs_completed": 3,
        "learning_rate": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_function": "CrossEntropyLoss",
        "seed": 42,
        "device": "cuda",
        "dataset_version": "v4 (Semantic Clean)",
        "status": "Interrupted at Epoch 3"
    }
    with open('model_config_v3.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 2. Augmentation Parameters
    aug_params = {
        "train": {
            "RandomResizedCrop": {"size": 256},
            "RandomHorizontalFlip": {"p": 0.5},
            "RandomRotation": {"degrees": 15},
            "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
            "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        }
    }
    with open('augmentation_params_v3.json', 'w') as f:
        json.dump(aug_params, f, indent=4)

def package_v3():
    print("Gathering artifacts for MobileNetV3...")
    create_metadata_files()
    
    pkg_dir = Path('experiment_package_v3')
    if pkg_dir.exists(): shutil.rmtree(pkg_dir)
    pkg_dir.mkdir()

    # 1. Model
    if MODEL_PATH.exists():
        shutil.copy(MODEL_PATH, pkg_dir / MODEL_PATH.name)
    
    # 2. Evaluation Results
    eval_dest = pkg_dir / 'evaluation'
    eval_dest.mkdir()
    if EVAL_DIR.exists():
        for file in EVAL_DIR.glob('*'):
            shutil.copy(file, eval_dest / file.name)

    # 3. Metadata
    shutil.move('model_config_v3.json', pkg_dir / 'model_config.json')
    shutil.move('augmentation_params_v3.json', pkg_dir / 'augmentation_params.json')
    shutil.copy('merges.json', pkg_dir / 'merges_applied.json')
    
    # 4. Docs
    docs_dest = pkg_dir / 'docs'
    docs_dest.mkdir()
    if DOCS_DIR.exists():
        for file in DOCS_DIR.glob('*'):
            shutil.copy(file, docs_dest / file.name)

    # Zip it
    shutil.make_archive(OUTPUT_ZIP.replace('.zip', ''), 'zip', pkg_dir)
    shutil.rmtree(pkg_dir)
    print(f"Done! Created {OUTPUT_ZIP}")

if __name__ == "__main__":
    package_v3()
