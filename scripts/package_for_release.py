"""
Package the Cleaned Foundation Dataset for Release.

1. Reads 'foundation_dataset_v1_clean.csv'.
2. Copies images to a new 'release/' folder structure:
   release/
     ├── train/
     │   ├── apple_scab/
     │   │   ├── image_001.jpg
     │   │   └── ...
     │   └── ...
     ├── test/
     │   └── ... (if we split, or just one folder if we let users split)
     └── metadata.csv

For a foundation model pre-training dataset, we typically release all data
in a single 'data/' folder or 'train/' folder and let users split it.
We will use a flat class structure: release/data/{class_name}/{image}
"""

import pandas as pd
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from pipeline.fs_utils import ensure_dir, copy_file

# Config
INPUT_CSV = Path("data/processed/dataset/foundation_dataset_v1_clean.csv")
RELEASE_DIR = Path("data/release/agri_foundation_v1")
SOURCE_ROOT = Path("data/processed/dataset")

def package_dataset():
    print(f"Loading clean index: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print("Error: Input CSV not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    print(f"Preparing release directory: {RELEASE_DIR}")
    ensure_dir(RELEASE_DIR / "data")
    
    print(f"Copying {len(df)} images to release structure...")
    
    new_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Source path
        src_path = SOURCE_ROOT / row['path']
        
        # Destination: release/data/{label}/{original_filename}
        # We use the label as the folder name to enforce the unification
        label = row['label']
        filename = Path(row['path']).name
        
        # Handle duplicate filenames in the same class (rare but possible after merge)
        # by prepending the source dataset name
        safe_filename = f"{row['source']}_{filename}"
        
        dest_folder = RELEASE_DIR / "data" / label
        ensure_dir(dest_folder)
        
        dest_path = dest_folder / safe_filename
        
        try:
            copy_file(src_path, dest_path)
            
            # Record relative path for the release CSV
            new_row = row.copy()
            new_row['release_path'] = f"data/{label}/{safe_filename}"
            new_records.append(new_row)
            
        except Exception as e:
            print(f"Error copying {src_path}: {e}")

    # Save release CSV
    release_df = pd.DataFrame(new_records)
    release_df.to_csv(RELEASE_DIR / "metadata.csv", index=False)
    
    print("\n" + "="*40)
    print("PACKAGING COMPLETE")
    print("="*40)
    print(f"Output Directory: {RELEASE_DIR}")
    print(f"Total Images:     {len(release_df)}")
    print(f"Classes:          {release_df['label'].nunique()}")
    print("\nNext: Zip this folder and upload to Hugging Face / Kaggle!")

if __name__ == "__main__":
    package_dataset()

