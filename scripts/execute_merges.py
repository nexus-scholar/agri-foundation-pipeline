"""
Execute the merging of classes identified in 'merge_candidates_report.csv'.
1. Moves files from variant folders to base folder.
2. Updates 'metadata.csv'.
3. Cleans up empty folders.
"""

import pandas as pd
import shutil
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from pipeline.fs_utils import ensure_dir
from pipeline.data_utils import load_csv

# Config
MERGE_REPORT_CSV = Path("data/processed/dataset/merge_candidates_report.csv")
DATASET_ROOT = Path("data/release/agri_foundation_v1/data")
METADATA_CSV = Path("data/release/agri_foundation_v1/metadata.csv")

def execute_merges():
    print("Loading merge report...")
    merge_df = load_csv(MERGE_REPORT_CSV)
    if merge_df.empty:
        return
    
    print("Loading metadata...")
    metadata_df = load_csv(METADATA_CSV)
    if metadata_df.empty:
        return
    
    print(f"Original Metadata Count: {len(metadata_df)}")
    
    # Get unique base labels
    base_labels = merge_df['base_label'].unique()
    
    files_moved = 0
    updated_records = 0
    
    print(f"Processing {len(base_labels)} merge groups...")
    
    for base in tqdm(base_labels):
        # Create base directory if it doesn't exist (it might be one of the variants)
        base_dir = DATASET_ROOT / base
        ensure_dir(base_dir)
        
        # Get all variants for this base
        variants = merge_df[merge_df['base_label'] == base]['original_label'].tolist()
        
        for variant in variants:
            if variant == base:
                continue # Skip if variant is already the base name
                
            variant_dir = DATASET_ROOT / variant
            
            if not variant_dir.exists():
                # Might have been moved already or didn't exist in release (empty)
                continue
                
            # Move all files
            for file_path in variant_dir.glob("*"):
                if file_path.is_file():
                    # Handle potential filename conflicts in base dir
                    dest_path = base_dir / file_path.name
                    if dest_path.exists():
                        # Conflict! Prepend variant name
                        new_name = f"{variant}_{file_path.name}"
                        dest_path = base_dir / new_name
                    
                    try:
                        shutil.move(str(file_path), str(dest_path))
                        files_moved += 1
                        
                        # Update metadata in memory
                        # We identify rows by 'label' = variant AND 'filename' containing the original name
                        
                        # Let's find the specific row. 
                        # Note: filenames in metadata.csv are "safe_filename"s e.g. "plantvillage_image.jpg"
                        # The file_path.name corresponds to this safe_filename.
                        
                        mask = (metadata_df['label'] == variant) & (metadata_df['release_path'].str.endswith(file_path.name))
                        metadata_df.loc[mask, 'release_path'] = f"data/{base}/{dest_path.name}"
                        metadata_df.loc[mask, 'label'] = base
                        updated_records += 1
                        
                    except Exception as e:
                        print(f"Error moving {file_path}: {e}")
            
            # Remove empty variant dir
            try:
                variant_dir.rmdir()
            except OSError:
                print(f"Warning: Could not remove {variant_dir}, might not be empty.")

    print("\n" + "="*40)
    print("MERGE COMPLETE")
    print("="*40)
    print(f"Files Moved:     {files_moved}")
    print(f"Records Updated: {updated_records}")
    
    # Save updated metadata
    metadata_df.to_csv(METADATA_CSV, index=False)
    print(f"Updated metadata saved to: {METADATA_CSV}")
    
    # Recalculate class count
    new_class_count = metadata_df['label'].nunique()
    print(f"New Class Count: {new_class_count} (was 288)")

if __name__ == "__main__":
    execute_merges()
