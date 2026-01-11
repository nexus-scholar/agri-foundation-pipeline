"""
Dataset Verification and Cleaning Script.

Performs:
1. Integrity Check: Tries to open every image.
2. Deduplication: Calculates MD5 hashes to find exact duplicates.
3. Pruning: Removes corrupt files and duplicates from the CSV.
4. Stratification Check: Warns about classes with too few samples.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os
import sys
import concurrent.futures

# Add project root to path to allow importing pipeline modules
sys.path.append(os.getcwd())

from pipeline.data_utils import calculate_md5, load_csv

# Configuration
INPUT_CSV = Path("data/processed/dataset/combined_dataset.csv")
OUTPUT_CSV = Path("data/processed/dataset/foundation_dataset_v1_clean.csv")
DATA_ROOT = Path("data/processed/dataset")

def process_image(row_tuple):
    """
    Worker function for parallel processing.
    Args:
        row_tuple: (index, row_series) or just row_dict
    Returns:
        dict with status info
    """
    idx, row = row_tuple
    rel_path = row['path']
    full_path = DATA_ROOT / rel_path
    
    result = {
        'idx': idx,
        'missing': False,
        'corrupt': False,
        'hash': None,
        'row': row
    }

    if not full_path.exists():
        result['missing'] = True
        return result

    try:
        with Image.open(full_path) as img:
            img.verify() 
        result['hash'] = calculate_md5(full_path)
    except Exception:
        result['corrupt'] = True
        
    return result

def verify_and_clean():
    print(f"Loading {INPUT_CSV}...")
    df = load_csv(INPUT_CSV)
    if df.empty:
        return

    print(f"Initial count: {len(df)} records")

    valid_rows = []
    seen_hashes = {} # hash -> label
    corrupt_count = 0
    missing_count = 0
    duplicate_count = 0
    
    print("\nVerifying images (Parallel)...")
    
    # Use ThreadPoolExecutor for I/O bound tasks
    # Adjust max_workers based on disk speed vs CPU. 
    # For many small files, overhead can be high, but threading usually helps waiting for I/O.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        futures = [executor.submit(process_image, (idx, row)) for idx, row in df.iterrows()]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = future.result()
            
            if res['missing']:
                missing_count += 1
                continue
            
            if res['corrupt']:
                corrupt_count += 1
                continue
                
            # Deduplication
            file_hash = res['hash']
            if file_hash:
                row = res['row']
                if file_hash in seen_hashes:
                    # If hash exists, check label
                    if seen_hashes[file_hash] == row['label']:
                        duplicate_count += 1
                        continue
                    else:
                        # Collision with different label
                        # print(f"Warning: Hash collision for distinct labels! {row['path']}")
                        duplicate_count += 1 # Treating as duplicate/ambiguous
                        continue
                
                seen_hashes[file_hash] = row['label']
                valid_rows.append(row)
            else:
                # Failed to hash but file exists? Treat as corrupt
                corrupt_count += 1

    # Reassemble DataFrame
    df_clean = pd.DataFrame(valid_rows)
    
    print("\n" + "="*40)
    print("CLEANUP REPORT")
    print("="*40)
    print(f"Original Records: {len(df)}")
    print(f"Missing Files:    {missing_count}")
    print(f"Corrupt Files:    {corrupt_count}")
    print(f"Duplicates:       {duplicate_count}")
    print(f"Final Clean Set:  {len(df_clean)}")
    print("-" * 40)
    
    # 4. Class Stratification Check
    if not df_clean.empty:
        class_counts = df_clean['label'].value_counts()
        rare_classes = class_counts[class_counts < 50]
        
        if not rare_classes.empty:
            print(f"\nWARNING: {len(rare_classes)} classes have < 50 images.")
        else:
            print("\nClass Balance: OK (All classes > 50 samples)")

        # Save
        df_clean.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved Clean Index to: {OUTPUT_CSV}")
    else:
        print("Error: No valid data remaining.")

if __name__ == "__main__":
    verify_and_clean()
