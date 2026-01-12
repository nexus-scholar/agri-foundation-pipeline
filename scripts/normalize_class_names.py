import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('data/release/agri_foundation_v1/data')
METADATA_FILE = DATA_DIR.parent / 'metadata.csv'

SUFFIXES = ['_google', '_bing', '_baidu', '_plantseg']

def normalize_classes():
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found.")
        return

    print(f"Scanning for classes with suffixes: {SUFFIXES}")
    
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    merges = []

    for cls in classes:
        for suffix in SUFFIXES:
            if cls.endswith(suffix):
                # Identify target name
                target = cls[:-len(suffix)]
                
                # Special case: double suffix (e.g. _google_bing - rare but possible)
                # Recurse? For now just one level.
                
                print(f"Found suffix class: {cls} -> {target}")
                merges.append((cls, target))
                break # Only handle one suffix match per class to avoid confusion

    if not merges:
        print("No classes with suffixes found.")
        return

    print(f"\nProcessing {len(merges)} normalization merges...")
    
    # Load metadata
    if METADATA_FILE.exists():
        df = pd.read_csv(METADATA_FILE)
    else:
        df = None
        print("Warning: metadata.csv not found.")

    rows_updated = 0

    for source, target in tqdm(merges):
        source_path = DATA_DIR / source
        target_path = DATA_DIR / target
        
        # 1. Create target if not exists
        if not target_path.exists():
            target_path.mkdir(parents=True, exist_ok=True)
            
        # 2. Move files
        for item in source_path.iterdir():
            if item.is_file():
                dest_file = target_path / item.name
                if dest_file.exists():
                    import time
                    new_name = f"{item.stem}_norm_{int(time.time())}{item.suffix}"
                    dest_file = target_path / new_name
                
                shutil.move(str(item), str(dest_file))
        
        # 3. Remove source dir
        try:
            source_path.rmdir()
        except OSError:
            print(f"Warning: Could not remove {source_path} (not empty?)")

        # 4. Update Metadata
        if df is not None:
            mask = df['label'] == source
            count = mask.sum()
            if count > 0:
                df.loc[mask, 'label'] = target
                df.loc[mask, 'release_path'] = df.loc[mask, 'release_path'].apply(
                    lambda p: p.replace(f"data/{source}/", f"data/{target}/")
                )
                rows_updated += count

    if df is not None and rows_updated > 0:
        df.to_csv(METADATA_FILE, index=False)
        print(f"\nUpdated metadata.csv: {rows_updated} rows modified.")

if __name__ == "__main__":
    normalize_classes()
