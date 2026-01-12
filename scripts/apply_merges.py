import os
import shutil
import json
import time
from pathlib import Path

DATA_DIR = Path('data/release/agri_foundation_v1/data')
MERGES_FILE = 'merges.json'

def merge_classes():
    if not os.path.exists(MERGES_FILE):
        print(f"Error: {MERGES_FILE} not found.")
        return

    with open(MERGES_FILE, 'r') as f:
        merges = json.load(f)

    print(f"Starting merge process for {len(merges)} rules in {DATA_DIR}...")
    
    for merge_rule in merges:
        source = merge_rule['source']
        target = merge_rule['target']
        
        source_path = DATA_DIR / source
        target_path = DATA_DIR / target
        
        if not source_path.exists():
            # Silent skip if already merged
            continue
            
        if not target_path.exists():
            print(f"Warning: Target {target} not found. Creating it.")
            target_path.mkdir(parents=True, exist_ok=True)
            
        print(f"Merging {source} -> {target}")
        
        # Move files
        moved_count = 0
        for item in source_path.iterdir():
            if item.is_file():
                # Handle potential naming conflicts
                target_file = target_path / item.name
                if target_file.exists():
                    # Append timestamp suffix to avoid overwrite
                    new_name = f"{item.stem}_merged_{int(time.time())}{item.suffix}"
                    target_file = target_path / new_name
                
                shutil.move(str(item), str(target_file))
                moved_count += 1
        
        print(f"  Moved {moved_count} files.")
        
        # Remove source directory if empty
        if not any(source_path.iterdir()):
            source_path.rmdir()
            print(f"  Removed empty source directory: {source}")
        else:
            print(f"  Source directory {source} not empty, skipping removal.")

if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found.")
    else:
        merge_classes()
