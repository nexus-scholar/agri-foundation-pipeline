import os
import difflib
from pathlib import Path
from itertools import combinations
import re

DATA_DIR = Path('data/release/agri_foundation_v1/data')

def normalize_name(name):
    """Normalize name for comparison (lower case, remove 'bing', 'copy', etc.)"""
    name = name.lower()
    name = re.sub(r'[()\s-]', '', name) # Remove parens, spaces, and hyphens
    # Remove common noise words
    noise = ['bing', 'copy', 'image', 'plantseg', 'google']
    for n in noise:
        name = name.replace(n, '')
    return name.strip()

def scan_for_duplicates():
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found.")
        return

    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Scanning {len(classes)} classes for similarities...")
    
    candidates = []
    
    # 1. Heuristic: "Crop_Disease" vs "Crop_Crop_Disease"
    print("\n--- Heuristic Check (Prefix Duplication) ---")
    for cls in classes:
        parts = cls.split('_')
        if len(parts) > 1 and parts[0] == parts[1]:
            simple_name = '_'.join(parts[1:])
            if simple_name in classes:
                count_complex = len(list((DATA_DIR / cls).glob('*')))
                count_simple = len(list((DATA_DIR / simple_name).glob('*')))
                print(f"Potential Merge: {simple_name} ({count_simple}) <-> {cls} ({count_complex})")
                candidates.append((simple_name, cls))

    # 2. Fuzzy Matching (slow but thorough)
    print("\n--- Fuzzy Matching (Similarity > 0.85) ---")
    # We use a set to avoid printing A-B and B-A
    seen_pairs = set()
    
    for cls1 in classes:
        # Get close matches
        matches = difflib.get_close_matches(cls1, classes, n=5, cutoff=0.85)
        for cls2 in matches:
            if cls1 == cls2: continue
            
            pair = tuple(sorted((cls1, cls2)))
            if pair in seen_pairs: continue
            seen_pairs.add(pair)
            
            # Skip if we already caught it in heuristic
            if (cls1, cls2) in candidates or (cls2, cls1) in candidates: continue

            count1 = len(list((DATA_DIR / cls1).glob('*')))
            count2 = len(list((DATA_DIR / cls2).glob('*')))
            
            print(f"Similarity Match: {cls1} ({count1}) <-> {cls2} ({count2})")

    # 3. Normalized Similarity (catches 'Corn_(maize)' vs 'Corn')
    print("\n--- Normalized Similarity ---")
    for cls1, cls2 in combinations(classes, 2):
        n1 = normalize_name(cls1)
        n2 = normalize_name(cls2)
        
        if n1 == n2 and cls1 != cls2:
             pair = tuple(sorted((cls1, cls2)))
             if pair in seen_pairs: continue
             seen_pairs.add(pair)
             
             count1 = len(list((DATA_DIR / cls1).glob('*')))
             count2 = len(list((DATA_DIR / cls2).glob('*')))
             print(f"Normalized Identity: {cls1} ({count1}) <-> {cls2} ({count2})")

if __name__ == "__main__":
    scan_for_duplicates()
