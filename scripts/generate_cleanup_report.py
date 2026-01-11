"""
Generate a detailed report of removed files and underrepresented classes.
Comparing 'combined_dataset.csv' (Original) vs 'foundation_dataset_v1_clean.csv' (Cleaned).
"""

import pandas as pd
from pathlib import Path

# Config
ORIGINAL_CSV = Path("data/processed/dataset/combined_dataset.csv")
CLEAN_CSV = Path("data/processed/dataset/foundation_dataset_v1_clean.csv")
OUTPUT_DIR = Path("data/processed/dataset")

def generate_report():
    if not ORIGINAL_CSV.exists() or not CLEAN_CSV.exists():
        print("Error: Input CSV files not found.")
        return

    print("Loading datasets...")
    df_orig = pd.read_csv(ORIGINAL_CSV)
    df_clean = pd.read_csv(CLEAN_CSV)

    print(f"Original: {len(df_orig)}")
    print(f"Clean:    {len(df_clean)}")

    # 1. Identify Removed Files (Duplicates/Corrupt)
    # We use the 'path' column as the unique identifier for the source files
    clean_paths = set(df_clean['path'])
    
    # Filter original rows where path is NOT in clean_paths
    removed_df = df_orig[~df_orig['path'].isin(clean_paths)]
    
    removed_csv = OUTPUT_DIR / "removed_files.csv"
    removed_df.to_csv(removed_csv, index=False)
    print(f"\n[1] Removed Files: {len(removed_df)}")
    print(f"    Saved list to: {removed_csv}")
    
    # Breakdown of removed files by source
    print("\n    Removed files by source dataset:")
    print(removed_df['source'].value_counts().to_string())

    # 2. Identify Underrepresented Classes (< 50 images) in the CLEAN dataset
    class_counts = df_clean['label'].value_counts()
    rare_classes = class_counts[class_counts < 50].reset_index()
    rare_classes.columns = ['label', 'count']
    
    rare_csv = OUTPUT_DIR / "underrepresented_classes.csv"
    rare_classes.to_csv(rare_csv, index=False)
    
    print(f"\n[2] Underrepresented Classes (< 50 images): {len(rare_classes)}")
    print(f"    Saved list to: {rare_csv}")
    print("\n    Top 5 rarest classes:")
    print(rare_classes.head(5).to_string(index=False))

if __name__ == "__main__":
    generate_report()
