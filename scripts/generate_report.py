import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Config
DATA_SOURCES_DIR = Path('data/processed/dataset')
RELEASE_DIR = Path('data/release/agri_foundation_v1/data')
OUTPUT_FILE = Path('docs/V5_Dataset_Report.md')

def df_to_markdown(df):
    """Convert pandas DataFrame to Markdown table string manually."""
    if df.empty:
        return ""
    cols = df.columns
    # header
    md = "| " + " | ".join(cols) + " |\n"
    # separator
    md += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    # rows
    for _, row in df.iterrows():
        md += "| " + " | ".join(str(row[c]) for c in cols) + " |\n"
    return md

def generate_report():
    print("Generating V5 Dataset Report...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Title
        f.write("# Agri-Foundation V5 Dataset Report\n\n")
        f.write("**Date:** " + pd.Timestamp.now().strftime("%Y-%m-%d") + "\n\n")
        
        # 1. Pipeline Overview
        f.write("## 1. Pipeline Overview\n\n")
        f.write("This dataset was constructed using a Reproducible Data Pipeline (V5) consisting of five stages:\n\n")
        f.write("1.  **Ingestion:** Raw extraction of 10 diverse agricultural datasets (PlantVillage, PlantDoc, RoCoLe, etc.).\n")
        f.write("2.  **Standardization:** Mapping source-specific folder structures to a unified `Crop_Disease` schema.\n")
        f.write("3.  **Verification:** Removing corrupt images and deduplicating identical files (MD5 hashing).\n")
        f.write("4.  **Normalization:** Cleaning naming conventions (e.g., removing `_google` suffixes, fixing typos).\n")
        f.write("5.  **Sematic Merging:** Consolidating duplicate classes (e.g., `Corn_(maize)` $\\to$ `Corn`) and ambiguous labels.\n\n")

        # 2. Source Datasets Summary
        f.write("## 2. Source Datasets Summary\n\n")
        f.write("| Dataset | Total Images | Description |\n")
        f.write("| :--- | :--- | :--- |\n")
        
        # Scan processed folder for metadata.json files
        source_stats = []
        if DATA_SOURCES_DIR.exists():
            for d in sorted(DATA_SOURCES_DIR.iterdir()):
                if d.is_dir() and d.name.endswith('_processed'):
                    meta_path = d / 'metadata.json'
                    if meta_path.exists():
                        with open(meta_path, 'r') as mf:
                            meta = json.load(mf)
                            name = d.name.replace('_processed', '')
                            count = meta.get('total_images', 0)
                            source_stats.append((name, count, d))
                            f.write(f"| {name} | {count} | |\n")
        
        f.write("\n")

        # 3. Detailed Breakdown by Source
        f.write("## 3. Detailed Breakdown by Source\n\n")
        
        for name, count, path in source_stats:
            f.write(f"### {name} ({count} images)\n\n")
            
            # Load labels.csv to get crop/disease stats
            labels_file = path / 'labels.csv'
            if labels_file.exists():
                try:
                    df = pd.read_csv(labels_file)
                    # Group by label
                    counts = df['label'].value_counts().reset_index()
                    counts.columns = ['Class', 'Count']
                    counts = counts.sort_values('Class')
                    
                    f.write(df_to_markdown(counts))
                    f.write("\n\n")
                except Exception as e:
                    f.write(f"*Error reading stats: {e}*\n\n")

        # 4. Final Combined Statistics (V5 Release)
        f.write("## 4. Final Combined Dataset (V5 Release)\n\n")
        
        if RELEASE_DIR.exists():
            final_classes = []
            total_final = 0
            for d in sorted(RELEASE_DIR.iterdir()):
                if d.is_dir():
                    # Count images
                    imgs = len([x for x in d.iterdir() if x.is_file()])
                    final_classes.append({'Class Name': d.name, 'Image Count': imgs})
                    total_final += imgs
            
            f.write(f"**Total Images:** {total_final}\n\n")
            f.write(f"**Total Classes:** {len(final_classes)}\n\n")
            
            df_final = pd.DataFrame(final_classes)
            f.write(df_to_markdown(df_final))
        else:
            f.write("*Release directory not found.*")

    print(f"Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()