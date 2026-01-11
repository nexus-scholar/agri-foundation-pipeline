"""Generate a simple data report from the combined dataset CSV."""
import sys
import os
import csv
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from pipeline.data_utils import load_csv

# Config
INPUT_CSV = Path("data/processed/dataset/combined_dataset.csv")
OUTPUT_MD = Path("data/processed/dataset/data_report.md")

def generate_report():
    print(f"Reading {INPUT_CSV}...")
    df = load_csv(INPUT_CSV)
    
    if df.empty:
        return

    # Data aggregation
    # Source counts
    source_counts = df['source'].value_counts()
    
    # Crop counts
    crop_counts = df['crop'].value_counts()
    
    # Detailed class counts per source
    # Group by label, source
    stats = df.groupby(['label', 'source']).size().unstack(fill_value=0)
    
    # Generate Markdown Report
    lines = []
    lines.append("# Dataset Data Report")
    lines.append("")
    lines.append(f"**Total Images:** {len(df):,}")
    lines.append("")
    
    lines.append("## 1. Summary by Source Dataset")
    lines.append("| Source Dataset | Total Images |")
    lines.append("|---|---|")
    for source, count in source_counts.items():
        lines.append(f"| {source} | {count:,} |")
    lines.append("")

    lines.append("## 2. Summary by Crop")
    lines.append("| Crop | Total Images |")
    lines.append("|---|---|")
    for crop, count in crop_counts.items():
        lines.append(f"| {crop} | {count:,} |")
    lines.append("")

    lines.append("## 3. Detailed Class Counts")
    lines.append("| Class Label | Source Dataset | Count |")
    lines.append("|---|---|---|")

    # Iterate through the pivot table
    for label in stats.index:
        for source in stats.columns:
            count = stats.loc[label, source]
            if count > 0:
                lines.append(f"| {label} | {source} | {count:,} |")
    
    report_content = "\n".join(lines)
    
    # Ensure output dir exists
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Report generated at: {OUTPUT_MD}")
    print("-" * 40)
    # Print head of report
    print("\n".join(lines[:20]))
    print("...")

if __name__ == "__main__":
    generate_report()
