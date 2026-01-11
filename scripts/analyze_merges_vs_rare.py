"""
Analyze how merging classes affects the underrepresented classes list.
"""

import pandas as pd
from pathlib import Path

# Config
MERGE_CANDIDATES_CSV = Path("data/processed/dataset/merge_candidates_report.csv")
RARE_CLASSES_CSV = Path("data/processed/dataset/underrepresented_classes.csv")

def analyze_impact():
    if not MERGE_CANDIDATES_CSV.exists() or not RARE_CLASSES_CSV.exists():
        print("Error: Input CSV files not found.")
        return

    print("Loading reports...")
    merge_df = pd.read_csv(MERGE_CANDIDATES_CSV)
    rare_df = pd.read_csv(RARE_CLASSES_CSV)
    
    print(f"Merge Candidates: {len(merge_df)}")
    print(f"Rare Classes (<50): {len(rare_df)}")
    
    # Check which rare classes are in the merge list
    rare_in_merge = pd.merge(rare_df, merge_df, left_on='label', right_on='original_label', how='inner')
    
    print(f"\nFound {len(rare_in_merge)} rare classes that are candidates for merging.")
    
    if rare_in_merge.empty:
        return

    # Calculate projected counts after merge
    # We need to group by base_label and sum the counts from the merge_df (which has all variants)
    
    print("\n--- Impact Analysis ---")
    print(f"{ 'Base Label':<50} | {'Current Count (Rare)':<20} | {'Projected Count (Merged)':<20} | {'Status'}")
    print("-" * 110)
    
    # Get unique base labels involved with rare classes
    bases_involved = rare_in_merge['base_label'].unique()
    
    saved_count = 0
    
    for base in bases_involved:
        # All variants for this base
        group = merge_df[merge_df['base_label'] == base]
        projected_total = group['count'].sum()
        
        # Specific rare variants in this group
        rare_variants = rare_in_merge[rare_in_merge['base_label'] == base]
        
        for _, row in rare_variants.iterrows():
            current_count = row['count_x'] # count from rare_df
            status = "SAVED (>50)" if projected_total >= 50 else "STILL RARE (<50)"
            if projected_total >= 50:
                saved_count += 1
                
            print(f"{base:<50} | {row['label'] + ' (' + str(current_count) + ')':<20} | {projected_total:<24} | {status}")

    print("-" * 110)
    print(f"Summary: Merging will save {saved_count} classes from being 'underrepresented'.")

if __name__ == "__main__":
    analyze_impact()
