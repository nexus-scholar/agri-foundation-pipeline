"""
Identify classes that are likely duplicates due to search engine suffixes
(e.g., '_google', '_baidu', '_bing') and propose merges.
"""

import pandas as pd
from pathlib import Path
import re

# Config
METADATA_CSV = Path("data/release/agri_foundation_v1/metadata.csv")
OUTPUT_CSV = Path("data/processed/dataset/merge_candidates_report.csv")

def identify_merges():
    if not METADATA_CSV.exists():
        print(f"Error: {METADATA_CSV} not found.")
        return

    print("Loading metadata...")
    df = pd.read_csv(METADATA_CSV)
    
    # Get unique classes
    unique_labels = df['label'].unique()
    print(f"Total unique classes: {len(unique_labels)}")
    
    # Regex to find suffixes like _google, _baidu, _bing, _copy
    # We look for these at the end of the string
    suffix_pattern = r'(_google|_baidu|_bing|_copy)$'
    
    candidates = []
    
    for label in unique_labels:
        # Check if label has a suffix
        match = re.search(suffix_pattern, label, re.IGNORECASE)
        if match:
            # Create the 'base' label by removing the suffix
            base_label = re.sub(suffix_pattern, '', label, flags=re.IGNORECASE)
            
            # Check if this base label exists (or if other variants exist)
            # Actually, we want to group ALL variants of a base label together
            candidates.append({
                'original_label': label,
                'base_label': base_label,
                'suffix': match.group(1),
                'count': len(df[df['label'] == label])
            })
        else:
            # Even if it doesn't have a suffix, it might be the 'base' for others
            # We'll add it to the list to check for grouping, but mark suffix as None
            # Only if it matches the pattern of a base label derived from others?
            # Simpler: Just put everything in a list and group by base_label
            # But we need to define 'base_label' for non-suffixed ones too.
            # Let's assume current label IS the base label if no suffix found.
            candidates.append({
                'original_label': label,
                'base_label': label,
                'suffix': None,
                'count': len(df[df['label'] == label])
            })
            
    cand_df = pd.DataFrame(candidates)
    
    # Filter groups where there is more than 1 label for a base_label
    # AND at least one of them has a suffix (to avoid listing normal unique classes)
    
    # Count variants per base_label
    group_counts = cand_df.groupby('base_label')['original_label'].nunique()
    multi_variant_bases = group_counts[group_counts > 1].index
    
    # Filter the dataframe
    merge_groups = cand_df[cand_df['base_label'].isin(multi_variant_bases)].sort_values(by=['base_label', 'original_label'])
    
    if merge_groups.empty:
        print("No merge candidates found with standard suffixes.")
    else:
        print(f"\nFound {len(merge_groups)} classes that can be merged into {len(multi_variant_bases)} groups.")
        
        # Save detailed report
        merge_groups.to_csv(OUTPUT_CSV, index=False)
        print(f"Report saved to: {OUTPUT_CSV}")
        
        # Print preview
        print("\n--- Merge Candidates Preview ---")
        # Group by base label for display
        for base, group in merge_groups.groupby('base_label'):
            print(f"\nBase: {base}")
            print(group[['original_label', 'count']].to_string(index=False))
            total = group['count'].sum()
            print(f"  -> Potential Merged Count: {total}")

if __name__ == "__main__":
    identify_merges()
