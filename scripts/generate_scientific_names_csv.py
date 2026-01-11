"""Generate scientific name mapping CSV from markdown reference."""
import re
import pandas as pd
from pathlib import Path

def generate_csv():
    md_path = Path("notebooks/Plant_Disease_Scientific_Names_and_References.md")
    output_path = Path("data/processed/dataset/scientific_name_mapping.csv")
    
    if not md_path.exists():
        print(f"Error: Markdown file not found at {md_path}")
        return

    content = md_path.read_text(encoding='utf-8')
    
    # Extract the table section
    # Look for the table header
    header_pattern = r"""\|\s*Crop\s*\|\s*Dataset Class Name\s*\|\s*Common Name\s*\|\s*Scientific Name of Pathogen\s*\|\s*Reference\s*\|"""
    match = re.search(header_pattern, content)
    
    if not match:
        print("Error: Could not find the reference table in the markdown file.")
        return
        
    start_pos = match.start()
    # Find the end of the table (empty line or end of section)
    table_lines = []
    lines = content[start_pos:].split('\n')
    
    # Skip header and separator line
    # Header is line 0
    # Separator is line 1 usually like |---|---|...    
    for line in lines[2:]:
        line = line.strip()
        if not line or not line.startswith('|'):
            break
        table_lines.append(line)
        
    data = []
    for line in table_lines:
        # Parse markdown table row
        # Split by '|' and remove first/last empty elements
        parts = [p.strip() for p in line.split('|')]
        # Filter out empty strings from split (usually first and last)
        parts = [p for p in parts if p]
        
        if len(parts) >= 4: # Crop, Class Name, Common Name, Scientific Name, Ref
            crop = parts[0].replace('**', '') # Remove bolding
            class_name = parts[1]
            common_name = parts[2]
            scientific_name = parts[3].replace('*', '') # Remove italics
            reference = parts[4] if len(parts) > 4 else ""
            
            data.append({
                'crop': crop,
                'dataset_class_name': class_name,
                'common_name': common_name,
                'scientific_name': scientific_name,
                'reference_id': reference
            })
            
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Successfully extracted {len(df)} rows to {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate_csv()
