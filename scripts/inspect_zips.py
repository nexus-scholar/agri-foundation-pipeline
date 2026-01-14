import zipfile
import os

ZIPS = [
    "data/raw/dataset/RoCoLe.zip",
    "data/raw/dataset/plant-pathology-2021-fgvc8.zip"
]

def inspect_zip(path):
    print(f"\n--- Inspecting {os.path.basename(path)} ---")
    try:
        with zipfile.ZipFile(path, 'r') as z:
            namelist = z.namelist()
            print(f"Total files: {len(namelist)}")
            print("First 15 files:")
            for n in namelist[:15]:
                print(f"  {n}")
            
            # Check for CSVs
            csvs = [n for n in namelist if n.endswith('.csv')]
            if csvs:
                print("\nFound CSVs:")
                for c in csvs:
                    print(f"  {c}")
                    # Peek at CSV content
                    try:
                        with z.open(c) as f:
                            head = [next(f).decode() for _ in range(3)]
                            print(f"    Header/Row 1: {head[0].strip()}")
                            if len(head) > 1: print(f"    Row 2: {head[1].strip()}")
                    except:
                        pass
    except Exception as e:
        print(f"Error reading zip: {e}")

for z in ZIPS:
    inspect_zip(z)
