#!/usr/bin/env python3
"""
Complete Reproducible Dataset Processing Pipeline

Processes PlantVillage and PlantDoc datasets:
- Unzips datasets from data/raw/dataset/*.zip to data/processed/dataset/
- Merges PlantDoc train/test into single dataset
- Renames images to simple sequential format: image-00001.jpg
- Extracts crop and disease from folder names
- Creates CSV with filename, label (full class name), crop, disease
- Organizes output by class label (crop_disease format)
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import importlib

from pipeline.config import load_default_config, parse_args, DatasetSource
from pipeline.logging_utils import configure_logging
from pipeline.zip_utils import unzip_dataset
from pipeline.download_utils import download_datasets


def _import_processor(processor_path: str):
    module_name, func_name = processor_path.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def create_combined_csv(processed_data: dict[str, list], output_path):
    """Create combined CSV with all data."""
    print("\n" + "=" * 60)
    print("CREATING COMBINED DATASET CSV")
    print("=" * 60)

    combined = []
    
    for source_name, data in processed_data.items():
        print(f"Adding {source_name}: {len(data)} images")
        for row in data:
            combined.append({
                'filename': row['filename'],
                'label': row['label'],
                'crop': row['crop'],
                'disease': row['disease'],
                'source': source_name,
                'path': row['path']
            })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label', 'crop', 'disease', 'source', 'path'])
        writer.writeheader()
        writer.writerows(combined)

    print(f"TOTAL:        {len(combined)} images")
    print(f"\nSaved: {output_path}")

    return combined


def print_summary(processed_data: dict[str, list]):
    """Print final summary with class overlap analysis."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_labels = set()
    dataset_labels = {}

    for source_name, data in processed_data.items():
        labels = set(r['label'] for r in data)
        dataset_labels[source_name] = labels
        all_labels.update(labels)
        
        tomato_classes = set(l for l in labels if l.startswith('tomato') or l.startswith('Tomato'))
        
        print(f"\n{source_name}:")
        print(f"  Total images: {len(data)}")
        print(f"  Total classes: {len(labels)}")
        print(f"  Tomato classes: {len(tomato_classes)}")

    # Check for overlaps
    print("\nClass Overlap Analysis:")
    for label in sorted(all_labels):
        sources_with_label = []
        for source_name, labels in dataset_labels.items():
            if label in labels:
                sources_with_label.append(source_name)
        
        if len(sources_with_label) > 1:
            counts = []
            for src in sources_with_label:
                count = sum(1 for r in processed_data[src] if r['label'] == label)
                counts.append(f"{src}={count}")
            print(f"  {label}: {', '.join(counts)}")

    print("\n" + "=" * 60)
    print("DATASET PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nOutput structure:")
    for source_name in processed_data.keys():
        print(f"  data/processed/dataset/{source_name}_processed/")
        print("    +-- {class_label}/image-00001.jpg, ...")
        print("    +-- labels.csv")
        print("    +-- metadata.json")
    print("  data/processed/dataset/combined_dataset.csv")
    print("\nTo reproduce: python process_datasets.py")


def main():
    config = parse_args()
    logger = configure_logging(config.paths.log_file, config.log_level)

    print("\n" + "=" * 60)
    print("REPRODUCIBLE DATASET PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Project root: {config.paths.project_root}")
    print(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if config.download:
        print("\n" + "=" * 60)
        print("STEP 0: DOWNLOADING DATASETS")
        print("=" * 60)
        download_datasets(config.datasets, config)
        if config.download_only:
            print("Download-only flag set; exiting after downloads.")
            return

    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTING DATASETS FROM ZIP FILES")
    print("=" * 60)

    processed_data = {}
    for dataset_name in config.datasets:
        source: DatasetSource | None = config.dataset_sources.get(dataset_name)
        if not source:
            print(f"WARNING: Dataset '{dataset_name}' not defined in datasets.json")
            continue
        zip_path = source.zip_path
        raw_dir = source.raw_dir
        print(f"\n--- {dataset_name.upper()} ---")
        extracted = unzip_dataset(zip_path, raw_dir, dataset_name)
        if not extracted:
            print(f"  Skipping {dataset_name} (extraction failed)")
            continue
        processor = _import_processor(source.processor_path)
        data = processor(extracted, source.processed_dir)
        processed_data[dataset_name] = data

    if not processed_data:
        print("\nERROR: No datasets were processed successfully.")
        return

    combined_path = config.paths.combined_csv
    create_combined_csv(processed_data, combined_path)
    print_summary(processed_data)


if __name__ == "__main__":
    main()
