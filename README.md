# Dataset Processing Pipeline

This repository provides a reproducible pipeline for preparing the PlantVillage, PlantDoc, and Tomato Leaf datasets for downstream machine learning experiments.

## Overview

The pipeline performs the following steps:

1. Extract raw archives from `data/raw/dataset/` using Windows long-path safe utilities.
2. Normalize raw directory structures into `data/processed/dataset/{PlantVillage,PlantDoc,TomatoLeaf}`.
3. Process each dataset into curated folders with sequential filenames, CSV labels, and metadata.
4. Merge all processed records into `data/processed/dataset/combined_dataset.csv` for cross-dataset analysis.

## Requirements

- Windows 10/11 (long path support enabled) or WSL/Ubuntu.
- Python 3.10+ (virtual environment recommended).
- Sufficient disk space (~10 GB) for extracted datasets.

Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Preparation

Place the raw archives in `data/raw/dataset/`:

- `plantvillage.zip`
- `plantdoc.zip`
- `Tomato Leaf Dataset  A dataset for multiclass disease detection and classification.zip`

The repository intentionally ships without processed outputs. They will be regenerated into `data/processed/dataset/` when you run the pipeline.

## Running the Pipeline

```bash
python process_datasets.py --datasets plantvillage plantdoc tomatoleaf --log-level INFO
```

Options:

- `--datasets`: subset of datasets to process (`plantvillage`, `plantdoc`, `tomatoleaf`). Defaults to all.
- `--log-level`: Python logging level (e.g., `INFO`, `DEBUG`).

Only the selected datasets will be extracted and processed. Re-running is idempotent; previous processed folders are cleared automatically.

## Module Overview

- `pipeline/config.py`: centralizes project paths, CLI parsing, and dataset selection.
- `pipeline/fs_utils.py`: Windows-safe helpers for long paths, recursive copy, and cleanup.
- `pipeline/zip_utils.py`: extraction pipeline with rename manifest when original names exceed limits.
- `pipeline/plantvillage.py`: converts PlantVillage folders into normalized labels, CSV, metadata.
- `pipeline/plantdoc.py`: merges PlantDoc train/test splits with label normalization.
- `pipeline/tomato_leaf.py`: ingests Tomato Leaf raw images and YOLO annotated splits.
- `process_datasets.py`: orchestrates extraction, processing, combined CSV, and summary.
- `tests/test_label_utils.py`: regression tests for label normalization behavior.

## Outputs

After a full run, expect the following structure (folders regenerated as needed):

```
data/
├── raw/
│   └── dataset/
│       ├── plantvillage.zip
│       ├── plantdoc.zip
│       └── Tomato Leaf Dataset  A dataset for multiclass disease detection and classification.zip
└── processed/
    └── dataset/
        ├── PlantVillage/
        ├── PlantVillage_processed/
        │   ├── {class_label}/image-00001.jpg
        │   ├── labels.csv
        │   └── metadata.json
        ├── PlantDoc/
        ├── PlantDoc_processed/
        │   ├── {class_label}/image-00001.jpg
        │   ├── labels.csv
        │   └── metadata.json
        ├── TomatoLeaf/
        ├── TomatoLeaf_processed/
        │   ├── images/
        │   ├── annotated/{train,test}/images
        │   ├── metadata.json
        │   └── labels/
        └── combined_dataset.csv
```

## Troubleshooting

- **Long path errors**: Ensure Windows long paths are enabled (`LongPathsAware=1`) or use WSL.
- **Missing zip files**: Verify archives exist under `data/raw/dataset/`; the script aborts if none are found.
- **Partial extraction**: Check `processing.log` and the `{dataset}_renamed_files.json` manifest to review files renamed due to long paths.
- **Performance**: Running on SSD and disabling real-time antivirus for the repo can significantly speed up extraction.

## Testing

```bash
.\.venv\Scripts\python.exe -m pytest tests
```

## License / Citation

Refer to the original dataset licenses (PlantVillage, PlantDoc, Tomato Leaf). This pipeline script is provided under the repository's default license.
