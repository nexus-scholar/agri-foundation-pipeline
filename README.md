# Agri-Foundation Dataset Pipeline

**A robust, reproducible pipeline for aggregating, normalizing, and cleaning large-scale agricultural image datasets.**

This repository contains the data engineering infrastructure used to create the **Agri-Foundation-145k** dataset—a unified benchmark for plant disease detection comprising 144,751 images across 215 distinct classes, aggregated from 8 open-access sources.

## 🌟 Key Features

*   **Multi-Source Aggregation:** Standardizes data from PlantVillage, PlantDoc, New Plant Diseases, Tomato Leaf, Cassava, Wheat, PlantSeg, and PlantWild.
*   **Taxonomic Alignment:** "Fuzzy Alignment" algorithm maps disparate folder names (e.g., `Tomato_Early_Blight` vs. `Tomato___Early_blight`) to canonical biological entities.
*   **Quality Control:** Automated MD5 deduplication, file corruption checks, and class imbalance analysis.
*   **Research Grade:** Fully reproducible workflow with detailed logging and artifacts.

## 📂 Repository Structure

```text
.
├── data/
│   ├── raw/                 # Downloaded zip files
│   ├── processed/           # Extracted and standardized datasets
│   └── release/             # Final, publication-ready dataset
├── pipeline/                # Core processing logic
│   ├── data_utils.py        # Hashing & CSV utilities
│   ├── fs_utils.py          # Long-path safe file operations
│   └── [dataset].py         # Per-dataset normalization rules
├── scripts/                 # Execution scripts
│   ├── process_datasets.py        # Master ingestion script
│   ├── verify_and_clean.py        # QC & Deduplication
│   └── package_for_release.py     # Final artifact generation
├── notebooks/               # Analysis & Visualization
└── docs/                    # Detailed documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Recommended: SSD storage (I/O intensive)

```bash
# Clone the repository
git clone https://github.com/yourusername/dataset-processing.git
cd dataset-processing

# Install dependencies
pip install -r requirements.txt
```

### Usage

**1. Ingest & Normalize**
Download sources and standardize them into a unified structure.
```bash
python process_datasets.py
```

**2. Verify & Clean**
Run integrity checks and remove duplicates.
```bash
python scripts/verify_and_clean_dataset.py
```

**3. Package**
Generate the final release folder.
```bash
python scripts/package_for_release.py
```

## 📊 Documentation

*   [**Dataset Preparation Guide**](docs/DATASET_PREPARATION.md): Step-by-step reproduction instructions.
*   [**Dataset Details**](docs/DATASETS.md): Specifics on the 8 source datasets.
*   [**Pipeline Architecture**](docs/PIPELINE_OVERVIEW.md): Technical design and data flow.

## 📝 Citation

If you use this pipeline or dataset in your research, please cite:

```bibtex
@misc{agri_foundation_pipeline,
  author = {Your Name},
  title = {Agri-Foundation Dataset Processing Pipeline},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/dataset-processing}}
}
```

## 📄 License

This code is released under the **MIT License**. The datasets processed by this pipeline retain the licenses of their original authors.