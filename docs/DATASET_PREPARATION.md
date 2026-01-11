# Dataset Preparation and Processing

This document details the creation of the **Agri-Foundation-145k** dataset, a large-scale, taxonomically harmonized agricultural dataset designed for pre-training foundation models.

## 1. Data Collection

The dataset is an aggregation of eight open-access agricultural repositories.

### Source Datasets

| Dataset | Source | Description |
| :--- | :--- | :--- |
| **PlantVillage** | Kaggle / PSU | High-quality lab-controlled images of healthy and diseased leaves. |
| **PlantDoc** | GitHub | "In-the-wild" field images with complex backgrounds and multiple leaves. |
| **New Plant Diseases** | Kaggle | An augmented version of PlantVillage with additional crops and diseases. |
| **Tomato Leaf** | Mendeley Data | Specific focus on tomato bacterial spot, early/late blight, and others (YOLO annotated). |
| **Cassava Leaf Disease** | Kaggle | Focuses on cassava mosaic disease, bacterial blight, and others. |
| **Wheat Leaf Disease** | Kaggle | Covers wheat rusts, septoria, and other pathologies. |
| **PlantSeg** | Research | Specialized segmentation dataset. |
| **PlantWild** | Research | Additional field-collected images for robustness. |

## 2. Processing Pipeline

The processing pipeline (`process_datasets.py`) standardizes these diverse sources into a unified structure.

### 2.1. Extraction & Handling
*   **Long Paths:** The pipeline (`pipeline/fs_utils.py`) handles Windows long path limitations using `\\?\` prefixes.
*   **Renaming:** Files are renamed to a sequential format (e.g., `image-00001.jpg`) to avoid encoding issues and filename collisions.
*   **Structure:** Each source dataset is extracted to `data/processed/dataset/{DatasetName}`.

### 2.2. Taxonomic Normalization ("Fuzzy Alignment")
Raw class names varied widely across datasets (e.g., `Tomato_Early_Blight` vs `Tomato___Early_blight` vs `Tomato Early Blight Leaf`).
We implemented a **Disease Normalization** strategy (`scripts/align_foundation_labels.py` logic) to map these to **215 Canonical Classes**.

**Normalization Steps:**
1.  **Tokenization:** Split folder names by `_`, `-`, or spaces.
2.  **Stop Word Removal:** Removed non-semantic terms like "leaf", "plant", "processed".
3.  **Grouping:** Classes with identical normalized tokens were grouped.
4.  **Scientific Validation:** Mappings were verified against phytopathological taxonomy (see `data/processed/dataset/scientific_name_mapping.csv`).

### 2.3. Quality Control
*   **Deduplication:** We computed MD5 hashes for all images. **51,323 duplicate images** (mostly cross-dataset overlaps) were removed to prevent data leakage.
*   **Corruption Check:** All images were verified using the PIL library to ensure they can be opened.
*   **Class Balance:** Classes with < 50 images were flagged. Merging synonyms (e.g., `_google` vs `_bing` variants) rescued 54 classes from being underrepresented.

## 3. How to Reproduce

Follow these steps to regenerate the clean foundation dataset from scratch.

### Prerequisites
*   Python 3.8+
*   Dependencies: `pandas`, `tqdm`, `Pillow`, `torchvision` (for loader testing)

```bash
pip install -r requirements.txt
```

### Step 1: Download & Process
This script downloads (where possible) or expects zips in `data/raw/dataset/`, extracts them, and standardizes the format.

```bash
python process_datasets.py
```
*Output:* `data/processed/dataset/combined_dataset.csv` (Index of all ~196k raw images)

### Step 2: Verify & Clean
This performs integrity checks, MD5 deduplication, and stratification analysis.

```bash
python scripts/verify_and_clean_dataset.py
```
*Output:* `data/processed/dataset/foundation_dataset_v1_clean.csv` (The filtered index of ~145k images)

### Step 3: Package for Release
This organizes the cleaned files into the final folder structure (`data/label/image.jpg`) ready for training/publication.

```bash
python scripts/package_for_release.py
```
*Output:* `data/release/agri_foundation_v1/`

## 4. Dataset Output

The final dataset is located at `data/release/agri_foundation_v1/`.

*   **Total Images:** 144,751
*   **Classes:** 215
*   **Metadata:** `metadata.csv` (Maps filename to source, crop, disease, and original path)

For detailed analysis of the dataset composition, run:
```bash
jupyter notebook notebooks/global_dataset_analysis.ipynb
```