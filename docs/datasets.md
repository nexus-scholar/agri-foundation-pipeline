# Dataset Notes

This document provides specific details on the structure and processing logic for each of the eight source datasets aggregated in the Agri-Foundation project.

## 1. PlantVillage
- **Source:** Kaggle / PSU
- **Structure:** Nested folders `PlantVillage/{ClassName}/image.jpg`.
- **Processing (`pipeline/plantvillage.py`):**
  - Recursively finds class folders.
  - Normalizes class names (e.g., `Tomato_Early_blight` -> `tomato_early_blight`).
  - Renames images sequentially.
  - Output: `PlantVillage_processed/{label}/`.

## 2. PlantDoc
- **Source:** GitHub
- **Structure:** Split into `train` and `test` folders.
- **Processing (`pipeline/plantdoc.py`):**
  - Merges `train` and `test` splits into a single unified dataset.
  - Handles long Windows filenames via a rename manifest.
  - Output: `PlantDoc_processed/{label}/`.

## 3. Tomato Leaf Dataset
- **Source:** Mendeley Data
- **Structure:** Contains both raw images and YOLO-formatted annotations (`.txt`).
- **Processing (`pipeline/tomato_leaf.py`):**
  - Maps YOLO class indices (0-6) to specific disease names (e.g., `0` -> `early_blight`).
  - Creates annotated labels based on `.txt` files.
  - Output: `TomatoLeaf_processed/{label}/`.

## 4. Cassava Leaf Disease
- **Source:** Kaggle
- **Structure:** Flat `train_images` folder with a separate `train.csv` and `label_num_to_disease_map.json`.
- **Processing (`pipeline/cassava.py`):**
  - Reads `train.csv` to map image IDs to numeric labels.
  - Uses the JSON map to convert numeric labels to text (e.g., `3` -> `Cassava Mosaic Disease`).
  - Sorts images into class folders.
  - Output: `Cassava_processed/{label}/`.

## 5. Wheat Leaf Disease
- **Source:** Kaggle
- **Structure:** Nested zip file structure.
- **Processing (`pipeline/wheat.py`):**
  - Extracts inner `Wheat Disease.zip`.
  - Scans for class folders (e.g., `Septoria`, `Stripe Rust`).
  - Normalizes names to `wheat_{disease}`.
  - Output: `Wheat_processed/{label}/`.

## 6. New Plant Diseases
- **Source:** Kaggle (Augmented PlantVillage)
- **Structure:** `New Plant Diseases Dataset(Augmented)/train/{Class}`.
- **Processing (`pipeline/new_plant_diseases.py`):**
  - Handles the `Crop___Disease` naming convention (triple underscores).
  - Recursively finds valid class folders.
  - Output: `NewPlantDiseases_processed/{label}/`.

## 7. PlantSeg
- **Source:** Research Dataset
- **Structure:** Images in `images/` with a `Metadata.csv` defining labels.
- **Processing (`pipeline/plantseg.py`):**
  - Parses `Metadata.csv` to link filenames to `Plant` and `Disease` columns.
  - Constructs labels as `{Plant}_{Disease}`.
  - Locates and moves images.
  - Output: `PlantSeg_processed/{label}/`.

## 8. PlantWild
- **Source:** Research Dataset
- **Structure:** Folders named by class (e.g., `apple black rot`).
- **Processing (`pipeline/plantwild.py`):**
  - Standard folder traversal.
  - Normalizes folder names to snake_case.
  - Output: `PlantWild_processed/{label}/`.

---

## Combined CSV Schema

The master index `data/processed/dataset/combined_dataset.csv` consolidates all above sources:

| Field | Description |
| :--- | :--- |
| `filename` | Standardized filename (e.g., `image-00001.jpg`). |
| `label` | The normalized, unified class label. |
| `crop` | The crop type extracted from the label. |
| `disease` | The specific pathology extracted from the label. |
| `source` | The dataset origin (e.g., `plantvillage`, `cassava`). |
| `path` | Relative path to the image file. |