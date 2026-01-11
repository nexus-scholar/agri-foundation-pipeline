"Generate Jupyter notebooks for dataset analysis."
import json
import os
from pathlib import Path

def generate_notebooks():
    # Load dataset definitions
    with open("datasets.json", "r") as f:
        datasets = json.load(f)

    # Notebook structure template
    def create_notebook_content(dataset_name, processed_dir):
        # We need to escape backslashes for the JSON string if on Windows, 
        # but forward slashes work fine in Python paths even on Windows.
        # The paths in the notebook code will be relative.
        
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {dataset_name.title()} Dataset Analysis\n",
                    "\n",
                    f"This notebook provides a breakdown and visualization of the **{dataset_name}** dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from pathlib import Path\n",
                    "from PIL import Image\n",
                    "import random\n",
                    "\n",
                    "# Setup plotting style\n",
                    "plt.style.use('ggplot')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Configuration\n",
                    f"DATASET_NAME = \"{dataset_name}\"\n",
                    f"PROCESSED_DIR = \"{processed_dir}\"\n",
                    "# Path relative to this notebook (notebooks/datasets/)\n",
                    "DATA_ROOT = Path(\"../../data/processed/dataset\")\n",
                    "CSV_PATH = DATA_ROOT / PROCESSED_DIR / \"labels.csv\""
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the dataset metadata\n",
                    "if not CSV_PATH.exists():\n",
                    "    print(f\"Error: Labels file not found at {CSV_PATH}\")\n",
                    "else:\n",
                    "    df = pd.read_csv(CSV_PATH)\n",
                    "    print(f\"Successfully loaded {len(df)} records.\")\n",
                    "    display(df.head())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Class Distribution\n",
                    "Let's look at how the images are distributed across different classes."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if 'df' in locals():\n",
                    "    class_counts = df['label'].value_counts()\n",
                    "    print(f\"Total Classes: {len(class_counts)}\")\n",
                    "    \n",
                    "    # Plot\n",
                    "    plt.figure(figsize=(15, 8))\n",
                    "    class_counts.plot(kind='bar')\n",
                    "    plt.title(f\"{DATASET_NAME} - Class Distribution\")\n",
                    "    plt.xlabel(\"Class Label\")\n",
                    "    plt.ylabel(\"Number of Images\")\n",
                    "    plt.xticks(rotation=90)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    display(class_counts.to_frame(name='Count'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Crop Analysis\n",
                    "Breakdown by crop type (extracted from labels)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if 'df' in locals() and 'crop' in df.columns:\n",
                    "    crop_counts = df['crop'].value_counts()\n",
                    "    \n",
                    "    plt.figure(figsize=(10, 6))\n",
                    "    crop_counts.plot(kind='bar', color='green')\n",
                    "    plt.title(f\"{DATASET_NAME} - Crop Distribution\")\n",
                    "    plt.ylabel(\"Count\")\n",
                    "    plt.show()\n",
                    "    display(crop_counts.to_frame(name='Count'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Sample Images\n",
                    "Displaying random samples from the dataset to verify quality and content."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def show_random_samples(df, n=9):\n",
                    "    if df is None or len(df) == 0: return\n",
                    "    \n",
                    "    plt.figure(figsize=(15, 15))\n",
                    "    samples = df.sample(min(n, len(df)))\n",
                    "    \n",
                    "    for i, (_, row) in enumerate(samples.iterrows()):\n",
                    "        plt.subplot(3, 3, i+1)\n",
                    "        # The 'path' column in csv is relative to data/processed/dataset\n",
                    "        img_path = DATA_ROOT / row['path']\n",
                    "        \n",
                    "        try:\n",
                    "            img = Image.open(img_path)\n",
                    "            plt.imshow(img)\n",
                    "            plt.title(f\"{row['label']}\\n({row['crop']})\", fontsize=10)\n",
                    "            plt.axis('off')\n",
                    "        except Exception as e:\n",
                    "            plt.text(0.5, 0.5, f\"Error loading image:\\n{str(e)}\", ha='center')\n",
                    "            plt.axis('off')\n",
                    "            \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "\n",
                    "if 'df' in locals():\n",
                    "    show_random_samples(df)"
                ]
            }
        ]

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        return notebook

    # Global "null" for JSON compatibility
    null = None

    output_dir = Path("notebooks/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        name = ds['name']
        processed_dir = ds['processed_dir']
        
        nb_content = create_notebook_content(name, processed_dir)
        
        filename = output_dir / f"{name}_analysis.ipynb"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(nb_content, f, indent=1)
            
        print(f"Generated notebook: {filename}")

if __name__ == "__main__":
    generate_notebooks()
