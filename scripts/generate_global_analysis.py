"""Generate a global analysis notebook for the combined dataset."""
import json
from pathlib import Path

def generate_global_analysis_notebook():
    null = None
    output_path = Path("notebooks/global_dataset_analysis.ipynb")
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Global Dataset Analysis\n",
                "\n",
                "This notebook provides a comprehensive analysis of the combined agricultural dataset, aggregating statistics from all source datasets."
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
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "from PIL import Image\n",
                "\n",
                "# Setup plotting style\n",
                "plt.style.use('ggplot')\n",
                "sns.set_theme(style=\"whitegrid\")\n",
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
                "DATA_ROOT = Path(\"../data/processed/dataset\")\n",
                "CSV_PATH = DATA_ROOT / \"combined_dataset.csv\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load Data\n",
                "if not CSV_PATH.exists():\n",
                "    print(f\"Error: Combined CSV not found at {CSV_PATH}\")\n",
                "else:\n",
                "    df = pd.read_csv(CSV_PATH)\n",
                "    print(f\"Loaded {len(df)} images from {len(df['source'].unique())} datasets.\")\n",
                "    display(df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Dataset Distribution\n",
                "How many images are contributed by each source dataset?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
                        "source": [
                            "if 'df' in locals():\n",
                            "    plt.figure(figsize=(12, 6))\n",
                            "    sns.countplot(data=df, x='source', order=df['source'].value_counts().index, hue='source', palette='viridis', legend=False)\n",
                            "    plt.title('Image Count by Source Dataset')\n",
                            "    plt.xticks(rotation=45)\n",
                            "    plt.ylabel('Number of Images')\n",
                            "    plt.show()\n",
                            "    \n",
                            "    display(df['source'].value_counts().to_frame(name='Count'))"
                        ]
                    },
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            "## 2. Crop Analysis\n",
                            "Which crops are most represented in the global dataset?"
                        ]
                    },
                    {
                        "cell_type": "code",
                        "execution_count": null,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "if 'df' in locals():\n",
                            "    # Count crops\n",
                            "    crop_counts = df['crop'].value_counts()\n",
                            "    \n",
                            "    plt.figure(figsize=(15, 8))\n",
                            "    sns.barplot(x=crop_counts.index, y=crop_counts.values, hue=crop_counts.index, palette='magma', legend=False)\n",
                            "    plt.title('Global Crop Distribution')\n",
                            "    plt.xticks(rotation=90)\n",
                            "    plt.ylabel('Number of Images')\n",
                            "    plt.show()"
                        ]
                    },
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            "## 3. Crop vs. Dataset Matrix\n",
                            "How do different datasets overlap in terms of crop coverage? This is crucial for domain adaptation tasks."
                        ]
                    },
                    {
                        "cell_type": "code",
                        "execution_count": null,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "if 'df' in locals():\n",
                            "    pivot_table = pd.crosstab(df['crop'], df['source'])\n",
                            "    \n",
                            "    plt.figure(figsize=(12, 10))\n",
                            "    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)\n",
                            "    plt.title('Crop vs. Source Dataset Heatmap')\n",
                            "    plt.show()"
                        ]
                    },
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            "## 4. Class Balance (Top 30 Classes)\n",
                            "Examining the most frequent classes to check for imbalance."
                        ]
                    },
                    {
                        "cell_type": "code",
                        "execution_count": null,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "if 'df' in locals():\n",
                            "    top_classes = df['label'].value_counts().head(30)\n",
                            "    \n",
                            "    plt.figure(figsize=(15, 8))\n",
                            "    sns.barplot(y=top_classes.index, x=top_classes.values, hue=top_classes.index, palette='coolwarm', orient='h', legend=False)\n",
                            "    plt.title('Top 30 Most Frequent Classes')\n",
                            "    plt.xlabel('Count')\n",
                            "    plt.show()"
                        ]
                    },
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            "## 5. Disease Distribution\n",
                            "Aggregating diseases (ignoring crop type) to see prevalent pathologies."
                        ]
                    },
                    {
                        "cell_type": "code",
                        "execution_count": null,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "if 'df' in locals():\n",
                            "    disease_counts = df['disease'].value_counts().head(20)\n",
                            "    \n",
                            "    plt.figure(figsize=(15, 8))\n",
                            "    sns.barplot(y=disease_counts.index, x=disease_counts.values, hue=disease_counts.index, palette='rocket', orient='h', legend=False)\n",
                            "    plt.title('Top 20 Disease Types (Global)')\n",
                            "    plt.xlabel('Count')\n",
                            "    plt.show()"
                        ]
                    }    ]

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

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
        
    print(f"Generated global analysis notebook: {output_path}")

if __name__ == "__main__":
    generate_global_analysis_notebook()
