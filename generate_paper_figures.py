#!/usr/bin/env python3
"""
Paper Figure Generator

Generates publication-ready figures from experiment results.

Usage:
    python generate_paper_figures.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("results/paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set Paper Style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})


def plot_benchmark_bar():
    """Figure 5: Architecture Comparison on Potato - Accuracy vs Latency Trade-off."""
    # Data from Table VII / Phase 5 results
    data = {
        'Architecture': ['MobileNetV3', 'EfficientNet', 'MobileViT'],
        'Accuracy (%)': [53.33, 62.22, 64.44],
        'Latency (ms)': [7.1, 18.0, 25.0]
    }
    plot_df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Bar plot for Accuracy
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']  # Green, Orange, Blue
    sns.barplot(x='Architecture', y='Accuracy (%)', data=plot_df, palette=colors, ax=ax1, alpha=0.8)
    ax1.set_ylim(40, 70)
    ax1.set_ylabel("Field Accuracy (%)", fontweight='bold')

    # Line plot for Latency
    ax2 = ax1.twinx()
    sns.lineplot(x='Architecture', y='Latency (ms)', data=plot_df, color='red',
                 marker='o', linewidth=3, ax=ax2, sort=False)
    ax2.set_ylabel("Inference Latency (ms) - Lower is Better", color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 30)

    plt.title("Trade-off: Accuracy vs. Edge Latency (Potato)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig5_Efficiency_Tradeoff.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "Fig5_Efficiency_Tradeoff.pdf")
    plt.close()
    print("[SAVED] Fig5_Efficiency_Tradeoff.png")


def plot_generalization_gap():
    """Figure 1: Generalization Gap Bar Chart."""
    # Load data if available
    try:
        df = pd.read_csv("results/analysis/full_experiment_data.csv")
        p1 = df[(df['phase'] == 'P1') & (df['strong_aug'] == False)]

        if p1.empty:
            print("[SKIP] No Phase 1 data for generalization gap plot")
            return

    except FileNotFoundError:
        print("[SKIP] No experiment data found. Run analysis_report.py first.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Pivot for grouped bar chart
    crops = ['tomato', 'potato', 'pepper']
    models = ['mobilenetv3', 'efficientnet', 'mobilevit']
    model_labels = ['MobileNetV3', 'EfficientNet', 'MobileViT']

    x = range(len(crops))
    width = 0.25
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        gaps = []
        for crop in crops:
            row = p1[(p1['crop'] == crop) & (p1['model'] == model)]
            if not row.empty:
                gaps.append(row['gap'].values[0])
            else:
                gaps.append(0)

        ax.bar([xi + i*width for xi in x], gaps, width, label=label, color=color, alpha=0.8)

    ax.set_xlabel('Crop', fontweight='bold')
    ax.set_ylabel('Generalization Gap (%)', fontweight='bold')
    ax.set_title('Lab-to-Field Generalization Gap by Architecture', fontweight='bold')
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([c.title() for c in crops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig1_Generalization_Gap.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "Fig1_Generalization_Gap.pdf")
    plt.close()
    print("[SAVED] Fig1_Generalization_Gap.png")


def plot_learning_curves():
    """Figure 3: Active Learning Trajectories."""
    try:
        df = pd.read_csv("results/analysis/full_experiment_data.csv")
    except FileNotFoundError:
        print("[SKIP] No experiment data found.")
        return

    # Need trajectory data from metrics.json - this requires loading raw data
    print("[INFO] Learning curves require raw metrics.json data.")
    print("       Use analysis_report.py for full trajectory plots.")


def plot_al_strategy_comparison():
    """Figure 2: AL Strategy Comparison (Random vs Entropy vs Hybrid)."""
    # Placeholder data - update with actual Phase 3 results
    data = {
        'Round': [0, 1, 2, 3, 4, 5] * 3,
        'Labels': [0, 10, 20, 30, 40, 50] * 3,
        'Accuracy': [
            # Random
            25.0, 28.0, 32.0, 35.0, 38.0, 42.0,
            # Entropy
            25.0, 22.0, 30.0, 38.0, 45.0, 48.0,
            # Hybrid
            25.0, 30.0, 36.0, 42.0, 48.0, 54.0,
        ],
        'Strategy': ['Random']*6 + ['Entropy']*6 + ['Hybrid']*6
    }
    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Random': '#7f7f7f', 'Entropy': '#ff7f0e', 'Hybrid': '#2ca02c'}
    markers = {'Random': 'o', 'Entropy': 's', 'Hybrid': '^'}

    for strategy in ['Random', 'Entropy', 'Hybrid']:
        subset = plot_df[plot_df['Strategy'] == strategy]
        ax.plot(subset['Labels'], subset['Accuracy'],
                label=strategy, color=colors[strategy],
                marker=markers[strategy], linewidth=2, markersize=8)

    ax.set_xlabel('Labeled Samples', fontweight='bold')
    ax.set_ylabel('Field Accuracy (%)', fontweight='bold')
    ax.set_title('Active Learning Strategy Comparison (Tomato)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig2_AL_Strategy_Comparison.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "Fig2_AL_Strategy_Comparison.pdf")
    plt.close()
    print("[SAVED] Fig2_AL_Strategy_Comparison.png")


def plot_fixmatch_improvement():
    """Figure 4: FixMatch Improvement on Potato (AL Only vs AL+FixMatch)."""
    # Placeholder data - update with actual Phase 4 results
    data = {
        'Round': [0, 1, 2, 3, 4, 5] * 2,
        'Labels': [0, 10, 20, 30, 40, 50] * 2,
        'Accuracy': [
            # AL Only
            35.0, 38.0, 42.0, 45.0, 48.0, 50.0,
            # AL + FixMatch
            35.0, 42.0, 50.0, 55.0, 58.0, 62.0,
        ],
        'Method': ['AL Only']*6 + ['AL + FixMatch']*6
    }
    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'AL Only': '#7f7f7f', 'AL + FixMatch': '#2ca02c'}

    for method in ['AL Only', 'AL + FixMatch']:
        subset = plot_df[plot_df['Method'] == method]
        linestyle = '--' if method == 'AL Only' else '-'
        ax.plot(subset['Labels'], subset['Accuracy'],
                label=method, color=colors[method],
                marker='o', linewidth=2.5, markersize=8, linestyle=linestyle)

    ax.set_xlabel('Labeled Samples', fontweight='bold')
    ax.set_ylabel('Field Accuracy (%)', fontweight='bold')
    ax.set_title('FixMatch Improvement on Potato (PDA Scenario)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate improvement
    ax.annotate('+12%', xy=(50, 62), fontsize=14, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig4_FixMatch_Improvement.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "Fig4_FixMatch_Improvement.pdf")
    plt.close()
    print("[SAVED] Fig4_FixMatch_Improvement.png")


def plot_confusion_matrix_potato():
    """Figure 6: Confusion Matrix for Potato FixMatch."""
    import numpy as np

    # Example confusion matrix - update with actual P4 results
    # Classes: Early Blight, Late Blight, Healthy (but healthy should be 0)
    cm = np.array([
        [25, 5, 0],   # Early Blight
        [8, 22, 0],   # Late Blight
        [0, 0, 0],    # Healthy (missing in field - should be all zeros)
    ])

    classes = ['Early Blight', 'Late Blight', 'Healthy*']

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)

    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title('Confusion Matrix: Potato with FixMatch\n*Healthy class absent in field data', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig6_Confusion_Matrix_Potato.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "Fig6_Confusion_Matrix_Potato.pdf")
    plt.close()
    print("[SAVED] Fig6_Confusion_Matrix_Potato.png")


def main():
    print("="*60)
    print("GENERATING PAPER FIGURES")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate all figures
    plot_generalization_gap()
    plot_al_strategy_comparison()
    plot_fixmatch_improvement()
    plot_benchmark_bar()
    plot_confusion_matrix_potato()

    print()
    print("="*60)
    print("[DONE] All figures saved to:", OUTPUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()

