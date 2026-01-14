import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import timm
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Configuration defaults
DEFAULT_DATA_DIR = 'data/release/agri_foundation_v1/data'
DEFAULT_MODEL_DIR = 'models/foundational'

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Agricultural Vision Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth model file')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Path to dataset (for class names and validation data)')
    parser.add_argument('--model_name', type=str, default='mobilenetv4_conv_small.e2400_r224_in1k', help='Timm model name (must match training)')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save reports')
    return parser.parse_args()

def evaluate():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup Data
    print(f"Loading data from: {args.data_dir}")
    # We need the class names from the folder structure
    # Standard ImageFolder to get classes
    full_dataset = ImageFolder(args.data_dir)
    class_names = full_dataset.classes
    print(f"Classes: {len(class_names)}")

    # Transforms (Val only)
    val_transforms = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.14)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # For evaluation, we ideally want a held-out test set. 
    # If you used random split in training, we can't easily perfectly reconstruct the *exact* validation set 
    # without the indices.
    # However, for a general performance check on the whole dataset (or a subset), we can just run on the full folder 
    # or try to re-simulate the split if the seed was fixed.
    # Since we fixed seed=42 in training, we can recreate the split!
    
    print("Recreating validation split (Seed=42)...")
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Apply transform
    val_dataset.dataset.transform = val_transforms
    
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Load Model
    print(f"Loading model: {args.model_name}")
    try:
        model = timm.create_model(args.model_name, pretrained=False, num_classes=len(class_names))
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Inference
    print("Running inference...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Metrics
    print("Generating report...")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Save CSV
    csv_path = os.path.join(args.output_dir, 'classification_report.csv')
    df_report.to_csv(csv_path)
    print(f"Report saved to {csv_path}")

    # 5. Ghost Class Analysis
    print("\n--- Ghost Class Analysis (<50 samples) ---")
    # Identify rare classes from the SUPPORT column in the report (which reflects the val set size)
    # Note: Val set is ~20% of total. So if total was 50, val support is ~10.
    
    # Let's check the global counts to be sure
    # We can get total counts from the full_dataset targets
    from collections import Counter
    total_counts = Counter(full_dataset.targets)
    
    ghost_stats = []
    for idx, name in enumerate(class_names):
        total_count = total_counts[idx]
        if total_count < 50:
            # Get metrics from report
            if name in report:
                metrics = report[name]
                ghost_stats.append({
                    'Class': name,
                    'Total_Images': total_count,
                    'Val_Support': metrics['support'],
                    'Precision': f"{metrics['precision']:.2f}",
                    'Recall': f"{metrics['recall']:.2f}",
                    'F1-Score': f"{metrics['f1-score']:.2f}"
                })
    
    if ghost_stats:
        df_ghost = pd.DataFrame(ghost_stats)
        print(df_ghost.to_string(index=False))
        df_ghost.to_csv(os.path.join(args.output_dir, 'ghost_class_analysis.csv'), index=False)
    else:
        print("No ghost classes found or none in validation set.")

    # 6. Global Accuracy
    acc = report['accuracy']
    print(f"\nGlobal Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
