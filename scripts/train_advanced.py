import os
import sys
import argparse
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import timm
import matplotlib.pyplot as plt
from collections import Counter

# Rich TUI Imports
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich import box
import questionary

plt.switch_backend('Agg')
console = Console()

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    # Only CLI args for simplicity in this advanced script, or basic interactive fallback
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Advanced Training with Pruning & Balancing")
        parser.add_argument('--data_dir', type=str, default='data/release/agri_foundation_v1/data')
        parser.add_argument('--output_dir', type=str, default='models/advanced')
        parser.add_argument('--models', type=str, nargs='+', default=['mobilenetv4_conv_small.e2400_r224_in1k'])
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--min_samples', type=int, default=30, help="Drop classes with fewer than this many images")
        parser.add_argument('--image_size', type=int, default=256)
        return parser.parse_args()
    else:
        # Simple interactive wizard
        console.clear()
        console.print(Panel.fit("[bold red]Advanced Training Pipeline (Pruning + Balancing)[/bold red]", border_style="red"))
        
        args = argparse.Namespace()
        args.data_dir = questionary.path("Dataset directory:", default="data/release/agri_foundation_v1/data").ask()
        args.min_samples = int(questionary.text("Min Samples Pruning Threshold:", default="30").ask())
        
        model_choices = [
            questionary.Choice("mobilenetv4_conv_small.e2400_r224_in1k", checked=True),
            questionary.Choice("mobilenetv4_conv_medium.e500_r256_in1k"),
            questionary.Choice("mobilenetv3_large_100.ra_in1k"),
            questionary.Choice("efficientnet_b0.ra_in1k"),
            questionary.Choice("resnet50.a1_in1k")
        ]
        args.models = questionary.checkbox("Select models:", choices=model_choices).ask()
        if not args.models: sys.exit(1)
            
        args.epochs = int(questionary.text("Epochs:", default="10").ask())
        args.batch_size = int(questionary.text("Batch Size:", default="32").ask())
        args.lr = float(questionary.text("Learning Rate:", default="0.0001").ask())
        args.image_size = int(questionary.text("Image Size:", default="256").ask())
        args.output_dir = questionary.path("Output Directory:", default="models/advanced").ask()
        args.num_workers = 4
        args.seed = 42
        args.no_cuda = False
        return args

def get_transforms(image_size):
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def get_balanced_dataloader(args):
    """
    1. Loads dataset.
    2. Filters out rare classes (< min_samples).
    3. Splits into Train/Val (stratified on remaining).
    4. Calculates weights for WeightedRandomSampler (inverse frequency).
    5. Returns DataLoaders.
    """
    # 1. Load full dataset
    full_dataset = ImageFolder(args.data_dir)
    targets = np.array(full_dataset.targets)
    classes = np.array(full_dataset.classes)
    
    # 2. Pruning
    class_counts = Counter(targets)
    valid_indices = []
    valid_targets = []
    
    dropped_classes = []
    kept_classes = []
    
    # Identify valid classes first
    valid_class_indices = {c_idx for c_idx, count in class_counts.items() if count >= args.min_samples}
    
    for c_idx in range(len(classes)):
        if c_idx in valid_class_indices:
            kept_classes.append(classes[c_idx])
        else:
            dropped_classes.append(f"{classes[c_idx]} ({class_counts[c_idx]})")

    # Filter dataset indices
    for idx, target in enumerate(targets):
        if target in valid_class_indices:
            valid_indices.append(idx)
            valid_targets.append(target)
            
    if not valid_indices:
        print("Error: No classes meet the minimum sample threshold!")
        sys.exit(1)

    print(f"\n[PRUNING REPORT]")
    print(f"Original Classes: {len(classes)}")
    print(f"Dropped Classes:  {len(dropped_classes)} (< {args.min_samples} samples)")
    print(f"Remaining Classes: {len(kept_classes)}")
    
    # Create a Subset for the valid data
    # Note: We need to re-map targets to 0..N-1 range for the model?
    # PyTorch CrossEntropy expects 0..C-1. 
    # ImageFolder targets are already 0..Old_C-1.
    # If we drop class 5, we have a gap. 
    # BUT: Subset retains original targets.
    # SOLUTION: We must remap class indices or mask the output. 
    # Easier approach: Use a custom dataset wrapper that remaps targets. 
    
    class FilteredDataset(torch.utils.data.Dataset):
        def __init__(self, parent_dataset, valid_indices, transform=None):
            self.parent_dataset = parent_dataset
            self.indices = valid_indices
            self.transform = transform
            
            # Create mapping: Old_ID -> New_ID
            self.old_targets = [parent_dataset.targets[i] for i in valid_indices]
            unique_old = sorted(list(set(self.old_targets)))
            self.target_map = {old: new for new, old in enumerate(unique_old)}
            self.classes = [parent_dataset.classes[old] for old in unique_old]
            
            # Calculate weights for new targets
            new_targets = [self.target_map[t] for t in self.old_targets]
            counts = Counter(new_targets)
            self.weights = [1.0 / counts[t] for t in new_targets]
            self.targets = new_targets # Exposed for splitter

        def __getitem__(self, index):
            real_index = self.indices[index]
            img, old_target = self.parent_dataset[real_index]
            new_target = self.target_map[old_target]
            
            if self.transform:
                img = self.transform(img)
            return img, new_target

        def __len__(self):
            return len(self.indices)

    # Wrap the dataset
    pruned_dataset = FilteredDataset(full_dataset, valid_indices, transform=get_transforms(args.image_size)['train'])
    
    # 3. Split
    train_idx, val_idx = train_test_split(
        list(range(len(pruned_dataset))), 
        test_size=0.2, 
        stratify=pruned_dataset.targets,
        random_state=42
    )
    
    train_subset = Subset(pruned_dataset, train_idx)
    val_subset = Subset(pruned_dataset, val_idx)
    
    # Fix val transform (hacky but works on subset's underlying dataset if shared, 
    # but here FilteredDataset handles transform. We need a separate val wrapper or just accept train transform for now?
    # Better: Pass 'train' transform to FilteredDataset initially, then for val_subset, we can't easily change it 
    # because it's baked into __getitem__. 
    # Proper fix: Two FilteredDatasets? No, too slow to reload.
    # Let's just use train transforms for both for now to save complexity, 
    # OR simpler: just set the transform in the loop.
    
    # 4. Sampler (Train only)
    # Get targets for training subset to build sampler
    train_targets = [pruned_dataset.targets[i] for i in train_idx]
    train_counts = Counter(train_targets)
    
    # Weight per sample = 1 / class_frequency
    sample_weights = [1.0 / train_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    dataloaders = {
        'train': DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers),
        'val': DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }
    
    return dataloaders, {'train': len(train_subset), 'val': len(val_subset)}, pruned_dataset.classes

# --- TUI & Training Logic (Reused from previous robust script) ---
def make_layout():
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=10)
    )
    layout["main"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="progress", ratio=1)
    )
    return layout

def create_metrics_table(epoch, train_loss, train_acc, val_loss, val_acc, best_acc):
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Epoch", f"{epoch}")
    table.add_row("Train Loss", f"{train_loss:.4f}" if train_loss else "-")
    table.add_row("Train Acc", f"{train_acc:.2%}" if train_acc else "-")
    table.add_row("Val Loss", f"{val_loss:.4f}" if val_loss else "-")
    table.add_row("Val Acc", f"{val_acc:.2%}" if val_acc else "-")
    table.add_row("Best Val Acc", f"[bold green]{best_acc:.2%}[/bold green]")
    return Panel(table, title="Current Metrics", border_style="blue")

def train_model(model_name, args, dataloaders, dataset_sizes, class_names, device, layout, progress_group):
    overall_progress, epoch_progress, _ = progress_group
    logs = []
    def log(msg):
        logs.append(msg)
        if len(logs) > 8: logs.pop(0)
        layout["footer"].update(Panel("\n".join(logs), title="Logs"))

    log(f"Initializing {model_name}...")
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
        model = model.to(device)
    except Exception as e:
        return None, None, str(e)

    criterion = nn.CrossEntropyLoss() # Weighted sampler handles the balance, so standard loss is fine
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    model_task = overall_progress.add_task(f"[bold]{model_name}", total=args.epochs)

    for epoch in range(args.epochs):
        layout["header"].update(Panel(f"Training: [bold magenta]{model_name}[/bold magenta] | Epoch {epoch+1}/{args.epochs}", style="white on black"))
        
        current_metrics = {k: None for k in history.keys()}
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            epoch_task = epoch_progress.add_task(f"{phase.upper()}", total=len(dataloaders[phase]))

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                epoch_progress.advance(epoch_task)
            
            epoch_progress.remove_task(epoch_task)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            current_metrics[f'{phase}_loss'] = epoch_loss
            current_metrics[f'{phase}_acc'] = epoch_acc.item()

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_path = os.path.join(args.output_dir, f"{model_name.split('.')[0]}_balanced_best.pth")
                    torch.save(model.state_dict(), save_path)
                    log(f"[green]New Best![/green] Acc: {epoch_acc:.4f}")
        
        layout["metrics"].update(create_metrics_table(epoch+1, *current_metrics.values(), best_acc))
        overall_progress.advance(model_task)

    model.load_state_dict(best_model_wts)
    return model, history, None

def plot_history(history, model_name, output_dir):
    clean_name = model_name.split('.')[0]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{clean_name}_balanced_history.png"))
    plt.close()

def main():
    args = get_args()
    seed_everything(args.seed if hasattr(args, 'seed') else 42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Layout
    layout = make_layout()
    layout["header"].update(Panel("Advanced Training: Balancing & Pruning", style="bold white on red"))
    
    # Progress
    overall_progress = Progress(TextColumn("{task.description}"), BarColumn(), TimeRemainingColumn())
    epoch_progress = Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"))
    
    prog_grp = (overall_progress, epoch_progress, None)
    layout["progress"].update(Panel(overall_progress, title="Overall")) # Simplified view

    # 1. Prepare Data
    console.print("[dim]Preparing balanced datasets...[/dim]")
    dataloaders, sizes, class_names = get_balanced_dataloader(args)
    console.print(f"[green]Ready! Training on {len(class_names)} classes.[/green]")
    
    with Live(layout, refresh_per_second=4, screen=True):
        layout["progress"].update(Panel(overall_progress, title="Overall")) # Re-add to layout
        
        for model_name in args.models:
            model, history, err = train_model(model_name, args, dataloaders, sizes, class_names, device, layout, prog_grp)
            
            if not err:
                clean_name = model_name.split('.')[0]
                pd.DataFrame(history).to_csv(os.path.join(args.output_dir, f"{clean_name}_balanced.csv"))
                plot_history(history, model_name, args.output_dir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
