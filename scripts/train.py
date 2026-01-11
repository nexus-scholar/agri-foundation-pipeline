import os
import argparse
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt

# Rich TUI Imports
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich import box
from rich.logging import RichHandler
import logging

# Set non-interactive backend for matplotlib
plt.switch_backend('Agg')

# Setup Console
console = Console()

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description="Train MobileNet Foundational Models")
    parser.add_argument('--data_dir', type=str, default='../data/release', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='../models/foundational', help='Directory to save models')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['mobilenetv4_conv_medium.e500_r256_in1k', 'mobilenetv5_base.e500_r256_in1k'],
                        help='List of timm model names to train')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()

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

def get_dataloaders(data_dir, batch_size, num_workers, image_size, console_log):
    data_transforms = get_transforms(image_size)
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    class_names = []

    console_log(f"Loading data from: [cyan]{data_dir}[/cyan]")
    
    for x in ['train', 'val']:
        path = os.path.join(data_dir, x)
        if not os.path.exists(path):
            console_log(f"[yellow]WARN[/yellow] {x} directory not found at {path}. Attempting fallback split...")
            try:
                full_dataset = ImageFolder(data_dir, transform=data_transforms['train'])
                train_size = int(0.8 * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
                
                # Hack for transforms
                console_log("[dim]Using random split. Validation set will use training transforms.[/dim]")
                
                image_datasets = {'train': train_dataset, 'val': val_dataset}
                class_names = full_dataset.classes
                break
            except Exception as e:
                console_log(f"[bold red]ERROR[/bold red] Failed to load dataset: {e}")
                exit(1)
        else:
            image_datasets[x] = ImageFolder(path, data_transforms[x])
            if x == 'train':
                class_names = image_datasets[x].classes

    for x in ['train', 'val']:
        dataloaders[x] = DataLoader(image_datasets[x], batch_size=batch_size,
                                     shuffle=True if x == 'train' else False, 
                                     num_workers=num_workers)
        dataset_sizes[x] = len(image_datasets[x])

    console_log(f"Data loaded: [green]{dataset_sizes['train']} train[/green], [blue]{dataset_sizes.get('val', 0)} val[/blue] images.")
    console_log(f"Classes ({len(class_names)}): {', '.join(class_names[:3])}...")
    
    return dataloaders, dataset_sizes, class_names

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
    table.add_row("Train Loss", f"{train_loss:.4f}" if train_loss is not None else "-")
    table.add_row("Train Acc", f"{train_acc:.2%}" if train_acc is not None else "-")
    table.add_row("Val Loss", f"{val_loss:.4f}" if val_loss is not None else "-")
    table.add_row("Val Acc", f"{val_acc:.2%}" if val_acc is not None else "-")
    table.add_row("Best Val Acc", f"[bold green]{best_acc:.2%}[/bold green]")
    
    return Panel(table, title="Current Metrics", border_style="blue")

def train_one_model(model_name, args, dataloaders, dataset_sizes, class_names, device, layout, progress_group):
    # Unpack progress components
    overall_progress, epoch_progress, step_progress = progress_group
    
    # Logs buffer
    logs = []
    def log(message):
        logs.append(message)
        if len(logs) > 8: logs.pop(0)
        layout["footer"].update(Panel("\n".join(logs), title="Logs", border_style="grey50"))

    log(f"Initializing {model_name}...")
    
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
        model = model.to(device)
    except Exception as e:
        log(f"[red]Error creating model {model_name}: {e}[/red]")
        return None, None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Add task for this model
    model_task = overall_progress.add_task(f"[bold]{model_name}", total=args.epochs)
    
    for epoch in range(args.epochs):
        layout["header"].update(Panel(f"Training: [bold magenta]{model_name}[/bold magenta] | Epoch {epoch+1}/{args.epochs}", style="white on black"))
        
        current_metrics = {'train_loss': None, 'train_acc': None, 'val_loss': None, 'val_acc': None}
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # Epoch progress
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
                
                # Update metrics live (approximate)
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                
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
                    
                    clean_name = model_name.split('.')[0]
                    save_path = os.path.join(args.output_dir, f"{clean_name}_best.pth")
                    torch.save(model.state_dict(), save_path)
                    log(f"[green]New Best Model![/green] Acc: {epoch_acc:.4f}")
                else:
                    log(f"Epoch {epoch+1} Val Acc: {epoch_acc:.4f}")
        
        # Update metrics table
        layout["metrics"].update(create_metrics_table(
            epoch + 1, 
            current_metrics['train_loss'], current_metrics['train_acc'],
            current_metrics['val_loss'], current_metrics['val_acc'],
            best_acc
        ))
        
        overall_progress.advance(model_task)

    time_elapsed = time.time() - since
    log(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, model_name, output_dir):
    clean_name = model_name.split('.')[0]
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{clean_name} Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title(f'{clean_name} Accuracy')
    plt.legend()
    
    save_path = os.path.join(output_dir, f"{clean_name}_history.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Layout Setup
    layout = make_layout()
    layout["header"].update(Panel("Dataset Processing & Training Pipeline", style="bold white on blue"))
    layout["footer"].update(Panel("Initializing...", title="Logs"))
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Progress Bars
    overall_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    epoch_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    
    progress_panel = Panel(
        Layout(name="progress_layout"), 
        title="Progress", border_style="green"
    )
    # We construct a layout inside the panel to hold the two progress bars vertically
    p_layout = Layout()
    p_layout.split_column(
        Layout(overall_progress),
        Layout(epoch_progress)
    )
    layout["progress"].update(Panel(p_layout, title="Progress", border_style="green"))

    with Live(layout, refresh_per_second=4, screen=True):
        dataloaders, dataset_sizes, class_names = get_dataloaders(args.data_dir, args.batch_size, args.num_workers, args.image_size, lambda x: None)
        
        # Pass progress group
        progress_group = (overall_progress, epoch_progress, None)
        
        for model_name in args.models:
            model, history = train_one_model(model_name, args, dataloaders, dataset_sizes, class_names, device, layout, progress_group)
            
            if model and history:
                clean_name = model_name.split('.')[0]
                csv_path = os.path.join(args.output_dir, f"{clean_name}_history.csv")
                pd.DataFrame(history).to_csv(csv_path, index=False)
                plot_history(history, model_name, args.output_dir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user.")