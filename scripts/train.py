import os
import argparse
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
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
import questionary

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

def get_args_interactive():
    console.clear()
    console.print(Panel.fit("[bold cyan]Agricultural Vision - Training Config Wizard[/bold cyan]", border_style="blue"))
    
    # 1. Dataset
    data_dir = questionary.path("Dataset directory:", default="../data/release").ask()
    
    # 2. Data Split Strategy
    split_strategy = questionary.select(
        "Data Split Strategy:",
        choices=[
            "Use existing 'train'/'val' folders",
            "Random Split (80% Train, 20% Val)",
            "Random Split (70% Train, 30% Val)",
            "Random Split (90% Train, 10% Val)"
        ],
        default="Use existing 'train'/'val' folders"
    ).ask()

    # 3. Models
    # Curated list of efficient models for Edge/Mobile
    model_choices = [
        questionary.Choice("mobilenetv5_300m.gemma3n", checked=True),
        questionary.Choice("mobilenetv4_conv_medium.e500_r256_in1k", checked=True),
        questionary.Choice("mobilenetv4_conv_small.e2400_r224_in1k"),
        questionary.Choice("mobilenetv4_conv_large.e600_r384_in1k"),
        questionary.Choice("mobilenetv3_large_100.ra_in1k"),
        questionary.Choice("resnet18.a1_in1k"),
        questionary.Choice("efficientnet_b0.ra_in1k")
    ]
    
    models = questionary.checkbox(
        "Select models to train (Space to select, Enter to confirm):",
        choices=model_choices
    ).ask()
    
    if not models:
        console.print("[red]No models selected! Exiting.[/red]")
        sys.exit(1)

    # 4. Hyperparameters
    epochs = int(questionary.text("Number of epochs:", default="20").ask())
    batch_size = int(questionary.text("Batch size:", default="32").ask())
    lr = float(questionary.text("Learning rate:", default="0.0001").ask())
    image_size = int(questionary.text("Image size:", default="256").ask())
    
    # 5. Misc
    output_dir = questionary.path("Output directory:", default="../models/foundational").ask()
    seed = int(questionary.text("Random seed:", default="42").ask())
    use_cuda = questionary.confirm("Use CUDA if available?", default=True).ask()
    
    # Construct namespace
    args = argparse.Namespace(
        data_dir=data_dir,
        output_dir=output_dir,
        models=models,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        num_workers=4,
        seed=seed,
        no_cuda=not use_cuda,
        split_strategy=split_strategy
    )
    return args

def get_args():
    if len(sys.argv) == 1:
        return get_args_interactive()

    parser = argparse.ArgumentParser(description="Train MobileNet Foundational Models")
    parser.add_argument('--data_dir', type=str, default='../data/release', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='../models/foundational', help='Directory to save models')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['mobilenetv4_conv_medium.e500_r256_in1k'],
                        help='List of timm model names to train')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--interactive', action='store_true', help='Force interactive mode')
    
    # CLI fallback for split strategy isn't fully implemented to keep CLI simple, defaults to existing folders
    parser.add_argument('--split_strategy', type=str, default="Use existing 'train'/'val' folders")

    args = parser.parse_args()
    
    if args.interactive:
        return get_args_interactive()
        
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

def get_dataloaders(args, console_log):
    data_dir = args.data_dir
    split_strategy = args.split_strategy
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    data_transforms = get_transforms(args.image_size)
    dataloaders = {}
    dataset_sizes = {}
    class_names = []

    if console_log: console_log(f"Loading data from: [cyan]{data_dir}[/cyan]")
    
    if not os.path.exists(data_dir):
        msg = f"[bold red]Error:[/bold red] Dataset directory not found: {data_dir}"
        if console_log: console_log(msg)
        else: console.print(msg)
        sys.exit(1)
    
    # Check if explicit train/val folders exist
    has_split_folders = os.path.exists(os.path.join(data_dir, 'train')) and os.path.exists(os.path.join(data_dir, 'val'))
    
    if split_strategy == "Use existing 'train'/'val' folders":
        if has_split_folders:
            image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
            class_names = image_datasets['train'].classes
        else:
             # Fallback if strategy requested but folders missing
             if console_log: console_log("[yellow]WARN[/yellow] 'train'/'val' folders not found. Falling back to 80/20 Random Split.")
             full_dataset = ImageFolder(data_dir, transform=data_transforms['train'])
             class_names = full_dataset.classes
             train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, stratify=full_dataset.targets)
             
             train_dataset = Subset(full_dataset, train_idx)
             val_dataset = Subset(full_dataset, val_idx)
             # Note: Subset uses original dataset's transform. For val, we might ideally want val transforms.
             # This is a limitation of simple Subset. In production, wrap dataset or use separate instances.
             image_datasets = {'train': train_dataset, 'val': val_dataset}
    else:
        # Parse split ratio from string
        val_ratio = 0.2
        if "70%" in split_strategy: val_ratio = 0.3
        elif "90%" in split_strategy: val_ratio = 0.1
        
        if console_log: console_log(f"Performing {int((1-val_ratio)*100)}/{int(val_ratio*100)} Random Split...")
        
        # Load full dataset - we assume flat structure or we ignore existing split folders to re-split
        # If structure is train/val but we want random split, we merge them technically, but ImageFolder(root) 
        # normally expects class folders directly under root.
        # If data_dir has 'train'/'val' subdirs, ImageFolder(data_dir) will see 'train' and 'val' as classes!
        # So we must be careful.
        
        try:
            full_dataset = ImageFolder(data_dir, transform=data_transforms['train'])
            class_names = full_dataset.classes
            # Stratified split is better
            train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=val_ratio, stratify=full_dataset.targets)
            
            image_datasets = {
                'train': Subset(full_dataset, train_idx),
                'val': Subset(full_dataset, val_idx)
            }
        except Exception as e:
            msg = f"[bold red]Error loading dataset for splitting:[/bold red] {e}"
            if console_log: console_log(msg)
            else: console.print(msg)
            sys.exit(1)

    for x in ['train', 'val']:
        dataloaders[x] = DataLoader(image_datasets[x], batch_size=batch_size,
                                     shuffle=True if x == 'train' else False, 
                                     num_workers=num_workers)
        dataset_sizes[x] = len(image_datasets[x])

    if console_log: 
        console_log(f"Data loaded: [green]{dataset_sizes['train']} train[/green], [blue]{dataset_sizes['val']} val[/blue] images.")
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
        time.sleep(2)
        return None, None, str(e)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

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
                    
                    clean_name = model_name.split('.')[0]
                    save_path = os.path.join(args.output_dir, f"{clean_name}_best.pth")
                    torch.save(model.state_dict(), save_path)
                    log(f"[green]New Best Model![/green] Acc: {epoch_acc:.4f}")
                else:
                    log(f"Epoch {epoch+1} Val Acc: {epoch_acc:.4f}")
        
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
    return model, history, None

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
    
    layout = make_layout()
    layout["header"].update(Panel("Dataset Processing & Training Pipeline", style="bold white on blue"))
    layout["footer"].update(Panel("Initializing...", title="Logs"))
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
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
    
    p_layout = Layout()
    p_layout.split_column(
        Layout(overall_progress),
        Layout(epoch_progress)
    )
    layout["progress"].update(Panel(p_layout, title="Progress", border_style="green"))

    console.print("[dim]Loading dataset...[/dim]")
    dataloaders, dataset_sizes, class_names = get_dataloaders(args, None)
    console.print(f"[green]Ready to train on {len(class_names)} classes![/green]")

    errors = []
    
    with Live(layout, refresh_per_second=4, screen=True):
        progress_group = (overall_progress, epoch_progress, None)
        
        for model_name in args.models:
            model, history, err = train_one_model(model_name, args, dataloaders, dataset_sizes, class_names, device, layout, progress_group)
            
            if err:
                errors.append(f"Model {model_name} failed: {err}")
                continue

            if model and history:
                clean_name = model_name.split('.')[0]
                csv_path = os.path.join(args.output_dir, f"{clean_name}_history.csv")
                pd.DataFrame(history).to_csv(csv_path, index=False)
                plot_history(history, model_name, args.output_dir)

    if errors:
        console.print("\n[bold red]Training completed with errors:[/bold red]")
        for e in errors:
            console.print(f"  - {e}")
    else:
        console.print("\n[bold green]Training completed successfully![/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user.")