import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
import pandas as pd
import numpy as np
import time
import copy
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set non-interactive backend for matplotlib
plt.switch_backend('Agg')

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

def get_dataloaders(data_dir, batch_size, num_workers, image_size):
    data_transforms = get_transforms(image_size)
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    class_names = []

    print(f"Loading data from: {data_dir}")
    
    for x in ['train', 'val']:
        path = os.path.join(data_dir, x)
        if not os.path.exists(path):
            print(f"  [WARN] {x} directory not found at {path}. Attempting fallback split...")
            try:
                full_dataset = ImageFolder(data_dir, transform=data_transforms['train'])
                train_size = int(0.8 * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
                
                # Hack to update transform for validation
                # Note: random_split doesn't deep copy the dataset, so this is tricky. 
                # For a robust script, we'd wrap this. For now, we accept train transforms on val 
                # OR we re-instantiate ImageFolder twice with different transforms if we knew the file list.
                # Let's simple use the fallback:
                print("  [INFO] Using random split. Validation set will use training transforms (suboptimal but functional).")
                
                image_datasets = {'train': train_dataset, 'val': val_dataset}
                class_names = full_dataset.classes
                break
            except Exception as e:
                print(f"  [ERROR] Failed to load dataset: {e}")
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

    print(f"  Classes ({len(class_names)}): {class_names[:5]}...")
    print(f"  Train size: {dataset_sizes['train']}")
    print(f"  Val size:   {dataset_sizes.get('val', 0)}")
    
    return dataloaders, dataset_sizes, class_names

def train_one_model(model_name, args, dataloaders, dataset_sizes, class_names, device):
    print(f"\n{'-'*60}")
    print(f"Initializing {model_name}...")
    print(f"{'-'*60}")

    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
        model = model.to(device)
    except Exception as e:
        print(f"[ERROR] Failed to create model {model_name}: {e}")
        return None, None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Start Training: {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Progress bar
            pbar = tqdm(dataloaders[phase], desc=f"  {phase.upper()}", leave=True, unit="batch")
            
            for inputs, labels in pbar:
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
                
                # Update progress bar description
                pbar.set_postfix(loss=loss.item())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save checkpoint
                clean_name = model_name.split('.')[0]
                save_path = os.path.join(args.output_dir, f"{clean_name}_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  [SAVED] New best model: {epoch_acc:.4f} -> {save_path}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, model_name, output_dir):
    clean_name = model_name.split('.')[0]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{clean_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{clean_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, f"{clean_name}_history.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"History plot saved to {save_path}")
    plt.close()

def main():
    args = get_args()
    seed_everything(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    dataloaders, dataset_sizes, class_names = get_dataloaders(args.data_dir, args.batch_size, args.num_workers, args.image_size)
    
    for model_name in args.models:
        model, history = train_one_model(model_name, args, dataloaders, dataset_sizes, class_names, device)
        
        if model and history:
            clean_name = model_name.split('.')[0]
            # Save history CSV
            csv_path = os.path.join(args.output_dir, f"{clean_name}_history.csv")
            pd.DataFrame(history).to_csv(csv_path, index=False)
            print(f"History data saved to {csv_path}")
            
            # Plot
            plot_history(history, model_name, args.output_dir)

if __name__ == "__main__":
    main()
