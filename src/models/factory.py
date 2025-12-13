"""
Unified model factory for PDA experiments.

Supports multiple edge-deployable architectures:
- MobileNetV3-Small (default, fastest)
- EfficientNet-B0 (balanced accuracy/speed)
- MobileViT-XS (transformer-based, highest accuracy)

All models are compatible with ImageNet pretrained weights and can be
fine-tuned for plant disease classification.
"""
from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

# Optional timm import for EfficientNet and MobileViT
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# Supported architectures
SUPPORTED_MODELS = ['mobilenetv3', 'efficientnet', 'mobilevit']


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Create a model by name with the specified number of output classes.

    This is the main entry point for model creation in experiments.

    Args:
        model_name: Architecture name ('mobilenetv3', 'efficientnet', 'mobilevit').
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        device: Target device (default: auto-detect CUDA/CPU).

    Returns:
        Model ready for training or inference.

    Raises:
        ValueError: If model_name is not supported.
        ImportError: If timm is required but not installed.
    """
    model_name = model_name.lower().strip()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    if model_name == 'mobilenetv3':
        model = _create_mobilenetv3(num_classes, pretrained)

    elif model_name == 'efficientnet':
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for EfficientNet. "
                "Install with: pip install timm"
            )
        model = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif model_name == 'mobilevit':
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for MobileViT. "
                "Install with: pip install timm"
            )
        model = timm.create_model(
            'mobilevit_xs',
            pretrained=pretrained,
            num_classes=num_classes,
        )

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    return model.to(device)


# Alias for backward compatibility with agri_refactor
get_edge_model = get_model


def _create_mobilenetv3(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV3-Small with custom classifier head."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def load_model(
    path: Union[str, Path],
    model_name: str,
    num_classes: int,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Load a model from a checkpoint file.

    Args:
        path: Path to the .pth checkpoint file.
        model_name: Architecture name to instantiate.
        num_classes: Number of output classes.
        device: Target device.

    Returns:
        Loaded model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = get_model(model_name, num_classes, pretrained=False, device='cpu')
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model.to(device)


def save_model(
    model: nn.Module,
    path: Union[str, Path],
    metadata: Optional[dict] = None,
) -> None:
    """
    Save model checkpoint with optional metadata.

    Args:
        model: Model to save.
        path: Destination path for .pth file.
        metadata: Optional dict to save as .json sidecar.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

    if metadata:
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze all layers except the classifier head.

    Works with MobileNetV3 (classifier), EfficientNet (classifier),
    and MobileViT (head).
    """
    classifier_keywords = ['classifier', 'head', 'fc']

    for name, param in model.named_parameters():
        is_classifier = any(kw in name.lower() for kw in classifier_keywords)
        param.requires_grad = is_classifier

    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_num_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_info(model_name: str) -> dict:
    """
    Get metadata about a model architecture.

    Args:
        model_name: Architecture name.

    Returns:
        Dictionary with model info (params, gflops estimate, etc.).
    """
    info = {
        'mobilenetv3': {
            'name': 'MobileNetV3-Small',
            'params_millions': 2.5,
            'gflops': 0.06,
            'input_size': 224,
            'notes': 'Fastest, good for edge deployment',
        },
        'efficientnet': {
            'name': 'EfficientNet-B0',
            'params_millions': 5.3,
            'gflops': 0.39,
            'input_size': 224,
            'notes': 'Balanced accuracy and speed',
        },
        'mobilevit': {
            'name': 'MobileViT-XS',
            'params_millions': 2.3,
            'gflops': 0.7,
            'input_size': 256,
            'notes': 'Transformer-based, highest accuracy',
        },
    }

    model_name = model_name.lower().strip()
    if model_name not in info:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(info.keys())}")

    return info[model_name]

