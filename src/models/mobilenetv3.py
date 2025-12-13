"""
Model definitions for PhD Active Adaptation experiments.

Primary model: MobileNetV3-Small
- Efficient architecture suitable for embedded deployment
- Pre-trained on ImageNet
- Custom classifier head for plant disease classification
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from pathlib import Path


def create_mobilenetv3(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create MobileNetV3-Small model with custom classifier.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        MobileNetV3-Small model with modified classifier
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def load_model(path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    model = create_mobilenetv3(num_classes, pretrained=True)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device)


def save_model(model: nn.Module, path: Path, metadata: dict = None):
    """Save model with optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

    if metadata:
        import json
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze all layers except classifier."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_num_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# Alias for backward compatibility
create_model = create_mobilenetv3

