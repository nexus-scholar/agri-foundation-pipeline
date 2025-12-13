"""
Models module for PhD Active Adaptation experiments.

Provides unified model factory and architecture utilities.
"""

from .factory import (
    get_model,
    get_edge_model,
    load_model,
    save_model,
    freeze_backbone,
    unfreeze_backbone,
    get_num_parameters,
    get_model_info,
    SUPPORTED_MODELS,
)

from .mobilenetv3 import (
    create_mobilenetv3,
    create_model,
)

__all__ = [
    # Factory (primary API)
    'get_model',
    'get_edge_model',
    'load_model',
    'save_model',
    'freeze_backbone',
    'unfreeze_backbone',
    'get_num_parameters',
    'get_model_info',
    'SUPPORTED_MODELS',
    # Legacy MobileNetV3
    'create_mobilenetv3',
    'create_model',
]

