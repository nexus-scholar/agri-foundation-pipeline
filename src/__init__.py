"""
PhD Active Adaptation - Source Package

A modular framework for plant disease classification with active learning
and domain adaptation.

Modules:
- config: Configuration and hyperparameters
- data: Data loading and transforms
- models: Neural network architectures
- strategies: Active learning and augmentation strategies
- utils: Metrics and utilities

Note: Submodules are imported lazily to avoid circular imports.
Import directly from submodules for specific functionality:
    from src.data import load_data_modules
    from src.models import get_model
"""

__version__ = "1.0.0"

# Define what's available - actual imports happen when accessed
__all__ = [
    'config',
    'data',
    'models',
    'strategies',
    'utils',
]


def __getattr__(name):
    """Lazy import of submodules to avoid circular imports."""
    if name == 'config':
        from . import config
        return config
    elif name == 'data':
        from . import data
        return data
    elif name == 'models':
        from . import models
        return models
    elif name == 'strategies':
        from . import strategies
        return strategies
    elif name == 'utils':
        from . import utils
        return utils
    raise AttributeError(f"module 'src' has no attribute '{name}'")

