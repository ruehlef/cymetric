"""
PyTorch implementation of Cymetric

This module provides PyTorch-based models for Calabi-Yau metric learning.

Usage:
    from cymetric.torch.models import MultFSModel, PhiFSModel
    from cymetric.torch.models.torchhelper import prepare_basis, train_model
    from cymetric.torch.models.callbacks import SigmaCallback, KaehlerCallback

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Install with: pip install cymetric[torch]"
    )

# Import all model classes
from .models.torchmodels import (
    PhiFSModel,
    MultFSModel, 
    FreeModel,
    MatrixFSModel,
    AddFSModel,
    PhiFSModelToric,
    MatrixFSModelToric
)

# Import helper functions
from .models.torchhelper import (
    prepare_basis,
    train_model,
    EarlyStopping
)

# Import callbacks
from .models.callbacks import (
    SigmaCallback,
    KaehlerCallback, 
    TransitionCallback,
    RicciCallback,
    VolkCallback,
    AlphaCallback
)

# Import metrics
from .models.metrics import (
    SigmaLoss,
    KaehlerLoss,
    TransitionLoss,
    RicciLoss,
    VolkLoss
)

# Expose commonly used classes
__all__ = [
    # Models
    'PhiFSModel',
    'MultFSModel',
    'FreeModel', 
    'MatrixFSModel',
    'AddFSModel',
    'PhiFSModelToric',
    'MatrixFSModelToric',
    # Helpers
    'prepare_basis',
    'train_model',
    # Callbacks
    'SigmaCallback',
    'KaehlerCallback',
    'TransitionCallback', 
    'RicciCallback',
    'VolkCallback',
    'AlphaCallback',
    # Metrics
    'SigmaLoss',
    'KaehlerLoss',
    'TransitionLoss',
    'RicciLoss',
    'VolkLoss'
]
