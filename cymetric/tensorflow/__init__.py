"""
TensorFlow implementation of Cymetric

This module provides TensorFlow-based models for Calabi-Yau metric learning.

Usage:
    from cymetric.tensorflow.models import MultFSModel, PhiFSModel
    from cymetric.tensorflow.models.tfhelper import prepare_basis, train_model
    from cymetric.tensorflow.models.callbacks import SigmaCallback, KaehlerCallback

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

# Check TensorFlow availability
try:
    import tensorflow as tf
    import tensorflow.keras as tfk
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    raise ImportError(
        "TensorFlow is not installed. Install with: pip install cymetric[tensorflow]"
    )

# Import all model classes
from .models.tfmodels import (
    PhiFSModel,
    MultFSModel,
    FreeModel,
    MatrixFSModel, 
    AddFSModel,
    PhiFSModelToric,
    MatrixFSModelToric
)

# Import helper functions
from .models.tfhelper import (
    prepare_basis,
    train_model
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
    VolkLoss,
    TotalLoss
)

# Expose TensorFlow objects
tf = tf
tfk = tfk

# Expose commonly used classes
__all__ = [
    # TensorFlow
    'tf',
    'tfk',
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
    'VolkLoss',
    'TotalLoss'
]
