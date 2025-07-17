"""
Cymetric: A package for Calabi-Yau metric learning

This package provides both PyTorch and TensorFlow implementations for 
learning Calabi-Yau metrics using neural networks.

Usage:
    # Import PyTorch implementation
    import cymetric.torch as cymetric_torch
    
    # Import TensorFlow implementation  
    import cymetric.tensorflow as cymetric_tf
    
    # Import shared utilities
    from cymetric.pointgen import PointGeneratorMathematica
    from cymetric.sage import sagelib
    from cymetric.wolfram import mathematicalib

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

__version__ = "1.0.0"
__author__ = "Fabian Ruehle"
__email__ = "f.ruehle@northeastern.edu"

# Framework availability checks
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

def check_torch():
    """Check if PyTorch is available."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Install it with: pip install cymetric[torch]"
        )

def check_tensorflow():
    """Check if TensorFlow is available."""
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. Install it with: pip install cymetric[tensorflow]"
        )

# Expose framework availability
TORCH_AVAILABLE = _TORCH_AVAILABLE
TENSORFLOW_AVAILABLE = _TF_AVAILABLE
