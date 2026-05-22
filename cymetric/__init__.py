"""
Cymetric: A package for Calabi-Yau metric learning

This package provides PyTorch, TensorFlow, and JAX implementations for 
learning Calabi-Yau metrics using neural networks.

Usage:
    # Import framework-specific implementations directly
    import cymetric.torch as cymetric_torch
    import cymetric.tensorflow as cymetric_tf
    import cymetric.jax as cymetric_jax
    
    # Import shared utilities
    from cymetric.pointgen import PointGeneratorMathematica
    from cymetric.sage import sagelib
    from cymetric.wolfram import mathematicalib
    
    # Use compatibility layer (defaults to TensorFlow if available, then JAX, then PyTorch)
    from cymetric.models import measures, callbacks
    
    # Framework selection options:
    
    # Option 1: Environment variable (before importing cymetric)
    import os
    os.environ['CYMETRIC_FRAMEWORK'] = 'jax'  # or 'torch', 'tensorflow'
    from cymetric.models import measures  # will use JAX
    
    # Option 2: Runtime switching
    import cymetric
    cymetric.set_preferred_framework('jax')  # or 'torch', 'tensorflow'
    from cymetric.models import measures  # will use JAX
    
    # Option 3: Check current framework
    print(f"Using framework: {cymetric.PREFERRED_FRAMEWORK}")

Framework Selection:
    The cymetric.models.* compatibility layer automatically selects the 
    appropriate framework implementation:
    
    - Default: TensorFlow (if available), then JAX, then PyTorch
    - Override: Set CYMETRIC_FRAMEWORK environment variable
    - Runtime: Use cymetric.set_preferred_framework()

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

__version__ = "0.4.0"
__author__ = "Fabian Ruehle"
__email__ = "f.ruehle@northeastern.edu"

import os

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

try:
    import jax
    import equinox  # noqa: F401
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

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

def check_jax():
    """Check if JAX is available."""
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is not installed. Install it with: pip install cymetric[jax]"
        )

def get_preferred_framework():
    """Get the preferred framework based on environment variable and availability.
    
    Returns:
        str: 'tensorflow', 'torch', or 'jax' based on preference and availability
    """
    # Check for user preference
    preferred = os.environ.get('CYMETRIC_FRAMEWORK', '').lower()
    
    if preferred == 'torch' or preferred == 'pytorch':
        if _TORCH_AVAILABLE:
            return 'torch'
        elif _TF_AVAILABLE:
            import warnings
            warnings.warn("PyTorch requested but not available, falling back to TensorFlow")
            return 'tensorflow'
        elif _JAX_AVAILABLE:
            import warnings
            warnings.warn("PyTorch requested but not available, falling back to JAX")
            return 'jax'
        else:
            raise ImportError("PyTorch requested but no framework available")
    
    elif preferred == 'tf' or preferred == 'tensorflow':
        if _TF_AVAILABLE:
            return 'tensorflow'
        elif _TORCH_AVAILABLE:
            import warnings
            warnings.warn("TensorFlow requested but not available, falling back to PyTorch")
            return 'torch'
        elif _JAX_AVAILABLE:
            import warnings
            warnings.warn("TensorFlow requested but not available, falling back to JAX")
            return 'jax'
        else:
            raise ImportError("TensorFlow requested but no framework available")
    
    elif preferred == 'jax' or preferred == 'equinox':
        if _JAX_AVAILABLE:
            return 'jax'
        elif _TF_AVAILABLE:
            import warnings
            warnings.warn("JAX requested but not available, falling back to TensorFlow")
            return 'tensorflow'
        elif _TORCH_AVAILABLE:
            import warnings
            warnings.warn("JAX requested but not available, falling back to PyTorch")
            return 'torch'
        else:
            raise ImportError("JAX requested but no framework available")
    
    # Default behavior: prefer TensorFlow, then JAX, then PyTorch
    if _TF_AVAILABLE:
        return 'tensorflow'
    elif _JAX_AVAILABLE:
        return 'jax'
    elif _TORCH_AVAILABLE:
        return 'torch'
    else:
        raise ImportError("No ML framework (PyTorch, TensorFlow, or JAX) available")

def set_preferred_framework(framework):
    """Set the preferred framework for the compatibility layer.
    
    Args:
        framework (str): 'torch', 'pytorch', 'tf', 'tensorflow', 'jax', or 'equinox'
        
    Note:
        This sets the CYMETRIC_FRAMEWORK environment variable and will
        affect future imports from cymetric.models.*
    """
    valid_frameworks = ['torch', 'pytorch', 'tf', 'tensorflow', 'jax', 'equinox']
    if framework.lower() not in valid_frameworks:
        raise ValueError(f"Framework must be one of {valid_frameworks}")
    
    os.environ['CYMETRIC_FRAMEWORK'] = framework.lower()
    
    # Clear any cached modules to force re-import
    import sys
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('cymetric.models.')]
    for module in modules_to_clear:
        if module != 'cymetric.models':  # Don't clear the package itself
            del sys.modules[module]

# Expose framework availability
TORCH_AVAILABLE = _TORCH_AVAILABLE
TENSORFLOW_AVAILABLE = _TF_AVAILABLE
JAX_AVAILABLE = _JAX_AVAILABLE
PREFERRED_FRAMEWORK = get_preferred_framework()
