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

import importlib
import os

# Framework availability checks. The probes are deferred so that importing
# cymetric.pointgen does not import an ML framework.
_AVAILABLE = {}

def _available(*modules):
    """Check if every module in modules can be imported.

    The result is cached, so each framework is probed at most once.

    Returns:
        bool: True if all of modules import successfully
    """
    if modules not in _AVAILABLE:
        try:
            for module in modules:
                importlib.import_module(module)
            _AVAILABLE[modules] = True
        except ImportError:
            _AVAILABLE[modules] = False
    return _AVAILABLE[modules]

def check_torch():
    """Check if PyTorch is available."""
    if not _available('torch'):
        raise ImportError(
            "PyTorch is not installed. Install it with: pip install cymetric[torch]"
        )

def check_tensorflow():
    """Check if TensorFlow is available."""
    if not _available('tensorflow'):
        raise ImportError(
            "TensorFlow is not installed. Install it with: pip install cymetric[tensorflow]"
        )

def check_jax():
    """Check if JAX is available."""
    if not _available('jax', 'equinox'):
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
        if _available('torch'):
            return 'torch'
        elif _available('tensorflow'):
            import warnings
            warnings.warn("PyTorch requested but not available, falling back to TensorFlow")
            return 'tensorflow'
        elif _available('jax', 'equinox'):
            import warnings
            warnings.warn("PyTorch requested but not available, falling back to JAX")
            return 'jax'
        else:
            raise ImportError("PyTorch requested but no framework available")
    
    elif preferred == 'tf' or preferred == 'tensorflow':
        if _available('tensorflow'):
            return 'tensorflow'
        elif _available('torch'):
            import warnings
            warnings.warn("TensorFlow requested but not available, falling back to PyTorch")
            return 'torch'
        elif _available('jax', 'equinox'):
            import warnings
            warnings.warn("TensorFlow requested but not available, falling back to JAX")
            return 'jax'
        else:
            raise ImportError("TensorFlow requested but no framework available")
    
    elif preferred == 'jax' or preferred == 'equinox':
        if _available('jax', 'equinox'):
            return 'jax'
        elif _available('tensorflow'):
            import warnings
            warnings.warn("JAX requested but not available, falling back to TensorFlow")
            return 'tensorflow'
        elif _available('torch'):
            import warnings
            warnings.warn("JAX requested but not available, falling back to PyTorch")
            return 'torch'
        else:
            raise ImportError("JAX requested but no framework available")
    
    # Default behavior: prefer TensorFlow, then JAX, then PyTorch
    if _available('tensorflow'):
        return 'tensorflow'
    elif _available('jax', 'equinox'):
        return 'jax'
    elif _available('torch'):
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

# Expose framework availability. Resolved on first access rather than at
# import time, and then cached in the module namespace, so that code which
# only uses cymetric.pointgen never triggers a framework import.
_LAZY_ATTRS = {
    'TORCH_AVAILABLE': lambda: _available('torch'),
    'TENSORFLOW_AVAILABLE': lambda: _available('tensorflow'),
    'JAX_AVAILABLE': lambda: _available('jax', 'equinox'),
    'PREFERRED_FRAMEWORK': get_preferred_framework,
}

def __getattr__(name):
    if name in _LAZY_ATTRS:
        value = _LAZY_ATTRS[name]()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))
