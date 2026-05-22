"""
Compatibility layer for cymetric.models imports.

This module provides a compatibility layer that automatically redirects imports
to the appropriate framework implementation (PyTorch, TensorFlow, or JAX).

Submodules are imported lazily: each one is only loaded when first accessed,
so importing cymetric.models itself never fails due to a missing framework.
Use explicit submodule imports in your code:

    from cymetric.models.models import PhiFSModel
    from cymetric.models.helper import train_model, prepare_basis
    from cymetric.models.callbacks import SigmaCallback
"""
