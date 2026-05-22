"""
Models compatibility layer.
Automatically imports from tensorflow, torch, or jax implementation.
Defaults to TensorFlow when available, then JAX, then PyTorch, but respects CYMETRIC_FRAMEWORK environment variable.
"""

from cymetric import TORCH_AVAILABLE, TENSORFLOW_AVAILABLE, JAX_AVAILABLE, get_preferred_framework

preferred = get_preferred_framework()

if preferred == 'tensorflow':
    from cymetric.tensorflow.models.models import *
elif preferred == 'torch':
    from cymetric.torch.models.models import *
elif preferred == 'jax':
    from cymetric.jax.models.models import *
else:
    raise ImportError("No framework (PyTorch, TensorFlow, or JAX) available for models module")
