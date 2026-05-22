"""
FubiniStudy module compatibility layer.
Automatically imports from tensorflow, torch, or jax implementation.
Defaults to TensorFlow when available, then JAX, then PyTorch, but respects CYMETRIC_FRAMEWORK environment variable.
"""

from cymetric import TORCH_AVAILABLE, TENSORFLOW_AVAILABLE, JAX_AVAILABLE, get_preferred_framework

preferred = get_preferred_framework()

if preferred == 'tensorflow':
    from cymetric.tensorflow.models.fubinistudy import *
elif preferred == 'torch':
    from cymetric.torch.models.fubinistudy import *
elif preferred == 'jax':
    from cymetric.jax.models.fubinistudy import *
else:
    raise ImportError("No framework (PyTorch, TensorFlow, or JAX) available for fubinistudy module")
