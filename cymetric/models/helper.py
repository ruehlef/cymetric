"""
Helper functions compatibility layer.
Automatically imports from tensorflow or torch implementation.
Defaults to TensorFlow when both are available, but respects CYMETRIC_FRAMEWORK environment variable.
"""

from cymetric import TORCH_AVAILABLE, TENSORFLOW_AVAILABLE, get_preferred_framework

preferred = get_preferred_framework()

if preferred == 'tensorflow':
    from cymetric.tensorflow.models.helper import *
elif preferred == 'torch':
    from cymetric.torch.models.helper import *
else:
    raise ImportError("No framework (PyTorch or TensorFlow) available for helper module")
