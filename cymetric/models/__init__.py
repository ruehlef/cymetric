"""
Compatibility layer for cymetric.models imports.

This module provides a compatibility layer that automatically redirects imports
to the appropriate framework implementation (PyTorch or TensorFlow).
"""

# Import all compatibility modules so they can be accessed directly
from . import measures
from . import callbacks
from . import losses
from . import metrics
from . import fubinistudy
from . import models  # Unified models module
from . import helper  # Unified helper module

# Re-export commonly used items
from .measures import *
from .callbacks import *
from .losses import *
from .metrics import *
from .fubinistudy import *
