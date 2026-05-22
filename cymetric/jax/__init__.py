"""
JAX implementation of Cymetric.

This module provides JAX/Equinox-based models for Calabi-Yau metric learning.
JIT compilation via jax.jit (eqx.filter_jit) plays the same role as
@tf.function in the TensorFlow implementation.

Usage:
    from cymetric.jax.models import FreeModel, MultFSModel, PhiFSModel
    from cymetric.jax.models.helper import prepare_basis, train_model
    from cymetric.jax.models.callbacks import SigmaCallback, KaehlerCallback
"""

try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
except ImportError as e:
    raise ImportError(
        "JAX dependencies are not installed. "
        "Install with: pip install jax equinox optax"
    ) from e

from .models.models import (
    FreeModel,
    MultFSModel,
    MatrixFSModel,
    AddFSModel,
    PhiFSModel,
    ToricModel,
    PhiFSModelToric,
    MatrixFSModelToric,
)

from .models.helper import (
    prepare_basis,
    train_model,
)

from .models.callbacks import (
    AlphaCallback,
    SigmaCallback,
    KaehlerCallback,
    TransitionCallback,
    RicciCallback,
    VolkCallback,
)
