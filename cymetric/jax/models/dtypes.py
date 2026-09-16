"""Precision policy for the JAX backend.

The JAX models used to hardcode ``jnp.complex64``/``jnp.float32`` everywhere,
so enabling ``jax_enable_x64`` and passing a ``complex128`` BASIS still gave
single-precision pullbacks and a partly single-precision ``ddbar phi`` (see
issue #5).  These helpers make the working precision follow JAX's own x64
switch, which is the knob users already expect to control it.

They are deliberately *functions* rather than module-level constants:
``jax.config.jax_enable_x64`` can be set after this module is imported, and a
constant captured at import time would silently miss it.  The value is a
static Python bool, so calling them inside a traced function is free.

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

import jax
import jax.numpy as jnp


def x64_enabled():
    """True when JAX is configured for 64-bit values."""
    return bool(jax.config.jax_enable_x64)


def complex_dtype():
    """complex128 when ``jax_enable_x64`` is set, else complex64."""
    return jnp.complex128 if x64_enabled() else jnp.complex64


def real_dtype():
    """float64 when ``jax_enable_x64`` is set, else float32."""
    return jnp.float64 if x64_enabled() else jnp.float32
