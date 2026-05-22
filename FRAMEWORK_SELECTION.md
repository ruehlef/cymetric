# Cymetric Framework Selection

This document describes how to control which framework (PyTorch, TensorFlow, or JAX) is used by the cymetric compatibility layer.

## Default Behavior

When multiple frameworks are installed, cymetric defaults to **TensorFlow**, then **JAX**, then **PyTorch**:

```python
import cymetric
from cymetric.models.measures import ricci_measure
print(ricci_measure.__module__)  # cymetric.tensorflow.models.measures
```

## Framework Selection Methods

### Method 1: Environment Variable (Recommended)

Set the `CYMETRIC_FRAMEWORK` environment variable before importing cymetric:

```bash
export CYMETRIC_FRAMEWORK=jax
python your_script.py
```

Or in Python:
```python
import os
os.environ['CYMETRIC_FRAMEWORK'] = 'jax'  # Must be before importing cymetric
import cymetric
from cymetric.models.measures import ricci_measure
```

Valid values: `'torch'`, `'pytorch'`, `'tf'`, `'tensorflow'`, `'jax'`, `'equinox'`

### Method 2: Runtime Switching

Change the framework after importing cymetric:

```python
import cymetric
from cymetric.models.measures import ricci_measure  # Uses default (TensorFlow)

# Switch to JAX
cymetric.set_preferred_framework('jax')
from cymetric.models.measures import ricci_measure  # Now uses JAX
```

### Method 3: Check Current Framework

```python
import cymetric
print(f"Available frameworks: PyTorch={cymetric.TORCH_AVAILABLE}, TensorFlow={cymetric.TENSORFLOW_AVAILABLE}, JAX={cymetric.JAX_AVAILABLE}")
print(f"Currently using: {cymetric.PREFERRED_FRAMEWORK}")
```

## Direct Framework Access

You can always import from specific frameworks directly:

```python
# Always use PyTorch
from cymetric.torch.models.measures import ricci_measure

# Always use TensorFlow  
from cymetric.tensorflow.models.measures import ricci_measure

# Always use JAX
from cymetric.jax.models.measures import ricci_measure
```

## Compatibility Layer Modules

The following modules support automatic framework selection:

- `cymetric.models.measures`
- `cymetric.models.callbacks`
- `cymetric.models.losses`
- `cymetric.models.metrics`
- `cymetric.models.fubinistudy`
- `cymetric.models.models`
- `cymetric.models.helper`

All of these automatically redirect to the appropriate framework implementation.

## JAX Backend Notes

The JAX backend uses [Equinox](https://docs.kidger.site/equinox/) for neural network modules and [Optax](https://optax.readthedocs.io/) for optimizers:

- Models are `equinox.Module` subclasses (not `keras.Model` or `torch.nn.Module`)
- Optimizers are `optax.GradientTransformation` objects (e.g. `optax.adam(lr)`)
- Model weights are saved/loaded with `equinox.tree_serialise_leaves` / `equinox.tree_deserialise_leaves` (`.eqx` files)
- Training uses JAX JIT compilation via `@equinox.filter_jit`

## Function Name Compatibility

The helper modules provide unified function names across frameworks:

```python
# These all work regardless of the selected framework
from cymetric.models.helper import prepare_basis, train_model
```

This ensures that existing code continues to work regardless of which framework is selected.
