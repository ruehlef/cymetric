# Cymetric Framework Selection

This document describes how to control which framework (PyTorch or TensorFlow) is used by the cymetric compatibility layer.

## Default Behavior

When both PyTorch and TensorFlow are installed, cymetric defaults to **TensorFlow**:

```python
import cymetric
from cymetric.models.measures import ricci_measure
print(ricci_measure.__module__)  # cymetric.tensorflow.models.measures
```

## Framework Selection Methods

### Method 1: Environment Variable (Recommended)

Set the `CYMETRIC_FRAMEWORK` environment variable before importing cymetric:

```bash
export CYMETRIC_FRAMEWORK=torch
python your_script.py
```

Or in Python:
```python
import os
os.environ['CYMETRIC_FRAMEWORK'] = 'torch'  # Must be before importing cymetric
import cymetric
from cymetric.models.measures import ricci_measure
```

Valid values: `'torch'`, `'pytorch'`, `'tf'`, `'tensorflow'`

### Method 2: Runtime Switching

Change the framework after importing cymetric:

```python
import cymetric
from cymetric.models.measures import ricci_measure  # Uses default (TensorFlow)

# Switch to PyTorch
cymetric.set_preferred_framework('torch')
from cymetric.models.measures import ricci_measure  # Now uses PyTorch
```

### Method 3: Check Current Framework

```python
import cymetric
print(f"Available frameworks: PyTorch={cymetric.TORCH_AVAILABLE}, TensorFlow={cymetric.TENSORFLOW_AVAILABLE}")
print(f"Currently using: {cymetric.PREFERRED_FRAMEWORK}")
```

## Direct Framework Access

You can always import from specific frameworks directly:

```python
# Always use PyTorch
from cymetric.torch.models.measures import ricci_measure

# Always use TensorFlow  
from cymetric.tensorflow.models.measures import ricci_measure
```

## Compatibility Layer Modules

The following modules support automatic framework selection:

- `cymetric.models.measures`
- `cymetric.models.callbacks`
- `cymetric.models.losses`
- `cymetric.models.metrics`
- `cymetric.models.fubinistudy`
- `cymetric.models.torchmodels` / `cymetric.models.tfmodels`
- `cymetric.models.torchhelper` / `cymetric.models.tfhelper`

All of these automatically redirect to the appropriate framework implementation.

## Function Name Compatibility

The helper modules provide unified function names across frameworks:

```python
# These all work regardless of the selected framework
from cymetric.models.torchhelper import prepare_torch_basis  # Works with both frameworks
from cymetric.models.tfhelper import prepare_tf_basis        # Works with both frameworks
from cymetric.models.torchhelper import prepare_tf_basis     # Cross-compatible alias
from cymetric.models.tfhelper import prepare_torch_basis     # Cross-compatible alias
```

This ensures that existing code using framework-specific function names continues to work regardless of which framework is selected.
