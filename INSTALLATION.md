# Framework-Aware Installation Guide

Cymetric supports PyTorch, TensorFlow, and JAX backends with intelligent installation that adapts to your Python version.

## Automatic Installation (Recommended)

```bash
pip install cymetric
```

This will automatically:
- ✅ Install PyTorch on Python 3.8+
- ✅ Install TensorFlow on Python 3.8-3.12
- ✅ Install JAX on Python 3.9+
- ⚠️  Skip TensorFlow on Python 3.13+ (not yet supported)
- ⚠️  Skip JAX on Python 3.8 (not yet supported)
- 📦 Always install core dependencies

## Manual Framework Selection

If you prefer to control which frameworks are installed:

### PyTorch Only
```bash
pip install cymetric[torch]
```

### TensorFlow Only (Python 3.8-3.12)
```bash
pip install cymetric[tensorflow]
```

### JAX Only
```bash
pip install cymetric[jax]
```

### All Compatible Frameworks (Recommended)
```bash
pip install cymetric
```

This installs PyTorch + TensorFlow + JAX depending on your Python version (see compatibility table below).

### Core Package Only
```bash
pip install cymetric[minimal]
```

## Python Version Compatibility

| Python Version | PyTorch Support | TensorFlow Support | JAX Support |
|---------------|----------------|-------------------|-------------|
| 3.8           | ✅ Yes          | ✅ Yes            | ❌ No        |
| 3.9           | ✅ Yes          | ✅ Yes            | ✅ Yes       |
| 3.10          | ✅ Yes          | ✅ Yes            | ✅ Yes       |
| 3.11          | ✅ Yes          | ✅ Yes            | ✅ Yes       |
| 3.12          | ✅ Yes          | ✅ Yes            | ✅ Yes       |
| 3.13+         | ✅ Yes          | ❌ Not yet        | ✅ Yes       |

## Framework Selection at Runtime

Even if both frameworks are installed, you can choose which one to use:

```python
import cymetric

# Set framework preference
cymetric.set_preferred_framework('torch')      # Use PyTorch
cymetric.set_preferred_framework('tensorflow') # Use TensorFlow
cymetric.set_preferred_framework('jax')        # Use JAX

# Or use environment variable (must be set before importing cymetric)
import os
os.environ['CYMETRIC_FRAMEWORK'] = 'torch'  # or 'tensorflow' or 'jax'
```

## Troubleshooting

### Python 3.13 Installation
On Python 3.13, the installer will automatically skip TensorFlow but still install PyTorch and JAX:
```
⚠️  Skipping TensorFlow (not compatible with this Python version)
📦 Including PyTorch dependencies
📦 Including JAX dependencies
```

### Python 3.8 Installation
On Python 3.8, JAX is not supported. PyTorch and TensorFlow will be installed:
```
📦 Including PyTorch dependencies
📦 Including TensorFlow dependencies
⚠️  Skipping JAX (not compatible with this Python version)
```

### Manual Framework Installation
If automatic detection fails, install frameworks manually:
```bash
# Install PyTorch
pip install torch torchvision

# Install JAX
pip install jax jaxlib equinox optax

# Then install cymetric core only
pip install cymetric[minimal]
```

### Check Installation
```python
import cymetric
print(cymetric.TORCH_AVAILABLE)      # True if PyTorch available
print(cymetric.TENSORFLOW_AVAILABLE) # True if TensorFlow available
print(cymetric.JAX_AVAILABLE)        # True if JAX available
print(cymetric.get_preferred_framework()) # Current framework
```
