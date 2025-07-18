# Framework-Aware Installation Guide

Cymetric supports both PyTorch and TensorFlow backends with intelligent installation that adapts to your Python version.

## Automatic Installation (Recommended)

```bash
pip install cymetric
```

This will automatically:
- ✅ Install PyTorch on Python 3.8+ 
- ✅ Install TensorFlow on Python 3.8-3.12
- ⚠️  Skip TensorFlow on Python 3.13+ (not yet supported)
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

### Both Frameworks (Python 3.8-3.12)
```bash
pip install cymetric[both]
```

### Core Package Only
```bash
pip install cymetric[minimal]
```

## Python Version Compatibility

| Python Version | PyTorch Support | TensorFlow Support |
|---------------|----------------|-------------------|
| 3.8           | ✅ Yes          | ✅ Yes            |
| 3.9           | ✅ Yes          | ✅ Yes            |
| 3.10          | ✅ Yes          | ✅ Yes            |
| 3.11          | ✅ Yes          | ✅ Yes            |
| 3.12          | ✅ Yes          | ✅ Yes            |
| 3.13+         | ✅ Yes          | ❌ Not yet        |

## Framework Selection at Runtime

Even if both frameworks are installed, you can choose which one to use:

```python
import cymetric

# Set framework preference
cymetric.set_preferred_framework('torch')     # Use PyTorch
cymetric.set_preferred_framework('tensorflow') # Use TensorFlow

# Or use environment variable
import os
os.environ['CYMETRIC_FRAMEWORK'] = 'torch'
```

## Troubleshooting

### Python 3.13 Installation
On Python 3.13, the installer will automatically skip TensorFlow:
```
⚠️  Skipping TensorFlow (not compatible with this Python version)
📦 Including PyTorch dependencies
```

### Manual Framework Installation
If automatic detection fails, install frameworks manually:
```bash
# Install PyTorch first
pip install torch torchvision

# Then install cymetric
pip install cymetric[minimal]
```

### Check Installation
```python
import cymetric
print(cymetric.TORCH_AVAILABLE)      # True if PyTorch available
print(cymetric.TENSORFLOW_AVAILABLE) # True if TensorFlow available
print(cymetric.get_preferred_framework()) # Current framework
```
