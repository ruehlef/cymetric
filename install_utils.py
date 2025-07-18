"""
Smart installation script for Cymetric with framework fallback support.

This script handles intelligent installation of PyTorch/TensorFlow dependencies
based on Python version compatibility and framework availability.
"""
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is supported."""
    version = sys.version_info
    if version < (3, 8):
        raise RuntimeError(f"Python {version.major}.{version.minor} is not supported. Requires Python 3.8+")
    return version

def can_install_tensorflow():
    """Check if TensorFlow can be installed on this Python version."""
    version = sys.version_info
    # TensorFlow officially supports Python 3.8-3.12 as of 2024
    return (3, 8) <= (version.major, version.minor) <= (3, 12)

def can_install_torch():
    """Check if PyTorch can be installed on this Python version."""
    version = sys.version_info
    # PyTorch generally has broader Python version support
    return version >= (3, 8)

def framework_available(framework_name):
    """Check if a framework is already installed."""
    try:
        spec = importlib.util.find_spec(framework_name)
        return spec is not None
    except ImportError:
        return False

def get_compatible_frameworks():
    """Determine which frameworks are compatible with current Python version."""
    compatible = []
    
    if can_install_torch():
        compatible.append('torch')
        
    if can_install_tensorflow():
        compatible.append('tensorflow')
        
    return compatible

def print_installation_info():
    """Print information about framework compatibility."""
    python_version = sys.version_info
    compatible = get_compatible_frameworks()
    
    print(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    print(f"Compatible frameworks: {', '.join(compatible) if compatible else 'None'}")
    
    if 'torch' in compatible:
        status = "✓ installed" if framework_available('torch') else "○ will be installed"
        print(f"PyTorch: {status}")
        
    if 'tensorflow' in compatible:
        status = "✓ installed" if framework_available('tensorflow') else "○ will be installed"
        print(f"TensorFlow: {status}")
    else:
        print("TensorFlow: ✗ not compatible with this Python version")
        
    if not compatible:
        print("⚠️  Warning: No frameworks compatible. Installing core package only.")

if __name__ == "__main__":
    print_installation_info()
