"""
Setup script for Cymetric - Unified PyTorch/TensorFlow Package

A package for learning Calabi-Yau metrics using neural networks with 
support for both PyTorch and TensorFlow backends.
"""
from setuptools import setup, find_packages
import os
import sys

# Import our installation utilities
try:
    from install_utils import get_compatible_frameworks, print_installation_info
except ImportError:
    # Fallback if install_utils is not available
    def get_compatible_frameworks():
        version = sys.version_info
        frameworks = []
        if version >= (3, 8):
            frameworks.append('torch')
        if (3, 8) <= (version.major, version.minor) <= (3, 12):
            frameworks.append('tensorflow')
        return frameworks
    
    def print_installation_info():
        pass

# Read version from VERSION file
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_file, 'r') as f:
        return f.read().strip()

# Read requirements from files
def get_requirements(filename):
    req_file = os.path.join(os.path.dirname(__file__), filename)
    with open(req_file, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_file, 'r', encoding='utf-8') as f:
        return f.read()

def get_smart_requirements():
    """Get requirements based on framework compatibility."""
    print("\n🔍 Checking framework compatibility...")
    print_installation_info()
    
    core_reqs = get_requirements('requirements-core.txt')
    compatible_frameworks = get_compatible_frameworks()
    
    framework_reqs = []
    
    if 'torch' in compatible_frameworks:
        framework_reqs.extend(get_requirements('requirements-torch.txt'))
        print("📦 Including PyTorch dependencies")
    else:
        print("⚠️  Skipping PyTorch (not compatible with this Python version)")
    
    if 'tensorflow' in compatible_frameworks:
        framework_reqs.extend(get_requirements('requirements-tensorflow.txt'))
        print("📦 Including TensorFlow dependencies")
    else:
        print("⚠️  Skipping TensorFlow (not compatible with this Python version)")
    
    if not compatible_frameworks:
        print("❌ No compatible frameworks found!")
        print("   Installing core package only. You'll need to manually install PyTorch or TensorFlow.")
    
    print(f"📋 Total requirements: {len(core_reqs + framework_reqs)} packages\n")
    return core_reqs + framework_reqs

setup(
    name="cymetric",
    version=get_version(),
    author="Fabian Ruehle",
    author_email="f.ruehle@northeastern.edu",
    description="A unified package for learning Calabi-Yau metrics using PyTorch and TensorFlow",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ruehlef/cymetric",
    packages=find_packages(include=['cymetric', 'cymetric.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=get_smart_requirements(),
    extras_require={
        "torch": get_requirements('requirements-torch.txt'),
        "tensorflow": get_requirements('requirements-tensorflow.txt'),
        "both": (get_requirements('requirements-torch.txt') + 
                get_requirements('requirements-tensorflow.txt')),
        "minimal": get_requirements('requirements-core.txt'),
        "optional": get_requirements('requirements-optional.txt'),
    },
    entry_points={
        "console_scripts": [
            "cymetric-torch=cymetric.torch.models.helper:main",
            "cymetric-tf=cymetric.tensorflow.models.helper:main",
        ],
    },
    include_package_data=True,
    package_data={
        'cymetric.wolfram': ['*.m'],
    },
    zip_safe=False,
)
