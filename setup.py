"""
Setup script for Cymetric - Unified PyTorch/TensorFlow Package

A package for learning Calabi-Yau metrics using neural networks with 
support for both PyTorch and TensorFlow backends.
"""
from setuptools import setup, find_packages
import os

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
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=(
        get_requirements('requirements-core.txt') +
        get_requirements('requirements-torch.txt') + 
        get_requirements('requirements-tensorflow.txt')
    ),
    extras_require={
        "torch": get_requirements('requirements-torch.txt'),
        "tensorflow": get_requirements('requirements-tensorflow.txt'),
        "minimal": get_requirements('requirements-core.txt'),
        "optional": get_requirements('requirements-optional.txt'),
    },
    entry_points={
        "console_scripts": [
            "cymetric-torch=cymetric.torch.models.torchhelper:main",
            "cymetric-tf=cymetric.tensorflow.models.tfhelper:main",
        ],
    },
    include_package_data=True,
    package_data={
        'cymetric.wolfram': ['*.m'],
    },
    zip_safe=False,
)
