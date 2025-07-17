"""
Wolfram Mathematica interface for Cymetric

This module provides utilities for interfacing with Wolfram Mathematica
for symbolic computations and point generation.

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

from .mathematicalib import get_torch_components, get_tensorflow_components

__all__ = [
    'get_torch_components',
    'get_tensorflow_components'
]
