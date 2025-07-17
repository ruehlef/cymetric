"""
SageMath interface for toric geometry computations

This module provides utilities for working with toric varieties
and computing geometric data needed for Calabi-Yau metrics.

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

from .sagelib import prepare_toric_cy_data

__all__ = [
    'prepare_toric_cy_data'
]
