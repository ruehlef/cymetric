"""
Point generation utilities for Calabi-Yau manifolds

This module provides point generators for both CICY and toric Calabi-Yau manifolds
using Mathematica as backend for numerical computations.

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

# Optional backend: needs wolframclient, which is not a hard dependency.
try:
    from .pointgen_mathematica import PointGeneratorMathematica, ToricPointGeneratorMathematica
except ImportError:  # pragma: no cover - optional
    pass
from .pointgen_cicy import CICYPointGenerator
from .pointgen_toric import ToricPointGenerator
from .pointgen import PointGenerator
from .pointgen_mc import CICYPointGeneratorMC, ToricPointGeneratorMC
from .nphelper import prepare_dataset, prepare_basis_pickle, get_levicivita_tensor

__all__ = [
    'PointGeneratorMathematica',
    'ToricPointGeneratorMathematica',
    'CICYPointGenerator',
    'ToricPointGenerator',
    'PointGenerator',
    'CICYPointGeneratorMC',
    'ToricPointGeneratorMC',
    'prepare_dataset',
    'prepare_basis_pickle',
    'get_levicivita_tensor'
]
