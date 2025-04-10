"""
PyEvo: Python Evolutionary Optimization Library

A collection of evolutionary optimization algorithms for black-box optimization.
"""

__version__ = "0.1.0"

from pyevo.optimizers import (
    Optimizer,
    SNES,
    get_default_population_count,
    CMA_ES,
    PSO
)

__all__ = [
    "Optimizer",
    "SNES",
    "get_default_population_count", 
    "CMA_ES",
    "PSO"
] 