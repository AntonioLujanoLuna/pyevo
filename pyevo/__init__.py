"""
PyEvo - Evolutionary Computation Algorithms in Python.

PyEvo provides a collection of evolutionary algorithms for black-box optimization,
focusing on clean, educational implementations with practical applications.

Core algorithms:
- SNES (Separable Natural Evolution Strategies)
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- PSO (Particle Swarm Optimization)

Visit https://github.com/AntonioLujanoLuna/pyevo for more information.
"""

from pyevo.optimizers import (
    SNES, 
    CMA_ES, 
    PSO, 
    Optimizer,
    get_default_population_count
)

# Import utilities
from pyevo.utils import (
    is_gpu_available,
    optimize_with_acceleration,
    parallel_evaluate,
    batch_process
)

__version__ = "0.1.0"
__all__ = [
    "Optimizer",
    "SNES",
    "CMA_ES",
    "PSO",
    "get_default_population_count",
    "is_gpu_available",
    "optimize_with_acceleration",
    "parallel_evaluate",
    "batch_process"
] 