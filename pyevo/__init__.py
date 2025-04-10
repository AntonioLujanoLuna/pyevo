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

# Import optimizers
from pyevo.optimizers import (
    SNES, 
    CMA_ES, 
    PSO, 
    Optimizer,
    get_default_population_count
)

# Import acceleration utilities
from pyevo.utils.acceleration import (
    is_gpu_available,
    get_gpu_memory_info,
    clear_gpu_memory,
    get_array_module,
    to_device,
    batch_process,
    parallel_evaluate,
    optimize_with_acceleration,
    save_checkpoint,
    load_checkpoint
)

# Import interactive utilities
from pyevo.utils.interactive import InteractiveOptimizer

# Import image utilities
from pyevo.utils.image import (
    calculate_ssim,
    convolve2d,
    get_optimal_image_functions
)

# Import constants
from pyevo.utils.constants import (
    DEFAULT_MAX_IMAGE_SIZE, 
    DEFAULT_RECT_COUNT, 
    DEFAULT_POPULATION_SIZE,
    DEFAULT_EPOCHS, 
    DEFAULT_ALPHA, 
    DEFAULT_EARLY_STOP, 
    DEFAULT_PATIENCE,
    DEFAULT_OUTPUT_DIR, 
    DEFAULT_CHECKPOINT_DIR,
    OPTIMIZERS,
    DEFAULT_OPTIMIZER
)

__version__ = "0.1.0"

__all__ = [
    # Optimizers
    "Optimizer",
    "SNES",
    "CMA_ES",
    "PSO",
    "get_default_population_count",
    
    # Acceleration utilities
    "is_gpu_available",
    "get_gpu_memory_info",
    "clear_gpu_memory",
    "get_array_module",
    "to_device",
    "batch_process",
    "parallel_evaluate",
    "optimize_with_acceleration",
    "save_checkpoint",
    "load_checkpoint",
    
    # Interactive utilities
    "InteractiveOptimizer",
    
    # Image utilities
    "calculate_ssim",
    "convolve2d",
    "get_optimal_image_functions",
    
    # Constants
    "DEFAULT_MAX_IMAGE_SIZE", 
    "DEFAULT_RECT_COUNT", 
    "DEFAULT_POPULATION_SIZE",
    "DEFAULT_EPOCHS", 
    "DEFAULT_ALPHA", 
    "DEFAULT_EARLY_STOP", 
    "DEFAULT_PATIENCE",
    "DEFAULT_OUTPUT_DIR", 
    "DEFAULT_CHECKPOINT_DIR",
    "OPTIMIZERS",
    "DEFAULT_OPTIMIZER"
] 