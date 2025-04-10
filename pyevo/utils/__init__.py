"""
PyEvo utility modules.

This package contains various utility modules for the PyEvo library:
- acceleration: GPU and parallel processing acceleration utilities
- image: Image processing utilities for image approximation examples
- interactive: Interactive control for optimizers
- constants: Common constants used throughout the project
"""

# Import from acceleration module
from pyevo.utils.acceleration import (
    is_gpu_available,
    get_array_module,
    to_device,
    batch_process,
    parallel_evaluate,
    optimize_with_acceleration,
)

# Import from constants module
from pyevo.utils.constants import (
    DEFAULT_MAX_IMAGE_SIZE, 
    DEFAULT_RECT_COUNT, 
    DEFAULT_POPULATION_SIZE,
    DEFAULT_EPOCHS, 
    DEFAULT_ALPHA, 
    DEFAULT_EARLY_STOP, 
    DEFAULT_PATIENCE,
    DEFAULT_OUTPUT_DIR, 
    DEFAULT_CHECKPOINT_DIR
)

# Import from interactive module
from pyevo.utils.interactive import InteractiveOptimizer

# Import from image module
from pyevo.utils.image import (
    calculate_ssim,
    convolve2d,
    get_optimal_image_functions
)

__all__ = [
    # Acceleration
    "is_gpu_available",
    "get_array_module",
    "to_device",
    "batch_process",
    "parallel_evaluate",
    "optimize_with_acceleration",
    
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
    
    # Interactive
    "InteractiveOptimizer",
    
    # Image
    "calculate_ssim",
    "convolve2d",
    "get_optimal_image_functions"
] 