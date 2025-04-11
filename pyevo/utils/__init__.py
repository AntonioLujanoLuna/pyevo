"""
PyEvo utility modules.

This package contains various utility modules for the PyEvo library:
- acceleration: GPU and parallel processing acceleration utilities
- image: Image processing utilities for image approximation examples
- interactive: Interactive control for optimizers
- constants: Common constants used throughout the project
"""

from typing import Dict, Type, Optional, Any, Union, List, Tuple, Callable

import numpy as np

# Import from acceleration module
from pyevo.utils.acceleration import (
    is_gpu_available,
    get_array_module,
    to_device,
    batch_process,
    parallel_evaluate,
    optimize_with_acceleration,
    save_checkpoint,
    load_checkpoint,
    get_gpu_memory_info,
    clear_gpu_memory,
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
    DEFAULT_CHECKPOINT_DIR,
    OPTIMIZERS,
    DEFAULT_OPTIMIZER,
)

# Import from interactive module
from pyevo.utils.interactive import InteractiveOptimizer

# Import from image module
from pyevo.utils.image import (
    calculate_ssim,
    convolve2d,
    get_optimal_image_functions,
)

# Import optimizers directly from the package to avoid circular imports
from pyevo.optimizers.base import Optimizer

def create_optimizer(optimizer_type: str = "snes", **kwargs: Any) -> Optimizer:
    """Create an optimizer instance based on the specified type.
    
    Args:
        optimizer_type: Type of optimizer to create (default: "snes")
        **kwargs: Additional arguments to pass to the optimizer constructor
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If the optimizer type is not supported
    """
    # Import optimizers from the package
    from pyevo.optimizers import SNES, CMA_ES, PSO, DE, SimulatedAnnealing, GeneticAlgorithm, CrossEntropyMethod
    
    # Available optimizer types
    optimizer_classes: Dict[str, Type[Optimizer]] = {
        "snes": SNES,
        "cmaes": CMA_ES,
        "pso": PSO,
        "de": DE,
        "sa": SimulatedAnnealing,
        "ga": GeneticAlgorithm,
        "cem": CrossEntropyMethod,
    }
    
    # Ensure optimizer_type is lowercase for case-insensitive matching
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type not in optimizer_classes:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                         f"Available types: {', '.join(optimizer_classes.keys())}")
    
    # Create and return the optimizer instance
    return optimizer_classes[optimizer_type](**kwargs)

__all__ = [
    # Acceleration
    "is_gpu_available",
    "get_array_module",
    "to_device",
    "batch_process",
    "parallel_evaluate",
    "optimize_with_acceleration",
    "save_checkpoint",
    "load_checkpoint",
    "get_gpu_memory_info",
    "clear_gpu_memory",
    
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
    "DEFAULT_OPTIMIZER",
    
    # Interactive
    "InteractiveOptimizer",
    
    # Image
    "calculate_ssim",
    "convolve2d",
    "get_optimal_image_functions",
    
    # Helper function
    "create_optimizer"
] 