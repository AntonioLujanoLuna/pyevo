"""
Utilities package for the PySNES project.

Contains various utility modules used throughout the codebase.
"""

from utils.image_processing import calculate_ssim, convolve2d
from utils.interactive import InteractiveOptimizer
from utils.constants import OPTIMIZERS, DEFAULT_OPTIMIZER

__all__ = ["calculate_ssim", "convolve2d", "InteractiveOptimizer", "create_optimizer"] 

def create_optimizer(optimizer_type=DEFAULT_OPTIMIZER, **kwargs):
    """Create an optimizer instance based on the specified type.
    
    Args:
        optimizer_type: Type of optimizer to create (default: from constants)
        **kwargs: Additional arguments to pass to the optimizer constructor
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If the optimizer type is not supported
    """
    import sys
    import os
    
    # Add the parent directory to the path to ensure we can import
    # modules from the root directory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from snes import SNES
    from optimizers import CMA_ES, PSO
    
    # Available optimizer types
    optimizer_classes = {
        "snes": SNES,
        "cmaes": CMA_ES,
        "pso": PSO,
    }
    
    # Ensure optimizer_type is lowercase for case-insensitive matching
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type not in optimizer_classes:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                         f"Available types: {', '.join(optimizer_classes.keys())}")
    
    # Create and return the optimizer instance
    return optimizer_classes[optimizer_type](**kwargs) 