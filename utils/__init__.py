"""
Utilities package for the PySNES project.

Contains various utility modules used throughout the codebase.
"""

from utils.image_processing import calculate_ssim, convolve2d
from utils.interactive import InteractiveOptimizer

__all__ = ["calculate_ssim", "convolve2d", "InteractiveOptimizer"] 