"""
Global constants for the PySNES package.
"""

# Image processing constants
DEFAULT_MAX_IMAGE_SIZE = 512  # Maximum dimension for image processing
DEFAULT_RECT_COUNT = 500      # Default number of rectangles for image approximation 
DEFAULT_POPULATION_SIZE = 50  # Default population size for SNES

# Optimization constants
DEFAULT_EPOCHS = 5000         # Default number of epochs/iterations
DEFAULT_ALPHA = 0.05          # Default learning rate
DEFAULT_EARLY_STOP = 1e-6     # Default early stopping tolerance
DEFAULT_PATIENCE = 10         # Default number of epochs to wait before early stopping

# File paths
DEFAULT_OUTPUT_DIR = "examples/output"        # Default output directory
DEFAULT_CHECKPOINT_DIR = "examples/checkpoints"  # Default checkpoint directory