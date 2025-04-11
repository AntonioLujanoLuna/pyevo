"""
Global constants for the PyEvo package.
"""

# Image processing constants
DEFAULT_MAX_IMAGE_SIZE = 256  # Maximum dimension for image processing
DEFAULT_RECT_COUNT = 250      # Default number of rectangles for image approximation 
DEFAULT_POPULATION_SIZE = 16  # Default population size for SNES

# Optimization constants
DEFAULT_EPOCHS = 1000         # Default number of epochs/iterations
DEFAULT_ALPHA = 0.05          # Default learning rate
DEFAULT_EARLY_STOP = 1e-6     # Default early stopping tolerance
DEFAULT_PATIENCE = 10         # Default number of epochs to wait before early stopping

# Optimizer options
OPTIMIZERS = {
    "snes": "Separable Natural Evolution Strategies",
    "cmaes": "Covariance Matrix Adaptation Evolution Strategy",
    "pso": "Particle Swarm Optimization",
    "de": "Differential Evolution",
    "sa": "Simulated Annealing",
    "ga": "Genetic Algorithm",
    "cem": "Cross-Entropy Method",
}
DEFAULT_OPTIMIZER = "snes"

# File paths
DEFAULT_OUTPUT_DIR = "examples/output"        # Default output directory
DEFAULT_CHECKPOINT_DIR = "examples/checkpoints"  # Default checkpoint directory 