#!/usr/bin/env python3
"""
Interactive Optimization Example

This example demonstrates how to use the interactive mode for SNES optimization,
which allows pausing/resuming and adjusting parameters during runtime.
"""

import numpy as np
import argparse
import time
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyevo.optimizers import SNES
from pyevo.utils.interactive import InteractiveOptimizer
from pyevo.utils.constants import (
    DEFAULT_POPULATION_SIZE, DEFAULT_ALPHA, DEFAULT_CHECKPOINT_DIR
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive SNES optimization example')
    parser.add_argument('--dimensions', type=int, default=20,
                      help='Number of dimensions in the problem')
    parser.add_argument('--population', type=int, default=DEFAULT_POPULATION_SIZE,
                      help=f'Population size (default: {DEFAULT_POPULATION_SIZE})')
    parser.add_argument('--max-iterations', type=int, default=1000,
                      help='Maximum number of iterations')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                      help=f'Initial learning rate (default: {DEFAULT_ALPHA})')
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                      help=f'Directory for checkpoints (default: {DEFAULT_CHECKPOINT_DIR})')
    parser.add_argument('--load', type=str, default=None,
                      help='Load a previous checkpoint file')
    parser.add_argument('--noise', type=float, default=0.0,
                      help='Add noise to fitness evaluation (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed')
    return parser.parse_args()

def rosenbrock(x):
    """
    Rosenbrock function (banana function) - a non-convex test function.
    Global minimum at (1,1,...,1) with value 0.
    
    Args:
        x: Input vector
        
    Returns:
        Negative of the Rosenbrock function value (for maximization)
    """
    x = np.asarray(x)
    return -np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def sphere(x):
    """
    Sphere function - a simple convex test function.
    Global minimum at (0,0,...,0) with value 0.
    
    Args:
        x: Input vector
        
    Returns:
        Negative of the sphere function value (for maximization)
    """
    return -np.sum(x**2)
    
def with_noise(func, noise_level):
    """
    Add random noise to a fitness function.
    
    Args:
        func: Original fitness function
        noise_level: Level of noise to add (0.0-1.0)
        
    Returns:
        Function with added noise
    """
    def noisy_func(x):
        base = func(x)
        noise = np.random.normal(0, noise_level * abs(base))
        return base + noise
    return noisy_func

def main():
    """Run the interactive optimization example."""
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create fitness function with optional noise
    fitness_func = rosenbrock
    if args.noise > 0:
        fitness_func = with_noise(fitness_func, args.noise)
    
    print("\n=== Interactive SNES Optimization Example ===")
    print("Problem: Rosenbrock function (banana function)")
    print(f"Dimensions: {args.dimensions}")
    print(f"Population size: {args.population}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Learning rate: {args.alpha}")
    print(f"Noise level: {args.noise}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Random seed: {args.seed}")
    print("\nAvailable commands during optimization:")
    print("  pause - Pause the optimization")
    print("  resume - Resume the optimization")
    print("  stop - Stop the optimization")
    print("  stats - Show detailed statistics")
    print("  save - Save a checkpoint")
    print("  alpha <value> - Adjust learning rate")
    print("  params - Show current parameters")
    print("  help - Show available commands")
    print("\nType commands in the terminal during optimization.\n")
    
    # Load from checkpoint or create new optimizer
    if args.load:
        print(f"Loading from checkpoint: {args.load}")
        interactive = InteractiveOptimizer.load_session(args.load, fitness_func)
    else:
        # Create new SNES optimizer
        optimizer = SNES(
            solution_length=args.dimensions,
            population_count=args.population,
            alpha=args.alpha,
            random_seed=args.seed
        )
        
        # Create interactive optimizer
        interactive = InteractiveOptimizer(
            optimizer=optimizer,
            fitness_function=fitness_func,
            max_iterations=args.max_iterations,
            checkpoint_dir=args.checkpoint_dir
        )
    
    # Start interactive optimization
    print("\nStarting optimization. Enter commands in the terminal.")
    print("=" * 50)
    best_solution, best_fitness = interactive.start()
    
    # Print final results
    print("\nOptimization complete!")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Best solution: {best_solution[:5]}... (showing first 5 dimensions)")
    
    # For Rosenbrock, the optimal solution is (1,1,...,1)
    if fitness_func == rosenbrock:
        optimal = np.ones_like(best_solution)
        distance = np.linalg.norm(best_solution - optimal)
        print(f"Distance from optimal solution: {distance:.6f}")
    
if __name__ == "__main__":
    main() 