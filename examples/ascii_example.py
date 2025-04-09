#!/usr/bin/env python3
"""
ASCII Evolution Example

This example demonstrates evolving ASCII characters to match a target string.
"""

import numpy as np
import argparse
import time
import sys
import os

# Add parent directory to path to import the SNES module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from snes import SNES

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ASCII evolution using SNES')
    parser.add_argument('--target', type=str, default='Hello world! Some target text.',
                        help='Target string to evolve towards')
    parser.add_argument('--population', type=int, default=64,
                        help='Population size')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of generations to evolve')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    return parser.parse_args()

def fitness(params, target):
    """Calculate negative squared error between parameters and target ASCII codes.
    
    Args:
        params: Array of parameter values
        target: Target string
        
    Returns:
        Negative squared error (higher is better)
    """
    target_codes = np.array([ord(c) for c in target])
    diff = params - target_codes
    return -np.sum(diff * diff)

def params_to_text(params):
    """Convert parameter values to ASCII text.
    
    Args:
        params: Array of parameter values
        
    Returns:
        String representation
    """
    min_ascii = 32   # space
    max_ascii = 126  # tilde
    ascii_values = np.clip(np.round(params), min_ascii, max_ascii).astype(int)
    return ''.join([chr(v) for v in ascii_values])

def main():
    """Run the ASCII evolution example."""
    args = parse_args()
    
    # Setup problem parameters
    target = args.target
    solution_length = len(target)
    
    # ASCII bounds
    min_ascii = 32   # space
    max_ascii = 126  # tilde
    mid_ascii = (min_ascii + max_ascii) / 2
    ascii_range = max_ascii - min_ascii
    
    # SNES parameters
    population_count = args.population
    alpha = args.alpha
    center = np.full(solution_length, mid_ascii, dtype=np.float32)
    sigma = np.full(solution_length, ascii_range, dtype=np.float32)
    epochs = args.epochs
    
    print(f"Target: '{target}'")
    print(f"Population size: {population_count}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate (alpha): {alpha}")
    print(f"Random seed: {args.seed}")
    print("\nEvolution progress:")
    print("-" * 50)
    
    # Initialize optimizer
    optimizer = SNES(
        solution_length=solution_length,
        population_count=population_count,
        alpha=alpha,
        center=center,
        sigma=sigma,
        random_seed=args.seed
    )
    
    start_time = time.time()
    
    # Run optimization
    for i in range(epochs):
        # Generate solutions
        solutions = optimizer.ask()
        
        # Evaluate fitness
        fitnesses = np.array([fitness(solution, target) for solution in solutions])
        
        # Update optimizer
        optimizer.tell(fitnesses)
        
        # Print progress
        if i == epochs - 1 or i % 10 == 0:
            best_params = optimizer.get_best_solution()
            best_text = params_to_text(best_params)
            print(f"Epoch {i:03d}: {best_text}")
    
    end_time = time.time()
    
    # Print final results
    best_params = optimizer.get_best_solution()
    best_text = params_to_text(best_params)
    
    print("-" * 50)
    print(f"Target:    '{target}'")
    print(f"Final:     '{best_text}'")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Calculate error
    target_codes = np.array([ord(c) for c in target])
    final_codes = np.clip(np.round(best_params), min_ascii, max_ascii).astype(int)
    error = np.sum((final_codes - target_codes) ** 2)
    
    print(f"Final error: {error}")
    
    # Calculate character-wise accuracy
    matches = sum(1 for a, b in zip(best_text, target) if a == b)
    accuracy = matches / len(target) * 100
    print(f"Character accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()