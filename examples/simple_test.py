"""
Simple test script to verify the optimization algorithms work correctly.
"""

import os
import sys
import numpy as np

# Add parent directory to path to import pyevo package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyevo.optimizers import SNES, CMA_ES, PSO

def simple_objective(x):
    """Simple objective function (sphere function)"""
    return -np.sum(x**2)  # Negative because optimizers maximize

def test_optimizer(optimizer_name, dimensions=5, max_iterations=50):
    """Test a specific optimizer on the sphere function."""
    print(f"\n--- Testing {optimizer_name} optimizer ---")
    
    # Create optimizer based on name
    if optimizer_name == "snes":
        optimizer = SNES(solution_length=dimensions, random_seed=42)
    elif optimizer_name == "cmaes":
        optimizer = CMA_ES(solution_length=dimensions, random_seed=42)
    elif optimizer_name == "pso":
        optimizer = PSO(solution_length=dimensions, random_seed=42)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_name}")
    
    # Run optimization
    for i in range(max_iterations):
        # Generate solutions
        solutions = optimizer.ask()
        
        # Evaluate fitness
        fitnesses = [simple_objective(solution) for solution in solutions]
        
        # Update optimizer
        optimizer.tell(fitnesses)
        
        # Print progress every 10 iterations
        if i % 10 == 0 or i == max_iterations - 1:
            best_solution = optimizer.get_best_solution()
            best_fitness = simple_objective(best_solution)
            print(f"Iteration {i:3d}: Best fitness = {-best_fitness:.6f}")
    
    print(f"Final solution: {best_solution[:2]}... (showing first 2 values)")
    print(f"Final fitness: {-best_fitness:.6f} (closer to 0 is better)")
    print("-" * 50)

def main():
    """Test all optimizers."""
    dimensions = 5
    max_iterations = 50
    
    # Test each optimizer
    test_optimizer("snes", dimensions, max_iterations)
    test_optimizer("cmaes", dimensions, max_iterations)
    test_optimizer("pso", dimensions, max_iterations)

if __name__ == "__main__":
    main() 