"""
Simple test script to verify the optimization algorithms work correctly.
"""

import numpy as np
from utils import create_optimizer

def simple_objective(x):
    """Simple objective function (sphere function)"""
    return -np.sum(x**2)  # Negative because optimizers maximize

def test_optimizer(optimizer_name, dimensions=5, max_iterations=50):
    """Test a specific optimizer on the sphere function."""
    print(f"\n--- Testing {optimizer_name} optimizer ---")
    
    # Create optimizer
    optimizer = create_optimizer(
        optimizer_type=optimizer_name,
        solution_length=dimensions,
        random_seed=42
    )
    
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