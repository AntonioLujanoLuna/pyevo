"""
Example demonstrating the standardized import pattern for PyEvo.

This example shows how to use PyEvo with the recommended import pattern,
which ensures the library can be used whether installed or not.
"""

# Standard libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Try importing from the installed package
try:
    from pyevo import SNES, CMA_ES, PSO, optimize_with_acceleration
except ImportError:
    # If not installed, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pyevo import SNES, CMA_ES, PSO, optimize_with_acceleration

# Simple objective function (Rastrigin function)
def rastrigin(x):
    """
    Rastrigin function - a non-convex function used as a performance test problem
    for optimization algorithms.
    
    It has many local minima but only one global minimum at x = 0.
    For optimization, we negate the result since PyEvo maximizes fitness.
    """
    A = 10
    n = len(x)
    return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def main():
    """Run optimization with different algorithms and compare results."""
    dimensions = 10
    max_iterations = 100
    
    optimizers = {
        "SNES": SNES(solution_length=dimensions, random_seed=42),
        "CMA-ES": CMA_ES(solution_length=dimensions, random_seed=42),
        "PSO": PSO(solution_length=dimensions, random_seed=42)
    }
    
    results = {}
    
    # Run each optimizer
    for name, optimizer in optimizers.items():
        print(f"\nRunning {name}...")
        best_solution, best_fitness, stats = optimize_with_acceleration(
            optimizer=optimizer,
            fitness_func=rastrigin,
            max_iterations=max_iterations
        )
        
        results[name] = {
            "best_solution": best_solution,
            "best_fitness": best_fitness,
            "stats": stats
        }
        
        print(f"{name} - Best fitness: {-best_fitness:.6f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(
            result["stats"]["iterations"], 
            [-f for f in result["stats"]["best_fitness"]], 
            label=name
        )
    
    plt.xlabel("Iterations")
    plt.ylabel("Fitness (lower is better)")
    plt.title("Optimization Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    
    # Save the plot
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/optimization_comparison.png")
    plt.close()
    
    print("\nOptimization complete. Results saved to output/optimization_comparison.png")

if __name__ == "__main__":
    main() 