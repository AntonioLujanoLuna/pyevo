"""
Example demonstrating all available optimizers in PyEvo on the Ackley function.

This example compares the performance of all optimization algorithms on the
Ackley function, a challenging multimodal benchmark optimization problem.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Try importing from the installed package
try:
    from pyevo import (
        SNES, CMA_ES, PSO, DE, SimulatedAnnealing, 
        GeneticAlgorithm, CrossEntropyMethod,
        optimize_with_acceleration
    )
except ImportError:
    # If not installed, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pyevo import (
        SNES, CMA_ES, PSO, DE, SimulatedAnnealing, 
        GeneticAlgorithm, CrossEntropyMethod,
        optimize_with_acceleration
    )

def ackley(x):
    """
    Ackley function - a widely used multimodal test function.
    
    It has many local minima but only one global minimum at x = 0.
    The function is characterized by an almost flat outer region and 
    a large hole at the center.
    
    For optimization, we negate the result since PyEvo maximizes fitness.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    
    return -(term1 + term2 + a + np.exp(1))

def run_optimizer(name, optimizer, dimensions, max_iterations):
    """Run an optimizer on the Ackley function with timing."""
    print(f"\nRunning {name}...")
    start_time = time()
    
    best_solution, best_fitness, stats = optimize_with_acceleration(
        optimizer=optimizer,
        fitness_func=ackley,
        max_iterations=max_iterations,
        callback=lambda opt, iter, imp: print(f"  Iteration {iter:3d}: Best fitness = {-opt.get_stats().get('best_fitness', 0):.6f}") 
                                     if iter % 20 == 0 or iter == max_iterations - 1 else None
    )
    
    elapsed_time = time() - start_time
    
    print(f"{name} - Best solution norm: {np.linalg.norm(best_solution):.6f}")
    print(f"{name} - Best fitness: {-best_fitness:.6f} (closer to 0 is better)")
    print(f"{name} - Time elapsed: {elapsed_time:.2f} seconds")
    
    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "stats": stats,
        "time": elapsed_time
    }

def main():
    """Run all optimizers and compare results."""
    # Problem setup
    dimensions = 10
    max_iterations = 150
    bounds = np.array([[-5.0] * dimensions, [5.0] * dimensions])
    
    # Create optimizers
    optimizers = {
        "SNES": SNES(
            solution_length=dimensions,
            random_seed=42
        ),
        "CMA-ES": CMA_ES(
            solution_length=dimensions,
            random_seed=42
        ),
        "PSO": PSO(
            solution_length=dimensions,
            bounds=bounds, 
            random_seed=42
        ),
        "DE": DE(
            solution_length=dimensions, 
            population_count=30,
            f=0.8,
            cr=0.9,
            bounds=bounds,
            random_seed=42
        ),
        "SimulatedAnnealing": SimulatedAnnealing(
            solution_length=dimensions,
            initial_temp=10.0,
            cooling_factor=0.97,
            step_size=0.3,
            bounds=bounds,
            random_seed=42
        ),
        "GeneticAlgorithm": GeneticAlgorithm(
            solution_length=dimensions,
            population_count=50,
            crossover_prob=0.8,
            mutation_prob=0.1,
            bounds=bounds,
            random_seed=42
        ),
        "CrossEntropyMethod": CrossEntropyMethod(
            solution_length=dimensions,
            population_count=50,
            elite_ratio=0.2,
            alpha=0.7,
            bounds=bounds,
            random_seed=42
        )
    }
    
    # Run optimizers
    results = {}
    for name, optimizer in optimizers.items():
        results[name] = run_optimizer(name, optimizer, dimensions, max_iterations)
    
    # Plot convergence
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        # Convert to log scale for better visualization
        fitness_values = [-f for f in result["stats"]["best_fitness"]]
        plt.plot(
            result["stats"]["iterations"],
            fitness_values,
            label=f"{name} (final = {fitness_values[-1]:.2e})"
        )
    
    plt.xlabel("Iterations")
    plt.ylabel("Fitness (log scale, lower is better)")
    plt.title("Optimizer Comparison on Ackley Function")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    
    # Save the plot
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/all_optimizers_comparison.png")
    plt.close()
    
    # Create performance summary table
    print("\nPerformance Summary:")
    print("-" * 85)
    print(f"{'Optimizer':<20} | {'Final Fitness':<15} | {'Solution Norm':<15} | {'Time (s)':<10} | {'Iterations':<10}")
    print("-" * 85)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]["best_fitness"]):
        final_fitness = -result["best_fitness"]
        solution_norm = np.linalg.norm(result["best_solution"])
        iterations = len(result["stats"]["iterations"])
        print(f"{name:<20} | {final_fitness:<15.6f} | {solution_norm:<15.6f} | {result['time']:<10.2f} | {iterations:<10}")
    
    print("-" * 85)
    print("\nOptimization complete. Results saved to output/all_optimizers_comparison.png")

if __name__ == "__main__":
    main() 