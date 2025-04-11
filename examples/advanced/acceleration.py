"""
Example demonstrating both GPU acceleration and parallel processing.

This example shows how to use the acceleration utilities for:
1. GPU acceleration via CuPy
2. Parallel processing on CPU
3. Combined acceleration
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Try importing directly, fall back to local import if not installed
try:
    from pyevo import (
        SNES, CMA_ES, PSO,
        optimize_with_acceleration,
        is_gpu_available,
        parallel_evaluate,
        batch_process
    )
except ImportError:
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pyevo import (
        SNES, CMA_ES, PSO,
        optimize_with_acceleration,
        is_gpu_available,
        parallel_evaluate,
        batch_process
    )

# Benchmark problem: Rastrigin function
def rastrigin(x, A=10.0):
    """
    Rastrigin function - a more challenging benchmark with many local optima.
    We use the negative value since our optimizers maximize.
    
    f(x) = -A*n - sum_i [x_i^2 - A*cos(2π*x_i)]
    
    Global minimum at x = 0 with f(x) = 0
    """
    n = len(x)
    return -A*n - np.sum(x**2 - A*np.cos(2*np.pi*x))

# Benchmark problem: Rosenbrock function
def rosenbrock(x):
    """
    Rosenbrock function - a challenging unimodal benchmark.
    We use the negative value since our optimizers maximize.
    
    f(x) = -sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    Global minimum at x = [1,...,1] with f(x) = 0
    """
    return -np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def run_benchmarks(dimensions=20, iterations=1000, population_size=100):
    """
    Run benchmarks comparing different acceleration methods.
    
    Args:
        dimensions: Number of dimensions for optimization
        iterations: Maximum number of iterations
        population_size: Population size
    """
    # Check if GPU is available
    gpu_available = is_gpu_available()
    print(f"GPU acceleration: {'Available' if gpu_available else 'Not available'}")
    print(f"CPU cores: {os.cpu_count()}")
    
    # Test different acceleration configurations
    configs = [
        {"name": "No acceleration", "use_gpu": False, "use_parallel": False},
        {"name": "Parallel CPU", "use_gpu": False, "use_parallel": True},
    ]
    
    if gpu_available:
        configs.append({"name": "GPU acceleration", "use_gpu": True, "use_parallel": False})
    
    # Choose a challenging problem
    print("\nOptimizing Rastrigin function (20D)")
    print(f"Dimensions: {dimensions}, Max iterations: {iterations}, Population: {population_size}\n")
    
    # Create optimizer
    optimizer = SNES(
        solution_length=dimensions,
        population_count=population_size,
        random_seed=42
    )
    
    results = {}
    
    # Run benchmarks for each configuration
    for config in configs:
        print(f"Testing {config['name']}...")
        
        # Reset optimizer for fair comparison
        optimizer.reset()
        
        # Time the optimization
        start_time = time.time()
        
        best_solution, best_fitness, stats = optimize_with_acceleration(
            optimizer=optimizer,
            fitness_func=rastrigin,
            max_iterations=iterations,
            use_gpu=config["use_gpu"],
            use_parallel=config["use_parallel"],
            callback=lambda opt, i, _: print(f"  Iteration {i}: Best fitness = {-opt.get_stats()['best_fitness']:.6f}") if i % 100 == 0 else None
        )
        
        elapsed_time = time.time() - start_time
        
        # Store results
        results[config["name"]] = {
            "time": elapsed_time,
            "best_fitness": -best_fitness,  # Convert back to minimization for reporting
            "iterations": len(stats["iterations"])
        }
        
        print(f"  Completed in {elapsed_time:.2f} seconds")
        print(f"  Best fitness: {-best_fitness:.6f}")
        print(f"  Speedup: {results['No acceleration']['time'] / elapsed_time:.2f}x (compared to no acceleration)")
        print()
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Bar chart of times
    bars = plt.bar(
        range(len(results)), 
        [results[config["name"]]["time"] for config in configs],
        color=['blue', 'green', 'red'][:len(configs)]
    )
    
    # Add time and speedup annotations
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{results[configs[i]['name']]['time']:.2f}s\n{results['No acceleration']['time'] / results[configs[i]['name']]['time']:.2f}x",
            ha='center',
            fontweight='bold'
        )
    
    plt.xticks(range(len(results)), [config["name"] for config in configs])
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison of Acceleration Methods')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('benchmark_results.png')
    print("Benchmark results saved to benchmark_results.png")
    
    # Show the plot if in interactive mode
    if plt.isinteractive():
        plt.show()

if __name__ == "__main__":
    run_benchmarks() 