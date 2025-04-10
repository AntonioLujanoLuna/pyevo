"""
Simple test script to verify the optimization algorithms work correctly.
"""

import os
import sys
import numpy as np

# Import from the installed package
try:
    # Try importing from installed package
    from pyevo import SNES, CMA_ES, PSO, optimize_with_acceleration, is_gpu_available
except ImportError:
    # If not installed, add parent directory to path 
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pyevo import SNES, CMA_ES, PSO, optimize_with_acceleration, is_gpu_available

def simple_objective(x):
    """Simple objective function (sphere function)"""
    return -np.sum(x**2)  # Negative because optimizers maximize

def test_optimizer(optimizer_name, dimensions=5, max_iterations=50, use_gpu=False, use_parallel=False):
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
    
    # Run optimization with acceleration utilities
    best_solution, best_fitness, stats = optimize_with_acceleration(
        optimizer=optimizer,
        fitness_func=simple_objective,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        use_parallel=use_parallel,
        callback=lambda opt, iter, imp: print(f"Iteration {iter:3d}: Best fitness = {-opt.get_stats()['best_fitness']:.6f}") if iter % 10 == 0 or iter == max_iterations - 1 else None
    )
    
    print(f"Final solution: {best_solution[:2]}... (showing first 2 values)")
    print(f"Final fitness: {-best_fitness:.6f} (closer to 0 is better)")
    print("-" * 50)

def main():
    """Test all optimizers."""
    dimensions = 5
    max_iterations = 50
    
    # Check if GPU is available
    gpu_available = is_gpu_available()
    if gpu_available:
        print("GPU acceleration is available")
    else:
        print("GPU acceleration is not available, using CPU only")
    
    use_gpu = False  # Set to True to use GPU if available
    use_parallel = True  # Set to True to use parallel processing
    
    # Test each optimizer
    test_optimizer("snes", dimensions, max_iterations, use_gpu, use_parallel)
    test_optimizer("cmaes", dimensions, max_iterations, use_gpu, use_parallel)
    test_optimizer("pso", dimensions, max_iterations, use_gpu, use_parallel)

if __name__ == "__main__":
    main() 