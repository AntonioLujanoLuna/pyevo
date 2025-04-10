"""
Utility functions for PyEvo optimizers.

This module provides common utilities for PyEvo optimizers, including:
1. GPU acceleration via CuPy (if available)
2. Parallel processing utilities
3. Common helper functions
"""

import os
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    
def is_gpu_available():
    """Check if GPU acceleration is available via CuPy."""
    return HAS_CUPY

def get_array_module(use_gpu=False):
    """
    Get the appropriate array module (NumPy or CuPy) based on availability and preference.
    
    Args:
        use_gpu (bool): Whether to use GPU acceleration if available
        
    Returns:
        module: NumPy or CuPy module
    """
    if use_gpu and HAS_CUPY:
        return cp
    return np

def to_device(array, use_gpu=False):
    """
    Move array to the appropriate device (CPU or GPU).
    
    Args:
        array: NumPy or CuPy array
        use_gpu (bool): Whether to use GPU acceleration if available
        
    Returns:
        array: Array on the appropriate device
    """
    if use_gpu and HAS_CUPY:
        if not isinstance(array, cp.ndarray):
            return cp.asarray(array)
        return array
    else:
        if HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    
def batch_process(fitness_func, solutions, batch_size=None, use_gpu=False, **kwargs):
    """
    Process solutions in batches for better performance, especially on GPU.
    
    Args:
        fitness_func: Function to evaluate fitness of solutions
        solutions: Array of solutions to evaluate
        batch_size: Size of batches (None for auto-configure)
        use_gpu: Whether to use GPU acceleration if available
        **kwargs: Additional arguments to pass to fitness_func
        
    Returns:
        array: Array of fitness values
    """
    # Move solutions to appropriate device
    xp = get_array_module(use_gpu)
    solutions_device = to_device(solutions, use_gpu)
    
    # Auto-configure batch size if not provided
    if batch_size is None:
        if use_gpu and HAS_CUPY:
            # Default to smaller batches on GPU to avoid memory issues
            batch_size = min(32, len(solutions))
        else:
            # Larger batches on CPU
            batch_size = min(128, len(solutions))
    
    # Process in batches
    n_solutions = len(solutions)
    fitnesses = xp.zeros(n_solutions, dtype=xp.float32)
    
    for i in range(0, n_solutions, batch_size):
        batch_end = min(i + batch_size, n_solutions)
        batch = solutions_device[i:batch_end]
        fitnesses[i:batch_end] = fitness_func(batch, **kwargs)
    
    # Ensure result is on CPU for optimizer
    return to_device(fitnesses, use_gpu=False)

def parallel_evaluate(fitness_func, solutions, max_workers=None, **kwargs):
    """
    Evaluate solutions in parallel using multiple CPU cores.
    
    Args:
        fitness_func: Function to evaluate fitness of a single solution
        solutions: Array of solutions to evaluate
        max_workers: Maximum number of worker processes (None for auto-configure)
        **kwargs: Additional arguments to pass to fitness_func
        
    Returns:
        list: List of fitness values
    """
    # Auto-configure number of workers if not provided
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)  # Leave one core free
    
    # Create a wrapper that passes kwargs to fitness_func
    def evaluate_solution(solution):
        return fitness_func(solution, **kwargs)
    
    # Use ProcessPoolExecutor for parallel evaluation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        fitnesses = list(executor.map(evaluate_solution, solutions))
    
    return fitnesses

def optimize_with_acceleration(optimizer, fitness_func, max_iterations=100, 
                               use_gpu=False, use_parallel=False, max_workers=None, 
                               batch_size=None, callback=None, **kwargs):
    """
    Run optimization with GPU acceleration and/or parallel processing.
    
    Args:
        optimizer: PyEvo optimizer instance
        fitness_func: Function to evaluate fitness of solutions
        max_iterations: Maximum number of iterations
        use_gpu: Whether to use GPU acceleration if available
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of worker processes for parallel evaluation
        batch_size: Batch size for GPU processing
        callback: Optional callback function called after each iteration
        **kwargs: Additional arguments to pass to fitness_func
        
    Returns:
        tuple: (best_solution, best_fitness, stats)
    """
    stats = {
        'iterations': [],
        'best_fitness': [],
        'improvement': []
    }
    
    for iteration in range(max_iterations):
        # Generate solutions
        solutions = optimizer.ask()
        
        # Evaluate fitness
        if use_gpu and HAS_CUPY:
            # Use GPU acceleration
            fitnesses = batch_process(fitness_func, solutions, batch_size, use_gpu, **kwargs)
        elif use_parallel:
            # Use parallel CPU processing
            fitnesses = parallel_evaluate(fitness_func, solutions, max_workers, **kwargs)
        else:
            # Standard single-threaded evaluation
            fitnesses = [fitness_func(solution, **kwargs) for solution in solutions]
        
        # Update optimizer
        improvement = optimizer.tell(fitnesses)
        
        # Update stats
        stats['iterations'].append(iteration)
        stats['best_fitness'].append(optimizer.get_stats().get('best_fitness', 0))
        stats['improvement'].append(improvement)
        
        # Call callback if provided
        if callback:
            callback(optimizer, iteration, improvement)
        
        # Early stopping check
        if improvement < 1e-8:
            break
    
    return optimizer.get_best_solution(), stats['best_fitness'][-1], stats 