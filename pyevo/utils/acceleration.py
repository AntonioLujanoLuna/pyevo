"""
Acceleration utilities for PyEvo optimizers.

This module provides acceleration capabilities for PyEvo optimizers:
1. GPU acceleration via CuPy (if available)
2. Parallel processing utilities
3. Memory management for GPU operations
"""

import os
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Tuple, Optional, Any, Union, TypeVar, Sequence

# Import Optimizer type only for type annotations while avoiding circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyevo.optimizers.base import Optimizer

# Type variable for array-like objects (can be numpy or cupy arrays)
ArrayLike = TypeVar('ArrayLike', bound=np.ndarray)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None  # type: ignore
    
def is_gpu_available() -> bool:
    """Check if GPU acceleration is available via CuPy."""
    return HAS_CUPY

def get_gpu_memory_info() -> Optional[Tuple[int, int]]:
    """
    Get GPU memory usage information.
    
    Returns:
        tuple: (free_bytes, total_bytes) or None if GPU not available
    """
    if not HAS_CUPY:
        return None
    
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return free_bytes, total_bytes
    except Exception:
        return None

def clear_gpu_memory() -> bool:
    """
    Release all unused GPU memory.
    
    Returns:
        bool: Whether the operation was successful
    """
    if not HAS_CUPY:
        return False
    
    try:
        cp.get_default_memory_pool().free_all_blocks()
        return True
    except Exception:
        return False

def get_array_module(use_gpu: bool = False) -> Any:
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

def to_device(array: Any, use_gpu: bool = False) -> Any:
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
    
def estimate_memory_usage(solutions: Any, element_size: int = 4) -> int:
    """
    Estimate memory usage for processing solutions.
    
    Args:
        solutions: Array of solutions
        element_size: Size of each element in bytes (default: 4 for float32)
        
    Returns:
        int: Estimated memory usage in bytes
    """
    if hasattr(solutions, 'nbytes'):
        return solutions.nbytes
    
    # Estimate based on shape and element size
    try:
        return int(np.prod(solutions.shape) * element_size)
    except AttributeError:
        # Fallback: assume a list or sequence
        return len(solutions) * len(solutions[0]) * element_size

def batch_process(
    fitness_func: Callable[[Any], Any],
    solutions: Any,
    batch_size: Optional[int] = None,
    use_gpu: bool = False,
    **kwargs: Any
) -> Any:
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
    
    # Auto-configure batch size if not provided
    if batch_size is None:
        if use_gpu and HAS_CUPY:
            # Check available GPU memory and adjust batch size accordingly
            mem_info = get_gpu_memory_info()
            if mem_info:
                free_bytes, total_bytes = mem_info
                estimated_bytes = estimate_memory_usage(solutions) * 3  # solutions, intermediate results, output
                
                # Use at most 75% of free memory
                max_safe_items = int(free_bytes * 0.75 / (estimated_bytes / len(solutions)))
                batch_size = min(32, max_safe_items, len(solutions))
            else:
                # Default if memory info not available
                batch_size = min(32, len(solutions))
        else:
            # Larger batches on CPU
            batch_size = min(128, len(solutions))
    
    # Process in batches
    n_solutions = len(solutions)
    solutions_device = None
    fitnesses = None
    
    try:
        # Move data to device
        solutions_device = to_device(solutions, use_gpu)
        fitnesses = xp.zeros(n_solutions, dtype=xp.float32)
        
        for i in range(0, n_solutions, batch_size):
            batch_end = min(i + batch_size, n_solutions)
            batch = solutions_device[i:batch_end]
            fitnesses[i:batch_end] = fitness_func(batch, **kwargs)
            
            # Clear intermediate results if using GPU
            if use_gpu and HAS_CUPY and i % (batch_size * 10) == 0:
                clear_gpu_memory()
        
        # Ensure result is on CPU for optimizer
        return to_device(fitnesses, use_gpu=False)
    finally:
        # Clean up GPU memory
        if use_gpu and HAS_CUPY:
            # Force releasing references to GPU arrays
            solutions_device = None
            fitnesses = None
            clear_gpu_memory()

def parallel_evaluate(
    fitness_func: Callable[[Any], Any],
    solutions: Any,
    max_workers: Optional[int] = None,
    **kwargs: Any
) -> List[Any]:
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

def optimize_with_acceleration(
    optimizer: 'Optimizer',
    fitness_func: Callable[[Any], Any],
    max_iterations: int = 100,
    use_gpu: bool = False,
    use_parallel: bool = False,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    callback: Optional[Callable[['Optimizer', int, float], None]] = None,
    checkpoint_freq: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[Any, float, Dict[str, List[Any]]]:
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
        checkpoint_freq: How often to save checkpoints (iterations)
        checkpoint_path: Directory to save checkpoints
        **kwargs: Additional arguments to pass to fitness_func
        
    Returns:
        tuple: (best_solution, best_fitness, stats)
    """
    stats: Dict[str, List[Any]] = {
        'iterations': [],
        'best_fitness': [],
        'improvement': [],
        'time_per_iteration': [],
        'memory_usage': []
    }
    
    start_time = None
    if HAS_CUPY:
        import time
        start_time = time.time()
    
    for iteration in range(max_iterations):
        # Generate solutions
        solutions = optimizer.ask()
        
        # Evaluate fitness
        if use_gpu and HAS_CUPY:
            # Track memory before GPU operation
            mem_info = get_gpu_memory_info()
            if mem_info:
                stats['memory_usage'].append(mem_info)
                
            # Use GPU acceleration
            fitnesses = batch_process(fitness_func, solutions, batch_size, use_gpu, **kwargs)
            
            # Explicitly clean up
            clear_gpu_memory()
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
        
        # Track timing if available
        if start_time is not None:
            import time
            current_time = time.time()
            stats['time_per_iteration'].append(current_time - start_time)
            start_time = current_time
        
        # Call callback if provided
        if callback:
            callback(optimizer, iteration, improvement)
        
        # Save checkpoint if requested
        if checkpoint_freq and checkpoint_path and iteration % checkpoint_freq == 0:
            save_checkpoint(optimizer, stats, f"{checkpoint_path}_{iteration}.npz")
        
        # Early stopping check
        if improvement < 1e-8:
            break
    
    # Final checkpoint
    if checkpoint_freq and checkpoint_path:
        save_checkpoint(optimizer, stats, f"{checkpoint_path}_final.npz")
    
    return optimizer.get_best_solution(), stats['best_fitness'][-1], stats

def save_checkpoint(optimizer: 'Optimizer', session_info: dict, filepath: str) -> bool:
    """
    Save optimizer state and session info in a single file.
    
    Args:
        optimizer: PyEvo optimizer instance
        session_info: Dictionary of session information
        filepath: Path to save the checkpoint
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Get optimizer state
        optimizer_state = {}
        for key, value in optimizer.__dict__.items():
            if isinstance(value, (np.ndarray, int, float, bool, str, list, dict)) or value is None:
                optimizer_state[key] = value
        
        # Convert session info to a serializable format
        serializable_info = {}
        for key, value in session_info.items():
            if isinstance(value, list) and all(isinstance(x, (int, float, bool, str)) for x in value):
                serializable_info[key] = value
            elif isinstance(value, (int, float, bool, str, dict)):
                serializable_info[key] = value
        
        # Save to file
        np.savez(filepath, optimizer_state=optimizer_state, session_info=serializable_info)
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False

def load_checkpoint(filepath: str) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Load optimizer state and session info from a checkpoint file.
    
    Args:
        filepath: Path to the checkpoint file
        
    Returns:
        tuple: (optimizer_state, session_info) or (None, None) if failed
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        optimizer_state = data['optimizer_state'].item()
        session_info = data['session_info'].item()
        return optimizer_state, session_info
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None 