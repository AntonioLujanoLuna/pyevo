"""
Tests for acceleration utilities.

These tests verify that the acceleration utilities (GPU and parallel) work correctly.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyevo import (
    SNES,
    is_gpu_available,
    get_gpu_memory_info,
    clear_gpu_memory,
    get_array_module,
    to_device,
    batch_process,
    parallel_evaluate,
    optimize_with_acceleration
)

class TestAcceleration(unittest.TestCase):
    """Test acceleration utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple objective function (sphere function)
        self.objective = lambda x: -np.sum(x**2)  # Negative because optimizers maximize
        
        # Create a test optimizer
        self.optimizer = SNES(solution_length=10, random_seed=42)
        
        # Check if GPU is available
        self.has_gpu = is_gpu_available()
        
    def test_gpu_detection(self):
        """Test GPU detection."""
        # This just verifies that is_gpu_available() runs without error
        result = is_gpu_available()
        self.assertIsInstance(result, bool)
        
    def test_array_module(self):
        """Test get_array_module function."""
        # Should always return numpy if use_gpu=False
        module = get_array_module(use_gpu=False)
        self.assertEqual(module.__name__, 'numpy')
        
        # If GPU is available and use_gpu=True, should return cupy
        if self.has_gpu:
            module = get_array_module(use_gpu=True)
            self.assertEqual(module.__name__, 'cupy')
    
    def test_device_transfer(self):
        """Test to_device function."""
        # Create a test array
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        # Should return the same array if not using GPU
        result = to_device(array, use_gpu=False)
        self.assertIs(type(result), np.ndarray)
        np.testing.assert_array_equal(result, array)
        
        # If GPU is available, test GPU transfer
        if self.has_gpu:
            import cupy as cp
            result = to_device(array, use_gpu=True)
            self.assertIs(type(result), cp.ndarray)
            # Transfer back to CPU for comparison
            result_cpu = to_device(result, use_gpu=False)
            np.testing.assert_array_equal(result_cpu, array)
    
    def test_batch_processing(self):
        """Test batch_process function with sphere function."""
        # Create test solutions
        solutions = np.random.randn(10, 5).astype(np.float32)
        
        # Simple fitness function
        def fitness_func(x):
            return np.sum(x**2, axis=1)
        
        # Test on CPU
        result = batch_process(fitness_func, solutions, batch_size=2, use_gpu=False)
        self.assertEqual(len(result), len(solutions))
        
        # Test expected values
        expected = np.array([np.sum(x**2) for x in solutions])
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # If GPU is available, test on GPU
        if self.has_gpu:
            result_gpu = batch_process(fitness_func, solutions, batch_size=2, use_gpu=True)
            self.assertEqual(len(result_gpu), len(solutions))
            np.testing.assert_allclose(result_gpu, expected, rtol=1e-5)
    
    def test_parallel_evaluation(self):
        """Test parallel_evaluate function."""
        # Create test solutions
        solutions = np.random.randn(10, 5).astype(np.float32)
        
        # Simple fitness function for a single solution
        def fitness_func(x):
            return np.sum(x**2)
        
        # Test parallel evaluation
        result = parallel_evaluate(fitness_func, solutions, max_workers=2)
        self.assertEqual(len(result), len(solutions))
        
        # Test expected values
        expected = [np.sum(x**2) for x in solutions]
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_optimize_with_acceleration(self):
        """Test optimize_with_acceleration function."""
        # Run optimization with CPU
        best_solution, best_fitness, stats = optimize_with_acceleration(
            optimizer=self.optimizer,
            fitness_func=self.objective,
            max_iterations=10,
            use_gpu=False,
            use_parallel=False
        )
        
        # Check that we got results
        self.assertEqual(len(stats['iterations']), 10)
        self.assertLessEqual(best_fitness, 0)  # Should be negative (maximizing -x^2)
        
        # Test with parallel processing
        self.optimizer.reset()
        best_solution_parallel, best_fitness_parallel, stats_parallel = optimize_with_acceleration(
            optimizer=self.optimizer,
            fitness_func=self.objective,
            max_iterations=10,
            use_gpu=False,
            use_parallel=True,
            max_workers=2
        )
        
        # Should get similar results
        self.assertEqual(len(stats_parallel['iterations']), 10)
        
        # If GPU is available, test with GPU
        if self.has_gpu:
            self.optimizer.reset()
            best_solution_gpu, best_fitness_gpu, stats_gpu = optimize_with_acceleration(
                optimizer=self.optimizer,
                fitness_func=self.objective,
                max_iterations=10,
                use_gpu=True,
                use_parallel=False
            )
            
            # Should get similar results
            self.assertEqual(len(stats_gpu['iterations']), 10)
    
    def test_gpu_memory_management(self):
        """Test GPU memory management functions."""
        if not self.has_gpu:
            self.skipTest("GPU not available")
            
        # Get initial memory info
        mem_info = get_gpu_memory_info()
        self.assertIsNotNone(mem_info)
        self.assertEqual(len(mem_info), 2)  # (free_bytes, total_bytes)
        
        # Clear memory
        result = clear_gpu_memory()
        self.assertTrue(result)
        
        # Check that memory was freed
        new_mem_info = get_gpu_memory_info()
        self.assertIsNotNone(new_mem_info)
        
        # Free memory should be at least as much as before
        self.assertGreaterEqual(new_mem_info[0], mem_info[0])

if __name__ == "__main__":
    unittest.main() 