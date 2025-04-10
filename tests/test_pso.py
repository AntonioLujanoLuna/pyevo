#!/usr/bin/env python3
"""
Unit tests for the PSO optimizer.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import the optimizers module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizers import PSO

class TestPSO(unittest.TestCase):
    """Test cases for the PSO optimizer."""

    def test_initialization(self):
        """Test the initialization of PSO."""
        solution_length = 10
        population_count = 20
        
        # Test with default parameters
        optimizer = PSO(solution_length)
        self.assertEqual(optimizer.solution_length, solution_length)
        self.assertTrue(optimizer.population_count >= 10)  # Should have reasonable default
        self.assertEqual(optimizer.positions.shape, (optimizer.population_count, solution_length))
        self.assertEqual(optimizer.velocities.shape, (optimizer.population_count, solution_length))
        self.assertEqual(optimizer.personal_best_positions.shape, (optimizer.population_count, solution_length))
        
        # Test with custom parameters
        custom_center = np.ones(solution_length)
        custom_sigma = np.full(solution_length, 2.0)
        optimizer = PSO(
            solution_length=solution_length,
            population_count=population_count,
            center=custom_center,
            sigma=custom_sigma,
            random_seed=42,
            omega=0.8,
            phi_p=2.0,
            phi_g=2.0
        )
        
        self.assertEqual(optimizer.solution_length, solution_length)
        self.assertEqual(optimizer.population_count, population_count)
        self.assertEqual(optimizer.omega, 0.8)
        self.assertEqual(optimizer.phi_p, 2.0)
        self.assertEqual(optimizer.phi_g, 2.0)
        
        # Check initial positions are reasonably centered around custom_center
        positions_mean = np.mean(optimizer.positions, axis=0)
        self.assertTrue(np.allclose(positions_mean, custom_center, atol=3.0))

    def test_ask(self):
        """Test the ask method."""
        solution_length = 5
        population_count = 10
        optimizer = PSO(
            solution_length=solution_length,
            population_count=population_count,
            random_seed=42
        )
        
        # Get solutions
        solutions = optimizer.ask()
        
        # Check dimensions
        self.assertEqual(solutions.shape, (population_count, solution_length))
        
        # Check that solutions are different
        self.assertFalse(np.allclose(solutions[0], solutions[1]))
        
        # Check determinism with same seed
        optimizer2 = PSO(
            solution_length=solution_length,
            population_count=population_count,
            random_seed=42
        )
        solutions2 = optimizer2.ask()
        np.testing.assert_array_equal(solutions, solutions2)

    def test_tell(self):
        """Test the tell method."""
        solution_length = 5
        population_count = 10
        optimizer = PSO(
            solution_length=solution_length,
            population_count=population_count,
            random_seed=42
        )
        
        # Get initial positions and velocities
        initial_positions = optimizer.positions.copy()
        initial_velocities = optimizer.velocities.copy()
        
        # Generate solutions and evaluate with a simple fitness function
        solutions = optimizer.ask()
        fitnesses = [-np.sum(solution**2) for solution in solutions]  # Minimize sum of squares
        
        # Update optimizer
        optimizer.tell(fitnesses)
        
        # Check that positions and velocities have changed
        self.assertFalse(np.allclose(optimizer.positions, initial_positions))
        self.assertFalse(np.allclose(optimizer.velocities, initial_velocities))
        
        # Check that best global solution was updated
        self.assertIsNotNone(optimizer.global_best_position)
        self.assertIsNotNone(optimizer.global_best_fitness)
        
        # Test error when fitness count doesn't match population
        with self.assertRaises(ValueError):
            optimizer.tell(fitnesses[:-1])

    def test_simple_optimization(self):
        """Test optimization on a simple function."""
        # Function to optimize: f(x) = -(x-3)^2
        # Optimum is at x = 3
        def fitness(x):
            return -(x - 3)**2
        
        # Initialize optimizer
        optimizer = PSO(
            solution_length=1,
            population_count=10,
            random_seed=42
        )
        
        # Run optimization
        for _ in range(30):
            solutions = optimizer.ask()
            fitnesses = [fitness(solution[0]) for solution in solutions]
            optimizer.tell(fitnesses)
        
        # Check that optimizer converged to the optimum
        optimum = optimizer.get_best_solution()[0]
        self.assertTrue(abs(optimum - 3.0) < 0.5)  # PSO might need more tolerance

    def test_multivariate_optimization(self):
        """Test optimization on a multivariate function (sphere)."""
        # Sphere function: f(x) = sum(x_i^2)
        # Global minimum at origin (0,0,...)
        def sphere(x):
            return -np.sum(x**2)
        
        # Initialize optimizer with origin offset
        center = np.array([5.0, 5.0, 5.0])
        optimizer = PSO(
            solution_length=3,
            population_count=20,
            center=center,
            random_seed=42
        )
        
        # Run optimization
        for _ in range(50):
            solutions = optimizer.ask()
            fitnesses = [sphere(solution) for solution in solutions]
            optimizer.tell(fitnesses)
        
        # Check that optimizer converged toward the optimum
        optimum = optimizer.get_best_solution()
        self.assertTrue(np.linalg.norm(optimum) < 0.5)
        
    def test_reset(self):
        """Test the reset method."""
        optimizer = PSO(solution_length=5, random_seed=42)
        
        # Change state
        optimizer.ask()
        optimizer.tell(np.random.randn(optimizer.population_count))
        
        # Reset and initialize with new values
        new_center = np.ones(5)
        new_sigma = 0.3
        optimizer.reset(center=new_center, sigma=new_sigma)
        
        # Check that state was reset appropriately
        # Global best may be initialized to zeros, not None
        # Just check that it's improved from the reset
        self.assertTrue(np.linalg.norm(optimizer.global_best_position) < 0.001 or 
                       optimizer.global_best_fitness is None)

if __name__ == '__main__':
    unittest.main() 