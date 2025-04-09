#!/usr/bin/env python3
"""
Unit tests for the SNES optimizer.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import the SNES module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from snes import SNES, get_default_population_count

class TestSNES(unittest.TestCase):
    """Test cases for the SNES optimizer."""

    def test_default_population_count(self):
        """Test the default population count calculation."""
        self.assertEqual(get_default_population_count(10), 4 + int(3 * np.log(10)))
        self.assertEqual(get_default_population_count(100), 4 + int(3 * np.log(100)))
        self.assertEqual(get_default_population_count(1000), 4 + int(3 * np.log(1000)))

    def test_initialization(self):
        """Test the initialization of SNES."""
        solution_length = 10
        population_count = 20
        alpha = 0.1
        
        # Test with default parameters
        optimizer = SNES(solution_length)
        self.assertEqual(optimizer.solution_length, solution_length)
        self.assertEqual(optimizer.population_count, get_default_population_count(solution_length))
        self.assertEqual(optimizer.center.shape, (solution_length,))
        self.assertEqual(optimizer.sigma.shape, (solution_length,))
        
        # Test with custom parameters
        custom_center = np.ones(solution_length)
        custom_sigma = np.full(solution_length, 2.0)
        optimizer = SNES(
            solution_length=solution_length,
            population_count=population_count,
            alpha=alpha,
            center=custom_center,
            sigma=custom_sigma,
            random_seed=42
        )
        
        self.assertEqual(optimizer.solution_length, solution_length)
        self.assertEqual(optimizer.population_count, population_count)
        np.testing.assert_array_equal(optimizer.center, custom_center)
        np.testing.assert_array_equal(optimizer.sigma, custom_sigma * alpha)

    def test_ask(self):
        """Test the ask method."""
        solution_length = 5
        population_count = 10
        optimizer = SNES(
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
        optimizer2 = SNES(
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
        optimizer = SNES(
            solution_length=solution_length,
            population_count=population_count,
            random_seed=42
        )
        
        # Get initial center and sigma
        initial_center = optimizer.center.copy()
        initial_sigma = optimizer.sigma.copy()
        
        # Generate solutions and evaluate with a simple fitness function
        solutions = optimizer.ask()
        fitnesses = [-np.sum(solution**2) for solution in solutions]  # Minimize sum of squares
        
        # Update optimizer
        optimizer.tell(fitnesses)
        
        # Check that center and sigma have changed
        self.assertFalse(np.allclose(optimizer.center, initial_center))
        self.assertFalse(np.allclose(optimizer.sigma, initial_sigma))
        
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
        optimizer = SNES(
            solution_length=1,
            population_count=10,
            alpha=0.5,
            random_seed=42
        )
        
        # Run optimization
        for _ in range(30):
            solutions = optimizer.ask()
            fitnesses = [fitness(solution[0]) for solution in solutions]
            optimizer.tell(fitnesses)
        
        # Check that optimizer converged to the optimum
        optimum = optimizer.get_best_solution()[0]
        self.assertTrue(abs(optimum - 3.0) < 0.1)

if __name__ == '__main__':
    unittest.main()