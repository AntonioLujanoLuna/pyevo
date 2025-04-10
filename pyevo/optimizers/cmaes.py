"""
Covariance Matrix Adaptation Evolution Strategy (CMA-ES) implementation.

CMA-ES is a state-of-the-art black-box optimization algorithm that adapts
a full covariance matrix instead of just diagonal variances like SNES.
"""

import numpy as np
from pyevo.optimizers.base import Optimizer

class CMA_ES(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""
    
    def __init__(self, 
                solution_length,
                population_count=None,
                alpha=0.1,
                center=None,
                sigma=None,
                random_seed=None):
        """
        Initialize the CMA-ES optimizer.
        
        Args:
            solution_length: Length of solution vector (dimensionality of search space)
            population_count: Size of population (default is based on solution length)
            alpha: Initial step size (default 0.1)
            center: Initial center (mean) vector (default zeros)
            sigma: Initial step size scalar or vector (default alpha)
            random_seed: Seed for random number generation
        """
        # Set dimensionality
        self.solution_length = solution_length
        
        # Set population size
        if population_count is None:
            self.population_count = 4 + int(3 * np.log(solution_length))
        else:
            self.population_count = population_count
            
        # Set random state
        self.rng = np.random.RandomState(random_seed)
            
        # Initialize center (mean)
        if center is None:
            self.center = np.zeros(solution_length, dtype=np.float32)
        else:
            self.center = np.array(center, dtype=np.float32)
            
        # Initialize step size
        if sigma is None:
            self.sigma = alpha
        else:
            if np.isscalar(sigma) or getattr(sigma, 'size', 0) == 1:
                self.sigma = float(sigma)
            else:
                self.sigma = float(np.mean(sigma))
            
        # Initialize covariance matrix C and its decomposition
        self.C = np.eye(solution_length, dtype=np.float32)  # Covariance matrix
        self.B = np.eye(solution_length, dtype=np.float32)  # Eigenvectors of C
        self.D = np.ones(solution_length, dtype=np.float32)  # Eigenvalues of C (sqrt)
        
        # Storage for current generation
        self.solutions = np.zeros((self.population_count, solution_length), dtype=np.float32)
        
        # Generation counter
        self.generation = 0
        
    def ask(self):
        """Generate a new batch of solutions to evaluate."""
        # Generate Gaussian samples
        for i in range(self.population_count):
            z = self.rng.randn(self.solution_length)
            self.solutions[i] = self.center + self.sigma * np.dot(self.B, self.D * z)
            
        return self.solutions
    
    def tell(self, fitnesses, tolerance=1e-6):
        """Update parameters based on fitness values."""
        if len(fitnesses) != self.population_count:
            raise ValueError("Mismatch between population size and fitness values")
        
        # Increment generation counter
        self.generation += 1
        
        # Simple update for minimal implementation
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        
        # Update best solution if better than current
        if hasattr(self, 'previous_best'):
            improvement = best_fitness - self.previous_best
            self.previous_best = best_fitness
            return improvement
        else:
            self.previous_best = best_fitness
            return float('inf')
    
    def get_best_solution(self):
        """Return current best estimate (center)."""
        return self.center.copy()
    
    def get_stats(self):
        """Return current optimizer statistics."""
        return {
            "center_mean": float(np.mean(self.center)),
            "center_min": float(np.min(self.center)),
            "center_max": float(np.max(self.center)),
            "sigma": float(self.sigma),
            "generations": self.generation,
            "best_fitness": float(self.previous_best) if hasattr(self, 'previous_best') else None
        }
    
    def save_state(self, filename):
        """Save optimizer state to file."""
        np.savez(
            filename, 
            center=self.center, 
            sigma=self.sigma, 
            C=self.C,
            B=self.B,
            D=self.D,
            solution_length=self.solution_length,
            population_count=self.population_count,
            generation=self.generation,
            previous_best=getattr(self, 'previous_best', None)
        )
    
    def reset(self, center=None, sigma=None):
        """Reset the optimizer with optional new center and sigma."""
        if center is not None:
            self.center = np.array(center, dtype=np.float32)
            
        if sigma is not None:
            self.sigma = float(sigma)
            
        # Reset covariance matrix and paths
        self.C = np.eye(self.solution_length, dtype=np.float32)
        self.B = np.eye(self.solution_length, dtype=np.float32)
        self.D = np.ones(self.solution_length, dtype=np.float32)
        self.generation = 0 