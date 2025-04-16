"""
Separable Natural Evolution Strategies (SNES) implementation.

SNES is an efficient black-box optimization algorithm especially suited for high-dimensional
continuous domains. It maintains a search distribution parameterized by a mean vector μ and 
standard deviation vector σ, which are updated based on the fitness ranking of sampled solutions.
"""

import numpy as np
from typing import Optional, Sequence, Any, ClassVar
from pyevo.optimizers.base import Optimizer

def get_default_population_count(solution_length: int) -> int:
    """Calculate default population size based on solution length.
    
    Args:
        solution_length: Dimensionality of the search space
        
    Returns:
        Recommended population size
    """
    return 4 + int(3 * np.log(solution_length))

class SNES(Optimizer):
    """Separable Natural Evolution Strategy optimizer.
    
    SNES optimizes a black-box objective function by maintaining a search distribution
    and iteratively adapting it based on the fitness of sampled solutions.
    """
    
    def __init__(
        self,
        solution_length: int,
        population_count: Optional[int] = None,
        alpha: float = 0.05,
        center: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initialize the Separable Natural Evolution Strategy optimizer.
        
        Args:
            solution_length: Length of solution vector (dimensionality of search space)
            population_count: Size of population (default is based on solution length)
            alpha: Learning rate (default 0.05)
            center: Initial center (mean) vector (default zeros)
            sigma: Initial standard deviation vector (default ones scaled by alpha)
            random_seed: Seed for random number generation
        """
        # Set population count
        if population_count is None:
            self.population_count: int = get_default_population_count(solution_length)
        else:
            self.population_count: int = population_count
            
        self.solution_length: int = solution_length
        
        # Set random state
        self.rng: np.random.RandomState = np.random.RandomState(random_seed)
        
        # Set parameters for update rates
        self.eta_center: float = 1.0
        # From the paper: https://people.idsia.ch/~juergen/xNES2010gecco.pdf
        self.eta_sigma: float = (3 + np.log(solution_length)) / (5 * np.sqrt(solution_length))
        
        # Initialize center (mu)
        if center is None:
            self.center: np.ndarray = np.zeros(solution_length, dtype=np.float32)
        else:
            self.center: np.ndarray = np.array(center, dtype=np.float32)
            
        # Initialize std deviation (sigma)
        if sigma is None:
            self.sigma: np.ndarray = np.ones(solution_length, dtype=np.float32) * alpha
        else:
            self.sigma: np.ndarray = np.array(sigma, dtype=np.float32) * alpha
            
        # Precalculate utility weights
        self.utility_weights: np.ndarray = self._get_weight_vector()
        
        # Storage for current generation
        self.gaussians: np.ndarray = np.zeros((self.population_count, solution_length), dtype=np.float32)
        self.solutions: np.ndarray = np.zeros((self.population_count, solution_length), dtype=np.float32)
    
    def _get_weight_vector(self) -> np.ndarray:
        """Calculate utility weights for ranking.
        
        Returns:
            Array of utility weights for each rank position
        """
        n = self.population_count
        weights = np.zeros(n, dtype=np.float32)
        
        # Calculate raw weights: max(0, log(n/2 + 1) - log(1 + i))
        for i in range(n):
            u = np.log(n/2 + 1) - np.log(1 + i)
            weights[i] = max(0, u)
            
        # Normalize weights
        sum_weights = np.sum(weights)
        weights = weights / sum_weights - 1.0 / n
        
        return weights
    
    def ask(self) -> np.ndarray:
        """Generate a new batch of solutions to evaluate.
        
        Returns:
            Array of solutions with shape (population_count, solution_length)
        """
        # Generate Gaussian noise
        self.gaussians = self.rng.randn(self.population_count, self.solution_length).astype(np.float32)
        
        # Create solutions by adding noise to center
        for i in range(self.population_count):
            self.solutions[i] = self.center + self.sigma * self.gaussians[i]
            
        return self.solutions
    
    def tell(self, fitnesses: Sequence[float], tolerance: float = 1e-6) -> float:
        """Update parameters based on fitness values (vectorized version).
        
        Args:
            fitnesses: Array or list of fitness values for each solution
                      (higher values are better)
            tolerance: Minimum improvement threshold for early stopping
            
        Returns:
            Improvement in best fitness (for early stopping)
        """
        if len(fitnesses) != self.population_count:
            raise ValueError("Mismatch between population size and fitness values")
        
        # Sort indices by fitness (descending order)
        indices = np.argsort(-np.array(fitnesses))
        
        # Get utilities for each rank
        utilities = self.utility_weights[np.arange(len(indices))]
        
        # Vectorized calculation of deltas
        for j in range(self.solution_length):
            noises = self.gaussians[indices, j]
            delta_mu = np.sum(utilities * noises)
            delta_sigma = np.sum(utilities * (noises**2 - 1))
            
            # Update center and sigma
            self.center[j] += self.eta_center * self.sigma[j] * delta_mu
            self.sigma[j] *= np.exp(0.5 * self.eta_sigma * delta_sigma)
        
        # Return improvement metric for early stopping
        best_fitness = np.max(fitnesses)
        if hasattr(self, 'previous_best'):
            improvement = best_fitness - self.previous_best
            self.previous_best = best_fitness
            return improvement
        else:
            self.previous_best = best_fitness
            return float('inf')
    
    def get_stats(self) -> dict[str, Any]:
        """Return current optimizer statistics.
        
        Returns:
            Dictionary containing statistics about the current state
        """
        return {
            "center_mean": float(np.mean(self.center)),
            "center_min": float(np.min(self.center)),
            "center_max": float(np.max(self.center)),
            "sigma_mean": float(np.mean(self.sigma)),
            "sigma_min": float(np.min(self.sigma)),
            "sigma_max": float(np.max(self.sigma)),
            "best_fitness": float(self.previous_best) if hasattr(self, 'previous_best') else None
        }
    
    def save_state(self, filename: str) -> None:
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        np.savez(
            filename, 
            center=self.center, 
            sigma=self.sigma, 
            solution_length=self.solution_length,
            population_count=self.population_count,
            previous_best=getattr(self, 'previous_best', None)
        )
    
    @classmethod
    def load_state(cls, filename: str) -> 'SNES':
        """Load optimizer state from file.
        
        Args:
            filename: Path to load the state from
            
        Returns:
            SNES instance with loaded state
        """
        data = np.load(filename)
        optimizer = cls(
            solution_length=int(data['solution_length']),
            population_count=int(data['population_count']),
            center=data['center'],
            sigma=data['sigma']
        )
        if 'previous_best' in data and data['previous_best'] is not None:
            optimizer.previous_best = float(data['previous_best'])
        return optimizer
    
    def get_best_solution(self) -> np.ndarray:
        """Return current best estimate (center).
        
        Returns:
            Current center vector (best estimate of the optimum)
        """
        return self.center.copy()
        
    def reset(self, center: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None) -> None:
        """Reset the optimizer with optional new center and sigma.
        
        Args:
            center: New center vector (default: keep current)
            sigma: New sigma vector (default: keep current)
        """
        if center is not None:
            self.center = np.array(center, dtype=np.float32)
            
        if sigma is not None:
            self.sigma = np.array(sigma, dtype=np.float32) 