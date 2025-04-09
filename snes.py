"""
Separable Natural Evolution Strategies (SNES) implementation in Python.

SNES is an efficient black-box optimization algorithm especially suited for high-dimensional
continuous domains. It maintains a search distribution parameterized by a mean vector μ and 
standard deviation vector σ, which are updated based on the fitness ranking of sampled solutions.
"""

import numpy as np

def get_default_population_count(solution_length):
    """Calculate default population size based on solution length.
    
    Args:
        solution_length: Dimensionality of the search space
        
    Returns:
        Recommended population size
    """
    return 4 + int(3 * np.log(solution_length))

class SNES:
    """Separable Natural Evolution Strategy optimizer.
    
    SNES optimizes a black-box objective function by maintaining a search distribution
    and iteratively adapting it based on the fitness of sampled solutions.
    """
    
    def __init__(self, 
                 solution_length,
                 population_count=None,
                 alpha=0.05,
                 center=None,
                 sigma=None,
                 random_seed=None):
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
            self.population_count = get_default_population_count(solution_length)
        else:
            self.population_count = population_count
            
        self.solution_length = solution_length
        
        # Set random state
        self.rng = np.random.RandomState(random_seed)
        
        # Set parameters for update rates
        self.eta_center = 1.0
        # From the paper: https://people.idsia.ch/~juergen/xNES2010gecco.pdf
        self.eta_sigma = (3 + np.log(solution_length)) / (5 * np.sqrt(solution_length))
        
        # Initialize center (mu)
        if center is None:
            self.center = np.zeros(solution_length, dtype=np.float32)
        else:
            self.center = np.array(center, dtype=np.float32)
            
        # Initialize std deviation (sigma)
        if sigma is None:
            self.sigma = np.ones(solution_length, dtype=np.float32) * alpha
        else:
            self.sigma = np.array(sigma, dtype=np.float32) * alpha
            
        # Precalculate utility weights
        self.utility_weights = self._get_weight_vector()
        
        # Storage for current generation
        self.gaussians = np.zeros((self.population_count, solution_length), dtype=np.float32)
        self.solutions = np.zeros((self.population_count, solution_length), dtype=np.float32)
    
    def _get_weight_vector(self):
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
    
    def ask(self):
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
    
    def tell(self, fitnesses):
        """Update parameters based on fitness values.
        
        Args:
            fitnesses: Array or list of fitness values for each solution
                       (higher values are better)
        """
        if len(fitnesses) != self.population_count:
            raise ValueError("Mismatch between population size and fitness values")
        
        # Sort indices by fitness (descending order)
        indices = np.argsort(-np.array(fitnesses))
        
        # Update parameters for each dimension
        for j in range(self.solution_length):
            delta_mu = 0
            delta_sigma = 0
            
            # Sum utility-weighted noise for this dimension
            for rank, idx in enumerate(indices):
                utility = self.utility_weights[rank]
                noise = self.gaussians[idx, j]
                delta_mu += utility * noise
                delta_sigma += utility * (noise*noise - 1)
            
            # Update center (mu)
            self.center[j] += self.eta_center * self.sigma[j] * delta_mu
            
            # Update sigma: multiplicative update via exponential
            self.sigma[j] *= np.exp(0.5 * self.eta_sigma * delta_sigma)
            
    def get_best_solution(self):
        """Return current best estimate (center).
        
        Returns:
            Current center vector (best estimate of the optimum)
        """
        return self.center.copy()
        
    def reset(self, center=None, sigma=None):
        """Reset the optimizer with optional new center and sigma.
        
        Args:
            center: New center vector (default: keep current)
            sigma: New sigma vector (default: keep current)
        """
        if center is not None:
            self.center = np.array(center, dtype=np.float32)
            
        if sigma is not None:
            self.sigma = np.array(sigma, dtype=np.float32)