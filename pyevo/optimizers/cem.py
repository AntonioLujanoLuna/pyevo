"""
Cross-Entropy Method (CEM) implementation.

CEM is a probabilistic optimization algorithm that iteratively updates a probability
distribution over the search space based on the performance of sampled solutions.
It works by maintaining a Gaussian distribution and updating its parameters to 
increase the likelihood of sampling high-performing solutions.
"""

import numpy as np
from pyevo.optimizers.base import Optimizer

class CrossEntropyMethod(Optimizer):
    """Cross-Entropy Method optimizer.
    
    CEM maintains a multivariate Gaussian distribution and iteratively updates 
    its parameters (mean and covariance) to maximize the probability of 
    generating high-performing solutions.
    """
    
    def __init__(self, 
                 solution_length,
                 population_count=None,
                 elite_ratio=0.2,
                 alpha=0.7,
                 mean=None,
                 sigma=None,
                 bounds=None,
                 diagonal_cov=False,
                 random_seed=None):
        """
        Initialize the Cross-Entropy Method optimizer.
        
        Args:
            solution_length: Length of solution vector (dimensionality of search space)
            population_count: Size of population (default: 10 * solution_length)
            elite_ratio: Ratio of top solutions used for distribution update (default: 0.2)
            alpha: Learning rate for distribution updates (default: 0.7)
            mean: Initial mean vector (default: zeros)
            sigma: Initial standard deviation (default: ones)
            bounds: Tuple of (min, max) bounds for each dimension, shape (2, solution_length)
            diagonal_cov: Whether to use diagonal covariance matrix (default: False)
            random_seed: Seed for random number generation
        """
        self.solution_length = solution_length
        
        # Set population size
        if population_count is None:
            self.population_count = 10 * solution_length
        else:
            self.population_count = population_count
            
        # Set parameters
        self.elite_ratio = elite_ratio
        self.elite_count = max(1, int(self.population_count * elite_ratio))
        self.alpha = alpha
        self.diagonal_cov = diagonal_cov
        
        # Set random state
        self.rng = np.random.RandomState(random_seed)
        
        # Handle bounds
        if bounds is None:
            self.bounds = None
        else:
            # Ensure bounds have the right shape
            if np.array(bounds).shape != (2, solution_length):
                raise ValueError(f"Bounds should have shape (2, {solution_length}), got {np.array(bounds).shape}")
            self.bounds = np.array(bounds, dtype=np.float32)
        
        # Initialize distribution parameters
        if mean is None:
            if self.bounds is not None:
                # Initialize in the middle of the bounds
                lower, upper = self.bounds
                self.mean = (lower + upper) / 2
            else:
                # Initialize at origin
                self.mean = np.zeros(solution_length, dtype=np.float32)
        else:
            self.mean = np.array(mean, dtype=np.float32)
            
        if sigma is None:
            if self.bounds is not None:
                # Set sigma based on bounds
                lower, upper = self.bounds
                self.sigma = (upper - lower) / 6  # 99.7% of samples within bounds
            else:
                # Default sigma
                self.sigma = np.ones(solution_length, dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)
            
        # Initialize covariance matrix
        if self.diagonal_cov:
            # Use diagonal covariance matrix (variances only)
            self.cov = np.diag(self.sigma**2).astype(np.float32)
        else:
            # Use full covariance matrix
            self.cov = np.diag(self.sigma**2).astype(np.float32)
            
        # Storage for current solutions
        self.solutions = np.zeros((self.population_count, solution_length), dtype=np.float32)
        
        # Generation counter
        self.generation = 0
        
    def ask(self):
        """Generate solutions for evaluation.
        
        Returns:
            Array of solutions with shape (population_count, solution_length)
        """
        if self.diagonal_cov:
            # Sample using diagonal covariance (faster)
            noise = self.rng.randn(self.population_count, self.solution_length)
            self.solutions = self.mean + noise * np.sqrt(np.diag(self.cov))
        else:
            # Sample from multivariate normal distribution
            try:
                # Ensure covariance matrix is positive definite
                L = np.linalg.cholesky(self.cov)
                noise = self.rng.randn(self.population_count, self.solution_length)
                self.solutions = self.mean + np.dot(noise, L.T)
            except np.linalg.LinAlgError:
                # Fallback to diagonal if Cholesky decomposition fails
                self.cov = np.diag(np.diag(self.cov))
                self.solutions = self.rng.multivariate_normal(
                    self.mean, self.cov, size=self.population_count
                )
        
        # Ensure float32 type for consistency
        self.solutions = self.solutions.astype(np.float32)
        
        # Apply bounds if specified
        if self.bounds is not None:
            lower, upper = self.bounds
            np.clip(self.solutions, lower, upper, out=self.solutions)
            
        return self.solutions
    
    def tell(self, fitnesses, tolerance=1e-6):
        """Update distribution parameters based on fitness values.
        
        Args:
            fitnesses: Array of fitness values for each solution
            tolerance: Minimum improvement threshold for early stopping
            
        Returns:
            Improvement in best fitness
        """
        if len(fitnesses) != self.population_count:
            raise ValueError("Mismatch between population size and fitness values")
        
        # Convert to array
        fitnesses = np.array(fitnesses)
        
        # Find the best solution and calculate improvement
        best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[best_idx]
        current_best = self.solutions[best_idx].copy()
        
        # Calculate improvement
        if hasattr(self, 'previous_best'):
            improvement = current_best_fitness - self.previous_best
        else:
            improvement = float('inf')
            
        self.previous_best = current_best_fitness
        self.best_solution = current_best
        
        # Select elite samples
        elite_indices = np.argsort(fitnesses)[-self.elite_count:]
        elite_samples = self.solutions[elite_indices]
        
        # Update mean vector
        old_mean = self.mean.copy()
        new_mean = np.mean(elite_samples, axis=0)
        self.mean = old_mean * (1 - self.alpha) + new_mean * self.alpha
        
        # Update covariance matrix
        if self.diagonal_cov:
            # Update only diagonal elements
            old_cov = np.diag(self.cov).copy()
            new_cov = np.var(elite_samples, axis=0)
            updated_cov = old_cov * (1 - self.alpha) + new_cov * self.alpha
            self.cov = np.diag(updated_cov)
        else:
            # Update full covariance matrix
            old_cov = self.cov.copy()
            centered = elite_samples - new_mean
            new_cov = np.dot(centered.T, centered) / self.elite_count
            
            # Apply regularization to ensure positive definiteness
            new_cov += 1e-6 * np.eye(self.solution_length)
            
            # Update covariance with learning rate
            self.cov = old_cov * (1 - self.alpha) + new_cov * self.alpha
        
        # Increment generation counter
        self.generation += 1
        
        return improvement
    
    def get_best_solution(self):
        """Return current best solution.
        
        Returns:
            Best solution found so far
        """
        if hasattr(self, 'best_solution'):
            return self.best_solution.copy()
        return self.mean.copy()
    
    def get_stats(self):
        """Return current optimizer statistics.
        
        Returns:
            Dictionary containing statistics about the current state
        """
        return {
            "generation": self.generation,
            "mean_norm": float(np.linalg.norm(self.mean)),
            "cov_trace": float(np.trace(self.cov)),
            "cov_determinant": float(np.linalg.det(self.cov)) if not self.diagonal_cov else float(np.prod(np.diag(self.cov))),
            "best_fitness": float(self.previous_best) if hasattr(self, 'previous_best') else None
        }
    
    def save_state(self, filename):
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        np.savez(
            filename, 
            mean=self.mean,
            cov=self.cov,
            generation=self.generation,
            solution_length=self.solution_length,
            population_count=self.population_count,
            elite_ratio=self.elite_ratio,
            elite_count=self.elite_count,
            alpha=self.alpha,
            diagonal_cov=self.diagonal_cov,
            bounds=self.bounds,
            previous_best=getattr(self, 'previous_best', None),
            best_solution=getattr(self, 'best_solution', None)
        )
    
    @classmethod
    def load_state(cls, filename):
        """Load optimizer state from file.
        
        Args:
            filename: Path to load the state from
            
        Returns:
            CrossEntropyMethod instance with loaded state
        """
        data = np.load(filename, allow_pickle=True)
        
        # Create optimizer with basic parameters
        optimizer = cls(
            solution_length=int(data['solution_length']),
            population_count=int(data['population_count']),
            elite_ratio=float(data['elite_ratio']),
            alpha=float(data['alpha']),
            mean=data['mean'],
            bounds=data['bounds'] if 'bounds' in data and data['bounds'] is not None else None,
            diagonal_cov=bool(data['diagonal_cov'])
        )
        
        # Load state
        optimizer.cov = data['cov']
        optimizer.generation = int(data['generation'])
        optimizer.elite_count = int(data['elite_count'])
        
        if 'previous_best' in data and data['previous_best'] is not None:
            optimizer.previous_best = float(data['previous_best'])
            
        if 'best_solution' in data and data['best_solution'] is not None:
            optimizer.best_solution = data['best_solution']
            
        return optimizer
    
    def reset(self, mean=None, cov=None, alpha=None):
        """Reset the optimizer.
        
        Args:
            mean: New mean vector (default: keep current)
            cov: New covariance matrix (default: keep current)
            alpha: New learning rate (default: keep current)
        """
        if mean is not None:
            if len(mean) != self.solution_length:
                raise ValueError(f"Mean vector length mismatch. Expected {self.solution_length}, got {len(mean)}")
            self.mean = np.array(mean, dtype=np.float32)
            
        if cov is not None:
            if self.diagonal_cov:
                if isinstance(cov, np.ndarray) and cov.ndim == 2:
                    # Extract diagonal from provided matrix
                    self.cov = np.diag(np.diag(cov)).astype(np.float32)
                else:
                    # Assume it's a vector of variances
                    self.cov = np.diag(np.array(cov, dtype=np.float32))
            else:
                if cov.shape != (self.solution_length, self.solution_length):
                    raise ValueError(f"Covariance matrix shape mismatch. Expected {(self.solution_length, self.solution_length)}, got {cov.shape}")
                self.cov = np.array(cov, dtype=np.float32)
                
        if alpha is not None:
            self.alpha = alpha
            
        # Reset state
        self.generation = 0
        if hasattr(self, 'previous_best'):
            delattr(self, 'previous_best')
        if hasattr(self, 'best_solution'):
            delattr(self, 'best_solution') 