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
        
        # Initialize evolution paths
        self.pc = np.zeros(solution_length, dtype=np.float32)  # Path for C
        self.ps = np.zeros(solution_length, dtype=np.float32)  # Path for sigma
        
        # Strategy parameters
        self.cc = 4.0 / solution_length  # Learning rate for rank-one update
        self.cs = 4.0 / solution_length  # Learning rate for step size control
        self.c1 = 2.0 / (solution_length**2)  # Learning rate for rank-one update
        self.cmu = 0.0  # Learning rate for rank-mu update - set to 0 for minimal impl
        
        # Weights for weighted recombination
        self.weights = np.array([np.log(self.population_count + 0.5) - np.log(i+1) 
                               for i in range(self.population_count)], dtype=np.float32)
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)  # Variance effective selection mass
        
        # Damping parameter for sigma update
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mueff - 1) / (solution_length + 1)) - 1) + self.cs
        
        # Initialize expectation of ||N(0,I)||
        self.chiN = np.sqrt(solution_length) * (1.0 - 1.0 / (4.0 * solution_length) + 
                                               1.0 / (21.0 * solution_length**2))
        
        # Storage for current generation
        self.solutions = np.zeros((self.population_count, solution_length), dtype=np.float32)
        self.z_samples = np.zeros((self.population_count, solution_length), dtype=np.float32)
        
        # Generation counter
        self.generation = 0
        
    def ask(self):
        """Generate a new batch of solutions to evaluate."""
        # Generate Gaussian samples
        for i in range(self.population_count):
            z = self.rng.randn(self.solution_length)
            self.z_samples[i] = z
            self.solutions[i] = self.center + self.sigma * np.dot(self.B, self.D * z)
            
        return self.solutions
    
    def tell(self, fitnesses, tolerance=1e-6):
        """Update parameters based on fitness values."""
        if len(fitnesses) != self.population_count:
            raise ValueError("Mismatch between population size and fitness values")
        
        # Increment generation counter
        self.generation += 1
        
        # Create array of fitness values
        fitnesses = np.array(fitnesses, dtype=np.float32)
        
        # Sort by fitness (descending order)
        indices = np.argsort(fitnesses)[::-1]
        sorted_z = self.z_samples[indices]
        
        # Get previous best fitness for improvement calculation
        if hasattr(self, 'previous_best'):
            prev_best = self.previous_best
        else:
            prev_best = float('-inf')
            
        # Update best fitness
        self.previous_best = fitnesses[indices[0]]
        improvement = self.previous_best - prev_best
        
        # Store the best solution
        self.best_solution = self.solutions[indices[0]].copy()
        
        # Weighted recombination
        z_weighted = np.zeros(self.solution_length, dtype=np.float32)
        for i in range(self.population_count):
            z_weighted += self.weights[i] * sorted_z[i]
        
        # Update mean (center)
        prev_center = self.center.copy()
        self.center += self.sigma * np.dot(self.B, self.D * z_weighted)
        
        # Cumulation for step size control (evolution path)
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  np.dot(self.B, z_weighted)
        
        # Cumulation for covariance matrix adaptation (evolution path)
        hsig = int(np.linalg.norm(self.ps) / 
                   np.sqrt(1 - (1 - self.cs)**(2 * self.generation)) / self.chiN < 1.4 + 2/(self.solution_length+1))
        
        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  np.dot(self.B, self.D * z_weighted)
        
        # Update covariance matrix C
        self.C = (1 - self.c1) * self.C + \
                 self.c1 * np.outer(self.pc, self.pc)
        
        # Update step size using cumulative step length adaptation
        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
        
        # Enforce bounds for numerical stability
        if self.sigma < 1e-20:
            self.sigma = 1e-20
        
        # Update the eigen decomposition periodically
        if self.generation % 10 == 0:
            self._update_eigensystem()
        
        return improvement
    
    def _update_eigensystem(self):
        """Update the eigen decomposition of C."""
        try:
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            
            # Ensure positive eigenvalues for numerical stability
            eigenvalues = np.maximum(eigenvalues, 1e-14)
            
            # Sort eigenvalues and eigenvectors
            indices = np.argsort(eigenvalues)[::-1]
            self.D = np.sqrt(eigenvalues[indices])
            self.B = eigenvectors[:, indices]
        except np.linalg.LinAlgError:
            # In case of numerical issues, reset to identity
            self.C = np.eye(self.solution_length, dtype=np.float32)
            self.B = np.eye(self.solution_length, dtype=np.float32)
            self.D = np.ones(self.solution_length, dtype=np.float32)
    
    def get_best_solution(self):
        """Return current best estimate."""
        if hasattr(self, 'best_solution'):
            return self.best_solution.copy()
        return self.center.copy()
    
    def get_stats(self):
        """Return current optimizer statistics."""
        return {
            "center_mean": float(np.mean(self.center)),
            "center_min": float(np.min(self.center)),
            "center_max": float(np.max(self.center)),
            "sigma": float(self.sigma),
            "sigma_min": float(np.min(self.sigma * self.D)),
            "sigma_max": float(np.max(self.sigma * self.D)),
            "sigma_mean": float(np.mean(self.sigma * self.D)),
            "condition_number": float(np.max(self.D) / np.min(self.D)),
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
            pc=self.pc,
            ps=self.ps,
            solution_length=self.solution_length,
            population_count=self.population_count,
            generation=self.generation,
            previous_best=getattr(self, 'previous_best', None),
            best_solution=getattr(self, 'best_solution', self.center.copy())
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
        self.pc = np.zeros(self.solution_length, dtype=np.float32)
        self.ps = np.zeros(self.solution_length, dtype=np.float32)
        self.generation = 0
        
    @classmethod
    def load_state(cls, filename):
        """Load optimizer state from file."""
        data = np.load(filename)
        optimizer = cls(
            solution_length=int(data['solution_length']),
            population_count=int(data['population_count'])
        )
        optimizer.center = data['center']
        optimizer.sigma = float(data['sigma'])
        optimizer.C = data['C']
        optimizer.B = data['B']
        optimizer.D = data['D']
        optimizer.pc = data['pc'] if 'pc' in data else np.zeros_like(optimizer.center)
        optimizer.ps = data['ps'] if 'ps' in data else np.zeros_like(optimizer.center)
        optimizer.generation = int(data['generation'])
        if 'previous_best' in data:
            optimizer.previous_best = float(data['previous_best'])
        if 'best_solution' in data:
            optimizer.best_solution = data['best_solution']
        return optimizer 