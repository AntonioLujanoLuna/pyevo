"""
Covariance Matrix Adaptation Evolution Strategy (CMA-ES) implementation.

CMA-ES is a state-of-the-art black-box optimization algorithm that adapts
a full covariance matrix instead of just diagonal variances like SNES.
"""

import numpy as np
from typing import Optional, Sequence, Any
from pyevo.optimizers.base import Optimizer, pack_optional, unpack_optional

class CMA_ES(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""
    
    def __init__(self, 
                solution_length: int,
                population_count: Optional[int] = None,
                alpha: float = 0.1,
                center: Optional[np.ndarray] = None,
                sigma: Optional[np.ndarray] = None,
                random_seed: Optional[int] = None):
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
            
        # Covariance matrix and its decomposition. These are kept in float64:
        # the eigendecomposition of an ill-conditioned C is not reliable in
        # float32, even though sampled solutions are returned as float32.
        self.C = np.eye(solution_length, dtype=np.float64)   # Covariance matrix
        self.B = np.eye(solution_length, dtype=np.float64)   # Eigenvectors of C
        self.D = np.ones(solution_length, dtype=np.float64)  # sqrt of eigenvalues of C

        # Initialize evolution paths
        self.pc = np.zeros(solution_length, dtype=np.float64)  # Path for C
        self.ps = np.zeros(solution_length, dtype=np.float64)  # Path for sigma

        # Weighted recombination over the mu best of lambda samples. Previously
        # every sample carried positive weight, so the worst solutions still
        # pulled the mean toward them; canonical CMA-ES truncates at mu.
        n = solution_length
        lam = self.population_count
        self.mu = max(1, lam // 2)

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = (weights / np.sum(weights)).astype(np.float64)
        self.mueff = float(1.0 / np.sum(self.weights ** 2))  # variance-effective selection mass

        # Strategy parameters (Hansen, "The CMA Evolution Strategy: A Tutorial").
        # The old cc = cs = 4/n exceeded 1 for n < 4, which inverted the path
        # cumulation and produced NaN at n = 1.
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        # Rank-mu update. This used to be hardcoded to 0, which removed the
        # single most important adaptation mechanism in modern CMA-ES.
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff),
        )

        # Damping parameter for sigma update
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs

        # Expectation of ||N(0,I)||
        self.chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        # Generation of the last eigendecomposition, for lazy updates
        self.eigeneval = 0
        
        # Storage for current generation
        self.solutions = np.zeros((self.population_count, solution_length), dtype=np.float32)
        self.z_samples = np.zeros((self.population_count, solution_length), dtype=np.float32)
        
        # Generation counter
        self.generation = 0
        
    def ask(self) -> np.ndarray:
        """Generate a new batch of solutions to evaluate."""
        # Generate Gaussian samples: z ~ N(0, I), then y = B * D * z.
        z = self.rng.randn(self.population_count, self.solution_length)
        self.z_samples = z.astype(np.float32)
        y = z @ (self.B * self.D).T
        self.solutions = (self.center + self.sigma * y).astype(np.float32)

        return self.solutions
    
    def tell(self, fitnesses: Sequence[float], tolerance: float = 1e-6) -> float:
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

        # Weighted recombination over the mu best samples only.
        elite_z = sorted_z[:self.mu].astype(np.float64)
        z_weighted = self.weights @ elite_z

        # y = B * D * z maps a sample from the unit-Gaussian space into the
        # search space scaled by the current covariance.
        y_weighted = self.B @ (self.D * z_weighted)

        # Update mean (center)
        self.center = (self.center + self.sigma * y_weighted).astype(np.float32)

        # Cumulation for step size control (evolution path)
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  (self.B @ z_weighted)

        # hsig guards against an over-long step inflating C early on.
        ps_norm = float(np.linalg.norm(self.ps))
        denom = np.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
        hsig = float(ps_norm / denom / self.chiN < 1.4 + 2.0 / (self.solution_length + 1))

        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_weighted

        # Rank-mu update: the elite samples themselves, mapped into y-space.
        elite_y = elite_z @ (self.B * self.D).T
        rank_mu = (elite_y * self.weights[:, None]).T @ elite_y

        # When hsig is 0 the rank-one term loses variance; the (1-hsig) term
        # compensates so C's total variance is preserved.
        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc)

        self.C = (1 - self.c1 - self.cmu) * self.C \
                 + self.c1 * (np.outer(self.pc, self.pc) + delta_hsig * self.C) \
                 + self.cmu * rank_mu

        # Keep C exactly symmetric; repeated updates accumulate asymmetry.
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

        # Update step size using cumulative step length adaptation
        self.sigma *= np.exp((ps_norm / self.chiN - 1) * self.cs / self.damps)

        # Enforce bounds for numerical stability
        if self.sigma < 1e-20:
            self.sigma = 1e-20

        # Lazily refresh the eigendecomposition. The old fixed "every 10
        # generations" let B and D drift arbitrarily far from C; this cadence
        # scales with how fast C actually changes.
        update_every = self.population_count / ((self.c1 + self.cmu) * self.solution_length * 10)
        if self.generation - self.eigeneval > update_every:
            self.eigeneval = self.generation
            self._update_eigensystem()

        return improvement
    
    def _update_eigensystem(self):
        """Update the eigen decomposition of C."""
        try:
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)

            # Ensure positive eigenvalues for numerical stability
            eigenvalues = np.maximum(eigenvalues, 1e-20)

            # Sort eigenvalues and eigenvectors
            indices = np.argsort(eigenvalues)[::-1]
            self.D = np.sqrt(eigenvalues[indices])
            self.B = eigenvectors[:, indices]
        except np.linalg.LinAlgError:
            # In case of numerical issues, reset to identity
            self.C = np.eye(self.solution_length, dtype=np.float64)
            self.B = np.eye(self.solution_length, dtype=np.float64)
            self.D = np.ones(self.solution_length, dtype=np.float64)
    
    def get_best_solution(self) -> np.ndarray:
        """Return current best estimate."""
        if hasattr(self, 'best_solution'):
            return self.best_solution.copy()
        return self.center.copy()
    
    def get_stats(self) -> dict:
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
    
    def save_state(self, filename: str):
        """Save optimizer state to file."""
        state = dict(
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
            eigeneval=self.eigeneval,
            best_solution=getattr(self, 'best_solution', self.center.copy()),
        )
        pack_optional(state, 'previous_best', getattr(self, 'previous_best', None))
        np.savez(filename, **state)
    
    def reset(self, center: Optional[np.ndarray] = None, sigma: Optional[float] = None):
        """Reset the optimizer with optional new center and sigma."""
        if center is not None:
            self.center = np.array(center, dtype=np.float32)
            
        if sigma is not None:
            self.sigma = float(sigma)
            
        # Reset covariance matrix and paths
        self.C = np.eye(self.solution_length, dtype=np.float64)
        self.B = np.eye(self.solution_length, dtype=np.float64)
        self.D = np.ones(self.solution_length, dtype=np.float64)
        self.pc = np.zeros(self.solution_length, dtype=np.float64)
        self.ps = np.zeros(self.solution_length, dtype=np.float64)
        self.generation = 0
        self.eigeneval = 0
        
    @classmethod
    def load_state(cls, filename: str) -> 'CMA_ES':
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
        optimizer.pc = data['pc'] if 'pc' in data else np.zeros(optimizer.solution_length)
        optimizer.ps = data['ps'] if 'ps' in data else np.zeros(optimizer.solution_length)
        optimizer.generation = int(data['generation'])
        optimizer.eigeneval = int(data['eigeneval']) if 'eigeneval' in data else 0

        previous_best = unpack_optional(data, 'previous_best')
        if previous_best is not None:
            optimizer.previous_best = float(previous_best)
        if 'best_solution' in data:
            optimizer.best_solution = data['best_solution']
        return optimizer 