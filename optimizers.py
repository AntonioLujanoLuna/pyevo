"""
Additional optimization algorithms for comparison with SNES.

This module contains various black-box optimization algorithms that can be used
as alternatives to SNES for comparing performance on different problems.
"""

import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base optimizer interface.
    
    All optimization algorithms should implement this interface to ensure
    compatibility with the existing codebase.
    """
    
    @abstractmethod
    def ask(self):
        """Generate a new batch of solutions to evaluate.
        
        Returns:
            Array of solutions with shape (population_count, solution_length)
        """
        pass
    
    @abstractmethod
    def tell(self, fitnesses, tolerance=1e-6):
        """Update parameters based on fitness values.
        
        Args:
            fitnesses: Array or list of fitness values for each solution
                      (higher values are better)
            tolerance: Minimum improvement threshold for early stopping
            
        Returns:
            Improvement in best fitness (for early stopping)
        """
        pass
    
    @abstractmethod
    def get_best_solution(self):
        """Return current best estimate of the solution.
        
        Returns:
            Current best solution vector
        """
        pass
    
    @abstractmethod
    def get_stats(self):
        """Return current optimizer statistics.
        
        Returns:
            Dictionary containing statistics about the current state
        """
        pass
    
    @abstractmethod
    def save_state(self, filename):
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        pass
    
    @abstractmethod
    def reset(self, center=None, sigma=None):
        """Reset the optimizer with optional new parameters.
        
        Args:
            center: New center/mean vector
            sigma: New standard deviation vector
        """
        pass


class CMA_ES(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
    CMA-ES is a state-of-the-art black-box optimization algorithm that adapts
    a full covariance matrix instead of just diagonal variances like SNES.
    This makes it more powerful on problems with parameter interactions but
    also more computationally expensive.
    """
    
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
        # Set random state
        self.rng = np.random.RandomState(random_seed)
        
        # Set dimensionality
        self.solution_length = solution_length
        
        # Set population size (lambda)
        if population_count is None:
            self.population_count = 4 + int(3 * np.log(solution_length))
        else:
            self.population_count = population_count
            
        # Set parent number (mu)
        self.parent_number = self.population_count // 2
            
        # Initialize center (mean)
        if center is None:
            self.center = np.zeros(solution_length, dtype=np.float32)
        else:
            self.center = np.array(center, dtype=np.float32)
            
        # Initialize step size
        if sigma is None:
            self.sigma = alpha
        else:
            # Handle both scalar and vector inputs
            if np.isscalar(sigma) or getattr(sigma, 'size', 0) == 1:
                self.sigma = float(sigma)
            else:
                # If sigma is a vector, use the mean value
                self.sigma = float(np.mean(sigma))
            
        # Initialize covariance matrix C and its decomposition
        self.C = np.eye(solution_length, dtype=np.float32)  # Covariance matrix
        self.B = np.eye(solution_length, dtype=np.float32)  # Eigenvectors of C
        self.D = np.ones(solution_length, dtype=np.float32)  # Eigenvalues of C (sqrt)
        
        # Strategy parameters
        self.cc = 4.0 / solution_length  # Learning rate for rank-one update
        self.cs = 4.0 / solution_length  # Learning rate for step size control
        self.c1 = 2.0 / (solution_length**2)  # Learning rate for rank-one update
        self.cmu = min(1 - self.c1, 0.2 * (self.parent_number / (solution_length**2)))  # Learning rate for rank-mu update
        
        # Evolution paths
        self.ps = np.zeros(solution_length, dtype=np.float32)  # Evolution path for sigma
        self.pc = np.zeros(solution_length, dtype=np.float32)  # Evolution path for C
        
        # Expected length of random vector
        self.chiN = np.sqrt(solution_length) * (1 - 1.0/(4*solution_length) + 1.0/(21*solution_length**2))
        
        # Weight vector for recombination
        self.weights = np.log(self.parent_number + 0.5) - np.log(np.arange(1, self.parent_number + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)  # Variance effective selection mass
        
        # Storage for current generation
        self.solutions = np.zeros((self.population_count, solution_length), dtype=np.float32)
        self.z = np.zeros((self.population_count, solution_length), dtype=np.float32)
        
    def ask(self):
        """Generate a new batch of solutions to evaluate.
        
        Returns:
            Array of solutions with shape (population_count, solution_length)
        """
        # Generate Gaussian samples
        for i in range(self.population_count):
            self.z[i] = self.rng.randn(self.solution_length)
            self.solutions[i] = self.center + self.sigma * np.dot(self.B, self.D * self.z[i])
            
        return self.solutions
    
    def tell(self, fitnesses, tolerance=1e-6):
        """Update parameters based on fitness values.
        
        Args:
            fitnesses: Array or list of fitness values for each solution
                      (higher values are better)
            tolerance: Minimum improvement threshold for early stopping
            
        Returns:
            Improvement in best fitness (for early stopping)
        """
        if len(fitnesses) != self.population_count:
            raise ValueError("Mismatch between population size and fitness values")
        
        # Sort by fitness (descending order)
        sorted_indices = np.argsort(-np.array(fitnesses))
        best_fitness = fitnesses[sorted_indices[0]]
            
        # Select top solutions (parents)
        selected_indices = sorted_indices[:self.parent_number]
        
        # Calculate weighted mean of selected solutions
        old_center = self.center.copy()
        self.center = np.zeros_like(self.center)
        for i, idx in enumerate(selected_indices):
            self.center += self.weights[i] * self.solutions[idx]
            
        # Update evolution paths
        y = self.center - old_center
        z = np.dot(np.linalg.inv(np.dot(self.B, np.diag(self.D))), y) / self.sigma
        
        # Update step size evolution path ps
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z
        
        # Update covariance matrix evolution path pc
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.solution_length + 1))) < 1.4 + 2 / (self.solution_length + 1)
        self.pc = (1 - self.cc) * self.pc
        if hsig:
            self.pc += np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y / self.sigma
            
        # Update covariance matrix
        artmp = (1 / self.sigma) * np.array([self.solutions[i] - old_center for i in selected_indices])
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * np.sum([self.weights[i] * np.outer(artmp[i], artmp[i]) for i in range(self.parent_number)], axis=0)
        
        # Update step size
        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / 2)
        
        # Eigendecomposition of C
        if np.any(~np.isfinite(self.C)):
            self.C = np.eye(self.solution_length)
            
        # Perform eigendecomposition every so often
        if np.random.rand() < 0.05:  # ~20 generations
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            self.D = np.sqrt(eigenvalues)
            self.B = eigenvectors
            
        # Return improvement metric for early stopping
        if hasattr(self, 'previous_best'):
            improvement = best_fitness - self.previous_best
            self.previous_best = best_fitness
            return improvement
        else:
            self.previous_best = best_fitness
            return float('inf')
    
    def get_best_solution(self):
        """Return current best estimate (center).
        
        Returns:
            Current center vector (best estimate of the optimum)
        """
        return self.center.copy()
    
    def get_stats(self):
        """Return current optimizer statistics.
        
        Returns:
            Dictionary containing statistics about the current state
        """
        return {
            "center_mean": float(np.mean(self.center)),
            "center_min": float(np.min(self.center)),
            "center_max": float(np.max(self.center)),
            "sigma": float(self.sigma),
            "condition_number": float(np.max(self.D) / np.min(self.D + 1e-10)),
            "best_fitness": float(self.previous_best) if hasattr(self, 'previous_best') else None
        }
    
    def save_state(self, filename):
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        np.savez(
            filename, 
            center=self.center, 
            sigma=self.sigma, 
            C=self.C,
            B=self.B,
            D=self.D,
            ps=self.ps,
            pc=self.pc,
            solution_length=self.solution_length,
            population_count=self.population_count,
            previous_best=getattr(self, 'previous_best', None)
        )
    
    @classmethod
    def load_state(cls, filename):
        """Load optimizer state from file.
        
        Args:
            filename: Path to load the state from
            
        Returns:
            CMA_ES instance with loaded state
        """
        data = np.load(filename)
        optimizer = cls(
            solution_length=int(data['solution_length']),
            population_count=int(data['population_count']),
            center=data['center'],
            sigma=float(data['sigma'])
        )
        
        # Load additional state
        optimizer.C = data['C']
        optimizer.B = data['B']
        optimizer.D = data['D']
        optimizer.ps = data['ps']
        optimizer.pc = data['pc']
        
        if 'previous_best' in data and data['previous_best'] is not None:
            optimizer.previous_best = float(data['previous_best'])
        return optimizer
    
    def reset(self, center=None, sigma=None):
        """Reset the optimizer with optional new center and sigma.
        
        Args:
            center: New center vector (default: keep current)
            sigma: New sigma scalar (default: keep current)
        """
        if center is not None:
            self.center = np.array(center, dtype=np.float32)
            
        if sigma is not None:
            self.sigma = float(sigma)
            
        # Reset covariance matrix and paths
        self.C = np.eye(self.solution_length, dtype=np.float32)
        self.B = np.eye(self.solution_length, dtype=np.float32)
        self.D = np.ones(self.solution_length, dtype=np.float32)
        self.ps = np.zeros(self.solution_length, dtype=np.float32)
        self.pc = np.zeros(self.solution_length, dtype=np.float32)


class PSO(Optimizer):
    """Particle Swarm Optimization (PSO).
    
    PSO is a population-based stochastic optimization technique inspired by
    social behavior of birds flocking or fish schooling. Each particle
    represents a potential solution and moves through the search space
    based on its own experience and the experience of neighboring particles.
    """
    
    def __init__(self, 
                solution_length,
                population_count=None,
                alpha=None,  # Not used but kept for interface compatibility
                center=None,
                sigma=None,
                random_seed=None,
                omega=0.7,    # Inertia weight
                phi_p=1.5,    # Cognitive parameter
                phi_g=1.5):   # Social parameter
        """
        Initialize the PSO optimizer.
        
        Args:
            solution_length: Length of solution vector (dimensionality of search space)
            population_count: Size of population/swarm (default is based on solution length)
            alpha: Not used but kept for interface compatibility
            center: Initial center for initialization range (default zeros)
            sigma: Initial range for particle positions (default ones)
            random_seed: Seed for random number generation
            omega: Inertia weight controlling particle momentum
            phi_p: Cognitive parameter (weight for particle's personal best)
            phi_g: Social parameter (weight for global best)
        """
        # Set random state
        self.rng = np.random.RandomState(random_seed)
        
        # Set dimensionality
        self.solution_length = solution_length
        
        # Set population size
        if population_count is None:
            self.population_count = 10 + int(2 * np.sqrt(solution_length))
        else:
            self.population_count = population_count
            
        # Initialize parameters
        if center is None:
            self.center = np.zeros(solution_length, dtype=np.float32)
        else:
            self.center = np.array(center, dtype=np.float32)
            
        if sigma is None:
            self.sigma = np.ones(solution_length, dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)
        
        # PSO specific parameters
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        
        # Initialize particle positions and velocities
        self.positions = np.zeros((self.population_count, solution_length), dtype=np.float32)
        self.velocities = np.zeros((self.population_count, solution_length), dtype=np.float32)
        
        # Initialize personal best positions and fitness
        self.personal_best_positions = np.zeros((self.population_count, solution_length), dtype=np.float32)
        self.personal_best_fitnesses = np.full(self.population_count, -float('inf'), dtype=np.float32)
        
        # Initialize global best
        self.global_best_position = np.zeros(solution_length, dtype=np.float32)
        self.global_best_fitness = -float('inf')
        
        # Initialize particles
        for i in range(self.population_count):
            self.positions[i] = self.center + self.rng.uniform(-1, 1, solution_length) * self.sigma
            self.velocities[i] = self.rng.uniform(-0.5, 0.5, solution_length) * self.sigma
            self.personal_best_positions[i] = self.positions[i].copy()
    
    def ask(self):
        """Generate/return current particle positions for evaluation.
        
        Returns:
            Array of positions with shape (population_count, solution_length)
        """
        return self.positions.copy()
    
    def tell(self, fitnesses, tolerance=1e-6):
        """Update particle positions based on fitness values.
        
        Args:
            fitnesses: Array or list of fitness values for each particle
                      (higher values are better)
            tolerance: Minimum improvement threshold for early stopping
            
        Returns:
            Improvement in best fitness (for early stopping)
        """
        if len(fitnesses) != self.population_count:
            raise ValueError("Mismatch between population size and fitness values")
        
        # Convert to numpy array
        fitnesses = np.array(fitnesses)
        
        # Find best fitness in current generation
        current_best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        
        # Update personal best positions
        for i in range(self.population_count):
            if fitnesses[i] > self.personal_best_fitnesses[i]:
                self.personal_best_fitnesses[i] = fitnesses[i]
                self.personal_best_positions[i] = self.positions[i].copy()
                
        # Update global best
        old_global_best_fitness = self.global_best_fitness
        
        if current_best_fitness > self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best_position = self.positions[current_best_idx].copy()
        
        # Update particle velocities and positions
        for i in range(self.population_count):
            # Generate random components
            r_p = self.rng.uniform(0, 1, self.solution_length)
            r_g = self.rng.uniform(0, 1, self.solution_length)
            
            # Update velocity
            cognitive_component = self.phi_p * r_p * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.phi_g * r_g * (self.global_best_position - self.positions[i])
            
            self.velocities[i] = self.omega * self.velocities[i] + cognitive_component + social_component
            
            # Update position
            self.positions[i] += self.velocities[i]
        
        # Calculate improvement for early stopping
        improvement = self.global_best_fitness - old_global_best_fitness
        
        return improvement
    
    def get_best_solution(self):
        """Return current best solution.
        
        Returns:
            Current global best position
        """
        return self.global_best_position.copy()
    
    def get_stats(self):
        """Return current optimizer statistics.
        
        Returns:
            Dictionary containing statistics about the current state
        """
        return {
            "global_best_fitness": float(self.global_best_fitness),
            "position_mean": float(np.mean(self.positions)),
            "position_min": float(np.min(self.positions)),
            "position_max": float(np.max(self.positions)),
            "velocity_mean": float(np.mean(self.velocities)),
            "velocity_min": float(np.min(self.velocities)),
            "velocity_max": float(np.max(self.velocities))
        }
    
    def save_state(self, filename):
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        np.savez(
            filename, 
            positions=self.positions,
            velocities=self.velocities,
            personal_best_positions=self.personal_best_positions,
            personal_best_fitnesses=self.personal_best_fitnesses,
            global_best_position=self.global_best_position,
            global_best_fitness=self.global_best_fitness,
            solution_length=self.solution_length,
            population_count=self.population_count,
            omega=self.omega,
            phi_p=self.phi_p,
            phi_g=self.phi_g
        )
    
    @classmethod
    def load_state(cls, filename):
        """Load optimizer state from file.
        
        Args:
            filename: Path to load the state from
            
        Returns:
            PSO instance with loaded state
        """
        data = np.load(filename)
        
        # Create optimizer with loaded parameters
        optimizer = cls(
            solution_length=int(data['solution_length']),
            population_count=int(data['population_count']),
            omega=float(data['omega']),
            phi_p=float(data['phi_p']),
            phi_g=float(data['phi_g'])
        )
        
        # Load additional state
        optimizer.positions = data['positions']
        optimizer.velocities = data['velocities']
        optimizer.personal_best_positions = data['personal_best_positions']
        optimizer.personal_best_fitnesses = data['personal_best_fitnesses']
        optimizer.global_best_position = data['global_best_position']
        optimizer.global_best_fitness = float(data['global_best_fitness'])
        
        return optimizer
    
    def reset(self, center=None, sigma=None):
        """Reset the optimizer with optional new center and sigma.
        
        Args:
            center: New center vector for initialization (default: keep current)
            sigma: New sigma vector for initialization range (default: keep current)
        """
        if center is not None:
            self.center = np.array(center, dtype=np.float32)
            
        if sigma is not None:
            self.sigma = np.array(sigma, dtype=np.float32)
            
        # Reset all particles
        for i in range(self.population_count):
            self.positions[i] = self.center + self.rng.uniform(-1, 1, self.solution_length) * self.sigma
            self.velocities[i] = self.rng.uniform(-0.5, 0.5, self.solution_length) * self.sigma
            self.personal_best_positions[i] = self.positions[i].copy()
            self.personal_best_fitnesses[i] = -float('inf')
            
        # Reset global best
        self.global_best_position = np.zeros(self.solution_length, dtype=np.float32)
        self.global_best_fitness = -float('inf') 