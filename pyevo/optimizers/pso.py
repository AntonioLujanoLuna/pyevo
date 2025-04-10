"""
Particle Swarm Optimization (PSO) implementation.

PSO is a population-based stochastic optimization technique inspired by
social behavior of birds flocking or fish schooling.
"""

import numpy as np
from pyevo.optimizers.base import Optimizer

class PSO(Optimizer):
    """Particle Swarm Optimization (PSO)."""
    
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
        """Generate/return current particle positions for evaluation."""
        return self.positions.copy()
    
    def tell(self, fitnesses, tolerance=1e-6):
        """Update particle positions based on fitness values."""
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
        """Return current best solution."""
        return self.global_best_position.copy()
    
    def get_stats(self):
        """Return current optimizer statistics."""
        return {
            "global_best_fitness": float(self.global_best_fitness),
            "position_mean": float(np.mean(self.positions)),
            "position_min": float(np.min(self.positions)),
            "position_max": float(np.max(self.positions)),
            "velocity_mean": float(np.mean(self.velocities))
        }
    
    def save_state(self, filename):
        """Save optimizer state to file."""
        np.savez(
            filename, 
            positions=self.positions,
            velocities=self.velocities,
            personal_best_positions=self.personal_best_positions,
            personal_best_fitnesses=self.personal_best_fitnesses,
            global_best_position=self.global_best_position,
            global_best_fitness=self.global_best_fitness,
            solution_length=self.solution_length,
            population_count=self.population_count
        )
    
    def reset(self, center=None, sigma=None):
        """Reset the optimizer with optional new center and sigma."""
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