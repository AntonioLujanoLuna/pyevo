"""
Simulated Annealing (SA) implementation.

SA is a probabilistic technique for approximating the global optimum of a given function.
It is inspired by the annealing process in metallurgy, where controlled cooling leads to a
more optimal state. SA can escape local optima by occasionally accepting worse solutions,
with the probability of acceptance decreasing as the algorithm progresses.
"""

import numpy as np
from typing import Optional, Sequence, Any
from pyevo.optimizers.base import Optimizer

class SimulatedAnnealing(Optimizer):
    """Simulated Annealing optimizer.
    
    SA starts with an initial solution and gradually improves it by
    generating neighboring solutions and accepting them based on a
    temperature parameter that decreases over time.
    """
    
    def __init__(self, 
                 solution_length: int,
                 initial_temp: float = 100.0,
                 cooling_factor: float = 0.95,
                 step_size: float = 0.1,
                 min_temp: float = 1e-6,
                 bounds: Optional[np.ndarray] = None,
                 initial_solution: Optional[np.ndarray] = None,
                 random_seed: Optional[int] = None) -> None:
        """
        Initialize the Simulated Annealing optimizer.
        
        Args:
            solution_length: Length of solution vector (dimensionality of search space)
            initial_temp: Initial temperature (higher = more exploration)
            cooling_factor: Temperature reduction per iteration (in [0.8, 0.999])
            step_size: Size of neighborhood exploration step
            min_temp: Minimum temperature for termination
            bounds: Tuple of (min, max) bounds for each dimension, shape (2, solution_length)
            initial_solution: Initial solution to start optimization (default: random)
            random_seed: Seed for random number generation
        """
        self.solution_length = solution_length
        
        # SA parameters
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.cooling_factor = cooling_factor
        self.step_size = step_size
        self.min_temp = min_temp
        
        # For compatibility with population-based methods
        self.population_count = 1
        
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
        
        # Initialize current solution
        if initial_solution is not None:
            if len(initial_solution) != solution_length:
                raise ValueError(f"Initial solution should have length {solution_length}")
            self.current_solution = np.array(initial_solution, dtype=np.float32)
        else:
            if self.bounds is None:
                # No bounds, initialize around zero
                self.current_solution = self.rng.randn(solution_length).astype(np.float32)
            else:
                # Initialize within bounds
                lower, upper = self.bounds
                self.current_solution = (lower + self.rng.random(solution_length) * 
                                        (upper - lower)).astype(np.float32)
        
        # Initialize candidate solution and best solution
        self.candidate_solution = np.copy(self.current_solution)
        self.best_solution = np.copy(self.current_solution)
        
        # Statistics
        self.iteration = 0
        self.accepted_count = 0
        self.rejected_count = 0
        
    def ask(self) -> np.ndarray:
        """Generate a new candidate solution to evaluate.
        
        Returns:
            Array of solutions to evaluate (just one solution for SA)
        """
        # Generate a candidate solution by perturbing the current solution
        perturbation = self.rng.randn(self.solution_length) * self.step_size
        self.candidate_solution = self.current_solution + perturbation
        
        # Enforce bounds
        if self.bounds is not None:
            lower, upper = self.bounds
            np.clip(self.candidate_solution, lower, upper, out=self.candidate_solution)
        
        # Return candidate solution for evaluation
        return np.array([self.candidate_solution])
    
    def _acceptance_probability(self, current_fitness: float, candidate_fitness: float) -> float:
        """Calculate the probability of accepting a worse solution.
        
        Args:
            current_fitness: Fitness of current solution
            candidate_fitness: Fitness of candidate solution
            
        Returns:
            Probability of accepting candidate solution
        """
        # If candidate is better, always accept
        if candidate_fitness > current_fitness:
            return 1.0
        
        # Otherwise, calculate acceptance probability based on temperature
        return np.exp((candidate_fitness - current_fitness) / self.temperature)
    
    def tell(self, fitnesses: Sequence[float], tolerance: float = 1e-6) -> float:
        """Update state based on fitness evaluation.
        
        Args:
            fitnesses: Array of fitness values (just one for SA)
            tolerance: Minimum improvement threshold for early stopping
            
        Returns:
            Improvement in best fitness
        """
        # Extract fitness (SA only generates one candidate solution)
        candidate_fitness = fitnesses[0]
        
        # Initialize current_fitness for first iteration
        if not hasattr(self, 'current_fitness'):
            self.current_fitness = candidate_fitness
            self.best_fitness = candidate_fitness
            return float('inf')
        
        # Calculate acceptance probability
        acceptance_prob = self._acceptance_probability(self.current_fitness, candidate_fitness)
        
        # Decide whether to accept the candidate solution
        if acceptance_prob > self.rng.random():
            improvement = candidate_fitness - self.current_fitness
            self.current_solution = np.copy(self.candidate_solution)
            self.current_fitness = candidate_fitness
            self.accepted_count += 1
            
            # Update best solution if needed
            if candidate_fitness > self.best_fitness:
                self.best_solution = np.copy(self.candidate_solution)
                self.best_fitness = candidate_fitness
                improvement = candidate_fitness - self.best_fitness
        else:
            improvement = 0.0
            self.rejected_count += 1
        
        # Cool down the temperature
        self.temperature *= self.cooling_factor
        
        # Update iteration counter
        self.iteration += 1
        
        return improvement
    
    def get_best_solution(self) -> np.ndarray:
        """Return the best solution found so far.
        Returns:
            The best solution vector found during optimization
        """
        return self.best_solution.copy()
    
    def get_stats(self) -> dict[str, Any]:
        """Return current optimizer statistics.
        
        Returns:
            Dictionary containing statistics about the current state
        """
        return {
            "temperature": float(self.temperature),
            "iteration": self.iteration,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "acceptance_rate": float(self.accepted_count / max(1, (self.accepted_count + self.rejected_count))),
            "best_fitness": float(self.best_fitness) if hasattr(self, 'best_fitness') else None,
            "current_fitness": float(self.current_fitness) if hasattr(self, 'current_fitness') else None
        }
    
    def save_state(self, filename: str) -> None:
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        np.savez(
            filename, 
            current_solution=self.current_solution,
            candidate_solution=self.candidate_solution,
            best_solution=self.best_solution,
            temperature=self.temperature,
            initial_temp=self.initial_temp,
            cooling_factor=self.cooling_factor,
            step_size=self.step_size,
            min_temp=self.min_temp,
            solution_length=self.solution_length,
            bounds=self.bounds,
            iteration=self.iteration,
            accepted_count=self.accepted_count,
            rejected_count=self.rejected_count,
            current_fitness=getattr(self, 'current_fitness', None),
            best_fitness=getattr(self, 'best_fitness', None)
        )
    
    @classmethod
    def load_state(cls, filename: str) -> 'SimulatedAnnealing':
        """Load optimizer state from file.
        
        Args:
            filename: Path to load the state from
            
        Returns:
            SimulatedAnnealing instance with loaded state
        """
        data = np.load(filename, allow_pickle=True)
        
        # Create optimizer with basic parameters
        optimizer = cls(
            solution_length=int(data['solution_length']),
            initial_temp=float(data['initial_temp']),
            cooling_factor=float(data['cooling_factor']),
            step_size=float(data['step_size']),
            min_temp=float(data['min_temp']),
            bounds=data['bounds'] if 'bounds' in data and data['bounds'] is not None else None,
            initial_solution=data['current_solution']
        )
        
        # Load state
        optimizer.temperature = float(data['temperature'])
        optimizer.candidate_solution = data['candidate_solution']
        optimizer.best_solution = data['best_solution']
        optimizer.iteration = int(data['iteration'])
        optimizer.accepted_count = int(data['accepted_count'])
        optimizer.rejected_count = int(data['rejected_count'])
        
        if 'current_fitness' in data and data['current_fitness'] is not None:
            optimizer.current_fitness = float(data['current_fitness'])
            
        if 'best_fitness' in data and data['best_fitness'] is not None:
            optimizer.best_fitness = float(data['best_fitness'])
            
        return optimizer
    
    def reset(self, initial_solution: Optional[np.ndarray] = None, temperature: Optional[float] = None, step_size: Optional[float] = None) -> None:
        """Reset the optimizer.
        
        Args:
            initial_solution: New initial solution (default: keep current)
            temperature: New temperature (default: reset to initial)
            step_size: New step size (default: keep current)
        """
        if initial_solution is not None:
            if len(initial_solution) != self.solution_length:
                raise ValueError(f"Initial solution should have length {self.solution_length}")
            self.current_solution = np.array(initial_solution, dtype=np.float32)
        
        self.temperature = temperature if temperature is not None else self.initial_temp
        self.step_size = step_size if step_size is not None else self.step_size
        
        # Reset candidate and best solutions
        self.candidate_solution = np.copy(self.current_solution)
        self.best_solution = np.copy(self.current_solution)
        
        # Reset statistics
        self.iteration = 0
        self.accepted_count = 0
        self.rejected_count = 0
        
        # Reset fitness values
        if hasattr(self, 'current_fitness'):
            delattr(self, 'current_fitness')
        if hasattr(self, 'best_fitness'):
            delattr(self, 'best_fitness') 