"""
Differential Evolution (DE) implementation.

DE is a population-based evolutionary algorithm effective at optimizing 
continuous domains. It uses vector differences between members of the population 
to generate new candidate solutions, and performs well on multimodal problems.
"""

import numpy as np
from typing import Optional, Sequence, Any
from pyevo.optimizers.base import Optimizer

class DE(Optimizer):
    """Differential Evolution optimizer.
    
    DE evolves a population of candidate solutions by creating new solutions
    through mutation and crossover of existing solutions, then selecting the
    best solutions to survive to the next generation.
    """
    
    def __init__(self, 
                 solution_length: int,
                 population_count: Optional[int] = None,
                 f: float = 0.5,
                 cr: float = 0.7,
                 strategy: str = "best/1/bin",
                 bounds: Optional[np.ndarray] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize the Differential Evolution optimizer.
        
        Args:
            solution_length: Length of solution vector (dimensionality of search space)
            population_count: Size of population (default: min(10 * solution_length, 100), minimum 4)
            f: Differential weight (mutation factor), typically in [0.4, 1.0]
            cr: Crossover probability, typically in [0.1, 1.0]
            strategy: DE variant to use (default: "best/1/bin")
            bounds: Tuple of (min, max) bounds for each dimension, shape (2, solution_length)
            random_seed: Seed for random number generation
        """
        self.solution_length = solution_length
        
        # Set population size (larger than SNES/CMA-ES by default)
        if population_count is None:
            self.population_count = min(10 * solution_length, 100)
        else:
            self.population_count = population_count

        # best/1 excludes both the target and the best individual before
        # drawing 2 donors; rand/1 excludes the target before drawing 3.
        min_pop = 4
        if self.population_count < min_pop:
            raise ValueError(
                f"DE needs population_count >= {min_pop} to form mutant vectors, "
                f"got {self.population_count}"
            )
            
        # Set random state
        self.rng = np.random.RandomState(random_seed)
        
        # DE-specific parameters
        self.f = f  # Differential weight
        self.cr = cr  # Crossover probability
        self.strategy = strategy  # DE variant
        
        # Handle bounds
        if bounds is None:
            self.bounds = None
        else:
            # Ensure bounds have the right shape
            if np.array(bounds).shape != (2, solution_length):
                raise ValueError(f"Bounds should have shape (2, {solution_length}), got {np.array(bounds).shape}")
            self.bounds = np.array(bounds, dtype=np.float32)
        
        # Initialize population
        self.initialize_population()
        
        # Store fitnesses
        self.fitnesses = np.zeros(self.population_count, dtype=np.float32)
        self.best_idx = 0
        self.generation = 0
        # False until the initial population has been scored by tell().
        self._scored = False
        self.trial_vectors: Optional[np.ndarray] = None
        
    def initialize_population(self) -> None:
        """Initialize the population randomly within bounds."""
        if self.bounds is None:
            # No bounds, initialize around zero with standard deviation 1.0
            self.population = self.rng.randn(self.population_count, self.solution_length).astype(np.float32)
        else:
            # Initialize within bounds
            lower, upper = self.bounds
            self.population = lower + self.rng.random((self.population_count, self.solution_length)) * (upper - lower)
            self.population = self.population.astype(np.float32)
    
    def ask(self) -> np.ndarray:
        """Generate solutions for evaluation.

        On the first call this returns the initial population so it can be
        scored. Every later call returns freshly built trial vectors, which
        `tell` then greedily selects against their parents.

        Returns:
            Array of solutions to evaluate
        """
        if not self._scored:
            # First generation: the population itself needs fitness values.
            self.trial_vectors = None
            return self.population

        self.trial_vectors = self._generate_trials()
        return self.trial_vectors
    
    def _enforce_bounds(self, vectors: np.ndarray) -> np.ndarray:
        """Enforce solution bounds if specified."""
        if self.bounds is not None:
            lower, upper = self.bounds
            np.clip(vectors, lower, upper, out=vectors)
        return vectors
    
    def tell(self, fitnesses: Sequence[float], tolerance: float = 1e-6) -> float:
        """Update population based on fitness values.
        
        Args:
            fitnesses: Array of fitness values for each solution
            tolerance: Minimum improvement threshold for early stopping
            
        Returns:
            Improvement in best fitness
        """
        if len(fitnesses) != self.population_count:
            raise ValueError("Mismatch between population size and fitness values")

        fitnesses = np.asarray(fitnesses, dtype=np.float32)

        if not self._scored:
            # First call: these are the fitnesses of the initial population.
            self.fitnesses = fitnesses
            self._scored = True
            improvement = float('inf')
        else:
            # Later calls: `fitnesses` scores the trial vectors from ask().
            # Greedy one-to-one selection - this replaces the population, which
            # previously never happened, leaving DE a no-op across generations.
            self._selection(fitnesses)
            improvement = float(np.max(self.fitnesses) - self.previous_best)

        self.best_idx = int(np.argmax(self.fitnesses))
        self.previous_best = float(self.fitnesses[self.best_idx])
        self.generation += 1

        return improvement

    def _generate_trials(self) -> np.ndarray:
        """Build one trial vector per population member using the DE strategy.

        Returns:
            Array of trial vectors, shape (population_count, solution_length)
        """
        trial_vectors = np.zeros_like(self.population)
        curr_best_idx = int(np.argmax(self.fitnesses))

        # Apply the selected DE strategy to create trial vectors
        if self.strategy.startswith("best"):
            # Use best individual as base vector
            for i in range(self.population_count):
                # Select random individuals for mutation, excluding current and best
                candidates = list(range(self.population_count))
                candidates.remove(i)
                if i != curr_best_idx:
                    candidates.remove(curr_best_idx)
                
                r1, r2 = self.rng.choice(candidates, size=2, replace=False)
                
                # Create mutant vector (best/1)
                mutant = self.population[curr_best_idx] + self.f * (self.population[r1] - self.population[r2])
                
                # Apply crossover (bin = binomial)
                crossover_mask = self.rng.random(self.solution_length) < self.cr
                # Ensure at least one dimension is changed
                crossover_mask[self.rng.randint(0, self.solution_length)] = True
                
                trial_vectors[i] = np.where(crossover_mask, mutant, self.population[i])
        else:
            # Default: rand/1/bin strategy
            for i in range(self.population_count):
                # Select random individuals for mutation, excluding current
                candidates = list(range(self.population_count))
                candidates.remove(i)
                r1, r2, r3 = self.rng.choice(candidates, size=3, replace=False)
                
                # Create mutant vector (rand/1)
                mutant = self.population[r1] + self.f * (self.population[r2] - self.population[r3])
                
                # Apply crossover (bin = binomial)
                crossover_mask = self.rng.random(self.solution_length) < self.cr
                # Ensure at least one dimension is changed
                crossover_mask[self.rng.randint(0, self.solution_length)] = True
                
                trial_vectors[i] = np.where(crossover_mask, mutant, self.population[i])
        
        # Enforce bounds
        return self._enforce_bounds(trial_vectors)

    def _selection(self, trial_fitnesses: np.ndarray) -> None:
        """Greedily replace parents by the trial vectors that beat them.

        Args:
            trial_fitnesses: Fitness of each trial vector, aligned with the
                             population by index
        """
        if self.trial_vectors is None:
            return

        improved = trial_fitnesses > self.fitnesses
        self.population[improved] = self.trial_vectors[improved]
        self.fitnesses[improved] = trial_fitnesses[improved]
    
    def get_best_solution(self) -> np.ndarray:
        """Return current best solution.
        
        Returns:
            Best solution found so far
        """
        if hasattr(self, 'best_idx'):
            return self.population[self.best_idx].copy()
        return self.population[0].copy()
    
    def get_stats(self) -> dict[str, Any]:
        """Return current optimizer statistics.
        
        Returns:
            Dictionary containing statistics about the current state
        """
        population_mean = np.mean(self.population, axis=0)
        population_std = np.std(self.population, axis=0)
        
        return {
            "generation": self.generation,
            "population_mean": float(np.mean(population_mean)),
            "population_min": float(np.min(self.population)),
            "population_max": float(np.max(self.population)),
            "population_diversity": float(np.mean(population_std)),
            "best_fitness": float(self.previous_best) if hasattr(self, 'previous_best') else None
        }
    
    def save_state(self, filename: str) -> None:
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        np.savez(
            filename, 
            population=self.population,
            fitnesses=self.fitnesses,
            best_idx=self.best_idx,
            generation=self.generation,
            solution_length=self.solution_length,
            population_count=self.population_count,
            f=self.f,
            cr=self.cr,
            strategy=self.strategy,
            has_bounds=self.bounds is not None,
            bounds=self.bounds if self.bounds is not None else np.zeros((2, self.solution_length), dtype=np.float32),
            scored=self._scored,
            previous_best=float(getattr(self, 'previous_best', 0.0)),
            has_previous_best=hasattr(self, 'previous_best')
        )
    
    @classmethod
    def load_state(cls, filename: str) -> 'DE':
        """Load optimizer state from file.
        
        Args:
            filename: Path to load the state from
            
        Returns:
            DE instance with loaded state
        """
        data = np.load(filename)

        # Create optimizer with basic parameters
        optimizer = cls(
            solution_length=int(data['solution_length']),
            population_count=int(data['population_count']),
            f=float(data['f']),
            cr=float(data['cr']),
            strategy=str(data['strategy']),
            bounds=data['bounds'] if bool(data['has_bounds']) else None
        )

        # Load state
        optimizer.population = data['population']
        optimizer.fitnesses = data['fitnesses']
        optimizer.best_idx = int(data['best_idx'])
        optimizer.generation = int(data['generation'])
        optimizer._scored = bool(data['scored'])

        if bool(data['has_previous_best']):
            optimizer.previous_best = float(data['previous_best'])

        return optimizer
    
    def reset(self, population: Optional[np.ndarray] = None, f: Optional[float] = None, cr: Optional[float] = None) -> None:
        """Reset the optimizer.
        
        Args:
            population: New initial population (default: reinitialize randomly)
            f: New differential weight (default: keep current)
            cr: New crossover probability (default: keep current)
        """
        if population is not None:
            if population.shape != (self.population_count, self.solution_length):
                raise ValueError(f"Population shape mismatch. Expected {(self.population_count, self.solution_length)}, got {population.shape}")
            self.population = population.astype(np.float32)
        else:
            self.initialize_population()
            
        if f is not None:
            self.f = f
            
        if cr is not None:
            self.cr = cr
            
        # Reset state
        self.fitnesses = np.zeros(self.population_count, dtype=np.float32)
        self.best_idx = 0
        self.generation = 0
        self._scored = False
        self.trial_vectors = None
        if hasattr(self, 'previous_best'):
            delattr(self, 'previous_best') 