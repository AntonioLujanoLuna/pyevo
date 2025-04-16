"""
Genetic Algorithm (GA) implementation.

GA is a search heuristic inspired by natural selection that uses mechanisms
such as selection, crossover, and mutation to evolve candidate solutions.
It's well-suited for optimization problems with discrete or combinatorial search spaces
but also works for continuous domains.
"""

import numpy as np
from typing import Optional, Sequence, Any, Tuple
from pyevo.optimizers.base import Optimizer

class GeneticAlgorithm(Optimizer):
    """Genetic Algorithm optimizer.
    
    GA maintains a population of candidate solutions and evolves them using
    selection, crossover, and mutation operations inspired by natural evolution.
    """
    
    def __init__(self, 
                 solution_length: int,
                 population_count: Optional[int] = None,
                 elite_count: Optional[int] = None,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.1,
                 mutation_strength: float = 0.1,
                 selection_method: str = "tournament",
                 tournament_size: int = 3,
                 bounds: Optional[Sequence[Sequence[float]]] = None,
                 discrete: bool = False,
                 random_seed: Optional[int] = None) -> None:
        """
        Initialize the Genetic Algorithm optimizer.
        
        Args:
            solution_length: Length of solution vector (dimensionality of search space)
            population_count: Size of population (default: 4 * solution_length)
            elite_count: Number of top solutions to preserve unchanged (default: population_count // 10)
            crossover_prob: Probability of crossover (default: 0.7)
            mutation_prob: Probability of mutation per gene (default: 0.1)
            mutation_strength: Scale of mutations (default: 0.1)
            selection_method: Method to select parents ("tournament", "roulette")
            tournament_size: Number of individuals in tournament selection
            bounds: Tuple of (min, max) bounds for each dimension, shape (2, solution_length)
            discrete: Whether to use discrete (integer) optimization
            random_seed: Seed for random number generation
        """
        self.solution_length = solution_length
        
        # Set population size
        if population_count is None:
            self.population_count = 4 * solution_length
        else:
            self.population_count = population_count
        
        # Set elite count
        if elite_count is None:
            self.elite_count = max(1, self.population_count // 10)
        else:
            self.elite_count = min(elite_count, self.population_count - 1)
        
        # Set GA parameters
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_strength = mutation_strength
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.discrete = discrete
        
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
        
        # Initialize population
        self.initialize_population()
        
        # Generation counter
        self.generation = 0
        
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
            
        # Convert to integers for discrete optimization
        if self.discrete and self.bounds is not None:
            self.population = np.floor(self.population).astype(np.int32).astype(np.float32)
    
    def ask(self) -> np.ndarray:
        """Generate solutions for evaluation.
        
        Returns:
            Array of solutions with shape (population_count, solution_length)
        """
        return self.population
    
    def _tournament_selection(self, fitnesses: Sequence[float], k: int = 3) -> int:
        """Tournament selection of individuals.
        
        Args:
            fitnesses: Array of fitness values
            k: Tournament size
            
        Returns:
            Index of selected individual
        """
        # Select k individuals randomly
        selected_indices = self.rng.choice(self.population_count, size=k, replace=False)
        
        # Return the index of the best individual
        return selected_indices[np.argmax(fitnesses[selected_indices])]
    
    def _roulette_selection(self, fitnesses: Sequence[float]) -> int:
        """Roulette wheel selection of individuals.
        
        Args:
            fitnesses: Array of fitness values
            
        Returns:
            Index of selected individual
        """
        # Handle negative fitnesses by shifting
        if np.min(fitnesses) < 0:
            shifted_fitnesses = fitnesses - np.min(fitnesses) + 1e-6
        else:
            shifted_fitnesses = fitnesses + 1e-6
            
        # Calculate selection probabilities
        probs = shifted_fitnesses / np.sum(shifted_fitnesses)
        
        # Select an individual based on fitness proportionate selection
        return self.rng.choice(self.population_count, p=probs)
    
    def _crossover(self, parent1: Sequence[float], parent2: Sequence[float]) -> Tuple[Sequence[float], Sequence[float]]:
        """Perform uniform crossover between two parents.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            Two offspring solutions
        """
        # Decide whether to perform crossover
        if self.rng.random() < self.crossover_prob:
            # Create crossover mask
            mask = self.rng.random(self.solution_length) < 0.5
            
            # Create offspring through uniform crossover
            offspring1 = np.where(mask, parent1, parent2)
            offspring2 = np.where(mask, parent2, parent1)
            
            return offspring1, offspring2
        else:
            # Return copies of parents if no crossover
            return parent1.copy(), parent2.copy()
    
    def _mutation(self, individual: Sequence[float]) -> Sequence[float]:
        """Perform mutation on an individual.
        
        Args:
            individual: Solution to mutate
            
        Returns:
            Mutated solution
        """
        # Generate mutation mask
        mutation_mask = self.rng.random(self.solution_length) < self.mutation_prob
        
        if np.any(mutation_mask):
            # Generate mutations
            mutations = self.rng.randn(self.solution_length) * self.mutation_strength
            
            # Apply mutations
            individual = individual.copy()
            individual[mutation_mask] += mutations[mutation_mask]
            
            # Enforce bounds
            if self.bounds is not None:
                lower, upper = self.bounds
                np.clip(individual, lower, upper, out=individual)
                
            # Handle discrete case
            if self.discrete and self.bounds is not None:
                individual = np.floor(individual).astype(np.int32).astype(np.float32)
                
        return individual
    
    def tell(self, fitnesses: Sequence[float], tolerance: float = 1e-6) -> float:
        """Update population based on fitness values.
        
        Args:
            fitnesses: Array of fitness values for each solution
            tolerance: Minimum improvement threshold for early stopping
            
        Returns:
            Improvement in best fitness
        """
        if len(fitnesses) != self.population_count:
            raise ValueError(f"Mismatch between population size ({self.population_count}) and fitness values ({len(fitnesses)})")
        
        # Convert fitnesses to array
        fitnesses = np.array(fitnesses)
        
        # Find the best individual and calculate improvement
        best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[best_idx]
        
        # Calculate improvement
        if hasattr(self, 'previous_best'):
            improvement = current_best_fitness - self.previous_best
        else:
            improvement = float('inf')
            
        self.previous_best = current_best_fitness
        self.best_solution = self.population[best_idx].copy()
        
        # Sort population by fitness (descending order)
        sorted_indices = np.argsort(fitnesses)[::-1]
        self.population = self.population[sorted_indices]
        sorted_fitnesses = fitnesses[sorted_indices]
        
        # Create new population
        new_population = np.zeros_like(self.population)
        
        # Elitism: keep the best individuals unchanged
        new_population[:self.elite_count] = self.population[:self.elite_count]
        
        # Fill the rest of the population through selection, crossover, and mutation
        for i in range(self.elite_count, self.population_count, 2):
            # Select parents
            if self.selection_method == "tournament":
                parent1_idx = self._tournament_selection(sorted_fitnesses, k=self.tournament_size)
                parent2_idx = self._tournament_selection(sorted_fitnesses, k=self.tournament_size)
            else:  # roulette wheel selection
                parent1_idx = self._roulette_selection(sorted_fitnesses)
                parent2_idx = self._roulette_selection(sorted_fitnesses)
                
            # Get parent individuals
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Perform crossover
            offspring1, offspring2 = self._crossover(parent1, parent2)
            
            # Perform mutation
            offspring1 = self._mutation(offspring1)
            offspring2 = self._mutation(offspring2)
            
            # Add to new population
            new_population[i] = offspring1
            if i + 1 < self.population_count:
                new_population[i + 1] = offspring2
                
        # Update population
        self.population = new_population
        
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
        return self.population[0].copy()
    
    def get_stats(self):
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
    
    def save_state(self, filename: str):
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        np.savez(
            filename, 
            population=self.population,
            generation=self.generation,
            solution_length=self.solution_length,
            population_count=self.population_count,
            elite_count=self.elite_count,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
            mutation_strength=self.mutation_strength,
            selection_method=self.selection_method,
            tournament_size=self.tournament_size,
            bounds=self.bounds,
            discrete=self.discrete,
            previous_best=getattr(self, 'previous_best', None),
            best_solution=getattr(self, 'best_solution', None)
        )
    
    @classmethod
    def load_state(cls, filename: str):
        """Load optimizer state from file.
        
        Args:
            filename: Path to load the state from
            
        Returns:
            GeneticAlgorithm instance with loaded state
        """
        data = np.load(filename, allow_pickle=True)
        
        # Create optimizer with basic parameters
        optimizer = cls(
            solution_length=int(data['solution_length']),
            population_count=int(data['population_count']),
            elite_count=int(data['elite_count']),
            crossover_prob=float(data['crossover_prob']),
            mutation_prob=float(data['mutation_prob']),
            mutation_strength=float(data['mutation_strength']),
            selection_method=str(data['selection_method']),
            tournament_size=int(data['tournament_size']),
            bounds=data['bounds'] if 'bounds' in data and data['bounds'] is not None else None,
            discrete=bool(data['discrete'])
        )
        
        # Load state
        optimizer.population = data['population']
        optimizer.generation = int(data['generation'])
        
        if 'previous_best' in data and data['previous_best'] is not None:
            optimizer.previous_best = float(data['previous_best'])
            
        if 'best_solution' in data and data['best_solution'] is not None:
            optimizer.best_solution = data['best_solution']
            
        return optimizer
    
    def reset(self, population: Optional[Sequence[Sequence[float]]] = None, crossover_prob: Optional[float] = None, mutation_prob: Optional[float] = None, mutation_strength: Optional[float] = None):
        """Reset the optimizer.
        
        Args:
            population: New initial population (default: reinitialize randomly)
            crossover_prob: New crossover probability (default: keep current)
            mutation_prob: New mutation probability (default: keep current)
            mutation_strength: New mutation strength (default: keep current)
        """
        if population is not None:
            if population.shape != (self.population_count, self.solution_length):
                raise ValueError(f"Population shape mismatch. Expected {(self.population_count, self.solution_length)}, got {population.shape}")
            self.population = population.astype(np.float32)
        else:
            self.initialize_population()
            
        # Update parameters if provided
        if crossover_prob is not None:
            self.crossover_prob = crossover_prob
            
        if mutation_prob is not None:
            self.mutation_prob = mutation_prob
            
        if mutation_strength is not None:
            self.mutation_strength = mutation_strength
            
        # Reset state
        self.generation = 0
        if hasattr(self, 'previous_best'):
            delattr(self, 'previous_best')
        if hasattr(self, 'best_solution'):
            delattr(self, 'best_solution') 