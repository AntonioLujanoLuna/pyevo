from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional
import numpy as np

class Optimizer(ABC):
    """Base optimizer interface.
    
    All optimization algorithms should implement this interface to ensure
    compatibility with the existing codebase.
    """
    
    @abstractmethod
    def ask(self) -> np.ndarray:
        """Generate a new batch of solutions to evaluate.
        
        Returns:
            Array of solutions with shape (population_count, solution_length)
        """
        pass
    
    @abstractmethod
    def tell(self, fitnesses: Sequence[float], tolerance: float = 1e-6) -> float:
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
    def get_best_solution(self) -> np.ndarray:
        """Return current best estimate of the solution.
        
        Returns:
            Current best solution vector
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return current optimizer statistics.
        
        Returns:
            Dictionary containing statistics about the current state
        """
        pass
    
    @abstractmethod
    def save_state(self, filename: str) -> None:
        """Save optimizer state to file.
        
        Args:
            filename: Path to save the state
        """
        pass
    
    @abstractmethod
    def reset(self, center: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None) -> None:
        """Reset the optimizer with optional new parameters.
        
        Args:
            center: New center/mean vector
            sigma: New standard deviation vector
        """
        pass 