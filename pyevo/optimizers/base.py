from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional
import numpy as np


def pack_optional(state: dict, name: str, value: Any) -> None:
    """Store an optional value in a dict destined for np.savez.

    np.savez turns ``None`` into a 0-d object array, which np.load then refuses
    to read back without allow_pickle=True. Storing an explicit presence flag
    alongside a placeholder keeps checkpoints loadable without pickling.

    Args:
        state: Dict of arrays being assembled for np.savez
        name: Key to store the value under
        value: The value, or None if absent
    """
    state[f"has_{name}"] = bool(value is not None)
    state[name] = value if value is not None else np.zeros(0, dtype=np.float32)


def unpack_optional(data: Any, name: str) -> Optional[np.ndarray]:
    """Read back a value written by pack_optional().

    Args:
        data: NpzFile returned by np.load
        name: Key the value was stored under

    Returns:
        The stored value, or None if it was absent when saved
    """
    flag = f"has_{name}"
    if flag in data:
        return data[name] if bool(data[flag]) else None
    # Backwards compatibility with checkpoints written before the flag existed.
    if name not in data:
        return None
    value = data[name]
    return None if value.dtype == object or value.shape == () else value


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

        Implementations may report any keys they like, but must include
        "best_fitness" (the best fitness seen so far, or None before the first
        tell()). Callers such as optimize_with_acceleration rely on it.

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
    
    @classmethod
    @abstractmethod
    def load_state(cls, filename: str) -> "Optimizer":
        """Reconstruct an optimizer from a file written by save_state.

        Args:
            filename: Path to load the state from

        Returns:
            An optimizer instance restored to the saved state
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