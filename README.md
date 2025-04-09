# PySNES: Separable Natural Evolution Strategies in Python

A Python implementation of Separable Natural Evolution Strategies (SNES), a powerful black-box optimization algorithm for high-dimensional continuous domains.

## What is SNES?

Separable Natural Evolution Strategies (SNES) is an evolutionary algorithm that excels at optimizing:
- High-dimensional problems (hundreds or thousands of parameters)
- Non-differentiable objective functions
- Multimodal landscapes with many local optima
- Problems where gradient information is unavailable

SNES works by:
1. Maintaining a normal distribution over the search space (with mean μ and standard deviation σ)
2. Sampling solutions from this distribution
3. Evaluating the fitness of each solution
4. Adjusting μ and σ based on the fitness rankings (not absolute values)

The "separable" aspect means each dimension is treated independently, which reduces computational complexity and makes the algorithm suitable for high-dimensional problems.

## Core Features

- Clean, NumPy-based implementation optimized for performance
- Flexible API with sensible defaults
- Comprehensive documentation and examples
- Minimal dependencies (just NumPy)
- Vectorized operations for efficient computation
- Parallel processing support for fitness evaluation
- Early stopping to avoid wasted computation
- Checkpointing for long-running optimizations
- Progress tracking and statistics

## Installation

```bash
pip install pysnes
```

Or directly from the repository:

```bash
git clone https://github.com/AntonioLujanoLuna/pysnes.git
cd pysnes
pip install -e .
```

## Quick Example

```python
import numpy as np
from pysnes import SNES

# Define objective function (minimize x^2)
def objective(x):
    return -np.sum(x**2)  # Negative because SNES maximizes

# Initialize optimizer (50-dimensional problem)
optimizer = SNES(solution_length=50)

# Run for 100 iterations
for i in range(100):
    # Get population of solutions
    solutions = optimizer.ask()
    
    # Evaluate fitness for each solution
    fitnesses = [objective(x) for x in solutions]
    
    # Update optimizer
    improvement = optimizer.tell(fitnesses)
    
    # Print progress
    if i % 10 == 0:
        best = optimizer.get_best_solution()
        print(f"Iteration {i}, Best fitness: {-np.sum(best**2):.6f}")
        
    # Early stopping if no improvement
    if improvement < 1e-6:
        print("Early stopping: no improvement")
        break
```

## Examples

This repository includes examples demonstrating SNES in action:

1. **ASCII Evolution**: Evolving text to match a target string
   ```bash
   python examples/ascii_example.py
   ```

2. **Image Approximation**: Approximating images using rectangles
   ```bash
   python examples/image_approx.py --image examples/resources/image.jpg --rects 200 --output-dir examples/output
   ```
   
   This example demonstrates approximating an image using rectangles evolved with SNES. The script includes:
   - Vectorized operations for efficient computation
   - Parallel processing for fitness evaluation
   - Support for creating animated GIFs of the optimization process
   - Automatic output directory management
   - Progress visualization with tqdm progress bar
   - Early stopping to avoid wasted computation
   - Checkpointing for long-running optimizations
   
   Additional options:
   - `--max-size`: Maximum size for the image (default: 128)
   - `--epochs`: Number of generations to evolve (default: 1000)
   - `--population`: Population size (default: 32)
   - `--alpha`: Learning rate (default: 0.05)
   - `--gif-frames`: Number of frames in the evolution GIF (default: 50)
   - `--gif-fps`: Frames per second in the GIF (default: 10)
   - `--workers`: Number of worker processes for parallel fitness evaluation
   - `--early-stop`: Early stopping tolerance (default: 1e-6)
   - `--patience`: Number of epochs to wait before early stopping (default: 10)
   - `--checkpoint`: Path to save optimizer checkpoint
   - `--no-display`: Disable live progress visualization

## Advanced Features

### Parallel Processing

For computationally intensive fitness functions, you can use parallel processing:

```python
from concurrent.futures import ProcessPoolExecutor
import functools

def parallel_fitness(solutions, objective_func, max_workers=None):
    """Evaluate fitness of multiple solutions in parallel."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(objective_func, solutions))

# In your optimization loop
fitnesses = parallel_fitness(solutions, objective, max_workers=4)
```

### Early Stopping

SNES supports early stopping to avoid wasted computation:

```python
# In your optimization loop
improvement = optimizer.tell(fitnesses, tolerance=1e-6)
if improvement < 1e-6:
    print("Early stopping: no improvement")
    break
```

### Checkpointing

Save and load optimizer state for long-running optimizations:

```python
# Save state
optimizer.save_state("checkpoint.npz")

# Load state
optimizer = SNES.load_state("checkpoint.npz")
```

### Statistics

Get statistics about the current optimizer state:

```python
stats = optimizer.get_stats()
print(f"Center mean: {stats['center_mean']:.6f}")
print(f"Sigma mean: {stats['sigma_mean']:.6f}")
print(f"Best fitness: {stats['best_fitness']:.6f}")
```

## How SNES Works

SNES maintains a probabilistic model of where good solutions are likely to be found in the search space. The model consists of:

- A mean vector μ that represents the current best estimate of the optimal solution
- A standard deviation vector σ that controls the exploration rate in each dimension

In each iteration:

1. Sample a population of solutions around μ using the normal distribution N(μ, σ²)
2. Evaluate the fitness of each solution
3. Rank solutions by fitness (not absolute values)
4. Calculate utility weights based on ranks (better solutions get higher weights)
5. Update μ by moving toward better solutions (weighted by utility)
6. Update σ by increasing/decreasing exploration based on solution utility

This process adapts the search distribution to focus on promising regions while maintaining appropriate exploration.

## SNES vs. Other Optimization Methods

- **Compared to gradient descent**: No need for gradients, better at escaping local optima
- **Compared to CMA-ES**: More efficient for high-dimensional problems but less powerful for correlated parameters
- **Compared to genetic algorithms**: More sample-efficient, with principled adaptation of the search distribution

## References

1. Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J., & Schmidhuber, J. (2014). "Natural Evolution Strategies." Journal of Machine Learning Research, 15(1), 949-980.
2. Schaul, T., Glasmachers, T., & Schmidhuber, J. (2011). "High dimensions and heavy tails for natural evolution strategies." In Proceedings of the 13th annual conference on Genetic and evolutionary computation (pp. 845-852).

## License

MIT License - See LICENSE file for details.