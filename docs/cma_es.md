# CMA-ES: Covariance Matrix Adaptation Evolution Strategy

## Mathematical Foundation
CMA-ES adapts a multivariate normal distribution to find the minimum of an objective function. It iteratively updates the mean, step-size (sigma), and covariance matrix of the distribution based on the fitness of sampled candidate solutions.

Key update equations involve:
- Evolution paths for step-size control (`p_sigma`) and covariance matrix adaptation (`p_c`).
- Rank-one update for the covariance matrix based on the evolution path `p_c`.
- Rank-mu update for the covariance matrix based on the best `mu` individuals.

*(Detailed equations and diagrams will be added here)*

## Parameter Guidance
- `population_count` (`lambda`): The number of offspring generated per iteration. A common default is `4 + floor(3*ln(N))`, where N is the problem dimension. Larger values provide more information but increase computational cost.
- `mu`: The number of parent individuals selected for recombination. Typically set to `floor(lambda / 2)`.
- `sigma`: The initial step size or standard deviation. This parameter is critical. Too small can lead to premature convergence, too large can slow down convergence initially. It should roughly match the expected distance to the optimum.
- `weights`: Recombination weights for the selected `mu` individuals. Usually set to decrease logarithmically.
- `cc`: Time constant for cumulation for the rank-one update path (`p_c`).
- `cs`: Time constant for cumulation for the step-size control path (`p_sigma`).
- `c1`: Learning rate for the rank-one update of the covariance matrix.
- `cmu`: Learning rate for the rank-mu update of the covariance matrix.

*(Further details on parameter tuning and sensitivity will be added)*

## Implementation Details
Our implementation follows the standard CMA-ES approach, often based on Hansen's tutorials and papers. Key components include:
1. Sampling new solutions from the current multivariate normal distribution.
2. Evaluating the fitness of the sampled solutions.
3. Selecting and recombining the best `mu` solutions to update the mean.
4. Updating the evolution path `p_sigma` for step-size control.
5. Updating the step-size `sigma`.
6. Updating the evolution path `p_c` for covariance matrix adaptation.
7. Updating the covariance matrix `C` using rank-one and rank-mu updates.
8. Performing eigenvalue decomposition of `C` for efficient sampling in subsequent iterations (optional but common).

*(Specific implementation choices and potential optimizations will be detailed here)*

## When to Use CMA-ES
CMA-ES is particularly effective for:
- Non-linear, non-convex optimization problems.
- Problems where gradient information is unavailable or unreliable.
- Moderate-dimensional problems (up to a few hundred dimensions, though performance can degrade).
- Problems requiring high precision in the solution.

**Strengths:**
- State-of-the-art performance on many benchmark functions.
- Relatively few parameters to tune compared to some other EAs.
- Invariant to rotations of the coordinate system.

**Weaknesses:**
- Can be computationally expensive, especially the covariance matrix update (O(N^2)) and eigendecomposition (O(N^3)).
- Performance can degrade in very high dimensions.
- May struggle with multi-modal problems if the population size is too small.

*(Comparisons to other algorithms like PSO, DE, SNES will be added)* 