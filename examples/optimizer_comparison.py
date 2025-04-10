"""
Optimizer Comparison Example

This script demonstrates how to compare different optimization algorithms
in PyEvo by applying them to standard benchmark functions.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import create_optimizer
from utils.constants import OPTIMIZERS, DEFAULT_OUTPUT_DIR

# Set random seed for reproducibility
RANDOM_SEED = 42

# Benchmark functions
def rosenbrock(x):
    """Rosenbrock function (banana function)
    
    Global minimum at (1, 1, ..., 1) with value 0
    Difficult function with a narrow valley
    """
    n = len(x)
    return -np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    """Rastrigin function
    
    Global minimum at (0, 0, ..., 0) with value 0
    Highly multimodal function with many local minima
    """
    n = len(x)
    return -10.0 * n - np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))

def sphere(x):
    """Sphere function
    
    Global minimum at (0, 0, ..., 0) with value 0
    Simple unimodal function with a single minimum
    """
    return -np.sum(x**2)

def ackley(x):
    """Ackley function
    
    Global minimum at (0, 0, ..., 0) with value 0
    Multimodal function with many local minima
    """
    n = len(x)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return -(term1 + term2 + 20 + np.exp(1))

# Available benchmark functions
BENCHMARK_FUNCTIONS = {
    "rosenbrock": {
        "function": rosenbrock,
        "bounds": (-5, 10),  # (min, max) bounds for initialization
        "optimum": 1.0,  # Value at each dimension of the optimum
    },
    "rastrigin": {
        "function": rastrigin,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
    },
    "sphere": {
        "function": sphere,
        "bounds": (-5, 5),
        "optimum": 0.0,
    },
    "ackley": {
        "function": ackley,
        "bounds": (-32.768, 32.768),
        "optimum": 0.0,
    },
}

def run_optimization(optimizer_type, benchmark_name, dimensions=10, max_iterations=500, trials=5):
    """Run optimization using the specified optimizer and benchmark function.
    
    Args:
        optimizer_type: Type of optimizer to use
        benchmark_name: Name of benchmark function
        dimensions: Number of dimensions for the problem
        max_iterations: Maximum number of iterations
        trials: Number of trials to run
        
    Returns:
        Dictionary with optimization results
    """
    if benchmark_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown benchmark function: {benchmark_name}")
    
    benchmark = BENCHMARK_FUNCTIONS[benchmark_name]
    fitness_function = benchmark["function"]
    bounds = benchmark["bounds"]
    
    # For storing results
    all_best_fitnesses = []
    all_runtimes = []
    
    for trial in range(trials):
        print(f"Running {optimizer_type} on {benchmark_name} (trial {trial+1}/{trials})...")
        
        start_time = time.time()
        
        # Initialize optimizer
        # Set center at middle of bounds and sigma to span 1/10 of bounds
        center = np.ones(dimensions) * (bounds[0] + bounds[1]) / 2
        sigma = np.ones(dimensions) * (bounds[1] - bounds[0]) / 10
        
        # Create optimizer
        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            solution_length=dimensions,
            center=center,
            sigma=sigma,
            random_seed=RANDOM_SEED + trial
        )
        
        # Track progress
        fitness_history = []
        
        # Run optimization
        for i in range(max_iterations):
            # Generate solutions
            solutions = optimizer.ask()
            
            # Evaluate fitness
            fitnesses = [fitness_function(solution) for solution in solutions]
            
            # Update optimizer
            improvement = optimizer.tell(fitnesses)
            
            # Track best fitness
            best_fitness = max(fitnesses)
            fitness_history.append(best_fitness)
            
            # Early stopping if improvement is minimal
            if abs(improvement) < 1e-10 and i > 100:
                break
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Save results from this trial
        all_best_fitnesses.append(fitness_history)
        all_runtimes.append(runtime)
    
    # Average results across trials
    max_length = max(len(history) for history in all_best_fitnesses)
    padded_histories = []
    
    for history in all_best_fitnesses:
        # Pad shorter histories with their last value
        if len(history) < max_length:
            padded = history + [history[-1]] * (max_length - len(history))
        else:
            padded = history
        padded_histories.append(padded)
    
    avg_fitness_history = np.mean(padded_histories, axis=0)
    std_fitness_history = np.std(padded_histories, axis=0)
    
    # Return results
    return {
        "optimizer": optimizer_type,
        "benchmark": benchmark_name,
        "dimensions": dimensions,
        "avg_fitness_history": avg_fitness_history,
        "std_fitness_history": std_fitness_history,
        "avg_runtime": np.mean(all_runtimes),
        "std_runtime": np.std(all_runtimes),
        "avg_iterations": np.mean([len(history) for history in all_best_fitnesses]),
        "final_best_fitness": avg_fitness_history[-1],
    }

def plot_results(results, output_dir=DEFAULT_OUTPUT_DIR):
    """Plot optimization results.
    
    Args:
        results: List of result dictionaries from run_optimization
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by benchmark
    benchmarks = {}
    for result in results:
        benchmark = result["benchmark"]
        if benchmark not in benchmarks:
            benchmarks[benchmark] = []
        benchmarks[benchmark].append(result)
    
    # Plot fitness history for each benchmark
    for benchmark, benchmark_results in benchmarks.items():
        plt.figure(figsize=(10, 6))
        
        for result in benchmark_results:
            optimizer = result["optimizer"]
            iterations = len(result["avg_fitness_history"])
            
            # Get color for this optimizer
            color = {"snes": "blue", "cmaes": "red", "pso": "green"}.get(optimizer, "purple")
            
            # Plot average fitness with error band
            x = np.arange(iterations)
            y = -result["avg_fitness_history"]  # Negate to show minimization
            err = result["std_fitness_history"]
            
            plt.plot(x, y, label=f"{OPTIMIZERS.get(optimizer, optimizer)}", color=color)
            plt.fill_between(x, y-err, y+err, alpha=0.2, color=color)
        
        plt.xlabel("Iterations")
        plt.ylabel("Function Value (lower is better)")
        plt.title(f"{benchmark.capitalize()} Function Optimization")
        plt.yscale("log")  # Log scale for better visualization
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        filename = os.path.join(output_dir, f"comparison_{benchmark}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {filename}")
        
        plt.close()
    
    # Plot runtime comparison
    plt.figure(figsize=(10, 6))
    
    # Collect data for bar chart
    optimizer_types = list(set(result["optimizer"] for result in results))
    benchmark_types = list(set(result["benchmark"] for result in results))
    
    bar_width = 0.8 / len(optimizer_types)
    
    for i, optimizer in enumerate(optimizer_types):
        runtimes = []
        errors = []
        
        for benchmark in benchmark_types:
            # Find result for this optimizer and benchmark
            result = next((r for r in results if r["optimizer"] == optimizer and r["benchmark"] == benchmark), None)
            
            if result:
                runtimes.append(result["avg_runtime"])
                errors.append(result["std_runtime"])
            else:
                runtimes.append(0)
                errors.append(0)
        
        x = np.arange(len(benchmark_types))
        offset = (i - len(optimizer_types)/2 + 0.5) * bar_width
        
        plt.bar(x + offset, runtimes, width=bar_width, yerr=errors, 
                label=OPTIMIZERS.get(optimizer, optimizer), 
                capsize=5)
    
    plt.xlabel("Benchmark Function")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison")
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(np.arange(len(benchmark_types)), [b.capitalize() for b in benchmark_types])
    plt.legend()
    
    # Save plot
    filename = os.path.join(output_dir, "comparison_runtime.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {filename}")
    
    plt.close()

def print_results_table(results):
    """Print a table of optimization results."""
    # Group results by benchmark
    benchmarks = {}
    for result in results:
        benchmark = result["benchmark"]
        if benchmark not in benchmarks:
            benchmarks[benchmark] = []
        benchmarks[benchmark].append(result)
    
    # Print header
    print("\n" + "=" * 80)
    print(f"{'Benchmark':<15} {'Optimizer':<30} {'Runtime (s)':<15} {'Iterations':<15} {'Best Value':<15}")
    print("-" * 80)
    
    # Print results for each benchmark
    for benchmark, benchmark_results in benchmarks.items():
        for result in benchmark_results:
            optimizer = result["optimizer"]
            optimizer_name = OPTIMIZERS.get(optimizer, optimizer)
            
            print(f"{benchmark.capitalize():<15} {optimizer_name:<30} "
                  f"{result['avg_runtime']:<15.2f} {result['avg_iterations']:<15.1f} "
                  f"{-result['final_best_fitness']:<15.6f}")
    
    print("=" * 80 + "\n")

def main():
    """Run the optimizer comparison example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare optimization algorithms on benchmark functions")
    parser.add_argument("--dimensions", type=int, default=10, help="Number of dimensions for the benchmark functions")
    parser.add_argument("--iterations", type=int, default=300, help="Maximum number of iterations")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials for each optimizer/benchmark combination")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["sphere", "rosenbrock", "rastrigin"],
                        help="Benchmark functions to use")
    parser.add_argument("--optimizers", type=str, nargs="+", default=["snes", "cmaes", "pso"],
                        help="Optimization algorithms to compare")
    
    args = parser.parse_args()
    
    # Validate inputs
    for benchmark in args.benchmarks:
        if benchmark not in BENCHMARK_FUNCTIONS:
            print(f"Warning: Unknown benchmark function '{benchmark}'. Available benchmarks:")
            for name in BENCHMARK_FUNCTIONS:
                print(f"  - {name}")
            return
    
    # Run optimizations
    results = []
    
    for optimizer in args.optimizers:
        for benchmark in args.benchmarks:
            result = run_optimization(
                optimizer_type=optimizer,
                benchmark_name=benchmark,
                dimensions=args.dimensions,
                max_iterations=args.iterations,
                trials=args.trials
            )
            results.append(result)
    
    # Print and plot results
    print_results_table(results)
    plot_results(results, args.output)

if __name__ == "__main__":
    main() 