#!/usr/bin/env python3
"""
Interactive Image Approximation Example

This example demonstrates approximating an image using rectangles evolved with SNES,
with interactive control during optimization.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import time
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from snes import SNES
from utils.interactive import InteractiveOptimizer
from examples.image_approx import (
    load_image, draw_solution, sigmoid, softplus, solution_hash
)

# Create a global solution cache
solution_cache = {}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive image approximation using SNES')
    parser.add_argument('--image', '-i', type=str, required=True,
                      help='Path to the image')
    parser.add_argument('--rects', '-r', type=int, default=100,
                      help='Number of rectangles')
    parser.add_argument('--max-size', '-m', type=int, default=128,
                      help='Maximum size for the image (larger images will be resized)')
    parser.add_argument('--population', '-p', type=int, default=32,
                      help='Population size')
    parser.add_argument('--alpha', '-a', type=float, default=0.05,
                      help='Learning rate')
    parser.add_argument('--max-iterations', type=int, default=1000,
                      help='Maximum iterations')
    parser.add_argument('--checkpoint-dir', type=str, default='examples/checkpoints',
                      help='Directory for checkpoints')
    parser.add_argument('--output-dir', '-d', type=str, default='examples/output',
                      help='Directory to save output files')
    parser.add_argument('--load', type=str, default=None,
                      help='Load a previous checkpoint file')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed')
    return parser.parse_args()

def create_fitness_function(target_image, num_rects, param_count, width, height):
    """Create a fitness function closure for the interactive optimizer."""
    def fitness(solution):
        """Calculate fitness for a solution (negative MSE)."""
        # Check cache first
        solution_key = solution_hash(solution, num_rects)
        if solution_key in solution_cache:
            return solution_cache[solution_key]
        
        # Draw solution
        img = draw_solution(solution, width, height, num_rects, param_count, use_gpu=False)
        
        # Calculate error (mean squared error)
        error = np.mean((img.astype(np.float32) - target_image.astype(np.float32)) ** 2)
        fitness_value = -error
        
        # Cache the result
        if len(solution_cache) >= 500:  # Smaller cache for interactive mode
            key_to_remove = next(iter(solution_cache))
            solution_cache.pop(key_to_remove)
        solution_cache[solution_key] = fitness_value
        
        return fitness_value
    
    return fitness

def main():
    """Run the interactive image approximation example."""
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load target image
    target_image = load_image(args.image, max_size=args.max_size)
    height, width = target_image.shape[:2]
    
    # Parameters
    num_rects = args.rects
    param_count = 7  # x, y, width, height, r, g, b
    solution_length = num_rects * param_count
    
    print("\n=== Interactive Image Approximation with SNES ===")
    print(f"Image: {args.image}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Number of rectangles: {num_rects}")
    print(f"Population size: {args.population}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Learning rate: {args.alpha}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Create fitness function
    fitness_func = create_fitness_function(target_image, num_rects, param_count, width, height)
    
    # Load from checkpoint or create new optimizer
    if args.load:
        print(f"Loading from checkpoint: {args.load}")
        interactive = InteractiveOptimizer.load_session(args.load, fitness_func)
    else:
        # Initialize center and sigma
        center = np.zeros(solution_length, dtype=np.float32)
        sigma = np.ones(solution_length, dtype=np.float32)
        
        # Initialize position parameters randomly - vectorized
        position_factor = 0.0  # 0.0 means start rectangles at the center
        position_indices = np.arange(0, solution_length, param_count)
        center[position_indices] = np.random.uniform(-1, 1, num_rects)
        center[position_indices + 1] = np.random.uniform(-1, 1, num_rects)
        sigma[position_indices] = position_factor
        sigma[position_indices + 1] = position_factor
        
        # Create new SNES optimizer
        optimizer = SNES(
            solution_length=solution_length,
            population_count=args.population,
            alpha=args.alpha,
            center=center,
            sigma=sigma,
            random_seed=args.seed
        )
        
        # Create interactive optimizer
        interactive = InteractiveOptimizer(
            optimizer=optimizer,
            fitness_function=fitness_func,
            max_iterations=args.max_iterations,
            checkpoint_dir=args.checkpoint_dir
        )
    
    # Setup visualization
    plt.figure(figsize=(8, 8))
    plt.ion()  # Interactive mode
    
    # Add visualization callback
    def update_visualization(optimizer, iteration):
        """Update the visualization periodically."""
        if iteration % 5 == 0:  # Update every 5 iterations
            current_solution = optimizer.get_best_solution()
            current_img = draw_solution(current_solution, width, height, num_rects, param_count)
            
            plt.clf()
            plt.imshow(current_img)
            plt.title(f"Iteration {iteration}")
            plt.pause(0.01)
    
    interactive.visualization_callback = update_visualization
    
    # Start interactive optimization
    print("\nStarting optimization. Enter commands in the terminal.")
    print("=" * 50)
    best_solution, best_fitness = interactive.start()
    
    # Display and save final result
    final_img = draw_solution(best_solution, width, height, num_rects, param_count)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(target_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(final_img)
    plt.title(f"Approximation with {num_rects} rectangles")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show(block=True)  # Block to prevent immediate exit
    
    # Save final image
    output_path = os.path.join(args.output_dir, f"interactive_result.jpg")
    Image.fromarray(final_img.astype(np.uint8)).save(output_path)
    print(f"Final image saved to: {output_path}")
    
    # Calculate error metrics
    final_error = np.mean((final_img.astype(np.float32) - target_image.astype(np.float32)) ** 2)
    print(f"Final mean squared error: {final_error:.2f}")
    
    # Calculate PSNR
    if final_error > 0:
        psnr = 10 * np.log10((255 ** 2) / final_error)
        print(f"PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    main() 