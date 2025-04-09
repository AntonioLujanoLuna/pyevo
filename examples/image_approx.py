#!/usr/bin/env python3
"""
Image Approximation Example

This example demonstrates approximating an image using rectangles evolved with SNES.
Includes support for creating animated GIFs of the optimization process.
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import argparse
import time
import sys
import os
import imageio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import functools
from tqdm import tqdm

# Add parent directory to path to import the SNES module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from snes import SNES

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Image approximation using SNES')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to the image (required)')
    parser.add_argument('--rects', type=int, default=200,
                        help='Number of rectangles')
    parser.add_argument('--max-size', type=int, default=128,
                        help='Maximum size for the image (larger images will be resized)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of generations to evolve')
    parser.add_argument('--population', type=int, default=32,
                        help='Population size')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the final image')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files (default: output)')
    parser.add_argument('--gif', type=str, default=None,
                        help='Path to save evolution GIF')
    parser.add_argument('--gif-frames', type=int, default=50,
                        help='Number of frames to include in the GIF')
    parser.add_argument('--gif-fps', type=float, default=10,
                        help='Frames per second in the GIF')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display live progress')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes for parallel fitness evaluation')
    parser.add_argument('--early-stop', type=float, default=1e-6,
                        help='Early stopping tolerance')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to save optimizer checkpoint')
    return parser.parse_args()

def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))

def softplus(x):
    """Softplus activation function."""
    return np.log(1.0 + np.exp(x))

def load_image(image_path, max_size=256):
    """Load and resize an image."""
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Resize image while maintaining aspect ratio
    ratio = min(1.0, max_size / img.width, max_size / img.height)
    new_width = int(img.width * ratio)
    new_height = int(img.height * ratio)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert to numpy array
    return np.array(img)

def draw_solution(solution, width, height, num_rects, param_count):
    """Draw rectangles based on solution parameters."""
    # Create a blank white image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Reshape solution into a 2D array for vectorized operations
    solution_reshaped = solution.reshape(num_rects, param_count)
    
    # Vectorized calculations for positions
    x = width / 2 + (solution_reshaped[:, 0] * width) / 2
    y = height / 2 + (solution_reshaped[:, 1] * height) / 2
    
    # Vectorized calculations for sizes
    w = (softplus(solution_reshaped[:, 2]) * width) / 8
    h = (softplus(solution_reshaped[:, 3]) * height) / 8
    
    # Vectorized calculations for colors
    r = (sigmoid(solution_reshaped[:, 4]) * 255).astype(np.uint8)
    g = (sigmoid(solution_reshaped[:, 5]) * 255).astype(np.uint8)
    b = (sigmoid(solution_reshaped[:, 6]) * 255).astype(np.uint8)
    
    # Draw rectangles (still need loop for PIL)
    for i in range(num_rects):
        draw.rectangle(
            [(x[i] - w[i]/2, y[i] - h[i]/2), (x[i] + w[i]/2, y[i] + h[i]/2)],
            fill=(r[i], g[i], b[i])
        )
    
    return np.array(img)

def fitness(solution, target_image, num_rects, param_count, width, height):
    """Calculate negative squared error between solution image and target."""
    # Draw solution
    img = draw_solution(solution, width, height, num_rects, param_count)
    
    # Calculate error (mean squared error) - already vectorized
    error = np.mean((img.astype(np.float32) - target_image.astype(np.float32)) ** 2)
    return -error

def create_comparison_image(target_image, solution_image, epoch, max_epochs):
    """Create a comparison image with target and solution side by side."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display target image
    ax1.imshow(target_image)
    ax1.set_title("Target Image")
    ax1.axis('off')
    
    # Display solution image
    ax2.imshow(solution_image)
    ax2.set_title(f"Epoch {epoch}/{max_epochs}")
    ax2.axis('off')
    
    # Add overall title
    fig.suptitle(f"Image Approximation Progress - Epoch {epoch}")
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    comparison = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return comparison

def parallel_fitness(solutions, target_image, num_rects, param_count, width, height, max_workers=None):
    """Evaluate fitness of multiple solutions in parallel."""
    partial_fitness = functools.partial(
        fitness, target_image=target_image, num_rects=num_rects, 
        param_count=param_count, width=width, height=height
    )
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(partial_fitness, solutions))

def main():
    """Run the image approximation example."""
    args = parse_args()
    
    if args.image is None:
        print("Error: --image argument is required")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default output paths if not specified
    if args.output is None:
        # Extract base name from input image
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join(args.output_dir, f"{base_name}_approximated.jpg")
    
    if args.gif is None:
        # Extract base name from input image
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.gif = os.path.join(args.output_dir, f"{base_name}_evolution.gif")
    
    # Parameters
    num_rects = args.rects
    param_count = 7  # x, y, width, height, r, g, b
    solution_length = num_rects * param_count
    population_count = args.population
    alpha = args.alpha
    max_epochs = args.epochs
    
    print(f"Loading image: {args.image}")
    print(f"Number of rectangles: {num_rects}")
    print(f"Maximum size: {args.max_size}x{args.max_size}")
    print(f"Population size: {population_count}")
    print(f"Epochs: {max_epochs}")
    print(f"Learning rate (alpha): {alpha}")
    if args.workers:
        print(f"Using {args.workers} worker processes")
    
    # Load target image
    target_image = load_image(args.image, max_size=args.max_size)
    height, width = target_image.shape[:2]
    
    print(f"Image dimensions after resizing: {width}x{height}")
    
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
    
    # Initialize optimizer
    optimizer = SNES(
        solution_length=solution_length,
        population_count=population_count,
        alpha=alpha,
        center=center,
        sigma=sigma,
        random_seed=args.seed
    )
    
    # Setup visualization
    if not args.no_display:
        plt.figure(figsize=(6, 6))
        plt.ion()  # Interactive mode
    
    # Setup for GIF creation
    creating_gif = args.gif is not None
    if creating_gif:
        # Calculate at which epochs we should capture frames
        frames_to_capture = args.gif_frames
        if frames_to_capture >= max_epochs:
            # Capture every epoch if we want more frames than epochs
            capture_frequency = 1
            frames_to_capture = max_epochs
        else:
            # Otherwise capture at regular intervals
            capture_frequency = max(1, max_epochs // frames_to_capture)
        
        # Initialize list to store frames
        gif_frames = []
        print(f"Will create GIF with {frames_to_capture} frames")
    
    # Run optimization
    start_time = time.time()
    best_fitness = float('-inf')
    best_solution = None
    stagnation_counter = 0
    
    # Pre-allocate fitness array
    fitnesses = np.zeros(population_count, dtype=np.float32)
    
    # Create progress bar
    pbar = tqdm(range(max_epochs), desc="Optimizing")
    
    for epoch in pbar:
        # Generate and evaluate solutions
        solutions = optimizer.ask()
        
        # Evaluate fitness (parallel if workers specified)
        if args.workers:
            fitnesses = parallel_fitness(
                solutions, target_image, num_rects, param_count, 
                width, height, max_workers=args.workers
            )
        else:
            # Vectorized fitness calculation
            for i in range(population_count):
                # Draw solution
                img = draw_solution(solutions[i], width, height, num_rects, param_count)
                # Calculate error (mean squared error)
                fitnesses[i] = -np.mean((img.astype(np.float32) - target_image.astype(np.float32)) ** 2)
        
        # Update optimizer and check for improvement
        improvement = optimizer.tell(fitnesses, tolerance=args.early_stop)
        
        # Early stopping check
        if improvement < args.early_stop:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            
        if stagnation_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}: no improvement for {args.patience} epochs")
            break
        
        # Track best solution
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_solution = solutions[current_best_idx].copy()
            
            # Save checkpoint if requested
            if args.checkpoint:
                checkpoint_path = os.path.join(args.output_dir, args.checkpoint)
                optimizer.save_state(checkpoint_path)
        
        # Current solution to visualize
        current_solution = optimizer.get_best_solution()
        current_img = draw_solution(current_solution, width, height, num_rects, param_count)
        
        # Capture frame for GIF if needed
        if creating_gif and (epoch % capture_frequency == 0 or epoch == max_epochs - 1):
            comparison = create_comparison_image(target_image, current_img, epoch, max_epochs)
            gif_frames.append(comparison)
        
        # Visualize progress periodically
        if (epoch % 50 == 0 or epoch == max_epochs - 1) and not args.no_display:
            plt.clf()
            plt.imshow(current_img)
            plt.title(f"Epoch {epoch}/{max_epochs}")
            plt.pause(0.01)
        
        # Update progress bar
        pbar.set_postfix({
            'error': f"{-best_fitness:.2f}",
            'improvement': f"{improvement:.2e}"
        })
    
    total_time = time.time() - start_time
    print(f"\nOptimization completed in {total_time:.2f} seconds")
    
    # Print final statistics
    stats = optimizer.get_stats()
    print("\nFinal optimizer statistics:")
    for key, value in stats.items():
        if value is not None:
            print(f"{key}: {value:.6f}")
    
    # Use either the best found solution or the final center
    final_solution = best_solution if best_solution is not None else optimizer.get_best_solution()
    final_img = draw_solution(final_solution, width, height, num_rects, param_count)
    
    # Display final result
    if not args.no_display:
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
        plt.show()
    
    # Save output image if requested
    if args.output:
        # Ensure the output path is in the output directory
        if not os.path.isabs(args.output):
            output_path = os.path.join(args.output_dir, args.output)
        else:
            output_path = args.output
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        Image.fromarray(final_img.astype(np.uint8)).save(output_path)
        print(f"Final image saved to: {output_path}")
    
    # Save GIF if requested
    if creating_gif and gif_frames:
        # Convert frames to 8-bit
        gif_frames = [frame.astype(np.uint8) for frame in gif_frames]
        
        # Ensure the gif path is in the output directory
        if not os.path.isabs(args.gif):
            gif_path = os.path.join(args.output_dir, args.gif)
        else:
            gif_path = args.gif
            
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
        
        # Save GIF
        imageio.mimsave(gif_path, gif_frames, fps=args.gif_fps)
        print(f"Evolution GIF saved to: {gif_path}")
    
    # Print final error metrics
    final_error = np.mean((final_img.astype(np.float32) - target_image.astype(np.float32)) ** 2)
    print(f"Final mean squared error: {final_error:.2f}")
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    if final_error > 0:
        psnr = 10 * np.log10((255 ** 2) / final_error)
        print(f"PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    main()