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
import hashlib

# Add parent directory to path to import the SNES module and utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyevo.optimizers import SNES
from pyevo.utils.image import calculate_ssim, convolve2d
from pyevo.utils.constants import (
    DEFAULT_MAX_IMAGE_SIZE, DEFAULT_RECT_COUNT, DEFAULT_POPULATION_SIZE,
    DEFAULT_EPOCHS, DEFAULT_ALPHA, DEFAULT_EARLY_STOP, DEFAULT_PATIENCE,
    DEFAULT_OUTPUT_DIR, DEFAULT_CHECKPOINT_DIR
)

# We're using our own pure NumPy SSIM implementation
HAS_SSIM = True

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy found - GPU acceleration available")
except ImportError:
    HAS_CUPY = False
    print("CuPy not found - using NumPy only")

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Image approximation using SNES')
    parser.add_argument('--image', '-i', type=str, default=None,
                        help='Path to the image (required)')
    parser.add_argument('--rects', '-r', type=int, default=DEFAULT_RECT_COUNT,
                        help=f'Number of rectangles (default: {DEFAULT_RECT_COUNT})')
    parser.add_argument('--max-size', '-m', type=int, default=DEFAULT_MAX_IMAGE_SIZE,
                        help=f'Maximum size for the image (larger images will be resized) (default: {DEFAULT_MAX_IMAGE_SIZE})')
    parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of generations to evolve (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--population', '-p', type=int, default=DEFAULT_POPULATION_SIZE,
                        help=f'Population size (default: {DEFAULT_POPULATION_SIZE})')
    parser.add_argument('--alpha', '-a', type=float, default=DEFAULT_ALPHA,
                        help=f'Learning rate (default: {DEFAULT_ALPHA})')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the final image')
    parser.add_argument('--output-dir', '-d', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save output files (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--gif', '-g', type=str, default=None,
                        help='Path to save evolution GIF')
    parser.add_argument('--mp4', '-v', type=str, default=None,
                        help='Path to save evolution as MP4 video')
    parser.add_argument('--gif-frames', type=int, default=50,
                        help='Number of frames to include in the animation')
    parser.add_argument('--gif-fps', type=float, default=10,
                        help='Frames per second in the animation')
    parser.add_argument('--display-interval', type=int, default=50,
                        help='How often to update the display (every N epochs)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display live progress')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of worker processes for parallel fitness evaluation')
    parser.add_argument('--early-stop', type=float, default=DEFAULT_EARLY_STOP,
                        help=f'Early stopping tolerance (default: {DEFAULT_EARLY_STOP})')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
                        help=f'Number of epochs to wait before early stopping (default: {DEFAULT_PATIENCE})')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to save optimizer checkpoint')
    parser.add_argument('--use-gpu', '-gpu', action='store_true',
                        help='Use GPU acceleration via CuPy if available')
    parser.add_argument('--cache-size', type=int, default=1000,
                        help='Size of solution cache for fitness evaluation')
    parser.add_argument('--quality-metrics', '-q', action='store_true',
                        help='Calculate and display additional quality metrics (SSIM)')
    parser.add_argument('--gpu-batch-size', type=int, default=None,
                        help='Batch size for GPU processing (default: auto-configured based on GPU memory)')
    return parser.parse_args()

def sigmoid(x):
    """
    Sigmoid activation function.
    
    Args:
        x (array-like): Input values.
        
    Returns:
        array-like: Output values in range (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))

def softplus(x):
    """
    Softplus activation function.
    
    Args:
        x (array-like): Input values.
        
    Returns:
        array-like: Smoothed positive values.
    """
    return np.log(1.0 + np.exp(x))

def load_image(image_path, max_size=DEFAULT_MAX_IMAGE_SIZE):
    """
    Load and resize an image.
    
    Args:
        image_path (str): Path to the image file.
        max_size (int, optional): Maximum dimension (width or height) for the image.
            Larger images will be resized while preserving aspect ratio.
            
    Returns:
        numpy.ndarray: Image as a NumPy array.
        
    Raises:
        SystemExit: If the image cannot be loaded.
    """
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

# Create a global solution cache
solution_cache = {}

def solution_hash(solution, num_rects, precision=3):
    """
    Create a hash for a solution to use as cache key.
    
    Args:
        solution (numpy.ndarray): Solution parameters.
        num_rects (int): Number of rectangles in the solution.
        precision (int, optional): Decimal precision for rounding to reduce
            cache misses for very similar solutions.
            
    Returns:
        str: MD5 hash of the rounded solution as a hexadecimal string.
    """
    # Round to reduce cache misses for very similar solutions
    rounded = np.round(solution, precision)
    return hashlib.md5(rounded.tobytes()).hexdigest()

def draw_solution_batch(solution, width, height, num_rects, param_count, use_gpu=False, gpu_batch_size=None):
    """
    Draw rectangles in batches for better performance using NumPy/CuPy operations.
    
    This method is optimized for larger numbers of rectangles by using
    batch processing and alpha compositing, with special optimizations for GPU.
    
    Args:
        solution (numpy.ndarray): Solution parameters.
        width (int): Image width.
        height (int): Image height.
        num_rects (int): Number of rectangles to draw.
        param_count (int): Number of parameters per rectangle.
        use_gpu (bool, optional): Whether to use GPU acceleration via CuPy.
        gpu_batch_size (int, optional): User-specified batch size for GPU processing.
        
    Returns:
        numpy.ndarray: RGB image as a NumPy array.
    """
    # Choose the right array library based on GPU usage
    xp = cp if use_gpu and HAS_CUPY else np
    
    # Ensure solution is on the correct device (CPU or GPU)
    if use_gpu and HAS_CUPY:
        # Only transfer to GPU if it's not already there
        if not isinstance(solution, cp.ndarray):
            solution_device = cp.asarray(solution)
        else:
            solution_device = solution
    else:
        solution_device = solution
    
    # Create a blank RGBA image (using alpha channel for faster compositing)
    img_array = xp.ones((height, width, 4), dtype=xp.uint8) * 255
    img_array[:, :, 3] = 0  # Start with transparent image
    
    # Reshape solution into a 2D array for vectorized operations
    solution_reshaped = solution_device.reshape(num_rects, param_count)
    
    # Vectorized calculations - precompute all parameters at once
    x = width / 2 + (solution_reshaped[:, 0] * width) / 2
    y = height / 2 + (solution_reshaped[:, 1] * height) / 2
    w = (softplus(solution_reshaped[:, 2]) * width) / 8
    h = (softplus(solution_reshaped[:, 3]) * height) / 8
    
    # Vectorized colors with alpha channel
    r = (sigmoid(solution_reshaped[:, 4]) * 255).astype(xp.uint8)
    g = (sigmoid(solution_reshaped[:, 5]) * 255).astype(xp.uint8)
    b = (sigmoid(solution_reshaped[:, 6]) * 255).astype(xp.uint8)
    
    # Process rectangles in batches for better memory efficiency
    # Group similar-sized rectangles together
    areas = w * h
    size_indices = xp.argsort(areas)
    
    # Optimize batch size for GPU vs CPU
    if use_gpu and HAS_CUPY:
        # Use user-specified batch size or default
        if gpu_batch_size is not None:
            batch_size = gpu_batch_size
        else:
            # Larger batches for GPU to better utilize parallelism
            batch_size = max(50, min(200, num_rects // 4))
    else:
        # Smaller batches for CPU to better use cache
        batch_size = max(1, min(50, num_rects // 10))
    
    # Process in batches from small to large
    for batch_start in range(0, num_rects, batch_size):
        batch_end = min(batch_start + batch_size, num_rects)
        batch_indices = size_indices[batch_start:batch_end]
        
        # Create a temporary buffer for this batch
        batch_buffer = xp.zeros((height, width, 4), dtype=xp.uint8)
        
        if use_gpu and HAS_CUPY and len(batch_indices) > 20:
            # For GPU, use vectorized operations for large batches
            
            # Pre-calculate rectangle coordinates for all in batch
            x1s = xp.maximum(0, (x[batch_indices] - w[batch_indices]/2).astype(xp.int32))
            y1s = xp.maximum(0, (y[batch_indices] - h[batch_indices]/2).astype(xp.int32))
            x2s = xp.minimum(width, (x[batch_indices] + w[batch_indices]/2).astype(xp.int32))
            y2s = xp.minimum(height, (y[batch_indices] + h[batch_indices]/2).astype(xp.int32))
            
            # Filter out too small rectangles
            valid_mask = (x2s > x1s) & (y2s > y1s)
            valid_indices = xp.where(valid_mask)[0]
            
            # Process valid rectangles
            for i in valid_indices:
                idx = batch_indices[i]
                x1, y1 = x1s[i], y1s[i]
                x2, y2 = x2s[i], y2s[i]
                
                # Assign color to the rectangle area (RGBA)
                batch_buffer[y1:y2, x1:x2, 0] = r[idx]
                batch_buffer[y1:y2, x1:x2, 1] = g[idx]
                batch_buffer[y1:y2, x1:x2, 2] = b[idx]
                batch_buffer[y1:y2, x1:x2, 3] = 255  # Fully opaque
        else:
            # For CPU or small batches, use loop-based approach
            for idx in batch_indices:
                # Calculate rectangle coordinates
                x1 = max(0, int(x[idx] - w[idx]/2))
                y1 = max(0, int(y[idx] - h[idx]/2))
                x2 = min(width, int(x[idx] + w[idx]/2))
                y2 = min(height, int(y[idx] + h[idx]/2))
                
                # Skip rectangles that are too small
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Assign color to the rectangle area (RGBA)
                batch_buffer[y1:y2, x1:x2, 0] = r[idx]
                batch_buffer[y1:y2, x1:x2, 1] = g[idx]
                batch_buffer[y1:y2, x1:x2, 2] = b[idx]
                batch_buffer[y1:y2, x1:x2, 3] = 255  # Fully opaque
        
        # Composite this batch with the main image - use in-place operations when possible
        alpha = batch_buffer[:, :, 3:4] / 255.0
        img_array[:, :, :3] = (1 - alpha) * img_array[:, :, :3] + alpha * batch_buffer[:, :, :3]
        img_array[:, :, 3] = xp.maximum(img_array[:, :, 3], batch_buffer[:, :, 3])
    
    # Extract RGB channels for result
    result = img_array[:, :, :3]
    
    # Convert back to NumPy array if using GPU
    if use_gpu and HAS_CUPY:
        result = cp.asnumpy(result)
        
    return result

def draw_solution(solution, width, height, num_rects, param_count, use_gpu=False, gpu_batch_size=None):
    """
    Draw rectangles based on solution parameters using NumPy for better performance.
    
    Args:
        solution (numpy.ndarray): Solution parameters.
        width (int): Image width.
        height (int): Image height.
        num_rects (int): Number of rectangles to draw.
        param_count (int): Number of parameters per rectangle.
        use_gpu (bool, optional): Whether to use GPU acceleration via CuPy.
        gpu_batch_size (int, optional): User-specified batch size for GPU processing.
        
    Returns:
        numpy.ndarray: RGB image as a NumPy array.
    """
    # Use batch drawing method for larger rectangle counts
    if num_rects > 50:
        return draw_solution_batch(solution, width, height, num_rects, param_count, use_gpu, gpu_batch_size)
    
    # Create a blank white image as NumPy array
    if use_gpu and HAS_CUPY:
        # Use CuPy for GPU acceleration
        img_array = cp.ones((height, width, 3), dtype=cp.uint8) * 255
        solution_reshaped = cp.asarray(solution.reshape(num_rects, param_count))
    else:
        img_array = np.ones((height, width, 3), dtype=np.uint8) * 255
        solution_reshaped = solution.reshape(num_rects, param_count)
    
    # Vectorized calculations for positions
    x = width / 2 + (solution_reshaped[:, 0] * width) / 2
    y = height / 2 + (solution_reshaped[:, 1] * height) / 2
    
    # Vectorized calculations for sizes
    w = (softplus(solution_reshaped[:, 2]) * width) / 8
    h = (softplus(solution_reshaped[:, 3]) * height) / 8
    
    # Vectorized calculations for colors
    r = (sigmoid(solution_reshaped[:, 4]) * 255).astype(np.uint8 if not use_gpu or not HAS_CUPY else cp.uint8)
    g = (sigmoid(solution_reshaped[:, 5]) * 255).astype(np.uint8 if not use_gpu or not HAS_CUPY else cp.uint8)
    b = (sigmoid(solution_reshaped[:, 6]) * 255).astype(np.uint8 if not use_gpu or not HAS_CUPY else cp.uint8)
    
    # Sort rectangles by size (smaller rectangles on top for potentially fewer pixel operations)
    areas = w * h
    sort_indices = np.argsort(areas) if not use_gpu or not HAS_CUPY else cp.argsort(areas).get()
    
    # Draw rectangles directly on the NumPy array
    for idx in sort_indices:
        # Calculate rectangle coordinates
        x1 = max(0, int(x[idx] - w[idx]/2))
        y1 = max(0, int(y[idx] - h[idx]/2))
        x2 = min(width, int(x[idx] + w[idx]/2))
        y2 = min(height, int(y[idx] + h[idx]/2))
        
        # Skip rectangles that are too small
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Assign color to the rectangle area
        img_array[y1:y2, x1:x2, 0] = r[idx]
        img_array[y1:y2, x1:x2, 1] = g[idx]
        img_array[y1:y2, x1:x2, 2] = b[idx]
    
    # Convert back to NumPy array if using GPU
    if use_gpu and HAS_CUPY:
        img_array = cp.asnumpy(img_array)
        
    return img_array

def fitness(solution, target_image, num_rects, param_count, width, height, use_gpu=False, use_cache=True, gpu_batch_size=None):
    """
    Calculate negative squared error between solution image and target.
    
    Args:
        solution (numpy.ndarray): Solution parameters.
        target_image (numpy.ndarray): Target image to match.
        num_rects (int): Number of rectangles in the solution.
        param_count (int): Number of parameters per rectangle.
        width (int): Image width.
        height (int): Image height.
        use_gpu (bool, optional): Whether to use GPU acceleration via CuPy.
        use_cache (bool, optional): Whether to use solution caching.
        gpu_batch_size (int, optional): User-specified batch size for GPU processing.
        
    Returns:
        float: Negative MSE (higher is better).
    """
    # Check cache first if enabled
    if use_cache:
        solution_key = solution_hash(solution, num_rects)
        if solution_key in solution_cache:
            return solution_cache[solution_key]
    
    # Ensure solution is on the correct device for drawing
    if use_gpu and HAS_CUPY:
        # Only transfer to GPU if it's not already there
        if not isinstance(solution, cp.ndarray):
            solution_gpu = cp.asarray(solution)
        else:
            solution_gpu = solution
        
        # Draw solution (returns CPU array due to final conversion in draw_solution)
        img = draw_solution(solution_gpu, width, height, num_rects, param_count, 
                          use_gpu=True, gpu_batch_size=gpu_batch_size)
        
        # Handle case where target_image is already on GPU
        if isinstance(target_image, cp.ndarray):
            target_gpu = target_image
        else:
            target_gpu = cp.asarray(target_image)
            
        # Convert drawn image back to GPU for comparison
        img_gpu = cp.asarray(img)
        
        # Calculate error (mean squared error) on GPU
        error = cp.mean((img_gpu.astype(cp.float32) - target_gpu.astype(cp.float32)) ** 2)
        error = float(error.get())  # Convert scalar back to CPU
    else:
        # Draw solution on CPU
        img = draw_solution(solution, width, height, num_rects, param_count, use_gpu=False)
        
        # Calculate error on CPU
        error = np.mean((img.astype(np.float32) - target_image.astype(np.float32)) ** 2)
    
    fitness_value = -error
    
    # Cache the result if enabled
    if use_cache:
        # Limit cache size
        if len(solution_cache) >= 1000:  # Can be adjusted as needed
            # Remove random entry (simple strategy for cache eviction)
            key_to_remove = next(iter(solution_cache))
            solution_cache.pop(key_to_remove)
        solution_cache[solution_key] = fitness_value
    
    return fitness_value

def create_comparison_image(target_image, solution_image, epoch, max_epochs):
    """
    Create a comparison image with target and solution side by side.
    
    Args:
        target_image (numpy.ndarray): Target image.
        solution_image (numpy.ndarray): Current solution image.
        epoch (int): Current epoch number.
        max_epochs (int): Maximum number of epochs.
        
    Returns:
        numpy.ndarray: Comparison image as a NumPy array.
    """
    # Save current backend and switch to non-interactive backend to avoid window flashing
    current_backend = plt.get_backend()
    plt.switch_backend('Agg')  # Use non-interactive backend
    
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
    
    # Switch back to original backend
    plt.switch_backend(current_backend)
    
    return comparison

def parallel_fitness(solutions, target_image, num_rects, param_count, width, height, max_workers=None, use_gpu=False, use_cache=True, gpu_batch_size=None):
    """
    Evaluate fitness of multiple solutions in parallel.
    
    Args:
        solutions (list): List of solutions to evaluate.
        target_image (numpy.ndarray): Target image to match.
        num_rects (int): Number of rectangles in the solution.
        param_count (int): Number of parameters per rectangle.
        width (int): Image width.
        height (int): Image height.
        max_workers (int, optional): Maximum number of worker processes.
        use_gpu (bool, optional): Whether to use GPU acceleration via CuPy.
        use_cache (bool, optional): Whether to use solution caching.
        gpu_batch_size (int, optional): User-specified batch size for GPU processing.
        
    Returns:
        list: List of fitness values for each solution.
    """
    # If using GPU, it's often more efficient to process in batches sequentially
    # rather than using parallel processes which can't efficiently share the GPU
    if use_gpu and HAS_CUPY:
        # Use larger batches for GPU processing
        batch_size = min(len(solutions), 16)  # Process 16 solutions at a time
        results = []
        
        # Process in batches
        for i in range(0, len(solutions), batch_size):
            batch_end = min(i + batch_size, len(solutions))
            batch = solutions[i:batch_end]
            
            # Process batch sequentially on GPU
            batch_results = []
            for solution in batch:
                # Transfer solution to GPU if it's not already there
                if not isinstance(solution, cp.ndarray):
                    solution_gpu = cp.asarray(solution)
                else:
                    solution_gpu = solution
                    
                # Ensure target image is on GPU
                if not isinstance(target_image, cp.ndarray):
                    target_gpu = cp.asarray(target_image)
                else:
                    target_gpu = target_image
                
                # Compute fitness
                fit_val = fitness(
                    solution_gpu, target_gpu, num_rects, 
                    param_count, width, height, use_gpu=True, 
                    use_cache=use_cache, gpu_batch_size=gpu_batch_size
                )
                batch_results.append(fit_val)
            
            results.extend(batch_results)
        return results
    else:
        # For CPU, use process pool for parallel execution
        partial_fitness = functools.partial(
            fitness, target_image=target_image, num_rects=num_rects, 
            param_count=param_count, width=width, height=height,
            use_gpu=False, use_cache=use_cache
        )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(partial_fitness, solutions))

def save_animation(frames, output_path, fps=10, format='gif'):
    """
    Save animation frames as GIF or MP4.
    
    Args:
        frames (list): List of frames (numpy arrays).
        output_path (str): Path to save the animation.
        fps (float): Frames per second.
        format (str): Animation format ('gif' or 'mp4').
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert frames to 8-bit if needed
        frames_8bit = [frame.astype(np.uint8) for frame in frames]
        
        if format.lower() == 'gif':
            # Save GIF
            imageio.mimsave(output_path, frames_8bit, fps=fps)
            print(f"Animation saved as GIF: {output_path}")
        elif format.lower() == 'mp4':
            # Save MP4 - requires imageio[ffmpeg]
            imageio.mimwrite(output_path, frames_8bit, fps=fps, quality=8)
            print(f"Animation saved as MP4: {output_path}")
        else:
            print(f"Unsupported animation format: {format}")
            return False
        return True
    except Exception as e:
        print(f"Error saving animation: {e}")
        return False

def main():
    """
    Run the image approximation example.
    
    This is the main function that coordinates the evolutionary image approximation process.
    """
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
    
    if args.gif is None and args.mp4 is None:
        # Set default animation output path
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
    
    # Validate GPU usage
    use_gpu = args.use_gpu and HAS_CUPY
    if args.use_gpu and not HAS_CUPY:
        print("Warning: CuPy not available, falling back to CPU")
    
    # Transfer target image to GPU if using GPU acceleration
    if use_gpu and HAS_CUPY:
        print("Using GPU acceleration with CuPy")
        target_image_gpu = cp.asarray(target_image)
    else:
        target_image_gpu = None
    
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
    
    # Setup for animation creation
    creating_animation = args.gif is not None or args.mp4 is not None
    if creating_animation:
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
        animation_frames = []
        print(f"Will create animation with {frames_to_capture} frames")
    
    # Configure solution cache
    global solution_cache
    solution_cache = {}  # Reset cache
    
    # Run optimization
    start_time = time.time()
    best_fitness = float('-inf')
    best_solution = None
    stagnation_counter = 0
    
    # Pre-allocate fitness array - on GPU if using GPU
    if use_gpu and HAS_CUPY:
        fitnesses = cp.zeros(population_count, dtype=cp.float32)
    else:
        fitnesses = np.zeros(population_count, dtype=np.float32)
    
    # Create progress bar
    pbar = tqdm(range(max_epochs), desc="Optimizing")
    
    # Warm-up GPU if needed
    if use_gpu and HAS_CUPY:
        # Small warm-up to initialize GPU kernels
        warmup_solution = cp.random.randn(solution_length).astype(cp.float32)
        _ = fitness(warmup_solution, target_image_gpu, num_rects, param_count, width, height, 
                  use_gpu=True, use_cache=False, gpu_batch_size=args.gpu_batch_size)
        print(f"\nGPU warm-up complete")
    
    # Display GPU batch size information if using GPU
    if use_gpu and HAS_CUPY:
        gpu_batch_size = args.gpu_batch_size if args.gpu_batch_size is not None else "auto"
        print(f"GPU batch size: {gpu_batch_size}")
    
    for epoch in pbar:
        # Generate and evaluate solutions - transfer to GPU if needed
        solutions = optimizer.ask()
        if use_gpu and HAS_CUPY:
            solutions_gpu = [cp.asarray(sol) for sol in solutions]
        else:
            solutions_gpu = solutions
        
        # Evaluate fitness (parallel if workers specified)
        if args.workers and not use_gpu:  # Only use parallel CPU workers if not using GPU
            fitnesses = parallel_fitness(
                solutions, target_image,
                num_rects, param_count, width, height, max_workers=args.workers,
                use_gpu=False, use_cache=args.cache_size > 0, gpu_batch_size=args.gpu_batch_size
            )
        else:
            # Process solutions
            for i in range(population_count):
                # Calculate fitness with optional GPU acceleration and caching
                fit_val = fitness(
                    solutions_gpu[i] if use_gpu and HAS_CUPY else solutions[i], 
                    target_image_gpu if use_gpu and HAS_CUPY else target_image,
                    num_rects, param_count, width, height, 
                    use_gpu=use_gpu, use_cache=args.cache_size > 0, gpu_batch_size=args.gpu_batch_size
                )
                
                # Store fitness value
                if use_gpu and HAS_CUPY:
                    fitnesses[i] = fit_val
                else:
                    fitnesses[i] = fit_val
        
        # If on GPU, transfer fitnesses back to CPU for optimizer
        if use_gpu and HAS_CUPY:
            fitnesses_cpu = cp.asnumpy(fitnesses)
            improvement = optimizer.tell(fitnesses_cpu, tolerance=args.early_stop)
        else:
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
        if use_gpu and HAS_CUPY:
            current_best_idx = int(cp.argmax(fitnesses).get())
            current_best_fitness = float(fitnesses[current_best_idx].get())
        else:
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            # Copy the best solution back to CPU if it's on GPU
            if use_gpu and HAS_CUPY:
                best_solution = cp.asnumpy(solutions_gpu[current_best_idx]).copy()
            else:
                best_solution = solutions[current_best_idx].copy()
            
            # Save checkpoint if requested
            if args.checkpoint:
                checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
                optimizer.save_state(checkpoint_path)
        
        # Current solution to visualize - get from optimizer
        current_solution = optimizer.get_best_solution()
        # Transfer to GPU if needed
        if use_gpu and HAS_CUPY:
            current_solution_gpu = cp.asarray(current_solution)
            current_img = draw_solution(current_solution_gpu, width, height, num_rects, param_count, 
                                      use_gpu=True, gpu_batch_size=args.gpu_batch_size)
        else:
            current_img = draw_solution(current_solution, width, height, num_rects, param_count, 
                                      use_gpu=False)
        
        # Capture frame for animation if needed
        if creating_animation and (epoch % capture_frequency == 0 or epoch == max_epochs - 1):
            comparison = create_comparison_image(target_image, current_img, epoch, max_epochs)
            animation_frames.append(comparison)
        
        # Visualize progress periodically
        if (epoch % args.display_interval == 0 or epoch == max_epochs - 1) and not args.no_display:
            plt.clf()
            plt.imshow(current_img)
            plt.title(f"Epoch {epoch}/{max_epochs}")
            plt.pause(0.01)
        
        # Update progress bar
        pbar.set_postfix({
            'error': f"{-best_fitness:.2f}",
            'improvement': f"{improvement:.2e}"
        })
        
        # Clear cache periodically to prevent memory issues
        if args.cache_size > 0 and epoch % 10 == 0:
            # Keep only best solutions in cache
            if len(solution_cache) > args.cache_size:
                solution_cache.clear()
    
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
    
    # Draw final image with GPU batch size if using GPU
    if use_gpu and HAS_CUPY:
        final_solution_gpu = cp.asarray(final_solution)
        final_img = draw_solution(final_solution_gpu, width, height, num_rects, param_count, 
                               use_gpu=True, gpu_batch_size=args.gpu_batch_size)
    else:
        final_img = draw_solution(final_solution, width, height, num_rects, param_count, 
                               use_gpu=False)
    
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
    
    # Save animation if requested
    if creating_animation and animation_frames:
        # Convert frames to 8-bit
        animation_frames_8bit = [frame.astype(np.uint8) for frame in animation_frames]
        
        if args.gif:
            # Ensure the gif path is in the output directory
            if not os.path.isabs(args.gif):
                gif_path = os.path.join(args.output_dir, args.gif)
            else:
                gif_path = args.gif
                
            # Make sure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
            
            # Save GIF
            imageio.mimsave(gif_path, animation_frames_8bit, fps=args.gif_fps)
            print(f"Evolution GIF saved to: {gif_path}")
        
        if args.mp4:
            # Ensure the mp4 path is in the output directory
            if not os.path.isabs(args.mp4):
                mp4_path = os.path.join(args.output_dir, args.mp4)
            else:
                mp4_path = args.mp4
                
            # Make sure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)
            
            try:
                # Save MP4 video
                imageio.mimwrite(mp4_path, animation_frames_8bit, fps=args.gif_fps, quality=8)
                print(f"Evolution MP4 saved to: {mp4_path}")
            except Exception as e:
                print(f"Error saving MP4 (may need to install ffmpeg): {e}")
                print("Try: pip install 'imageio[ffmpeg]'")
    
    # Print final error metrics
    final_error = np.mean((final_img.astype(np.float32) - target_image.astype(np.float32)) ** 2)
    print(f"Final mean squared error: {final_error:.2f}")
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    if final_error > 0:
        psnr = 10 * np.log10((255 ** 2) / final_error)
        print(f"PSNR: {psnr:.2f} dB")
    
    # Calculate SSIM if requested and available
    if args.quality_metrics and HAS_SSIM:
        ssim_value = calculate_ssim(final_img, target_image)
        if ssim_value is not None:
            print(f"SSIM: {ssim_value:.4f}")

if __name__ == "__main__":
    main()