"""
Advanced PyEvo Example

This example demonstrates the reorganized utilities and improved package structure,
showing how to use the different modules together:
1. Optimizers from pyevo.optimizers
2. Acceleration utilities from pyevo.utils.acceleration
3. Image processing from pyevo.utils.image
4. Interactive optimization from pyevo.utils.interactive
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Try importing directly, fall back to local import if not installed
try:
    from pyevo import (
        # Optimizers
        SNES, CMA_ES, PSO,
        
        # Acceleration
        optimize_with_acceleration,
        is_gpu_available,
        save_checkpoint, load_checkpoint,
        
        # Image processing
        calculate_ssim, get_optimal_image_functions,
        
        # Interactive mode
        InteractiveOptimizer,
        
        # Constants
        DEFAULT_ALPHA, DEFAULT_EPOCHS
    )
except ImportError:
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pyevo import (
        # Optimizers
        SNES, CMA_ES, PSO,
        
        # Acceleration
        optimize_with_acceleration,
        is_gpu_available,
        save_checkpoint, load_checkpoint,
        
        # Image processing
        calculate_ssim, get_optimal_image_functions,
        
        # Interactive mode
        InteractiveOptimizer,
        
        # Constants
        DEFAULT_ALPHA, DEFAULT_EPOCHS
    )

# Optimization benchmark problems

def rastrigin(x, A=10.0):
    """Rastrigin test function (minimized at x=0)."""
    n = len(x)
    return -A*n - np.sum(x**2 - A*np.cos(2*np.pi*x))

def rosenbrock(x):
    """Rosenbrock test function (minimized at x=[1,1,...])."""
    return -np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    """Ackley test function (minimized at x=0)."""
    a, b, c = 20, 0.2, 2*np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -(-a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1))

# Image example setup

def generate_test_image(size=128):
    """Generate a test image for optimization."""
    # Create a simple test image with shapes
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add a red circle
    x, y = np.ogrid[:size, :size]
    mask = (x - size/2)**2 + (y - size/2)**2 <= (size/3)**2
    img[mask] = [255, 0, 0]
    
    # Add a blue rectangle
    img[size//4:3*size//4, size//4:3*size//4, 0] = 0
    img[size//4:3*size//4, size//4:3*size//4, 2] = 255
    
    return img

def draw_circles(solution, width, height, num_circles):
    """Draw circles based on solution parameters."""
    # Create a blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Reshape solution into a 2D array (num_circles x 7 parameters)
    params = solution.reshape(num_circles, 7)
    
    # Draw each circle
    for i in range(num_circles):
        # Extract parameters for this circle
        x = int((params[i, 0] + 1) * width / 2)   # Map from [-1, 1] to [0, width]
        y = int((params[i, 1] + 1) * height / 2)  # Map from [-1, 1] to [0, height]
        r = int(np.abs(params[i, 2]) * min(width, height) / 4)  # Radius
        
        # RGB color (map from [-1, 1] to [0, 255])
        red = int((params[i, 3] + 1) * 127.5)
        green = int((params[i, 4] + 1) * 127.5)
        blue = int((params[i, 5] + 1) * 127.5)
        
        # Alpha/opacity (map from [-1, 1] to [0, 1])
        alpha = (params[i, 6] + 1) / 2
        
        # Create a mask for this circle
        mask_x, mask_y = np.ogrid[:height, :width]
        mask = (mask_x - y)**2 + (mask_y - x)**2 <= r**2
        
        # Apply the circle with alpha blending
        if np.any(mask):
            # Create a color array for this circle
            color = np.zeros((height, width, 3), dtype=np.float32)
            color[mask] = [red, green, blue]
            
            # Alpha blend with the existing image
            img = img.astype(np.float32) * (1 - alpha * mask[:, :, np.newaxis]) + color * (alpha * mask[:, :, np.newaxis])
            
            # Convert back to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

def image_fitness(solution, target_image, num_circles):
    """Calculate fitness based on similarity to target image."""
    # Draw the image from solution
    width, height = target_image.shape[1], target_image.shape[0]
    img = draw_circles(solution, width, height, num_circles)
    
    # Calculate SSIM (get optimal implementation)
    ssim_func, _ = get_optimal_image_functions()
    similarity = ssim_func(img, target_image)
    
    return similarity

# Visualization

def visualize_optimization(optimizer, iteration, target_image, num_circles):
    """Visualize the current best solution."""
    best_solution = optimizer.get_best_solution()
    width, height = target_image.shape[1], target_image.shape[0]
    img = draw_circles(best_solution, width, height, num_circles)
    
    plt.figure(figsize=(12, 6))
    
    # Plot target image
    plt.subplot(1, 2, 1)
    plt.imshow(target_image)
    plt.title("Target Image")
    plt.axis('off')
    
    # Plot current solution
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title(f"Iteration {iteration}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"output/progress_{iteration:04d}.png")
    plt.close()

# Main functions

def demo_standard_optimization():
    """Demonstrate standard optimization with acceleration."""
    print("\n=== Standard Optimization Demo ===")
    
    # Set up the optimization
    dimensions = 20
    optimizer = CMA_ES(solution_length=dimensions, random_seed=42)
    
    # Choose the Ackley function
    print(f"Optimizing Ackley function ({dimensions}D)")
    
    # Check for GPU availability
    gpu_available = is_gpu_available()
    print(f"GPU acceleration: {'Available' if gpu_available else 'Not available'}")
    print(f"CPU cores: {os.cpu_count()}")
    
    # Run optimization
    start_time = time.time()
    best_solution, best_fitness, stats = optimize_with_acceleration(
        optimizer=optimizer,
        fitness_func=ackley,
        max_iterations=100,
        use_gpu=gpu_available,
        use_parallel=True,
        checkpoint_freq=25,
        checkpoint_path="checkpoints/ackley",
        callback=lambda opt, i, imp: print(f"  Iteration {i}: Best fitness = {-opt.get_stats()['best_fitness']:.6f}") if i % 10 == 0 else None
    )
    elapsed_time = time.time() - start_time
    
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    print(f"Best fitness: {-best_fitness:.6f}")
    print(f"Best solution: {best_solution[:3]}... (first 3 values)")

def demo_interactive_optimization():
    """Demonstrate interactive optimization."""
    print("\n=== Interactive Optimization Demo ===")
    
    # Set up the optimization
    dimensions = 10
    optimizer = SNES(solution_length=dimensions, random_seed=42)
    
    # Choose the Rosenbrock function
    print(f"Optimizing Rosenbrock function ({dimensions}D)")
    print("Type 'help' for available commands")
    
    # Create interactive optimizer
    interactive_optimizer = InteractiveOptimizer(
        optimizer=optimizer,
        fitness_function=rosenbrock,
        max_iterations=100,
        checkpoint_dir="checkpoints"
    )
    
    # Run optimization interactively
    best_solution, best_fitness = interactive_optimizer.start()
    
    print(f"Final best fitness: {best_fitness:.6f}")

def demo_image_optimization():
    """Demonstrate image optimization with circle approximation."""
    print("\n=== Image Optimization Demo ===")
    
    # Generate a test image
    target_image = generate_test_image(size=128)
    
    # Save the target image
    os.makedirs("output", exist_ok=True)
    Image.fromarray(target_image).save("output/target_image.png")
    print("Target image saved to output/target_image.png")
    
    # Number of circles to use
    num_circles = 10
    
    # Optimization parameters
    solution_length = num_circles * 7  # 7 parameters per circle
    
    # Create optimizer
    optimizer = SNES(solution_length=solution_length, random_seed=42)
    
    # Create a fitness function that takes a solution
    fitness_func = lambda x: image_fitness(x, target_image, num_circles)
    
    # Set up visualization callback
    def visualization_callback(opt, iteration):
        if iteration % 10 == 0 or iteration == 1:
            visualize_optimization(opt, iteration, target_image, num_circles)
    
    # Create interactive optimizer
    interactive_optimizer = InteractiveOptimizer(
        optimizer=optimizer,
        fitness_function=fitness_func,
        max_iterations=100,
        checkpoint_dir="checkpoints"
    )
    
    # Set visualization callback
    interactive_optimizer.visualization_callback = visualization_callback
    
    # Run optimization interactively
    print("Starting interactive image optimization")
    print("Images will be saved to output/progress_*.png")
    print("Type 'help' for available commands")
    
    best_solution, best_fitness = interactive_optimizer.start()
    
    # Create final image
    width, height = target_image.shape[1], target_image.shape[0]
    final_img = draw_circles(best_solution, width, height, num_circles)
    Image.fromarray(final_img).save("output/final_image.png")
    
    print(f"Final image saved to output/final_image.png")
    print(f"Final similarity score: {best_fitness:.4f}")

if __name__ == "__main__":
    # Create output and checkpoint directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Run demos
    demo_standard_optimization()
    demo_interactive_optimization()
    demo_image_optimization() 