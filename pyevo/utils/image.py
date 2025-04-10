"""
Image Processing Utilities

This module contains image processing functions used throughout the codebase,
including SSIM calculation and convolution operations.
"""

import numpy as np
from PIL import Image

# Try to import scipy for optimized versions
try:
    from scipy.ndimage import convolve as scipy_convolve
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SCIPY = True
    print("SciPy/scikit-image detected - using optimized image processing")
except ImportError:
    HAS_SCIPY = False
    print("SciPy/scikit-image not found - using pure NumPy implementation")

def get_optimal_image_functions():
    """
    Get the optimal image processing functions based on available packages.
    
    Returns:
        tuple: (optimal_ssim_func, optimal_convolve_func)
    """
    if HAS_SCIPY:
        return scipy_ssim, scipy_convolve2d
    else:
        return calculate_ssim, convolve2d

def calculate_ssim(img1, img2, win_size=11, k1=0.01, k2=0.03, L=255, downsample=True):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Pure NumPy implementation with no external dependencies.
    
    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        win_size (int): Size of the Gaussian window (default: 11).
        k1 (float): First stability constant (default: 0.01).
        k2 (float): Second stability constant (default: 0.03).
        L (int): Dynamic range of pixel values (default: 255).
        downsample (bool): Whether to downsample large images for faster processing.
        
    Returns:
        float: SSIM value in range [0, 1], higher is better.
    """
    # Optional downsampling for faster processing on large images
    if downsample and (img1.shape[0] > 256 or img1.shape[1] > 256):
        factor = min(1, 256 / max(img1.shape[0], img1.shape[1]))
        new_size = (int(img1.shape[1] * factor), int(img1.shape[0] * factor))
        
        # Use PIL for high-quality resizing
        img1_pil = Image.fromarray(img1)
        img2_pil = Image.fromarray(img2)
        img1_small = np.array(img1_pil.resize(new_size, Image.LANCZOS))
        img2_small = np.array(img2_pil.resize(new_size, Image.LANCZOS))
        
        return calculate_ssim(img1_small, img2_small, win_size, k1, k2, L, downsample=False)
    
    # Convert to grayscale if color
    if img1.ndim == 3 and img1.shape[2] == 3:
        # Simple grayscale conversion using weighted sum
        img1_gray = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
        img2_gray = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    else:
        img1_gray = img1.astype(np.float32)
        img2_gray = img2.astype(np.float32)
    
    # Constants for stabilizing division
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    # Generate a Gaussian kernel for window
    x = np.arange(-(win_size // 2), win_size // 2 + 1)
    gauss = np.exp(-(x ** 2) / (2 * 1.5 ** 2))
    gauss = gauss / np.sum(gauss)
    window = np.outer(gauss, gauss)
    
    # Use our own optimized convolution function
    filter_func = convolve2d
    
    # Compute means
    mu1 = filter_func(img1_gray, window)
    mu2 = filter_func(img2_gray, window)
    
    # Compute variances and covariance
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = filter_func(img1_gray * img1_gray, window) - mu1_sq
    sigma2_sq = filter_func(img2_gray * img2_gray, window) - mu2_sq
    sigma12 = filter_func(img1_gray * img2_gray, window) - mu1_mu2
    
    # Compute SSIM
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    # Avoid division by zero
    ssim_map = np.where(denominator > 0, numerator / denominator, 0)
    
    # Return mean SSIM
    return float(np.mean(ssim_map))

def scipy_ssim(img1, img2, **kwargs):
    """
    Calculate SSIM using scikit-image implementation (faster).
    
    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        **kwargs: Additional arguments passed to structural_similarity.
        
    Returns:
        float: SSIM value in range [0, 1], higher is better.
    """
    if not HAS_SCIPY:
        return calculate_ssim(img1, img2)
    
    # Handle multichannel images
    if img1.ndim == 3 and img1.shape[2] == 3:
        # skimage SSIM expects multichannel parameter for color images
        return skimage_ssim(img1, img2, multichannel=True, **kwargs)
    else:
        return skimage_ssim(img1, img2, **kwargs)

def scipy_convolve2d(img, kernel):
    """
    Fast 2D convolution using SciPy.
    
    Args:
        img (numpy.ndarray): Input image.
        kernel (numpy.ndarray): 2D kernel/window.
        
    Returns:
        numpy.ndarray: Convolved result.
    """
    if not HAS_SCIPY:
        return convolve2d(img, kernel)
    
    # SciPy expects kernel rotated by 180 degrees for convolution
    kernel_flipped = kernel[::-1, ::-1]
    return scipy_convolve(img, kernel_flipped, mode='reflect')

def convolve2d(img, kernel):
    """
    Optimized 2D convolution using only NumPy.
    
    This implementation uses a more efficient approach than the simple 
    nested loops version, with better memory management.
    
    Args:
        img (numpy.ndarray): Input image.
        kernel (numpy.ndarray): 2D kernel/window.
        
    Returns:
        numpy.ndarray: Convolved result.
    """
    # Get dimensions
    k_height, k_width = kernel.shape
    i_height, i_width = img.shape
    
    # Calculate padding
    pad_height = k_height // 2
    pad_width = k_width // 2
    
    # Pad the image
    padded = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    
    # Prepare result array
    result = np.zeros((i_height, i_width), dtype=np.float32)
    
    # For small kernels, we can vectorize much more efficiently (common case)
    if k_height <= 15 and k_width <= 15:
        # Flip kernel for convolution (not correlation)
        kernel_flipped = kernel[::-1, ::-1]
        
        # For each pixel in the kernel window, multiply by appropriate kernel value
        for i in range(k_height):
            for j in range(k_width):
                result += padded[i:i+i_height, j:j+i_width] * kernel_flipped[i, j]
                
    else:
        # For larger kernels, process in blocks to manage memory better
        block_size = 64
        for i_start in range(0, i_height, block_size):
            i_end = min(i_start + block_size, i_height)
            for j_start in range(0, i_width, block_size):
                j_end = min(j_start + block_size, i_width)
                
                # Process this block
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        # Extract the local region
                        local_region = padded[i:i+k_height, j:j+k_width]
                        # Apply the kernel
                        result[i, j] = np.sum(local_region * kernel)
    
    return result 