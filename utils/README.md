# PyEvo Utilities

This directory contains utility modules that provide common functionality used throughout the PyEvo codebase.

## Available Modules

### Image Processing (`image_processing.py`)

Contains image processing utilities that are implemented using pure NumPy with no external dependencies beyond PIL/Pillow:

- **calculate_ssim**: Calculates the Structural Similarity Index (SSIM) between two images. This is a pure NumPy implementation that does not rely on scipy or scikit-image.
- **convolve2d**: Optimized 2D convolution function using only NumPy. Provides two different implementation strategies based on kernel size.

## Usage

You can import these utilities in your code like this:

```python
from utils.image_processing import calculate_ssim, convolve2d

# Calculate SSIM between two images
ssim_value = calculate_ssim(image1, image2)

# Perform 2D convolution
result = convolve2d(image, kernel)
``` 