"""
Convolution kernel generators.

This module provides functions for creating Gaussian kernels
and their derivatives for filtering operations.
"""

import numpy as np

def get_kernel_half_width(sigma: float) -> int:
    """
    Compute the half-width of a Gaussian kernel given its
    standard deviation.

    The half-width determines the spatial support of the
    kernel such that most of the Gaussian's energy is captured.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian function.
        Must be positive.

    Returns
    -------
    int
        Half-width of the kernel. The full kernel size is
        `2 * half_width + 1`.

    Notes
    -----
    - The factor 2.5 provides a practical truncation where
      the Gaussian values are already negligible.
    - This choice balances accuracy and computational cost.
    """

    return int(2.5 * sigma + 0.5)

def create_gaussian_kernel(sigma: float) -> np.ndarray:
    """
    Generate a normalized 1D Gaussian kernel.

    The kernel is centered at zero and truncated according to
    the half-width computed from the standard deviation.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian function.
        Must be positive.

    Returns
    -------
    np.ndarray
        1D Gaussian kernel of shape (2 * half_width + 1,)
        with dtype float32 and sum equal to 1.

    Notes
    -----
    - The kernel is explicitly normalized to ensure energy
      preservation during convolution.
    - The truncation radius is determined by
      `get_kernel_half_width`.
    """

    half_width = get_kernel_half_width(sigma)
    size = 2 * half_width + 1

    kernel = np.zeros(size, dtype=np.float32)
    norm = 0.0

    for i in range(size):
        x = i - half_width
        kernel[i] = np.exp(-(x * x) / (2.0 * sigma * sigma))
        norm += kernel[i]

    kernel /= norm
    return kernel

def create_gaussian_kernel_radius(sigma: float, radius: int | None = None):
    """
    Generate a normalized 1D Gaussian kernel with explicit radius control.

    Unlike `create_gaussian_kernel`, this function allows manual
    specification of the kernel radius for finer control over
    the spatial extent.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian function.
        Must be positive.

    radius : int or None, optional
        Half-width of the kernel. If None, defaults to int(3 * sigma).
        The full kernel size will be 2 * radius + 1.

    Returns
    -------
    np.ndarray
        1D Gaussian kernel of shape (2 * radius + 1,) with dtype
        float32 and sum equal to 1.

    Notes
    -----
    - Useful when you need explicit control over kernel size.
    - Energy is preserved through normalization.
    """
    if radius is None:
        radius = int(3 * sigma)
    
    size = 2 * radius + 1
    
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    
    return kernel

def create_gaussian_derivative_kernel(sigma: float) -> np.ndarray:
    """
    Generate a 1D first-order Gaussian derivative kernel.

    This kernel corresponds to the first derivative of a Gaussian
    function with respect to x and is commonly used for gradient
    estimation and edge detection when combined with separable
    convolution.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian function.
        Must be positive.

    Returns
    -------
    np.ndarray
        1D Gaussian derivative kernel of shape
        (2 * half_width + 1,) and dtype float32.

    Notes
    -----
    - The kernel represents the first derivative of a Gaussian:
        dG(x)/dx = -x * exp(-x^2 / (2 * sigma^2))
    - The kernel is antisymmetric and has zero DC response
      (its elements sum to zero).
    - Normalization is performed using the sum of absolute
      weighted values to provide a stable response magnitude.
    - The truncation radius is determined by
      `get_kernel_half_width(sigma)`.
    """

    half_width = get_kernel_half_width(sigma)
    size = 2 * half_width + 1

    kernel = np.zeros(size, dtype=np.float32)
    norm = 0.0

    for i in range(size):
        x = i - half_width
        value = x * np.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel[i] = value
        norm += abs(x * value)

    kernel /= norm
    return kernel

def create_gaussian_second_derivative_kernel(sigma: float) -> np.ndarray:
    """
    Generate a 1D second-order Gaussian derivative kernel.

    This kernel corresponds to the second derivative of a Gaussian
    function with respect to x and is commonly used for zero-crossing
    detection and as a building block for the Laplacian of Gaussian (LoG)
    operator.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian function.
        Must be positive.

    Returns
    -------
    np.ndarray
        1D second-order Gaussian derivative kernel of shape
        (2 * half_width + 1,) and dtype float32.

    Notes
    -----
    - The kernel implements the second derivative of a Gaussian:
        d²G(x)/dx² = (x² / sigma⁴ - 1 / sigma²) * exp(-x² / (2 sigma²))
    - The kernel is symmetric and has zero DC response
      (its elements sum approximately to zero).
    - Normalization is performed using the sum of absolute
      values of the kernel coefficients.
    - The truncation radius is determined by
      `get_kernel_half_width(sigma)`.
    """

    half_width = get_kernel_half_width(sigma)
    size = 2 * half_width + 1

    kernel = np.zeros(size, dtype=np.float32)
    norm = 0.0

    for i in range(size):
        x = i - half_width
        value = ((x * x) / (sigma ** 4) - 1.0 / (sigma ** 2)) * np.exp(
            -(x * x) / (2.0 * sigma * sigma)
        )
        kernel[i] = value
        norm += abs(value)

    kernel /= norm
    kernel -= kernel.sum() / size  # Ensure zero mean
    return kernel