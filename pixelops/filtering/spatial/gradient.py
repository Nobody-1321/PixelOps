"""
Image gradient computation algorithms.

This module provides functions for computing image gradients
using various methods: Gaussian derivatives, Sobel operators,
and Laplacian of Gaussian (LoG).
"""

import numpy as np
import cv2 as cv
from ..kernels import create_gaussian_derivative_kernel, create_gaussian_second_derivative_kernel
from ..utils import convolve_separable
from .gaussian import create_gaussian_kernel
from pixelops.core import (
    to_float32,
    validate_grayscale
)

def _gaussian_gradient_core(
    img: np.ndarray,
    sigma_s: float,
    sigma_d: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the image gradient using separable Gaussian smoothing
    and first-order Gaussian derivative filters.

    This function performs pure numerical computation and returns
    floating-point results without normalization.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).
        Expected dtype is uint8 or float.

    sigma_s : float
        Standard deviation of the Gaussian smoothing kernel.

    sigma_d : float
        Standard deviation of the Gaussian derivative kernel.

    Returns
    -------
    Gx : np.ndarray
        Gradient in the X direction (float32).
        values may be positive or negative, not normalized.

    Gy : np.ndarray
        Gradient in the Y direction (float32).
        values may be positive or negative, not normalized.

    Gmag : np.ndarray
        Gradient magnitude (float32).
        values are non-negative, not normalized.

    Gphase : np.ndarray
        Gradient phase in radians (float32).
        values are in range [-pi, pi], not normalized.

    Notes
    -----
    - No normalization or clipping is performed.
    - Suitable for quantitative analysis and further processing.
    """

    gauss = create_gaussian_kernel(sigma_s)
    gauss_deriv = create_gaussian_derivative_kernel(sigma_d)

    Gx = convolve_separable(img, gauss_deriv, gauss)
    Gy = convolve_separable(img, gauss, gauss_deriv)

    Gmag = np.sqrt(Gx * Gx + Gy * Gy)
    Gphase = np.arctan2(Gy, Gx)

    return Gx, Gy, Gmag, Gphase

def _sobel_gradient_core(
    img: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute image gradients using the Sobel operator.

    This function performs pure numerical computation and returns
    floating-point gradient components without normalization.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).
        Expected dtype is uint8 or float.

    Returns
    -------
    Gx : np.ndarray
        Gradient in the X direction (float32).

    Gy : np.ndarray
        Gradient in the Y direction (float32).

    Gmag : np.ndarray
        Gradient magnitude (float32).

    Gphase : np.ndarray
        Gradient phase in radians (float32).

    Notes
    -----
    - Sobel kernels are normalized by 1/8.
    - This operator approximates first-order derivatives
      using fixed discrete masks.
    """

    sobel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32
    ) * (1.0 / 8.0)

    sobel_y = np.array(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=np.float32
    ) * (1.0 / 8.0)

    Gx = cv.filter2D(img, cv.CV_32F, sobel_x)
    Gy = cv.filter2D(img, cv.CV_32F, sobel_y)

    Gmag = np.sqrt(Gx * Gx + Gy * Gy)
    Gphase = np.arctan2(Gy, Gx)

    return Gx, Gy, Gmag, Gphase

def gaussian_gradient(
    img: np.ndarray,
    sigma_s: float,
    sigma_d: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute image gradients using Gaussian derivatives.

    This function computes the first-order image derivatives
    smoothed by a Gaussian kernel and returns all results in
    floating point format.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).

    sigma_s : float
        Standard deviation of the Gaussian smoothing kernel.
        Must be positive.

    sigma_d : float
        Standard deviation of the Gaussian derivative kernel.
        Must be positive.

    Returns
    -------
    Gx : np.ndarray
        Gradient in X direction (float32).

    Gy : np.ndarray
        Gradient in Y direction (float32).

    Gmag : np.ndarray
        Gradient magnitude (float32).

    Gphase : np.ndarray
        Gradient phase in radians (float32).

    Notes
    -----
    - No normalization or quantization is applied.
    - Suitable for numerical analysis and advanced pipelines.
    - Boundary handling depends on the convolution backend.
    """

    if sigma_s <= 0 or sigma_d <= 0:
        raise ValueError("Sigma values must be positive.")

    validate_grayscale(img)
    img = to_float32(img)

    return _gaussian_gradient_core(img, sigma_s, sigma_d)

def sobel_gradient(
    img: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Sobel image gradients.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).

    Returns
    -------
    Gx : np.ndarray
        Gradient in X direction (float32).

    Gy : np.ndarray
        Gradient in Y direction (float32).

    Gmag : np.ndarray
        Gradient magnitude (float32).

    Gphase : np.ndarray
        Gradient phase in radians (float32).

    Notes
    -----
    - No normalization or quantization is applied.
    - Intended for numerical processing, not visualization.
    """

    validate_grayscale(img)
    img = to_float32(img)

    return _sobel_gradient_core(img)
