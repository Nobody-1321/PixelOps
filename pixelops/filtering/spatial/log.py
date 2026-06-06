import numpy as np
import cv2 as cv
from ..kernels import create_gaussian_second_derivative_kernel
from ..utils import convolve_separable
from .gaussian import create_gaussian_kernel
from pixelops.core import (
    to_float32,
    validate_grayscale
)

def _log_core(
    img: np.ndarray,
    sigma_s: float,
    sigma_d: float
) -> np.ndarray:
    """
    Compute the Laplacian of Gaussian (LoG) using separable convolution.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).

    sigma_s : float
        Standard deviation of the Gaussian smoothing kernel.

    sigma_d : float
        Standard deviation of the second-order Gaussian derivative.

    Returns
    -------
    np.ndarray
        Laplacian of Gaussian response (float32).

    Notes
    -----
    - The LoG is computed as:
        LoG = d²G/dx² * I + d²G/dy² * I
    - No normalization or clipping is applied.
    """

    img_f = img.astype(np.float32)

    gauss = create_gaussian_kernel(sigma_s)
    gauss_2nd = create_gaussian_second_derivative_kernel(sigma_d)

    Gxx = convolve_separable(img_f, gauss_2nd, gauss)
    Gyy = convolve_separable(img_f, gauss, gauss_2nd)

    return Gxx + Gyy

def log(
    img: np.ndarray,
    sigma_s: float,
    sigma_d: float
) -> np.ndarray:
    """
    Compute the Laplacian of Gaussian (LoG).

    The Laplacian of Gaussian highlights regions of rapid
    intensity change and produces a signed response.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).

    sigma_s : float
        Standard deviation of the Gaussian smoothing kernel.
        Must be positive.

    sigma_d : float
        Standard deviation of the second derivative kernel.
        Must be positive.

    Returns
    -------
    np.ndarray
        Signed Laplacian of Gaussian response (float32).

    Notes
    -----
    - No normalization or clipping is applied.
    - Output contains both positive and negative values.
    - Suitable for zero-crossing detection and blob detection.
    """

    if sigma_s <= 0 or sigma_d <= 0:
        raise ValueError("Sigma values must be positive.")

    validate_grayscale(img)
    img = to_float32(img)

    return _log_core(img, sigma_s, sigma_d)
