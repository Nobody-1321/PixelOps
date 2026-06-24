"""
Frequency Domain Masks Generator.

This module provides Numba-optimized functions to generate centered 
frequency transfer matrices (masks) such as Ideal, Gaussian, Butterworth, 
and Lanczos shapes.
"""

import numpy as np
from numba import njit, prange
import numpy as np
from ..utils import _distance_matrix

@njit(parallel=True, fastmath=True, cache=True)
def _generate_butterworth_lp(distance: np.ndarray, cutoff: float, order: int) -> np.ndarray:
    """
    Compute a Butterworth Low-Pass transfer function grid.

    Parameters
    ----------
    distance : np.ndarray
        2D float32 matrix containing distances to the center frequency.
    cutoff : float
        Cutoff radius frequency (D0).
    order : int
        Filter order controlling the steepness of the transition band.

    Returns
    -------
    np.ndarray
        Float32 2D array representing the transfer filter mask.
    """
    H, W = distance.shape
    mask = np.empty((H, W), dtype=np.float32)
    for i in prange(H):
        for j in range(W):
            mask[i, j] = 1.0 / (1.0 + (distance[i, j] / (cutoff + 1e-5)) ** (2 * order))
    return mask

@njit(parallel=True, fastmath=True, cache=True)
def _generate_lanczos_lp(distance: np.ndarray, cutoff: float, a: float) -> np.ndarray:
    """
    Compute a Lanczos windowed Low-Pass transfer function grid.

    Parameters
    ----------
    distance : np.ndarray
        2D float32 matrix containing distances to the center frequency.
    cutoff : float
        Cutoff radius frequency (D0).
    a : float
        Lanczos kernel size parameter.

    Returns
    -------
    np.ndarray
        Float32 2D array representing the Lanczos mask.
    """
    H, W = distance.shape
    mask = np.empty((H, W), dtype=np.float32)
    for i in prange(H):
        for j in range(W):
            normalized = distance[i, j] / cutoff
            if normalized == 0.0:
                mask[i, j] = 1.0
            elif normalized > a:
                mask[i, j] = 0.0
            else:
                sinc1 = np.sin(np.pi * normalized) / (np.pi * normalized)
                sinc2 = np.sin(np.pi * (normalized / a)) / (np.pi * (normalized / a))
                mask[i, j] = sinc1 * sinc2
    return mask

def ideal_lowpass_mask(shape: tuple, cutoff_frequency: float) -> np.ndarray:
    """
    Generate a binary Ideal Low-Pass Filter mask.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_frequency : float
        The radius defining the sharp frequency cutoff boundaries.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the centered binary mask.
    """
    distance = _distance_matrix(shape)
    return (distance <= cutoff_frequency).astype(np.float32)

def gaussian_lowpass_mask(shape: tuple, cutoff_frequency: float) -> np.ndarray:
    """
    Generate a smooth Gaussian Low-Pass Filter mask.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_frequency : float
        The standard deviation (sigma) or cutoff radius of the Gaussian curve.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the centered Gaussian distribution mask.
    """
    distance = _distance_matrix(shape)
    return np.exp(-(distance ** 2) / (2.0 * (cutoff_frequency ** 2))).astype(np.float32)

def butterworth_lowpass_mask(shape: tuple, cutoff_frequency: float, order: int = 2) -> np.ndarray:
    """
    Generate a Butterworth Low-Pass Filter mask.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_frequency : float
        Cutoff radius frequency (D0).
    order : int, optional
        Filter order controlling transition steepness. Default is 2.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the centered Butterworth mask.
    """
    distance = _distance_matrix(shape)
    return _generate_butterworth_lp(distance, cutoff_frequency, order)

def ideal_highpass_mask(shape: tuple, cutoff_frequency: float) -> np.ndarray:
    """
    Generate an Ideal High-Pass Filter mask via complement.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_frequency : float
        The sharp frequency cutoff boundaries radius.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the centered binary high-pass mask.
    """
    return 1.0 - ideal_lowpass_mask(shape, cutoff_frequency)

def gaussian_highpass_mask(shape: tuple, cutoff_frequency: float) -> np.ndarray:
    """
    Generate a Gaussian High-Pass Filter mask via complement.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_frequency : float
        The cutoff radius of the Gaussian curve.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the centered Gaussian high-pass mask.
    """
    return 1.0 - gaussian_lowpass_mask(shape, cutoff_frequency)

def butterworth_highpass_mask(shape: tuple, cutoff_frequency: float, order: int = 2) -> np.ndarray:
    """
    Generate a Butterworth High-Pass Filter mask via complement.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_frequency : float
        Cutoff radius frequency (D0).
    order : int, optional
        Filter order. Default is 2.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the centered Butterworth high-pass mask.
    """
    return 1.0 - butterworth_lowpass_mask(shape, cutoff_frequency, order)

def lanczos_lowpass_mask(shape: tuple, cutoff_frequency: float, a: int = 3) -> np.ndarray:
    """
    Generate a Lanczos windowed Low-Pass Filter mask.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_frequency : float
        Cutoff factor for the main frequency lobe.
    a : int, optional
        Lanczos kernel size window index. Default is 3.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the windowed Lanczos mask.
    """
    distance = _distance_matrix(shape)
    return _generate_lanczos_lp(distance, cutoff_frequency, float(a))

def laplacian_of_gaussian_mask(shape: tuple, cutoff_freq: float) -> np.ndarray:
    """
    Generate a Laplacian of Gaussian (LoG) sharpening mask in the frequency domain.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_freq : float
        Frequency scaling parameter for the LoG kernel structure.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the frequency transfer function for LoG.
    """
    H, W = shape
    u = np.fft.fftshift(np.fft.fftfreq(W).reshape(1, -1))
    v = np.fft.fftshift(np.fft.fftfreq(H).reshape(-1, 1))
    f_squared = u * u + v * v
    kernel = -4.0 * (np.pi ** 2) * f_squared * np.exp(-f_squared / (2.0 * (cutoff_freq ** 2)))
    return kernel.astype(np.float32)

def unsharp_masking_mask(shape: tuple, cutoff_freq: float, alpha: float = 1.0, method: str = "gaussian") -> np.ndarray:
    """
    Generate an Unsharp Masking high-frequency emphasis mask.

    Parameters
    ----------
    shape : tuple
        Dimensions of the output mask layout (H, W).
    cutoff_freq : float
        Cutoff frequency matching the low-pass base filter.
    alpha : float, optional
        Scaling weight for the high-frequency boost. Default is 1.0.
    method : str, optional
        Type of low-pass component: 'gaussian', 'ideal', or 'butterworth'. Default is 'gaussian'.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the unsharp matrix operator.
    """
    if method == "gaussian":
        lowpass = gaussian_lowpass_mask(shape, cutoff_freq)
    elif method == "ideal":
        lowpass = ideal_lowpass_mask(shape, cutoff_freq)
    elif method == "butterworth":
        lowpass = butterworth_lowpass_mask(shape, cutoff_freq, order=2)
    else:
        raise ValueError("Unsupported method. Choose 'gaussian', 'ideal', or 'butterworth'.")
    return (1.0 + alpha * (1.0 - lowpass)).astype(np.float32)
