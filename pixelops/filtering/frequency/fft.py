"""
Frequency Domain Filtering Module.

This module implements 2D Discrete Fourier Transforms (DFT) and various
frequency-domain masks (Ideal, Gaussian, Butterworth, Lanczos, etc.) 
for advanced edge-preserving, smoothing, and homomorphic image filtering.
All public filters expect and return normalized float32 images in the [0, 1] range.
"""

import numpy as np
from numba import njit, prange
from pixelops.core import (
    to_float32,
    validate_image
)

# =====================================================================
# CORE NUMBA FUNCTIONS (Optimized Performance)
# =====================================================================

def _fft_pass(image_f: np.ndarray) -> np.ndarray:
    """
    Compute the core 2D Fast Fourier Transform bypassing high-level overhead.

    Parameters
    ----------
    image_f : np.ndarray
        A 2D float32 array representing a single image channel.

    Returns
    -------
    np.ndarray
        Complex128 2D array containing the raw unshifted spectrum.
    """
    fft_rows = np.fft.fft(image_f, axis=1)
    return np.fft.fft(fft_rows, axis=0)

def _ifft_pass(f_transform: np.ndarray) -> np.ndarray:
    """
    Compute the core 2D Inverse FFT and extract its magnitude.

    Parameters
    ----------
    f_transform : np.ndarray
        Complex 2D array representing the frequency domain spectrum.

    Returns
    -------
    np.ndarray
        Float32 2D array containing the reconstructed spatial domain magnitude.
    """
    ifft_rows = np.fft.ifft(f_transform, axis=1)
    ifft2d = np.fft.ifft(ifft_rows, axis=0)
    return np.abs(ifft2d)

def fourier_transform_2d(image: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Discrete Fourier Transform using successive 1D FFT passes.

    Supports both 2D grayscale and multi-channel image layouts.

    Parameters
    ----------
    image : np.ndarray
        Input spatial image of shape (H, W) or (H, W, C).

    Returns
    -------
    np.ndarray
        Complex128 array containing the raw, unshifted Fourier spectrum.
    """
    validate_image(image)
    img_f = to_float32(image)
    
    if img_f.ndim == 2:
        return _fft_pass(img_f)
    else:
        out = np.empty_like(img_f, dtype=np.complex128)
        for c in range(img_f.shape[2]):
            out[:, :, c] = _fft_pass(img_f[:, :, c])
        return out

def inverse_fourier_transform_2d(f_transform: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Inverse Fourier Transform and return its magnitude space.

    Supports both 2D grayscale and multi-channel spectral inputs.

    Parameters
    ----------
    f_transform : np.ndarray
        Input frequency transform spectrum of shape (H, W) or (H, W, C).

    Returns
    -------
    np.ndarray
        Float32 array matching the channel dimensions of the input spectrum.
    """
    if f_transform.ndim == 2:
        return _ifft_pass(f_transform)
    elif f_transform.ndim == 3:
        out = np.empty_like(f_transform, dtype=np.float32)
        for c in range(f_transform.shape[2]):
            out[:, :, c] = _ifft_pass(f_transform[:, :, c])
        return out
    else:
        raise ValueError("Input transform must be 2D or 3D.")

def fourier_spectra(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude and phase spectra of a 2D Fourier Transform.

    Extracts centered and log-scaled frequency characteristics for evaluation or plotting.

    Parameters
    ----------
    image : np.ndarray
        Input 2D grayscale image array of shape (H, W).

    Returns
    -------
    magnitude : np.ndarray
        2D array (float32) containing the centered, log-scaled, and normalized [0.0, 1.0] magnitude map.
    phase : np.ndarray
        2D array (float32) containing the centered, normalized [0.0, 1.0] phase distribution angle map.

    Raises
    ------
    ValueError
        If the input image layout is not 2D grayscale.
    """
    validate_image(image)
    img_f = to_float32(image)

    if img_f.ndim != 2:
        raise ValueError("Spectra visualization is only supported for 2D grayscale images.")

    dft = _fft_pass(img_f)

    # Process log-magnitude mapping features
    magnitude = np.abs(dft)
    magnitude = np.log1p(magnitude)
    magnitude_shifted = np.fft.fftshift(magnitude)
    
    max_val = np.max(magnitude_shifted)
    if max_val > 0:
        magnitude_out = (magnitude_shifted / max_val).astype(np.float32)
    else:
        magnitude_out = magnitude_shifted.astype(np.float32)

    # Process phase mapping features
    phase = np.angle(dft)
    phase_shifted = np.fft.fftshift(phase)
    phase_out = ((phase_shifted + np.pi) / (2.0 * np.pi)).astype(np.float32)

    return magnitude_out, phase_out