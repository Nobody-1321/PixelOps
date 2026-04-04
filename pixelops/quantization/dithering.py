"""
Dithering and quantization algorithms.

This module provides error-diffusion and ordered dithering methods
for converting grayscale images to binary or reduced-level images.
"""

import numpy as np
from numba import njit, prange


@njit
def floyd_steinberg_dithering(image: np.ndarray) -> np.ndarray:
    """
    Apply Floyd-Steinberg error diffusion dithering.

    Converts a grayscale image to binary using error diffusion
    with the classic Floyd-Steinberg coefficients.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W).
        Any numeric dtype is accepted.

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W) and dtype uint8.
        Values are 0 or 255.

    Notes
    -----
    - Error diffusion pattern: 7/16, 3/16, 5/16, 1/16.
    - Scans left-to-right, top-to-bottom.
    - Threshold is fixed at 127.
    """
    img = image.astype(np.float32)
    h, w = img.shape

    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255.0 if old_pixel > 127.0 else 0.0
            img[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < w:
                img[y, x + 1] += error * 0.4375      # 7/16
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += error * 0.1875  # 3/16
                img[y + 1, x] += error * 0.3125          # 5/16
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 0.0625  # 1/16

    for y in range(h):
        for x in range(w):
            img[y, x] = min(max(img[y, x], 0.0), 255.0)

    return img.astype(np.uint8)


@njit
def atkinson_dithering(image: np.ndarray) -> np.ndarray:
    """
    Apply Atkinson error diffusion dithering.

    Converts a grayscale image to binary using Atkinson's
    error diffusion pattern, which diffuses only 6/8 of the
    error for higher contrast.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W).

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W) and dtype uint8.
        Values are 0 or 255.

    Notes
    -----
    - Diffuses only 75% of error (higher contrast than Floyd-Steinberg).
    - Uses 6-neighbor diffusion pattern.
    - Popular for its distinctive look in early Macintosh graphics.
    """
    img = image.astype(np.float32)
    h, w = img.shape

    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255.0 if old_pixel > 127.0 else 0.0
            img[y, x] = new_pixel
            error = (old_pixel - new_pixel) / 8.0

            if x + 1 < w:
                img[y, x + 1] += error
            if x + 2 < w:
                img[y, x + 2] += error
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += error
                img[y + 1, x] += error
                if x + 1 < w:
                    img[y + 1, x + 1] += error
            if y + 2 < h:
                img[y + 2, x] += error

    for y in range(h):
        for x in range(w):
            img[y, x] = min(max(img[y, x], 0.0), 255.0)

    return img.astype(np.uint8)


@njit(parallel=True)
def bayer_dithering(image: np.ndarray, matrix_size: int = 4) -> np.ndarray:
    """
    Apply ordered Bayer dithering.

    Converts a grayscale image to binary using a threshold matrix
    (Bayer matrix) for ordered dithering. Produces a regular
    halftone-like pattern.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W) and dtype uint8.

    matrix_size : int, optional
        Size of the Bayer matrix. Default is 4 (4x4 matrix).
        Currently only 4x4 is implemented.

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W) and dtype uint8.
        Values are 0 or 255.

    Notes
    -----
    - Parallelized for performance.
    - No sequential dependencies (unlike error diffusion).
    - Produces characteristic crosshatch patterns.
    """
    # Bayer 4x4 matrix (normalized)
    bayer_4x4 = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ], dtype=np.float32) / 16.0

    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)

    for y in prange(h):
        for x in range(w):
            threshold = bayer_4x4[y % 4, x % 4] * 255.0
            output[y, x] = 255 if image[y, x] > threshold else 0

    return output


@njit
def uniform_quantize(image: np.ndarray, levels: int = 8) -> np.ndarray:
    """
    Apply uniform quantization to reduce intensity levels.

    Maps pixel values to a reduced set of uniformly-spaced
    intensity levels.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W).

    levels : int, optional
        Number of output levels. Default is 8.
        Must be positive.

    Returns
    -------
    np.ndarray
        Quantized image of same shape and dtype as input.
        Values are centered within each quantization bin.

    Notes
    -----
    - Output values are: step/2, 3*step/2, ..., (2*levels-1)*step/2
    - No dithering is applied.
    """
    step = 256.0 / levels
    output = np.zeros_like(image)
    h, w = image.shape

    for y in range(h):
        for x in range(w):
            output[y, x] = np.uint8((image[y, x] // step) * step + step / 2)

    return output


def floyd_steinberg_serpentine(image: np.ndarray) -> np.ndarray:
    """
    Apply Floyd-Steinberg dithering with serpentine scanning.

    Alternates scan direction between rows to reduce visual
    artifacts that can appear with unidirectional scanning.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W).

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W) and dtype uint8.
        Values are 0 or 255.

    Notes
    -----
    - Even rows scan left-to-right.
    - Odd rows scan right-to-left.
    - Reduces "wormy" artifacts common in standard Floyd-Steinberg.
    """
    img = image.astype(np.float32)
    h, w = img.shape

    for y in range(h):

        if y % 2 == 0:
            x_range = range(w)
            direction = 1
        else:
            x_range = range(w - 1, -1, -1)
            direction = -1

        for x in x_range:
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            img[y, x] = new_pixel

            error = old_pixel - new_pixel

            nx = x + direction
            if 0 <= nx < w:
                img[y, nx] += error * 7 / 16

            if y + 1 < h:
                img[y + 1, x] += error * 5 / 16

                nx = x - direction
                if 0 <= nx < w:
                    img[y + 1, nx] += error * 3 / 16

                nx = x + direction
                if 0 <= nx < w:
                    img[y + 1, nx] += error * 1 / 16

    return np.clip(img, 0, 255).astype(np.uint8)
