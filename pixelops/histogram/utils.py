"""
Histogram utilities.

This module provides functions for computing and manipulating
histograms, including CLAHE-related clipping operations.
"""

import numpy as np
from numba import njit


def clip_histogram(hist: np.ndarray, clip_limit: int) -> np.ndarray:
    """
    Clip a histogram and redistribute the excess uniformly.

    Parameters
    ----------
    hist : np.ndarray
        Histogram array of shape (256,).
    clip_limit : int
        Maximum allowed value per histogram bin.

    Returns
    -------
    np.ndarray
        Clipped histogram with redistributed excess.
    """
    if hist.shape[0] != 256:
        raise ValueError("Histogram must have 256 bins.")

    hist = hist.astype(np.int64, copy=True)

    excess = np.maximum(hist - clip_limit, 0)
    total_excess = excess.sum()

    hist = np.minimum(hist, clip_limit)

    redist = total_excess // 256
    remainder = total_excess % 256

    hist += redist
    hist[:remainder] += 1

    return hist

def cal_histogram(channel):
    """
    Computes the histogram of a single image channel.

    The function operates on any 2D channel (e.g., grayscale,
    R/G/B, or H/S/V), assuming uint8 intensities in the range [0, 255].

    Parameters
    ----------
    channel : np.ndarray
        Single-channel image (H×W), dtype uint8.

    Returns
    -------
    hist : np.ndarray
        1D array of length 256 containing the histogram counts.
    """
    if not isinstance(channel, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if channel.ndim != 2:
        raise ValueError("cal_histogram expects a single-channel image (H×W).")

    if channel.dtype != np.uint8:
        raise ValueError("cal_histogram expects uint8 intensities in [0, 255].")

    hist, _ = np.histogram(
        channel.ravel(),
        bins=256,
        range=(0, 256)
    )

    return hist

# ----------------------------------------------
#
#        Numba-optimized versions
#
# ----------------------------------------------


@njit(cache=True, fastmath=True)
def clip_histogram_numba(hist: np.ndarray, clip_limit: int) -> np.ndarray:
    """
    Clip histogram and redistribute excess (Numba-optimized).

    Fast JIT-compiled version of `clip_histogram` for use in
    performance-critical histogram processing.

    Parameters
    ----------
    hist : np.ndarray
        Histogram array of shape (256,). Modified in-place.

    clip_limit : int
        Maximum allowed value per histogram bin.

    Returns
    -------
    np.ndarray
        Same histogram array, modified in-place.

    Notes
    -----
    - Remainder is not distributed (simpler than clip_histogram).
    - Designed for use within other Numba functions.
    """
    excess = 0

    for i in range(256):
        if hist[i] > clip_limit:
            excess += hist[i] - clip_limit
            hist[i] = clip_limit

    incr = excess // 256
    for i in range(256):
        hist[i] += incr

    return hist


@njit(cache=True, fastmath=True)
def cal_histogram_numba(block: np.ndarray) -> np.ndarray:
    """
    Compute histogram of an image block (Numba-optimized).

    Fast JIT-compiled histogram computation for use in
    CLAHE and other block-based algorithms.

    Parameters
    ----------
    block : np.ndarray
        Image block of shape (H, W) and dtype uint8.

    Returns
    -------
    np.ndarray
        Histogram array of shape (256,) and dtype int32.

    Notes
    -----
    - Assumes uint8 input values in [0, 255].
    - Designed for use within other Numba functions.
    """
    hist = np.zeros(256, dtype=np.int32)
    h, w = block.shape

    for y in range(h):
        for x in range(w):
            hist[block[y, x]] += 1

    return hist

