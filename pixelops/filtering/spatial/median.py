import numpy as np
from numba import njit, prange
from ..utils import reflect

@njit(cache=True)
def median_from_histogram(hist: np.ndarray, total: int) -> int:
    """
    Compute the median value from a discrete histogram.

    This function assumes a histogram representing the frequency
    of intensity values in a local window of a grayscale image.
    The median is obtained by accumulating histogram bins until
    half of the total number of samples is exceeded.

    Parameters
    ----------
    hist : np.ndarray
        Histogram array of length 256, where each index represents
        an intensity value in the range [0, 255] and each element
        stores its frequency within the window.

    total : int
        Total number of samples in the window (e.g., window_size²).

    Returns
    -------
    int
        Median intensity value in the range [0, 255].

    Notes
    -----
    - This function runs in O(256) time, independent of the window size.
    - It is designed to be used inside Numba-compiled code.
    """

    cum = 0
    mid = total // 2
    for i in range(256):
        cum += hist[i]
        if cum > mid:
            return i
    return 255

@njit(parallel=True, fastmath=True, cache=True)
def median_filter_core(
    img: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Core median filter using histogram sliding-window with reflection.

    Applies a median filter to a grayscale image using a histogram-based
    approach. Uses reflection at boundaries to avoid memory allocation
    for padding.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W) and dtype uint8.

    window_size : int
        Size of the square median window. Must be an odd integer.

    Returns
    -------
    np.ndarray
        Median-filtered image of shape (H, W) and dtype uint8.

    Notes
    -----
    - Uses reflect boundary condition (no padding allocation).
    - The histogram has 256 bins, assuming uint8 pixel values.
    - Horizontal sliding avoids recomputing the histogram from scratch.
    - Rows are parallelized using Numba's `prange`.
    """

    H, W = img.shape
    pad = window_size // 2
    out = np.zeros((H, W), dtype=np.uint8)

    for y in prange(H):
        hist = np.zeros(256, dtype=np.int32)

        # Initialize histogram for the first window in the row
        for dy in range(window_size):
            yy = reflect(y + dy - pad, H)
            for dx in range(window_size):
                xx = reflect(dx - pad, W)
                val = img[yy, xx]
                hist[val] += 1

        out[y, 0] = median_from_histogram(hist, window_size * window_size)

        # Slide the window horizontally
        for x in range(1, W):
            # Remove leftmost column
            for dy in range(window_size):
                yy = reflect(y + dy - pad, H)
                xx_old = reflect(x - 1 - pad, W)
                old_val = img[yy, xx_old]
                hist[old_val] -= 1

            # Add rightmost column
            for dy in range(window_size):
                yy = reflect(y + dy - pad, H)
                xx_new = reflect(x + pad, W)
                new_val = img[yy, xx_new]
                hist[new_val] += 1

            out[y, x] = median_from_histogram(hist, window_size * window_size)

    return out

def median_filter(
    image: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Apply a median filter to a grayscale or multi-channel image using
    a fast histogram-based algorithm accelerated with Numba.

    The filter is applied independently per channel when the input
    image has multiple channels.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, C), dtype uint8.

    window_size : int
        Size of the square median window. Must be an odd integer.

    Returns
    -------
    np.ndarray
        Median-filtered image with the same shape and dtype as input.

    Raises
    ------
    ValueError
        If window_size is not odd.
        If image has unsupported dimensions.
    """

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")

    if image.dtype != np.uint8:
        raise ValueError("Input image must be uint8.")

    # Grayscale
    if image.ndim == 2:
        return median_filter_core(image, window_size)

    # Multi-channel (e.g. BGR, RGB, etc.)
    if image.ndim == 3:
        h, w, c = image.shape
        out = np.empty_like(image)

        for ch in range(c):
            out[:, :, ch] = median_filter_core(
                image[:, :, ch], window_size
            )

        return out

    raise ValueError(
        "Input image must be 2D (grayscale) or 3D (multi-channel)."
    )
