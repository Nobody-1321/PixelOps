import numpy as np
from numba import njit, prange

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
def median_filter_grayscale_numba(
    padded: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Apply a median filter to a padded grayscale image using a
    histogram sliding-window approach.

    The function processes each row independently and computes
    the median value for each pixel by maintaining and updating
    a local histogram as the window slides horizontally.

    Parameters
    ----------
    padded : np.ndarray
        Padded grayscale image of shape (H + 2*pad, W + 2*pad)
        and dtype uint8. Padding must already be applied.

    window_size : int
        Size of the square median window. Must be an odd integer.

    Returns
    -------
    np.ndarray
        Median-filtered image of shape (H, W) and dtype uint8.

    Notes
    -----
    - The histogram has 256 bins, assuming uint8 pixel values.
    - Horizontal sliding avoids recomputing the histogram from scratch.
    - Rows are parallelized using Numba's `prange`.
    - The algorithm runs in O(H * W * (window_size + 256)) time.
    """

    pad = window_size // 2
    H = padded.shape[0] - 2 * pad
    W = padded.shape[1] - 2 * pad
    out = np.zeros((H, W), dtype=np.uint8)

    for y in prange(H):
        hist = np.zeros(256, dtype=np.int32)

        # Initialize histogram for the first window in the row
        for dy in range(window_size):
            for dx in range(window_size):
                val = padded[y + dy, dx]
                hist[val] += 1

        out[y, 0] = median_from_histogram(
            hist, window_size * window_size
        )

        # Slide the window horizontally
        for x in range(1, W):
            # Remove leftmost column
            for dy in range(window_size):
                old_val = padded[y + dy, x - 1]
                hist[old_val] -= 1

            # Add rightmost column
            for dy in range(window_size):
                new_val = padded[y + dy, x + window_size - 1]
                hist[new_val] += 1

            out[y, x] = median_from_histogram(
                hist, window_size * window_size
            )

    return out

def median_filter_grayscale(
    image: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Apply a median filter to a grayscale image using a fast
    histogram-based algorithm accelerated with Numba.

    This function acts as a high-level wrapper that performs
    boundary padding and delegates the core computation to a
    Numba-optimized backend.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W) and dtype uint8.
        Pixel values are expected to be in the range [0, 255].

    window_size : int
        Size of the square median window. Must be an odd integer.

    Returns
    -------
    np.ndarray
        Median-filtered image of shape (H, W) and dtype uint8.

    Raises
    ------
    ValueError
        If `window_size` is not an odd integer.

    Notes
    -----
    - Padding is performed using edge replication.
    - This implementation is significantly faster than naive
      sort-based median filters, especially for larger windows.
    - Suitable for real-time or large-scale image processing tasks.
    """

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    pad = window_size // 2
    padded = np.pad(image, pad, mode="edge")

    return median_filter_grayscale_numba(padded, window_size)

def median_filter_bgr(
    image: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Apply a median filter to a BGR image using a fast histogram-based
    algorithm accelerated with Numba.

    The median filter is applied independently to each color channel
    (B, G, R). No color space conversion or channel mixing is performed.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image of shape (H, W, 3) and dtype uint8.
        Pixel values are expected to be in the range [0, 255].
        Channel order must be BGR.

    window_size : int
        Size of the square median window. Must be an odd integer.

    Returns
    -------
    np.ndarray
        Median-filtered BGR image of shape (H, W, 3) and dtype uint8.

    Raises
    ------
    ValueError
        If `window_size` is not an odd integer.

    Notes
    -----
    - Each channel is processed independently using the same window size.
    - The implementation relies on a histogram sliding-window approach,
      which is significantly faster than naive median filtering.
    - Padding is performed using edge replication.
    """

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    pad = window_size // 2
    out = np.empty_like(image, dtype=np.uint8)

    for c in range(3):
        padded = np.pad(image[:, :, c], pad, mode="edge")
        out[:, :, c] = median_filter_grayscale_numba(
            padded, window_size
        )

    return out