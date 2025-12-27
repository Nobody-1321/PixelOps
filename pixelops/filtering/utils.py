import numpy as np
from scipy.ndimage import convolve1d
from numba import njit, prange

#----------------------------------------------
#
#        Numba-optimized versions
#
#----------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def convolve_separable_numba(
    img_padded: np.ndarray,
    kernel_h: np.ndarray,
    kernel_v: np.ndarray,
    pad: int
) -> np.ndarray:
    """
    Perform separable convolution on a padded grayscale image.

    Parameters
    ----------
    img_padded : np.ndarray
        Padded input image of shape (H + 2*pad, W + 2*pad).
        Must be float32 or float64.

    kernel_h : np.ndarray
        1D horizontal convolution kernel.

    kernel_v : np.ndarray
        1D vertical convolution kernel.

    pad : int
        Padding size (half kernel width).

    Returns
    -------
    np.ndarray
        Convolved image of shape (H, W), without padding.
    """

    H = img_padded.shape[0] - 2 * pad
    W = img_padded.shape[1] - 2 * pad
    ksize = kernel_h.shape[0]

    tmp = np.zeros((H + 2 * pad, W + 2 * pad), dtype=img_padded.dtype)
    out = np.zeros((H, W), dtype=img_padded.dtype)

    # Horizontal convolution
    for y in prange(H):
        yy = y + pad
        for x in range(W):
            xx = x + pad
            acc = 0.0
            for i in range(ksize):
                acc += kernel_h[i] * img_padded[yy, xx + i - pad]
            tmp[yy, xx] = acc

    # Vertical convolution
    for y in prange(H):
        yy = y + pad
        for x in range(W):
            xx = x + pad
            acc = 0.0
            for i in range(ksize):
                acc += kernel_v[i] * tmp[yy + i - pad, xx]
            out[y, x] = acc

    return out

def convolve_separable(
    img: np.ndarray,
    kernel_h: np.ndarray,
    kernel_v: np.ndarray
) -> np.ndarray:
    """
    Apply separable convolution using a Numba-optimized backend.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).

    kernel_h : np.ndarray
        Horizontal 1D kernel.

    kernel_v : np.ndarray
        Vertical 1D kernel.

    Returns
    -------
    np.ndarray
        Convolved image of shape (H, W).
    """

    pad = kernel_h.shape[0] // 2
    img_padded = np.pad(img, pad, mode="reflect")

    return convolve_separable_numba(img_padded, kernel_h, kernel_v, pad)
