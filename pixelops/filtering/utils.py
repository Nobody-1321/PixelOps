import numpy as np
from numba import njit, prange

#----------------------------------------------
#
#        Numba-optimized versions
#
#----------------------------------------------

@njit(inline="always")
def reflect(idx: int, size: int) -> int:
    if idx < 0:
        return -idx - 1
    if idx >= size:
        return 2 * size - idx - 1
    return idx

@njit(parallel=True, fastmath=True)
def convolve_separable(
    src: np.ndarray,
    kernel_h: np.ndarray,
    kernel_v: np.ndarray
) -> np.ndarray:
    """
    Separable convolution returning a new array.
    Uses kernel_h for horizontal pass, kernel_v for vertical pass.
    """
    H, W = src.shape
    k_h = kernel_h.shape[0]
    k_v = kernel_v.shape[0]
    r_h = k_h // 2
    r_v = k_v // 2

    tmp = np.empty_like(src)
    dst = np.empty_like(src)

    # Horizontal pass
    for y in prange(H):
        for x in range(W):
            acc = 0.0
            for i in range(k_h):
                xx = reflect(x + i - r_h, W)
                acc += kernel_h[i] * src[y, xx]
            tmp[y, x] = acc

    # Vertical pass
    for y in prange(H):
        for x in range(W):
            acc = 0.0
            for i in range(k_v):
                yy = reflect(y + i - r_v, H)
                acc += kernel_v[i] * tmp[yy, x]
            dst[y, x] = acc

    return dst

@njit(parallel=True, fastmath=True)
def convolve_separable_inplace(
    src: np.ndarray,
    kernel: np.ndarray,
    dst: np.ndarray
) -> None:
    """
    Separable convolution writing result into pre-allocated dst array.
    Uses the same kernel for both horizontal and vertical passes.
    """
    H, W = src.shape
    k = kernel.shape[0]
    r = k // 2

    tmp = np.empty_like(src)

    # Horizontal pass
    for y in prange(H):
        for x in range(W):
            acc = 0.0
            for i in range(k):
                xx = reflect(x + i - r, W)
                acc += kernel[i] * src[y, xx]
            tmp[y, x] = acc

    # Vertical pass
    for y in prange(H):
        for x in range(W):
            acc = 0.0
            for i in range(k):
                yy = reflect(y + i - r, H)
                acc += kernel[i] * tmp[yy, x]
            dst[y, x] = acc

@njit(parallel=True, fastmath=True, cache=True)
def convolve_horizontal_1d(
    img_padded: np.ndarray,
    kernel: np.ndarray,
    pad: int
) -> np.ndarray:
    """
    Apply 1D horizontal convolution to a padded image.

    Parameters
    ----------
    img_padded : np.ndarray
        Padded image of shape (H_pad, W_pad).

    kernel : np.ndarray
        1D convolution kernel.

    pad : int
        Half kernel size.

    Returns
    -------
    np.ndarray
        Horizontally convolved image (same shape as img_padded).
    """

    H_pad, W_pad = img_padded.shape
    ksize = kernel.shape[0]

    tmp = np.zeros_like(img_padded)

    for y in prange(H_pad):
        for x in range(W_pad):
            acc = 0.0
            for i in range(ksize):
                xx = x + i - pad
                if 0 <= xx < W_pad:
                    acc += kernel[i] * img_padded[y, xx]
            tmp[y, x] = acc

    return tmp
