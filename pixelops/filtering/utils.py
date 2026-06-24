import numpy as np
from numba import njit, prange
from .frequency.fft import fourier_transform_2d, inverse_fourier_transform_2d
from pixelops.core import (
    to_float32,
    validate_image
)

# ----------------------------------------------
#
#        Numba-optimized filtering utilities
#
# ----------------------------------------------

@njit(inline="always")
def reflect(idx: int, size: int) -> int:
    """
    Compute reflected index for boundary handling.

    Implements reflection boundary condition where indices
    outside [0, size) are mirrored back into valid range.

    Parameters
    ----------
    idx : int
        Input index (may be negative or >= size).

    size : int
        Size of the dimension.

    Returns
    -------
    int
        Reflected index in range [0, size).

    Notes
    -----
    - Equivalent to OpenCV's BORDER_REFLECT_101.
    - Handles one level of reflection (single boundary crossing).
    """
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
    Perform separable 2D convolution on a grayscale image.

    Applies horizontal convolution followed by vertical convolution,
    exploiting separability for O(k) complexity instead of O(k²).

    Parameters
    ----------
    src : np.ndarray
        Input grayscale image of shape (H, W) and dtype float32.

    kernel_h : np.ndarray
        1D horizontal kernel.

    kernel_v : np.ndarray
        1D vertical kernel.

    Returns
    -------
    np.ndarray
        Convolved image of shape (H, W) and dtype float32.

    Notes
    -----
    - Uses reflection boundary handling.
    - Parallelized with Numba for performance.
    - Result is a new array (src is not modified).
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
    Perform separable convolution writing result to pre-allocated array.

    Uses the same kernel for both horizontal and vertical passes.
    More memory-efficient than `convolve_separable` when the
    destination buffer can be reused.

    Parameters
    ----------
    src : np.ndarray
        Input grayscale image of shape (H, W) and dtype float32.

    kernel : np.ndarray
        1D kernel used for both horizontal and vertical passes.

    dst : np.ndarray
        Pre-allocated output array of same shape as src.
        Will be overwritten with the convolution result.

    Returns
    -------
    None
        Result is written to dst in-place.

    Notes
    -----
    - Uses reflection boundary handling.
    - Parallelized with Numba for performance.
    - Internally allocates a temporary buffer.
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

@njit(parallel=True, fastmath=True, cache=True)
def _distance_matrix(shape: tuple) -> np.ndarray:
    """
    Compute a centered Euclidean distance matrix from the frequency grid origin.

    Parameters
    ----------
    shape : tuple
        Dimensions of the matrix as (height, width).

    Returns
    -------
    np.ndarray
        Float32 2D array representing the distance D(u, v) to the center.
    """
    H, W = shape
    crow, ccol = H // 2, W // 2
    distance = np.empty((H, W), dtype=np.float32)
    
    for i in prange(H):
        for j in range(W):
            dy = float(i - crow)
            dx = float(j - ccol)
            distance[i, j] = np.sqrt(dx * dx + dy * dy)
            
    return distance

def apply_frequency_filter(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a centered frequency-domain mask to an image.

    This function automatically handles forward FFT, shifting operations,
    mask multiplication, and recovery back into normalized spatial space.

    Parameters
    ----------
    image : np.ndarray
        Input spatial image of shape (H, W) or (H, W, C).
    mask : np.ndarray
        Centered filter mask matrix with shape (H, W).

    Returns
    -------
    np.ndarray
        Filtered spatial image (float32) normalized to the [0.0, 1.0] interval.
    """
    validate_image(image)
    img_f = to_float32(image)

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D matrix.")
    if img_f.shape[:2] != mask.shape:
        raise ValueError("Image spatial dimensions and mask shape must match.")

    if img_f.ndim == 2:
        dft = fourier_transform_2d(img_f)
        dft_shifted = np.fft.fftshift(dft)
        filtered_shifted = dft_shifted * mask
        filtered_dft = np.fft.ifftshift(filtered_shifted)
        out = inverse_fourier_transform_2d(filtered_dft)
    else:
        out = np.empty_like(img_f)
        for c in range(img_f.shape[2]):
            dft = fourier_transform_2d(img_f[:, :, c])
            dft_shifted = np.fft.fftshift(dft)
            filtered_shifted = dft_shifted * mask
            filtered_dft = np.fft.ifftshift(filtered_shifted)
        out[:, :, c] = inverse_fourier_transform_2d(filtered_dft)

    return np.clip(out, 0.0, 1.0)