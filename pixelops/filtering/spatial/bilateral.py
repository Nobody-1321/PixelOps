import numpy as np
from scipy.special import comb
from numba import njit, prange
from ..utils import convolve_separable_inplace
from ..kernels import create_gaussian_kernel_radius


def binomial_coeffs(n: int, dtype=np.float32) -> np.ndarray:
    """
    Compute normalized binomial coefficients for bilateral filtering.

    These coefficients are used in the trigonometric polynomial
    approximation of the bilateral filter's range kernel.

    Parameters
    ----------
    n : int
        Order of the binomial expansion. Must be positive.

    dtype : np.dtype, optional
        Data type for the output array. Default is float32.

    Returns
    -------
    np.ndarray
        Array of shape (n + 1,) containing normalized binomial
        coefficients.

    Notes
    -----
    - Normalization factor is 2^(2n-2).
    - Higher n provides better approximation at increased cost.
    """
    coeffs = np.empty(n + 1, dtype=dtype)
    norm = 2 ** (2 * n - 2)
    for i in range(n + 1):
        coeffs[i] = comb(n, i, exact=False) / norm
    return coeffs

@njit(parallel=True, fastmath=True, cache=True)
def bilateral_filter_core(
    I: np.ndarray,
    binom: np.ndarray,
    gauss_kernel: np.ndarray,
    n_iter: int,
    sr: float
):
    """
    Core bilateral filter using trigonometric polynomial approximation.

    Implements fast bilateral filtering via Fourier series expansion
    of the range kernel, enabling separable convolution.

    Parameters
    ----------
    I : np.ndarray
        Input grayscale image as float32, values in [0, 1].
        Modified in-place.

    binom : np.ndarray
        Binomial coefficients for polynomial approximation.

    gauss_kernel : np.ndarray
        1D Gaussian kernel for spatial filtering.

    n_iter : int
        Number of iterations.

    sr : float
        Range sigma (intensity similarity bandwidth).

    Returns
    -------
    np.ndarray
        Filtered image as float32.

    Notes
    -----
    - Uses trigonometric recurrence for efficiency.
    - Numba-compiled for performance.
    """
    H, W = I.shape
    n = binom.shape[0] - 1
    sqrt_n = np.sqrt(n)
    g = 1.0 / sr

    # Buffers
    num = np.empty_like(I)
    den = np.empty_like(I)

    Gi0 = np.empty_like(I)
    Gi1 = np.empty_like(I)
    Gir0 = np.empty_like(I)
    Gir1 = np.empty_like(I)
    Hir0 = np.empty_like(I)
    Hir1 = np.empty_like(I)

    # Trig base (computed ONCE) - ensure same dtype as I
    cos_a = np.cos(g * I / sqrt_n).astype(I.dtype)
    sin_a = np.sin(g * I / sqrt_n).astype(I.dtype)

    cos_i = np.ones_like(I)
    sin_i = np.zeros_like(I)

    for _ in range(n_iter):
        num[:] = 0.0
        den[:] = 0.0
        cos_i[:] = 1.0
        sin_i[:] = 0.0

        for i in range(n + 1):
            b = binom[i]

            # Gi components
            Gi0[:] = I * cos_i
            Gi1[:] = I * sin_i

            convolve_separable_inplace(Gi0, gauss_kernel, Gir0)
            convolve_separable_inplace(Gi1, gauss_kernel, Gir1)
            convolve_separable_inplace(cos_i, gauss_kernel, Hir0)
            convolve_separable_inplace(sin_i, gauss_kernel, Hir1)

            num += b * (cos_i * Gir0 + sin_i * Gir1)
            den += b * (cos_i * Hir0 + sin_i * Hir1)

            # Trigonometric recurrence
            tmp = cos_i * cos_a - sin_i * sin_a
            sin_i = sin_i * cos_a + cos_i * sin_a
            cos_i = tmp

        # Stable division
        for y in prange(H):
            for x in range(W):
                if den[y, x] > 1e-6:
                    I[y, x] = num[y, x] / den[y, x]
                else:
                    I[y, x] = 0.0

    return I

def bilateral_filter(
    img: np.ndarray,
    ss: float,
    sr: float,
    n_iter: int = 1,
    n: int = 3
) -> np.ndarray:
    """
    Apply bilateral filter to a grayscale or multi-channel image.

    Edge-preserving smoothing filter that considers both spatial
    proximity and intensity similarity. Uses fast polynomial
    approximation for efficiency.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W) or (H, W, C).
        Any numeric dtype is accepted.

    ss : float
        Spatial sigma (spatial bandwidth in pixels).
        Must be positive. Controls spatial extent of smoothing.

    sr : float
        Range sigma (intensity bandwidth).
        Must be positive. Controls sensitivity to intensity differences.

    n_iter : int, optional
        Number of filtering iterations. Default is 1.
        Must be positive.

    n : int, optional
        Polynomial order for approximation. Default is 3.
        Must be a positive odd integer. Higher values improve
        accuracy but increase computation.

    Returns
    -------
    np.ndarray
        Filtered image with same shape as input and dtype float32.
        Values are in range [0, 1].

    Raises
    ------
    ValueError
        If ss <= 0, sr <= 0, n_iter <= 0, or n is not a positive
        odd integer.

    Notes
    -----
    - Input is normalized to [0, 1] internally.
    - No normalization or quantization is applied to output.
    - Each channel is processed independently.

    References
    ----------
    .. [1] Chaudhury, K., "Fast O(1) Bilateral Filtering Using
           Trigonometric Range Kernels", TIP 2011.
    """

    if ss <= 0:
        raise ValueError("Spatial sigma (ss) must be positive.")

    if sr <= 0:
        raise ValueError("Range sigma (sr) must be positive.")

    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer.")

    img_f = img.astype(np.float32) / 255.0

    binom = binomial_coeffs(n, img_f.dtype)
    gauss_kernel = create_gaussian_kernel_radius(ss)

    # Grayscale
    if img_f.ndim == 2:
        out = bilateral_filter_core(
            img_f, binom, gauss_kernel, n_iter, sr
        )

    # Multi-channel
    elif img_f.ndim == 3:
        out = np.empty_like(img_f)
        for c in range(img_f.shape[2]):
            out[:, :, c] = bilateral_filter_core(
                img_f[:, :, c],
                binom,
                gauss_kernel,
                n_iter,
                sr
            )
    else:
        raise ValueError("Input image must be 2D or 3D.")

    return out