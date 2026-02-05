import numpy as np
from scipy.special import comb
from numba import njit, prange
from ..utils import convolve_separable_inplace
from ..kernels import create_gaussian_kernel_radius
import numpy as np
from scipy.special import comb
from numba import njit, prange

def binomial_coeffs(n: int, dtype=np.float32) -> np.ndarray:
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

def bilateral_filter_grayscale(
    img: np.ndarray,
    ss: float,
    sr: float,
    n_iter: int = 1,
    n: int = 3
) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    binom = binomial_coeffs(n, img_f.dtype)
    gauss_kernel = create_gaussian_kernel_radius(ss)

    out = bilateral_filter_core(
        img_f, binom, gauss_kernel, n_iter, sr
    )

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def bilateral_filter_bgr(
    img: np.ndarray,
    ss: float,
    sr: float,
    n_iter: int = 1,
    n: int = 3
) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    binom = binomial_coeffs(n, img_f.dtype)
    gauss_kernel = create_gaussian_kernel_radius(ss)

    out = np.empty_like(img_f)

    for c in range(3):
        channel = img_f[:, :, c].copy()
        out[:, :, c] = bilateral_filter_core(channel, binom, gauss_kernel, n_iter, sr)

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
