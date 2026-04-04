"""
Anisotropic diffusion filtering.

This module implements the Perona-Malik anisotropic diffusion
algorithm for edge-preserving image smoothing.
"""

import numpy as np
from numba import njit, prange

@njit(inline="always")
def diffusivity_exp(gradient_sq: float, kappa_sq: float) -> float:
    """
    Exponential diffusivity function (Perona-Malik option 1).
    
    g(∇I) = exp(-(|∇I|/κ)²)
    
    Privileges high-contrast edges over low-contrast ones.
    """
    return np.exp(-gradient_sq / kappa_sq)

@njit(inline="always")
def diffusivity_inv(gradient_sq: float, kappa_sq: float) -> float:
    """
    Inverse quadratic diffusivity function (Perona-Malik option 2).
    
    g(∇I) = 1 / (1 + (|∇I|/κ)²)
    
    Privileges wide regions over smaller ones.
    """
    return 1.0 / (1.0 + gradient_sq / kappa_sq)

@njit(parallel=True, fastmath=True, cache=True)
def anisotropic_diffusion_core(
    img: np.ndarray,
    n_iter: int,
    kappa: float,
    gamma: float,
    option: int
) -> np.ndarray:
    """
    Core Perona-Malik anisotropic diffusion filter.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image as float32, values in [0, 1].

    n_iter : int
        Number of diffusion iterations.

    kappa : float
        Conductance parameter (edge threshold).
        Controls sensitivity to edges. Typical values: 20-100.

    gamma : float
        Integration constant (0 < gamma <= 0.25 for stability).
        Controls the speed of diffusion per iteration.

    option : int
        Diffusivity function to use:
        - 1: Exponential function (privileges high-contrast edges)
        - 2: Inverse quadratic (privileges wide regions)

    Returns
    -------
    np.ndarray
        Filtered image as float32,
        values may exceed [0, 1] and should be normalized/clipped by caller.
    """
    H, W = img.shape
    out = img.copy()
    
    kappa_sq = kappa * kappa

    # Buffers for gradients
    nabla_n = np.empty_like(out)
    nabla_s = np.empty_like(out)
    nabla_e = np.empty_like(out)
    nabla_w = np.empty_like(out)

    for _ in range(n_iter):
        # Compute gradients (finite differences)
        for y in prange(H):
            for x in range(W):
                # North gradient (y-1)
                if y > 0:
                    nabla_n[y, x] = out[y - 1, x] - out[y, x]
                else:
                    nabla_n[y, x] = 0.0

                # South gradient (y+1)
                if y < H - 1:
                    nabla_s[y, x] = out[y + 1, x] - out[y, x]
                else:
                    nabla_s[y, x] = 0.0

                # East gradient (x+1)
                if x < W - 1:
                    nabla_e[y, x] = out[y, x + 1] - out[y, x]
                else:
                    nabla_e[y, x] = 0.0

                # West gradient (x-1)
                if x > 0:
                    nabla_w[y, x] = out[y, x - 1] - out[y, x]
                else:
                    nabla_w[y, x] = 0.0

        # Update image with diffusion
        for y in prange(H):
            for x in range(W):
                gn_sq = nabla_n[y, x] * nabla_n[y, x]
                gs_sq = nabla_s[y, x] * nabla_s[y, x]
                ge_sq = nabla_e[y, x] * nabla_e[y, x]
                gw_sq = nabla_w[y, x] * nabla_w[y, x]

                if option == 1:
                    cn = diffusivity_exp(gn_sq, kappa_sq)
                    cs = diffusivity_exp(gs_sq, kappa_sq)
                    ce = diffusivity_exp(ge_sq, kappa_sq)
                    cw = diffusivity_exp(gw_sq, kappa_sq)
                else:
                    cn = diffusivity_inv(gn_sq, kappa_sq)
                    cs = diffusivity_inv(gs_sq, kappa_sq)
                    ce = diffusivity_inv(ge_sq, kappa_sq)
                    cw = diffusivity_inv(gw_sq, kappa_sq)

                out[y, x] += gamma * (
                    cn * nabla_n[y, x] +
                    cs * nabla_s[y, x] +
                    ce * nabla_e[y, x] +
                    cw * nabla_w[y, x]
                )

    return out

def anisotropic_diffusion(
    image: np.ndarray,
    n_iter: int = 10,
    kappa: float = 50.0,
    gamma: float = 0.1,
    option: int = 1
) -> np.ndarray:
    """
    Apply Perona-Malik anisotropic diffusion to an image.

    This filter smooths homogeneous regions while preserving edges.
    It is suitable for denoising while maintaining important structures.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, C).
        Any numeric dtype is accepted.

    n_iter : int, optional
        Number of diffusion iterations. Must be positive.

    kappa : float, optional
        Conductance parameter controlling edge sensitivity.
        Smaller values preserve more edges.

    gamma : float, optional
        Integration constant. Must be in (0, 0.25] for stability.

    option : int, optional
        Diffusivity function:
        - 1: Exponential
        - 2: Inverse quadratic

    Returns
    -------
    np.ndarray
        Diffused image with same shape as input and dtype float32,
        values may exceed [0, 1] and should be normalized/clipped by caller.

    Notes
    -----
    - Implements Perona-Malik anisotropic diffusion.
    - No normalization or clipping is applied.
    - The meaning of `kappa` depends on the scale of `image`.
    """

    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    if kappa <= 0:
        raise ValueError("kappa must be positive.")

    if not (0 < gamma <= 0.25):
        raise ValueError("gamma must be in (0, 0.25].")

    if option not in (1, 2):
        raise ValueError("option must be 1 or 2.")

    img_f = image.astype(np.float32)

    if img_f.ndim == 2:
        return anisotropic_diffusion_core(
            img_f, n_iter, kappa, gamma, option
        )

    elif img_f.ndim == 3:
        out = np.empty_like(img_f)

        for c in range(img_f.shape[2]):
            out[:, :, c] = anisotropic_diffusion_core(
                img_f[:, :, c],
                n_iter,
                kappa,
                gamma,
                option
            )

        return out

    else:
        raise ValueError("Invalid image dimensions.")
