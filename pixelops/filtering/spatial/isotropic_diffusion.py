import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def isotropic_diffusion_core(
    img: np.ndarray,
    n_iter: int,
    gamma: float
) -> np.ndarray:
    """
    Core isotropic diffusion filter (Heat equation / Gaussian smoothing).

    Implements the heat equation: ∂I/∂t = ∇²I (Laplacian)
    
    This is equivalent to iteratively applying Gaussian blur.
    After n iterations with step gamma, the result approximates a 
    Gaussian blur with σ = √(2 * n * gamma).

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image as float32, values in [0, 1].

    n_iter : int
        Number of diffusion iterations.

    gamma : float
        Integration constant (0 < gamma <= 0.25 for stability).
        Controls the speed of diffusion per iteration.

    Returns
    -------
    np.ndarray
        Filtered image as float32.
    """
    H, W = img.shape
    out = img.copy()

    # Buffer for Laplacian
    laplacian = np.empty_like(out)

    for _ in range(n_iter):
        # Compute discrete Laplacian using 4-connectivity
        for y in prange(H):
            for x in range(W):
                # Get neighbor values with boundary handling
                north = out[y - 1, x] if y > 0 else out[y, x]
                south = out[y + 1, x] if y < H - 1 else out[y, x]
                east = out[y, x + 1] if x < W - 1 else out[y, x]
                west = out[y, x - 1] if x > 0 else out[y, x]
                
                # Laplacian: ∇²I = I_n + I_s + I_e + I_w - 4*I_center
                laplacian[y, x] = north + south + east + west - 4.0 * out[y, x]

        # Update: I(t+1) = I(t) + gamma * ∇²I
        for y in prange(H):
            for x in range(W):
                out[y, x] += gamma * laplacian[y, x]

    return out

def isotropic_diffusion(
    image: np.ndarray,
    n_iter: int = 10,
    gamma: float = 0.25
) -> np.ndarray:
    """
    Apply isotropic diffusion (heat equation) to an image.

    This filter performs uniform smoothing equivalent to Gaussian blur.
    Unlike anisotropic diffusion, it does NOT preserve edges.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, C).
        Any numeric dtype is accepted.

    n_iter : int, optional
        Number of diffusion iterations. Must be positive.

    gamma : float, optional
        Integration constant. Must be in (0, 0.25] for stability.

    Returns
    -------
    np.ndarray
        Diffused image with same shape as input and dtype float32.

    Notes
    -----
    - Equivalent to solving the heat equation:
        ∂I/∂t = ∇²I
    - Effective Gaussian sigma is approximately:
        σ ≈ √(2 * n_iter * gamma)
    - No normalization or clipping is applied.
    """

    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    if not (0 < gamma <= 0.25):
        raise ValueError("gamma must be in (0, 0.25].")

    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)
        
    if img_f.ndim == 2:
        return isotropic_diffusion_core(img_f, n_iter, gamma)

    elif img_f.ndim == 3:
        out = np.empty_like(img_f)

        for c in range(img_f.shape[2]):
            out[:, :, c] = isotropic_diffusion_core(
                img_f[:, :, c],
                n_iter,
                gamma
            )

        return out

    else:
        raise ValueError("Invalid image dimensions.")

'''
def isotropic_diffusion_grayscale(
    image: np.ndarray,
    n_iter: int = 10,
    gamma: float = 0.25
) -> np.ndarray:
    """
    Apply isotropic diffusion (heat equation) to a grayscale image.

    This filter performs uniform smoothing equivalent to Gaussian blur.
    Unlike anisotropic diffusion, it does NOT preserve edges - all regions
    are smoothed equally.

    The effective Gaussian sigma after n iterations is approximately:
        σ_effective ≈ √(2 * n_iter * gamma)

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W) and dtype uint8.

    n_iter : int, optional
        Number of diffusion iterations. Default is 10.
        More iterations = more smoothing.

    gamma : float, optional
        Integration constant. Default is 0.25 (maximum stable value).
        Must be in (0, 0.25] for stability.

    Returns
    -------
    np.ndarray
        Filtered grayscale image of shape (H, W) and dtype uint8.

    Raises
    ------
    ValueError
        If gamma is not in (0, 0.25].
        If image is not 2D.

    Examples
    --------
    >>> from pixelops.filtering import isotropic_diffusion_grayscale
    >>> # Light smoothing (σ ≈ 2.2)
    >>> out = isotropic_diffusion_grayscale(img, n_iter=10, gamma=0.25)
    >>> # Strong smoothing (σ ≈ 5)
    >>> out = isotropic_diffusion_grayscale(img, n_iter=50, gamma=0.25)

    Notes
    -----
    - This is equivalent to solving the heat equation ∂I/∂t = ∇²I.
    - For edge-preserving smoothing, use anisotropic_diffusion_grayscale.
    - The algorithm is unconditionally stable for gamma ≤ 0.25.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale (2D array).")
    
    if not (0 < gamma <= 0.25):
        raise ValueError("gamma must be in (0, 0.25] for stability.")

    img_f = image.astype(np.float32) / 255.0
    
    out = isotropic_diffusion_core(img_f, n_iter, gamma)
    
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def isotropic_diffusion_bgr(
    image: np.ndarray,
    n_iter: int = 10,
    gamma: float = 0.25
) -> np.ndarray:
    """
    Apply isotropic diffusion (heat equation) to a BGR image.

    The filter is applied independently to each color channel.
    This performs uniform Gaussian-like smoothing without edge preservation.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image of shape (H, W, 3) and dtype uint8.

    n_iter : int, optional
        Number of diffusion iterations. Default is 10.

    gamma : float, optional
        Integration constant. Default is 0.25.
        Must be in (0, 0.25] for stability.

    Returns
    -------
    np.ndarray
        Filtered BGR image of shape (H, W, 3) and dtype uint8.

    Raises
    ------
    ValueError
        If gamma is not in (0, 0.25].
        If image is not 3D with 3 channels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be BGR (3D array with 3 channels).")
    
    if not (0 < gamma <= 0.25):
        raise ValueError("gamma must be in (0, 0.25] for stability.")

    img_f = image.astype(np.float32) / 255.0
    out = np.empty_like(img_f)

    for c in range(3):
        out[:, :, c] = isotropic_diffusion_core(
            img_f[:, :, c].copy(), n_iter, gamma
        )

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
'''