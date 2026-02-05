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
        Filtered image as float32.
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

def anisotropic_diffusion_grayscale(
    image: np.ndarray,
    n_iter: int = 10,
    kappa: float = 50.0,
    gamma: float = 0.1,
    option: int = 1
) -> np.ndarray:
    """
    Apply Perona-Malik anisotropic diffusion to a grayscale image.

    This filter smooths homogeneous regions while preserving edges,
    making it ideal for denoising without blurring important features.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W) and dtype uint8.

    n_iter : int, optional
        Number of diffusion iterations. Default is 10.
        More iterations = more smoothing.

    kappa : float, optional
        Conductance parameter (edge threshold). Default is 50.
        - Low values: preserve more edges (even weak ones)
        - High values: smooth more aggressively

    gamma : float, optional
        Integration constant. Default is 0.1.
        Must be in (0, 0.25] for stability.
        Higher values = faster diffusion per iteration.

    option : int, optional
        Diffusivity function. Default is 1.
        - 1: Exponential (exp(-(∇I/κ)²)) - preserves high-contrast edges
        - 2: Inverse quadratic (1/(1+(∇I/κ)²)) - preserves wide regions

    Returns
    -------
    np.ndarray
        Filtered grayscale image of shape (H, W) and dtype uint8.

    Raises
    ------
    ValueError
        If gamma is not in (0, 0.25].
        If option is not 1 or 2.
        If image is not 2D.

    Examples
    --------
    >>> from pixelops.filtering import anisotropic_diffusion_grayscale
    >>> # Light smoothing preserving edges
    >>> out = anisotropic_diffusion_grayscale(img, n_iter=5, kappa=30)
    >>> # Strong smoothing
    >>> out = anisotropic_diffusion_grayscale(img, n_iter=20, kappa=100)

    Notes
    -----
    - The algorithm is iterative; each iteration applies a small amount
      of diffusion controlled by gamma.
    - Kappa should be chosen based on the noise level and edge contrast
      in the image.
    - Option 1 (exponential) tends to create piecewise constant regions.
    - Option 2 (inverse) tends to preserve smoother gradients.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale (2D array).")
    
    if not (0 < gamma <= 0.25):
        raise ValueError("gamma must be in (0, 0.25] for stability.")
    
    if option not in (1, 2):
        raise ValueError("option must be 1 or 2.")

    img_f = image.astype(np.float32) / 255.0
    
    out = anisotropic_diffusion_core(img_f, n_iter, kappa / 255.0, gamma, option)
    
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def anisotropic_diffusion_bgr(
    image: np.ndarray,
    n_iter: int = 10,
    kappa: float = 50.0,
    gamma: float = 0.1,
    option: int = 1
) -> np.ndarray:
    """
    Apply Perona-Malik anisotropic diffusion to a BGR image.

    The filter is applied independently to each color channel.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image of shape (H, W, 3) and dtype uint8.

    n_iter : int, optional
        Number of diffusion iterations. Default is 10.

    kappa : float, optional
        Conductance parameter (edge threshold). Default is 50.

    gamma : float, optional
        Integration constant. Default is 0.1.
        Must be in (0, 0.25] for stability.

    option : int, optional
        Diffusivity function (1 or 2). Default is 1.

    Returns
    -------
    np.ndarray
        Filtered BGR image of shape (H, W, 3) and dtype uint8.

    Raises
    ------
    ValueError
        If gamma is not in (0, 0.25].
        If option is not 1 or 2.
        If image is not 3D with 3 channels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be BGR (3D array with 3 channels).")
    
    if not (0 < gamma <= 0.25):
        raise ValueError("gamma must be in (0, 0.25] for stability.")
    
    if option not in (1, 2):
        raise ValueError("option must be 1 or 2.")

    img_f = image.astype(np.float32) / 255.0
    out = np.empty_like(img_f)
    
    kappa_norm = kappa / 255.0

    for c in range(3):
        out[:, :, c] = anisotropic_diffusion_core(
            img_f[:, :, c].copy(), n_iter, kappa_norm, gamma, option
        )

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
