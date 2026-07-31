import numpy as np
import cv2 as cv
from numba import njit, prange
from pixelops.core import (
    to_float32,
    validate_grayscale
)

@njit(parallel=True, fastmath=True, cache=True)
def _update_etf_core(
    etf: np.ndarray,
    grad_mag: np.ndarray,
    r: int
) -> np.ndarray:
    """
    Core function to update the Edge Tangent Flow (ETF) using non-linear smoothing.
    """
    h, w, _ = etf.shape
    new_etf = np.zeros_like(etf)
    
    for y in prange(h):
        for x in range(w):
            t_cur = etf[y, x]
            g_cur = grad_mag[y, x]
            
            t_accum_x = 0.0
            t_accum_y = 0.0
            
            y_min = max(0, y - r)
            y_max = min(h - 1, y + r)
            x_min = max(0, x - r)
            x_max = min(w - 1, x + r)
            
            for ny in range(y_min, y_max + 1):
                for nx in range(x_min, x_max + 1):
                    t_nbr = etf[ny, nx]
                    g_nbr = grad_mag[ny, nx]
                    
                    # Weight calculations
                    w_m = 0.5 * (1.0 + np.tanh(g_nbr - g_cur))
                    dot_product = t_cur[0] * t_nbr[0] + t_cur[1] * t_nbr[1]
                    w_d = abs(dot_product)
                    phi = 1.0 if dot_product >= 0.0 else -1.0
                    
                    weight = w_m * w_d
                    
                    t_accum_x += phi * t_nbr[0] * weight
                    t_accum_y += phi * t_nbr[1] * weight
            
            norm = np.sqrt(t_accum_x**2 + t_accum_y**2)
            if norm > 0.0:
                new_etf[y, x, 0] = t_accum_x / norm
                new_etf[y, x, 1] = t_accum_y / norm
            else:
                new_etf[y, x, 0] = t_cur[0]
                new_etf[y, x, 1] = t_cur[1]
                
    return new_etf

def compute_etf(
    img: np.ndarray,
    r: int = 5,
    iterations: int = 3
) -> np.ndarray:
    """
    Initialize and refine the Edge Tangent Flow (ETF) vector field.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).
    r : int, optional
        Radius for the spatial neighborhood smoothing (default is 5).
    iterations : int, optional
        Number of refinement iterations (default is 3).

    Returns
    -------
    np.ndarray
        Refined ETF vector field of shape (H, W, 2) and dtype float32.
    """
    if r <= 0 or iterations <= 0:
        raise ValueError("Radius (r) and iterations must be positive.")

    validate_grayscale(img)
    img_f = to_float32(img)
    
    # Calculate spatial gradients
    sobel_x = cv.Sobel(img_f, cv.CV_32F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(img_f, cv.CV_32F, 0, 1, ksize=3)
    
    grad_mag = cv.magnitude(sobel_x, sobel_y)
    max_mag = grad_mag.max()
    if max_mag > 0:
        grad_mag /= max_mag
        
    h, w = img_f.shape
    etf = np.zeros((h, w, 2), dtype=np.float32)
    etf[:, :, 0] = -sobel_y
    etf[:, :, 1] = sobel_x
    
    norms = np.sqrt(etf[:, :, 0]**2 + etf[:, :, 1]**2)
    mask = norms > 0
    etf[mask, 0] /= norms[mask]
    etf[mask, 1] /= norms[mask]
    
    for _ in range(iterations):
        etf = _update_etf_core(etf, grad_mag, r)
        
    return etf