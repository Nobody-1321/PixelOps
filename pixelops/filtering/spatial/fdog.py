import numpy as np
import math
from numba import njit, prange
from pixelops.core import validate_grayscale

@njit(parallel=True, fastmath=True, cache=True)
def _fdog_core(
    img: np.ndarray,
    etf: np.ndarray,
    sigma_c: float,
    sigma_s: float,
    sigma_m: float,
    rho: float
) -> np.ndarray:
    """
    Core FDoG filter evaluating flow-based integral curves.
    """
    h, w = img.shape
    H_out = np.zeros((h, w), dtype=np.float32)
    
    S_len = int(math.ceil(3.0 * sigma_m))
    T_len = int(math.ceil(3.0 * sigma_s))
    
    gauss_c = np.zeros(T_len + 1, dtype=np.float32)
    gauss_s = np.zeros(T_len + 1, dtype=np.float32)
    gauss_m = np.zeros(S_len + 1, dtype=np.float32)
    
    for t in range(T_len + 1):
        gauss_c[t] = math.exp(-0.5 * (t / sigma_c)**2) / (math.sqrt(2.0 * math.pi) * sigma_c)
        gauss_s[t] = math.exp(-0.5 * (t / sigma_s)**2) / (math.sqrt(2.0 * math.pi) * sigma_s)
    for s in range(S_len + 1):
        gauss_m[s] = math.exp(-0.5 * (s / sigma_m)**2) / (math.sqrt(2.0 * math.pi) * sigma_m)

    for y in prange(h):
        for x in range(w):
            sum_h = 0.0
            w_h_sum = 0.0
            
            # --- Integración Longitudinal (Paso s a lo largo del flujo ETF) ---
            for s in range(-S_len, S_len + 1):
                cx, cy = float(x), float(y)
                step = float(s)
                
                remaining = abs(step)
                while remaining > 0.0:
                    ix, iy = int(round(cx)), int(round(cy))
                    if ix < 0 or ix >= w or iy < 0 or iy >= h:
                        break
                    t_vec = etf[iy, ix]
                    direction = 1.0 if step >= 0 else -1.0
                    cx += direction * t_vec[0]
                    cy += direction * t_vec[1]
                    remaining -= 1.0
                    
                ix, iy = int(round(cx)), int(round(cy))
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    continue
                
                t_local = etf[iy, ix]
                g_local_x, g_local_y = -t_local[1], t_local[0]
                
                # --- Integración Transversal (Paso t perpendicular al flujo) ---
                sum_f = 0.0
                for t in range(-T_len, T_len + 1):
                    lx = cx + t * g_local_x
                    ly = cy + t * g_local_y
                    
                    ilx, ily = int(round(lx)), int(round(ly))
                    if ilx < 0 or ilx >= w or ily < 0 or ily >= h:
                        val = 255.0  # Asumir fondo blanco si sale de la imagen[cite: 7]
                    else:
                        val = float(img[ily, ilx])
                        
                    # Aplicar núcleo DoG 1D
                    w_c = gauss_c[abs(t)]
                    w_s = gauss_s[abs(t)]
                    sum_f += val * (w_c - rho * w_s)
                
                # Ponderación longitudinal
                w_m = gauss_m[abs(s)]
                sum_h += sum_f * w_m
                w_h_sum += w_m
                
            if w_h_sum > 0:
                H_out[y, x] = sum_h / w_h_sum
                
    return H_out

def apply_fdog(
    img: np.ndarray,
    etf: np.ndarray,
    sigma_m: float = 3.0,
    sigma_c: float = 1.0,
    rho: float = 0.99,
    tau: float = 0.5
) -> np.ndarray:
    """
    Apply Flow-based Difference of Gaussians (FDoG) along ETF vector field.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.
    etf : np.ndarray
        Edge Tangent Flow vector field of shape (H, W, 2).
    sigma_m : float, optional
        Standard deviation for integration along the flow.
    sigma_c : float, optional
        Standard deviation for center of DoG.
    rho : float, optional
        Noise parameter for DoG.
    tau : float, optional
        Thresholding parameter for the final binarization.

    Returns
    -------
    np.ndarray
        Binarized line drawing image (uint8, values 0 or 255).
    """
    if sigma_m <= 0 or sigma_c <= 0:
        raise ValueError("Sigma values must be positive.")

    validate_grayscale(img)
    
    # IMPORTANTE: Usamos astype en lugar de to_float32 para mantener 
    # la escala original [0, 255] que necesita la binarización hiperbólica[cite: 7].
    img_f = img.astype(np.float32)
    sigma_s = 1.6 * sigma_c
    
    H_out = _fdog_core(img_f, etf, sigma_c, sigma_s, sigma_m, rho)
    
    # Binarización basada en la función hiperbólica tanh[cite: 7]
    output = np.ones_like(img, dtype=np.uint8) * 255
    condition = (H_out < 0) & ((1.0 + np.tanh(H_out)) < tau)
    output[condition] = 0
    
    return output