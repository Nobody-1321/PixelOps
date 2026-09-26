"""
Implementation based on the "Flow-Based Image Abstraction" algorithm.

Academic Reference:
    Kang, H., Lee, S., & Chui, C. K. (2009). 
    "Flow-Based Image Abstraction". 
    IEEE Transactions on Visualization and Computer Graphics.
    
Description:
    This module implements a complete stylistic image abstraction.
    It splits the process into two branches:
    1. Line extraction using Flow-based Difference of Gaussians (FDoG).
    2. Region smoothing using Flow-Based Bilateral (FBL) filtering 
       followed by luminance quantization.
    Finally, it merges both branches for a cartoon-like illustration.
"""

import numpy as np
import cv2 as cv
import math
from numba import njit, prange
from pixelops.core import validate_image
from pixelops.filtering.spatial.etf import compute_etf
from pixelops.filtering.spatial.fdog import apply_fdog

@njit(parallel=True, fastmath=True, cache=True)
def _fbl_flow_pass(img: np.ndarray, etf: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """
    Applies 1D Bilateral Filter along the ETF flow curves (Ce in paper).
    Protects and cleans up shape boundaries.
    """
    h, w, _ = img.shape
    out = np.zeros_like(img)
    S_len = int(math.ceil(3.0 * sigma_s))
    
    gauss_s = np.zeros(S_len + 1, dtype=np.float32)
    for s in range(S_len + 1):
        gauss_s[s] = math.exp(-0.5 * (s / sigma_s)**2)
        
    for y in prange(h):
        for x in range(w):
            center_color = img[y, x]
            sum_color = np.zeros(3, dtype=np.float32)
            sum_weight = 0.0
            
            for direction in (-1.0, 1.0):
                cx, cy = float(x), float(y)
                step = 1
                while step <= S_len:
                    ix, iy = int(round(cx)), int(round(cy))
                    if ix < 0 or ix >= w or iy < 0 or iy >= h:
                        break
                    
                    t_vec = etf[iy, ix]
                    cx += direction * t_vec[0]
                    cy += direction * t_vec[1]
                    
                    nx, ny = int(round(cx)), int(round(cy))
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        break
                        
                    nbr_color = img[ny, nx]
                    
                    # Euclidean distance in RGB color space
                    dr = center_color[0] - nbr_color[0]
                    dg = center_color[1] - nbr_color[1]
                    db = center_color[2] - nbr_color[2]
                    color_dist_sq = dr**2 + dg**2 + db**2
                    
                    w_r = math.exp(-0.5 * color_dist_sq / (sigma_r**2))
                    w_s = gauss_s[step]
                    
                    weight = w_s * w_r
                    sum_color[0] += nbr_color[0] * weight
                    sum_color[1] += nbr_color[1] * weight
                    sum_color[2] += nbr_color[2] * weight
                    sum_weight += weight
                    step += 1

            # Center pixel addition
            sum_color[0] += center_color[0] * gauss_s[0]
            sum_color[1] += center_color[1] * gauss_s[0]
            sum_color[2] += center_color[2] * gauss_s[0]
            sum_weight += gauss_s[0]
            
            out[y, x, 0] = sum_color[0] / sum_weight
            out[y, x, 1] = sum_color[1] / sum_weight
            out[y, x, 2] = sum_color[2] / sum_weight
            
    return out

@njit(parallel=True, fastmath=True, cache=True)
def _fbl_grad_pass(img: np.ndarray, etf: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """
    Applies 1D Bilateral Filter perpendicular to the flow/gradient axis (Cg in paper).
    Smooths out region interiors.
    """
    h, w, _ = img.shape
    out = np.zeros_like(img)
    S_len = int(math.ceil(3.0 * sigma_s))
    
    gauss_s = np.zeros(S_len + 1, dtype=np.float32)
    for s in range(S_len + 1):
        gauss_s[s] = math.exp(-0.5 * (s / sigma_s)**2)
        
    for y in prange(h):
        for x in range(w):
            t_vec = etf[y, x]
            dx = -t_vec[1]
            dy = t_vec[0]
            
            center_color = img[y, x]
            sum_color = np.zeros(3, dtype=np.float32)
            sum_weight = 0.0
            
            for direction in (-1.0, 1.0):
                for step in range(1, S_len + 1):
                    nx = int(round(x + direction * step * dx))
                    ny = int(round(y + direction * step * dy))
                    
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        break
                        
                    nbr_color = img[ny, nx]
                    
                    dr = center_color[0] - nbr_color[0]
                    dg = center_color[1] - nbr_color[1]
                    db = center_color[2] - nbr_color[2]
                    color_dist_sq = dr**2 + dg**2 + db**2
                    
                    w_r = math.exp(-0.5 * color_dist_sq / (sigma_r**2))
                    w_s = gauss_s[step]
                    
                    weight = w_s * w_r
                    sum_color[0] += nbr_color[0] * weight
                    sum_color[1] += nbr_color[1] * weight
                    sum_color[2] += nbr_color[2] * weight
                    sum_weight += weight

            # Center pixel addition
            sum_color[0] += center_color[0] * gauss_s[0]
            sum_color[1] += center_color[1] * gauss_s[0]
            sum_color[2] += center_color[2] * gauss_s[0]
            sum_weight += gauss_s[0]
            
            out[y, x, 0] = sum_color[0] / sum_weight
            out[y, x, 1] = sum_color[1] / sum_weight
            out[y, x, 2] = sum_color[2] / sum_weight
            
    return out

def quantize_luminance(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Stylize the filtered image by region flattening using uniform-sized-bin 
    luminance quantization.
    """
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    
    l_float = l.astype(np.float32)
    bin_size = 256.0 / bins
    l_quant = np.floor(l_float / bin_size) * bin_size + (bin_size / 2.0)
    l_quant = np.clip(l_quant, 0, 255).astype(np.uint8)
    
    lab_quant = cv.merge([l_quant, a, b])
    return cv.cvtColor(lab_quant, cv.COLOR_LAB2BGR)

def flow_based_abstraction(
    image: np.ndarray,
    etf_r: int = 5,
    etf_iter: int = 3,
    fdog_iter: int = 2,
    fbl_iter: int = 3,
    quant_bins: int = 8,
    sigma_m: float = 3.0,
    sigma_c: float = 1.0,
    rho: float = 0.99,
    tau: float = 0.5,
    sigma_e: float = 2.0,
    r_e: float = 50.0,
    sigma_g: float = 2.0,
    r_g: float = 10.0
) -> np.ndarray:
    """
    Generates a stylized visual abstraction from a photograph.

    Parameters
    ----------
    image : np.ndarray
        Input BGR color image.
    etf_r, etf_iter : int
        ETF computation parameters.
    fdog_iter, sigma_m, sigma_c, rho, tau : int/float
        Parameters for the line extraction branch (FDoG).
    fbl_iter, sigma_e, r_e, sigma_g, r_g : int/float
        Parameters for the region smoothing branch (FBL).
    quant_bins : int
        Number of bins for luminance quantization.

    Returns
    -------
    np.ndarray
        Combined abstracted image (Cartoon-style).
    """
    validate_image(image)
    if len(image.shape) != 3:
        raise ValueError("Image must be a color BGR image for full abstraction.")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 1. Shared Edge Tangent Flow (ETF) construction
    etf = compute_etf(gray, r=etf_r, iterations=etf_iter)
    
    # 2. Branch A: Line Extraction (FDoG)
    current_gray = gray.copy()
    line_map = None
    for i in range(fdog_iter):
        if i > 0:
            current_gray = cv.GaussianBlur(current_gray, (3, 3), 0)
        line_map = apply_fdog(current_gray, etf, sigma_m, sigma_c, rho, tau)
        if i < fdog_iter - 1:
            current_gray = np.where(line_map == 0, 0, gray)
            
    # 3. Branch B: Region Smoothing (FBL)
    smoothed_color = image.astype(np.float32)
    for _ in range(fbl_iter):
        # Alternate C_e (Flow) and C_g (Gradient) passes
        smoothed_color = _fbl_flow_pass(smoothed_color, etf, sigma_e, r_e)
        smoothed_color = _fbl_grad_pass(smoothed_color, etf, sigma_g, r_g)
        
    smoothed_color = np.clip(smoothed_color, 0, 255).astype(np.uint8)
    
    # 4. Stylization: Luminance Quantization
    flat_color = quantize_luminance(smoothed_color, bins=quant_bins)
    
    # 5. Merging Branches: Overlay black lines over flattened colors
    line_mask = (line_map == 0)
    final_abstraction = flat_color.copy()
    final_abstraction[line_mask] = [0, 0, 0] # Set line pixels to black
    
    return final_abstraction

def main():
    import matplotlib.pyplot as plt
    from pixelops.io import imread
    from pixelops.visualization import show_image

    img = imread("./data/img/media_NO.jpg", mode="rgb")
    out = flow_based_abstraction(img, etf_r=5, etf_iter=3, fdog_iter=2, fbl_iter=3, quant_bins=8)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    show_image(ax[0], img, title="Original")
    show_image(ax[1], out, title="Flow-Based Abstraction")
    plt.show()

if __name__ == "__main__":
    main()