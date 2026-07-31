"""
Implementation based on the "Coherent Line Drawing" algorithm.

Academic Reference:
    Kang, H., Lee, S., & Chui, C. K. (2007). 
    "Coherent Line Drawing". 
    In Proceedings of the 5th international symposium on Non-photorealistic 
    animation and rendering (NPAR '07). ACM.
    
Description:
    This module implements the workflow described in the original paper,
    including the construction of the Edge Tangent Flow (ETF) and the 
    Flow-based Difference of Gaussians (FDoG) filtering to extract 
    continuous lines and preserve edges.
"""
import numpy as np
import cv2 as cv
from pixelops.core import validate_image
from pixelops.filtering.spatial.etf import compute_etf
from pixelops.filtering.spatial.fdog import apply_fdog

def coherent_line_drawing(
    image: np.ndarray,
    etf_r: int = 3,
    fdog_iter: int = 2,
    sigma_m: float = 5.9,
    sigma_c: float = 0.6,
    rho: float = 0.99,
    tau: float = 0.9
) -> np.ndarray:
    """
    Extract coherent lines from an image using ETF and FDoG filtering.

    Parameters
    ----------
    image : np.ndarray
        Input BGR or grayscale image.
    etf_r : int, optional
        Radius for ETF calculation.
    fdog_iter : int, optional
        Number of feedback iterations for FDoG.
    sigma_m : float, optional
        FDoG longitudinal smoothing parameter.
    sigma_c : float, optional
        FDoG transversal center parameter.
    rho : float, optional
        FDoG noise control parameter.
    tau : float, optional
        Threshold for line binarization.

    Returns
    -------
    np.ndarray
        Line drawing representation of the input image.
    """
    validate_image(image)
    
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    etf = compute_etf(gray, r=etf_r, iterations=3)
    current_img = gray.copy()
    
    line_map = None
    for i in range(fdog_iter):
        if i > 0:
            current_img = cv.GaussianBlur(current_img, (3, 3), 0)
            
        line_map = apply_fdog(current_img, etf, sigma_m, sigma_c, rho, tau)
        
        if i < fdog_iter - 1:
            current_img = np.where(line_map == 0, 0, gray)
            
    return line_map