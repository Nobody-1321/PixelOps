import numpy as np
from ..utils import convolve_separable
from ..kernels import create_gaussian_kernel
import cv2

def gaussian_filter_core(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Core Gaussian filter (float-only).
    No normalization, no clipping, no dtype conversion.
    """

    gauss_kernel = create_gaussian_kernel(sigma)
    return convolve_separable(img, gauss_kernel, gauss_kernel)

def gaussian_filter_grayscale(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply a Gaussian smoothing filter to a grayscale image using
    a separable convolution.

    The image is internally normalized to the range [0, 1],
    filtered using a 1D Gaussian kernel in both axes, and then
    rescaled back to uint8.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W) and dtype uint8.
        Pixel values are expected to be in the range [0, 255].

    sigma : float
        Standard deviation of the Gaussian kernel. Must be positive.
        Larger values produce stronger smoothing.

    Returns
    -------
    np.ndarray
        Smoothed grayscale image of shape (H, W) and dtype uint8.

    Notes
    -----
    - The Gaussian filter is applied using separable convolution
      for computational efficiency.
    - The functions `create_gaussian_kernel` and
      `convolve_separable_opt` are assumed to implement a
      normalized 1D Gaussian kernel and an optimized separable
      convolution, respectively.
    """

    img_f = img.astype(np.float32) / 255.0

    gauss_kernel = create_gaussian_kernel(sigma)
    img_smoothed = convolve_separable(img_f, gauss_kernel, gauss_kernel)

    return np.clip(img_smoothed * 255, 0, 255).astype(np.uint8)

def gaussian_filter_bgr(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply a Gaussian smoothing filter to a BGR image using
    separable convolution applied independently to each channel.

    The image is internally normalized to the range [0, 1],
    filtered channel-wise using a 1D Gaussian kernel, and then
    rescaled back to uint8.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image of shape (H, W, 3) and dtype uint8.
        Channel order is assumed to be BGR.
        Pixel values are expected to be in the range [0, 255].

    sigma : float
        Standard deviation of the Gaussian kernel. Must be positive.
        Larger values produce stronger smoothing.

    Returns
    -------
    np.ndarray
        Smoothed BGR image of shape (H, W, 3) and dtype uint8.

    Notes
    -----
    - Each color channel is filtered independently.
    - No color space conversion is performed.
    - The convolution is separable for improved performance.
    """

    img_f = img.astype(np.float32) / 255.0
    gauss_kernel = create_gaussian_kernel(sigma)

    out = np.empty_like(img_f)

    for c in range(3):
        out[:, :, c] = convolve_separable(
            img_f[:, :, c],
            gauss_kernel,
            gauss_kernel
        )

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def gaussian_filter_lab_luminance(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to the luminance channel only while preserving chrominance.
    
    This function converts the input image to LAB color space, applies Gaussian 
    filtering to the L (luminance) channel, and converts back to BGR while keeping 
    the A and B (chrominance) channels unchanged. This approach preserves color 
    information while smoothing only the brightness component.
    
    Parameters
    ----------
    img : np.ndarray
        Input image in BGR color space.
    sigma : float
        Standard deviation for the Gaussian kernel. If sigma <= 0, returns a copy 
        of the original image unchanged.
    
    Returns
    -------
    np.ndarray
        Filtered image in BGR color space with Gaussian blur applied only to the 
        luminance channel.
    
    Notes
    -----
    - OpenCV's LAB color space uses L in range [0, 255]
    - Requires helper functions: create_gaussian_kernel() and convolve_separable()
    """

    if sigma <= 0:
        return img.copy()

    # Convert to LAB (OpenCV uses L in [0,255])
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    L, A, B = cv2.split(lab)

    # Normalize luminance to [0, 1]
    L /= 255.0

    # Gaussian blur on luminance only
    gauss_kernel = create_gaussian_kernel(sigma)
    L_blur = convolve_separable(L, gauss_kernel, gauss_kernel)

    # Restore range
    L_blur = np.clip(L_blur * 255.0, 0, 255)

    # Merge back without modifying chroma
    lab_blur = cv2.merge([
        L_blur.astype(np.uint8),
        A.astype(np.uint8),
        B.astype(np.uint8)
    ])

    return cv2.cvtColor(lab_blur, cv2.COLOR_LAB2BGR)