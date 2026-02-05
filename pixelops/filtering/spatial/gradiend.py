import numpy as np
import cv2 as cv
from ..kernels import create_gaussian_derivative_kernel, create_gaussian_second_derivative_kernel
from ..utils import convolve_separable
from  .gaussian import create_gaussian_kernel

def gaussian_gradient_core(
    img: np.ndarray,
    sigma_s: float,
    sigma_d: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the image gradient using separable Gaussian smoothing
    and first-order Gaussian derivative filters.

    This function performs pure numerical computation and returns
    floating-point results without normalization.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).
        Expected dtype is uint8 or float.

    sigma_s : float
        Standard deviation of the Gaussian smoothing kernel.

    sigma_d : float
        Standard deviation of the Gaussian derivative kernel.

    Returns
    -------
    Gx : np.ndarray
        Gradient in the X direction (float32).

    Gy : np.ndarray
        Gradient in the Y direction (float32).

    Gmag : np.ndarray
        Gradient magnitude (float32).

    Gphase : np.ndarray
        Gradient phase in radians (float32).

    Notes
    -----
    - No normalization or clipping is performed.
    - Suitable for quantitative analysis and further processing.
    """

    img_f = img.astype(np.float32)

    gauss = create_gaussian_kernel(sigma_s)
    gauss_deriv = create_gaussian_derivative_kernel(sigma_d)

    Gx = convolve_separable(img_f, gauss_deriv, gauss)
    Gy = convolve_separable(img_f, gauss, gauss_deriv)

    Gmag = np.sqrt(Gx * Gx + Gy * Gy)
    Gphase = np.arctan2(Gy, Gx)

    return Gx, Gy, Gmag, Gphase

def gaussian_gradient(
    img: np.ndarray,
    sigma_s: float,
    sigma_d: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the image gradient using Gaussian derivatives and
    return normalized uint8 images for visualization.

    This function is intended for display purposes only.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).

    sigma_s : float
        Standard deviation of the Gaussian smoothing kernel.

    sigma_d : float
        Standard deviation of the Gaussian derivative kernel.

    Returns
    -------
    Gx : np.ndarray
        Normalized gradient in X direction (uint8).

    Gy : np.ndarray
        Normalized gradient in Y direction (uint8).

    Gmag : np.ndarray
        Normalized gradient magnitude (uint8).

    Gphase : np.ndarray
        Normalized gradient phase (uint8).

    Notes
    -----
    - Internally calls `compute_gaussian_image_gradient_float`.
    - Normalization is performed independently for each output.
    - Not suitable for numerical analysis.
    """

    Gx, Gy, Gmag, Gphase = gaussian_gradient_core(
        img, sigma_s, sigma_d
    )

    Gx = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)
    Gy = cv.normalize(Gy, None, 0, 255, cv.NORM_MINMAX)
    Gmag = cv.normalize(Gmag, None, 0, 255, cv.NORM_MINMAX)
    Gphase = cv.normalize(Gphase, None, 0, 255, cv.NORM_MINMAX)

    return (
        Gx.astype(np.uint8),
        Gy.astype(np.uint8),
        Gmag.astype(np.uint8),
        Gphase.astype(np.uint8),
    )

def sobel_gradient_core(
    img: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute image gradients using the Sobel operator.

    This function performs pure numerical computation and returns
    floating-point gradient components without normalization.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).
        Expected dtype is uint8 or float.

    Returns
    -------
    Gx : np.ndarray
        Gradient in the X direction (float32).

    Gy : np.ndarray
        Gradient in the Y direction (float32).

    Gmag : np.ndarray
        Gradient magnitude (float32).

    Gphase : np.ndarray
        Gradient phase in radians (float32).

    Notes
    -----
    - Sobel kernels are normalized by 1/8.
    - This operator approximates first-order derivatives
      using fixed discrete masks.
    """

    img_f = img.astype(np.float32)

    sobel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32
    ) * (1.0 / 8.0)

    sobel_y = np.array(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=np.float32
    ) * (1.0 / 8.0)

    Gx = cv.filter2D(img_f, cv.CV_32F, sobel_x)
    Gy = cv.filter2D(img_f, cv.CV_32F, sobel_y)

    Gmag = np.sqrt(Gx * Gx + Gy * Gy)
    Gphase = np.arctan2(Gy, Gx)

    return Gx, Gy, Gmag, Gphase

def sobel_gradient(
    img: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Sobel image gradients and return normalized
    uint8 images for visualization.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).

    Returns
    -------
    Gx : np.ndarray
        Normalized gradient in X direction (uint8).

    Gy : np.ndarray
        Normalized gradient in Y direction (uint8).

    Gmag : np.ndarray
        Normalized gradient magnitude (uint8).

    Gphase : np.ndarray
        Normalized gradient phase (uint8).
    """

    Gx, Gy, Gmag, Gphase = sobel_gradient_core(img)

    Gx = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)
    Gy = cv.normalize(Gy, None, 0, 255, cv.NORM_MINMAX)
    Gmag = cv.normalize(Gmag, None, 0, 255, cv.NORM_MINMAX)
    Gphase = cv.normalize(Gphase, None, 0, 255, cv.NORM_MINMAX)

    return (
        Gx.astype(np.uint8),
        Gy.astype(np.uint8),
        Gmag.astype(np.uint8),
        Gphase.astype(np.uint8),
    )

def log_gradient_core(
    img: np.ndarray,
    sigma_s: float,
    sigma_d: float
) -> np.ndarray:
    """
    Compute the Laplacian of Gaussian (LoG) using separable convolution.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).

    sigma_s : float
        Standard deviation of the Gaussian smoothing kernel.

    sigma_d : float
        Standard deviation of the second-order Gaussian derivative.

    Returns
    -------
    np.ndarray
        Laplacian of Gaussian response (float32).

    Notes
    -----
    - The LoG is computed as:
        LoG = d²G/dx² * I + d²G/dy² * I
    - No normalization or clipping is applied.
    """

    img_f = img.astype(np.float32)

    gauss = create_gaussian_kernel(sigma_s)
    gauss_2nd = create_gaussian_second_derivative_kernel(sigma_d)

    Gxx = convolve_separable(img_f, gauss_2nd, gauss)
    Gyy = convolve_separable(img_f, gauss, gauss_2nd)

    return Gxx + Gyy

def log_gradient(
    img: np.ndarray,
    sigma_s: float,
    sigma_d: float
) -> np.ndarray:
    """
    Compute the Laplacian of Gaussian and return a normalized
    uint8 image for visualization.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.

    sigma_s : float
        Standard deviation of the Gaussian smoothing kernel.

    sigma_d : float
        Standard deviation of the second derivative kernel.

    Returns
    -------
    np.ndarray
        Normalized LoG response (uint8).
    """

    log = log_gradient_core(img, sigma_s, sigma_d)
    log = cv.normalize(log, None, 0, 255, cv.NORM_MINMAX)

    return log.astype(np.uint8)
