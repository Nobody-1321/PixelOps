import numpy as np
from ..utils import convolve_separable
from ..kernels import create_gaussian_kernel


def gaussian_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to an image using separable convolution.

    Performs spatial smoothing by convolving the image with a
    Gaussian kernel. The filter is applied separably for efficiency.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W) or (H, W, C).
        Any numeric dtype is accepted; internally converted to float32.

    sigma : float
        Standard deviation of the Gaussian kernel.
        Must be positive. Larger values produce stronger blur.

    Returns
    -------
    np.ndarray
        Blurred image with same shape as input and dtype float32.

    Raises
    ------
    ValueError
        If sigma <= 0 or image has invalid dimensions.

    Notes
    -----
    - Kernel size is automatically determined from sigma.
    - Uses reflection boundary handling.
    - Equivalent to OpenCV's GaussianBlur with BORDER_REFLECT.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    img_f = img.astype(np.float32)

    kernel = create_gaussian_kernel(sigma)

    if img_f.ndim == 2:
        out = convolve_separable(img_f, kernel, kernel)

    elif img_f.ndim == 3:
        out = np.empty_like(img_f)
        for c in range(img_f.shape[2]):
            out[:, :, c] = convolve_separable(
                img_f[:, :, c], kernel, kernel
            )
    else:
        raise ValueError("Invalid image dimensions.")
    
    return out