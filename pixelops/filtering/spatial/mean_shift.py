import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def mean_shift_filter_core(
    image_f: np.ndarray,
    hs: int,
    hr: float,
    max_iter: int,
    eps: float
) -> np.ndarray:
    """
    Core mean shift filter for grayscale images.

    Applies mean shift filtering using spatial and range kernels.
    This is the Numba-optimized core function.

    Parameters
    ----------
    image_f : np.ndarray
        Input grayscale image of shape (H, W) as float32, 
        values in range [0, 255].

    hs : int
        Spatial bandwidth (kernel radius in pixels).

    hr : float
        Range bandwidth (intensity similarity threshold).

    max_iter : int
        Maximum number of iterations per pixel.

    eps : float
        Convergence threshold for shift magnitude.

    Returns
    -------
    np.ndarray
        Filtered image of shape (H, W) as float32.
    """
    H, W = image_f.shape
    output = np.empty_like(image_f)

    # Pre-compute spatial kernel
    kernel_size = 2 * hs + 1
    spatial_kernel = np.empty((kernel_size, kernel_size), dtype=np.float32)
    hs_sq_2 = 2.0 * hs * hs
    for i in range(kernel_size):
        for j in range(kernel_size):
            dy = i - hs
            dx = j - hs
            spatial_kernel[i, j] = np.exp(-(dx * dx + dy * dy) / hs_sq_2)

    hr_sq_2 = 2.0 * hr * hr

    for i in prange(H):
        for j in range(W):
            xc = float(j)
            yc = float(i)
            vc = image_f[i, j]

            for _ in range(max_iter):
                x_min = int(max(xc - hs, 0))
                x_max = int(min(xc + hs + 1, W))
                y_min = int(max(yc - hs, 0))
                y_max = int(min(yc + hs + 1, H))

                # Spatial kernel slice indices
                sk_y0 = int(hs - (yc - y_min))
                sk_y1 = sk_y0 + (y_max - y_min)
                sk_x0 = int(hs - (xc - x_min))
                sk_x1 = sk_x0 + (x_max - x_min)

                total_weight = 0.0
                mean_x = 0.0
                mean_y = 0.0
                mean_v = 0.0

                for wi in range(y_max - y_min):
                    for wj in range(x_max - x_min):
                        py = y_min + wi
                        px = x_min + wj
                        val = image_f[py, px]
                        dv = val - vc
                        
                        sk_val = spatial_kernel[sk_y0 + wi, sk_x0 + wj]
                        w = sk_val * np.exp(-(dv * dv) / hr_sq_2)

                        total_weight += w
                        mean_x += w * px
                        mean_y += w * py
                        mean_v += w * val

                if total_weight < 1e-5:
                    break

                mean_x /= total_weight
                mean_y /= total_weight
                mean_v /= total_weight

                shift = np.sqrt(
                    (mean_x - xc) ** 2 + 
                    (mean_y - yc) ** 2 + 
                    (mean_v - vc) ** 2
                )
                xc, yc, vc = mean_x, mean_y, mean_v

                if shift < eps:
                    break

            output[i, j] = vc

    return output

def mean_shift_filter(
    image: np.ndarray,
    hs: int,
    hr: float,
    max_iter: int = 5,
    eps: float = 1.0
) -> np.ndarray:
    """
    Apply mean shift filtering to a grayscale or multi-channel image.

    Mean shift is an edge-preserving filter that iteratively shifts
    each pixel toward the local mode of the joint spatial-range
    distribution.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, C).
        Any numeric dtype is accepted; internally converted to float32.

    hs : int
        Spatial bandwidth (kernel radius in pixels).
        Must be positive.

    hr : float
        Range bandwidth (intensity similarity threshold).
        Must be positive.

    max_iter : int, optional
        Maximum number of mean shift iterations per pixel.
        Must be positive. Default is 5.

    eps : float, optional
        Convergence threshold. Iteration stops when the shift
        magnitude is below this value. Default is 1.0.

    Returns
    -------
    np.ndarray
        Mean-shift filtered image (float32) with the same shape
        as the input.

    Notes
    -----
    - No normalization, clipping, or quantization is applied.
    - Each channel is processed independently for multi-channel inputs.
    - Suitable for advanced image processing pipelines.
    """

    if hs <= 0:
        raise ValueError("Spatial bandwidth (hs) must be positive.")

    if hr <= 0:
        raise ValueError("Range bandwidth (hr) must be positive.")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if eps <= 0:
        raise ValueError("eps must be positive.")

    img_f = image.astype(np.float32)

    # Grayscale
    if img_f.ndim == 2:
        out = mean_shift_filter_core(
            img_f, hs, hr, max_iter, eps
        )

    # Multi-channel
    elif img_f.ndim == 3:
        out = np.empty_like(img_f)
        for c in range(img_f.shape[2]):
            out[:, :, c] = mean_shift_filter_core(
                img_f[:, :, c],
                hs, hr, max_iter, eps
            )
    else:
        raise ValueError("Input image must be 2D or 3D.")

    return out

'''
def mean_shift_filter_grayscale(
    image: np.ndarray,
    hs: int,
    hr: float,
    max_iter: int = 5,
    eps: float = 1.0
) -> np.ndarray:
    """
    Apply mean shift filtering to a grayscale image.

    This function acts as a high-level wrapper that validates
    input and delegates the core computation to a Numba-optimized
    backend.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image of shape (H, W) and dtype uint8.
        Pixel values are expected to be in the range [0, 255].

    hs : int
        Spatial bandwidth (kernel radius in pixels).
        Controls spatial smoothing extent.

    hr : float
        Range bandwidth (intensity similarity threshold).
        Controls how similar intensities must be to be averaged.

    max_iter : int, optional
        Maximum number of mean shift iterations per pixel. Default is 5.

    eps : float, optional
        Convergence threshold. Iteration stops when shift magnitude
        is below this value. Default is 1.0.

    Returns
    -------
    np.ndarray
        Filtered grayscale image of shape (H, W) and dtype uint8.

    Raises
    ------
    ValueError
        If `image` is not a 2D array.
        If `hs` is not positive.
        If `hr` is not positive.

    Notes
    -----
    - Mean shift is an edge-preserving filter similar to bilateral filtering.
    - Larger `hs` increases spatial smoothing.
    - Larger `hr` allows more intensity variation to be averaged.
    - The algorithm converges when pixel shifts become smaller than `eps`.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale (2D array).")
    
    if hs <= 0:
        raise ValueError("Spatial bandwidth (hs) must be positive.")
    
    if hr <= 0:
        raise ValueError("Range bandwidth (hr) must be positive.")

    image_f = image.astype(np.float32)
    
    output = mean_shift_filter_core(image_f, hs, hr, max_iter, eps)
    
    return np.clip(output, 0, 255).astype(np.uint8)

def mean_shift_filter_bgr(
    image: np.ndarray,
    hs: int,
    hr: float,
    max_iter: int = 5,
    eps: float = 1.0
) -> np.ndarray:
    """
    Apply mean shift filtering to a BGR image.

    This function processes each channel independently using
    the grayscale mean shift filter.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image of shape (H, W, 3) and dtype uint8.
        Pixel values are expected to be in the range [0, 255].

    hs : int
        Spatial bandwidth (kernel radius in pixels).
        Controls spatial smoothing extent.

    hr : float
        Range bandwidth (intensity similarity threshold).
        Controls how similar intensities must be to be averaged.

    max_iter : int, optional
        Maximum number of mean shift iterations per pixel. Default is 5.

    eps : float, optional
        Convergence threshold. Iteration stops when shift magnitude
        is below this value. Default is 1.0.
    Returns
    -------
    np.ndarray
        Filtered BGR image of shape (H, W, 3) and dtype uint8.
    Raises
    ------
    ValueError
        If `image` is not a 3D array with 3 channels.
    Notes
    -----
    - Each channel is processed independently using the same parameters.                

    - No color space conversion or channel mixing is performed.
    """ 

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be BGR (3D array with 3 channels).")

    out = np.empty_like(image, dtype=np.uint8)

    for c in range(3):
        out[:, :, c] = mean_shift_filter_grayscale(
            image[:, :, c], hs, hr, max_iter, eps
        )

    return out
'''