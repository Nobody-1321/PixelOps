import numpy as np
import cv2 as cv
from .utils import cal_histogram_numba, clip_histogram_numba, cal_histogram, clip_histogram
from numba import njit

#----------------------------------------------
#        HISTOGRAM EQUALIZATION
#----------------------------------------------

def histogram_equalization_channel(channel):
    """
    Applies histogram equalization to a single image channel.

    Parameters
    ----------
    channel : np.ndarray
        Single-channel image (H×W), dtype uint8.

    Returns
    -------
    np.ndarray
        Equalized channel with the same shape and dtype as the input.

    Raises
    ------
    ValueError
        If the input is not a 2D uint8 array.
    """
    if not isinstance(channel, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if channel.ndim != 2:
        raise ValueError("Expected a single-channel image (H×W).")

    if channel.dtype != np.uint8:
        raise ValueError("Expected channel of type uint8.")

    hist, _ = np.histogram(
        channel.ravel(),
        bins=256,
        range=(0, 256)
    )

    cdf = hist.cumsum()

    # Avoid division by zero
    if cdf[-1] == 0:
        return channel.copy()

    #cdf_min = cdf[cdf > 0][0]  # First non-zero value in CDF
    cdf_min = cdf[cdf > 0].min()  # First non-zero value in CDF
    if cdf_min == cdf[-1]:
        return channel.copy()
    
    cdf_normalized = (cdf - cdf_min) / (cdf[-1] - cdf_min)
    lut = np.round(cdf_normalized * 255).astype(np.uint8)

    return lut[channel]

def histogram_equalization(image):
    """
    Applies histogram equalization to a grayscale or BGR image.

    For grayscale images, histogram equalization is applied directly.
    For BGR images, the image is converted to YCrCb color space, histogram
    equalization is applied to the Y (luminance) channel, and the image is
    converted back to BGR.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H×W) or BGR (H×W×3) image, dtype uint8.

    Returns
    -------
    np.ndarray
        Equalized image with the same shape and dtype as the input.

    Raises
    ------
    ValueError
        If the input image has an unsupported shape or type.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if image.ndim == 2:
        return histogram_equalization_channel(image)

    elif image.ndim == 3 and image.shape[2] == 3:
        ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb)

        y_eq = histogram_equalization_channel(y)

        ycrcb_eq = cv.merge([y_eq, cr, cb])
        return cv.cvtColor(ycrcb_eq, cv.COLOR_YCrCb2BGR)

    else:
        raise ValueError("Unsupported image shape. Expected (H, W) or (H, W, 3).")

#----------------------------------------------
#        CLAHE IMPLEMENTATION
#----------------------------------------------

@njit
def create_mapping_numba(hist, num_pixels):
    lut = np.empty(256, dtype=np.uint8)

    if num_pixels <= 0:
        for i in range(256):
            lut[i] = i
        return lut

    non_zero_bins = 0 
    single_value = -1
    for i in range(256):
        if hist[i] > 0:
            non_zero_bins += 1
            single_value = i
    
    if non_zero_bins == 1:
        for i in range(256):
            lut[i] = single_value
        return lut
    
    cdf = 0
    cdf_min = -1

    for i in range(256):
        cdf += hist[i]
        if cdf_min < 0 and cdf > 0:
            cdf_min = cdf

    if cdf_min < 0:
        for i in range(256):
            lut[i] = i
        return lut

    denom = num_pixels - cdf_min
    if denom <= 0:
        for i in range(256):
            lut[i] = i
        return lut

    scale = 255.0 / denom
    cdf = 0

    for i in range(256):
        cdf += hist[i]
        v = int((cdf - cdf_min) * scale + 0.5)
        if v < 0:
            v = 0
        elif v > 255:
            v = 255
        lut[i] = v

    return lut

def compute_block_mappings(
    image,
    n_rows,
    n_cols,
    cell_h,
    cell_w,
    clip_limit
):
    mappings = np.empty((n_rows, n_cols, 256), dtype=np.uint8)

    for i in range(n_rows):
        for j in range(n_cols):
            r0 = i * cell_h
            r1 = min((i + 1) * cell_h, image.shape[0])
            c0 = j * cell_w
            c1 = min((j + 1) * cell_w, image.shape[1])

            block = image[r0:r1, c0:c1]

            hist = cal_histogram_numba(block)
            hist = clip_histogram_numba(hist, clip_limit)
            mappings[i, j] = create_mapping_numba(hist, block.size)

    return mappings

@njit(cache=True, fastmath=True)
def apply_interpolation_numba(
    image,
    mappings,
    n_rows,
    n_cols,
    cell_h,
    cell_w
):
    h, w = image.shape
    out = np.empty_like(image)

    for y in range(h):
        fy = (y + 0.5) / cell_h - 0.5
        i0 = int(fy)

        if i0 < 0:
            i0 = 0
            i1 = 0
            wy = 0.0
        elif i0 >= n_rows - 1:
            i0 = n_rows - 1
            i1 = i0
            wy = 0.0
        else:
            i1 = i0 + 1
            wy = fy - i0

        for x in range(w):
            fx = (x + 0.5) / cell_w - 0.5
            j0 = int(fx)

            if j0 < 0:
                j0 = 0
                j1 = 0
                wx = 0.0
            elif j0 >= n_cols - 1:
                j0 = n_cols - 1
                j1 = j0
                wx = 0.0
            else:
                j1 = j0 + 1
                wx = fx - j0

            p = image[y, x]

            tl = mappings[i0, j0, p]
            tr = mappings[i0, j1, p]
            bl = mappings[i1, j0, p]
            br = mappings[i1, j1, p]

            top = tl * (1.0 - wx) + tr * wx
            bot = bl * (1.0 - wx) + br * wx

            out[y, x] = int(top * (1.0 - wy) + bot * wy + 0.5)

    return out

def clahe_core(
    image: np.ndarray,
    clip_limit: int,
    grid_size: tuple[int, int]
) -> np.ndarray:
    """
    Apply CLAHE to a grayscale uint8 image.

    Notes
    -----
    - Expects uint8 input in [0, 255]
    - Intended for contrast enhancement (visualization)
    """
    if image.ndim != 2:
        raise ValueError("Expected a grayscale image (H, W).")

    if image.dtype != np.uint8:
        raise TypeError("CLAHE expects uint8 image.")

    n_rows, n_cols = grid_size
    h, w = image.shape

    if h < n_rows or w < n_cols:
        raise ValueError("Grid size larger than image.")

    cell_h = h // n_rows
    cell_w = w // n_cols

    mappings = compute_block_mappings(
        image, n_rows, n_cols, cell_h, cell_w, clip_limit
    )

    return apply_interpolation_numba(
        image,
        np.array(mappings),
        n_rows,
        n_cols,
        cell_h,
        cell_w
    )

def clahe(
    image: np.ndarray,
    clip_limit: int = 10,
    grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE to a grayscale or BGR image.

    This function is intended for visualization and contrast enhancement.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or BGR (H, W, 3) uint8 image.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image (uint8).
    """

    if image.dtype != np.uint8:
        raise TypeError("CLAHE expects uint8 image.")

    if image.ndim == 2:
        return clahe_core(image, clip_limit, grid_size)

    elif image.ndim == 3 and image.shape[2] == 3:
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)

        l_eq = clahe_core(l, clip_limit, grid_size)

        lab_eq = cv.merge((l_eq, a, b))
        return cv.cvtColor(lab_eq, cv.COLOR_LAB2BGR)

    else:
        raise ValueError("Invalid image dimensions.")
