import numpy as np
import cv2 as cv
from ..histogram.utils import cal_histogram


def ridler_calvard_threshold(img, max_iterations=100, tolerance=1e-3):
    """
    Computes an automatic threshold using the Ridler-Calvard (ISODATA) method.

    The algorithm iteratively updates the threshold by computing the mean
    intensities of two groups separated by the current threshold until
    convergence.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (2D array).
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    tolerance : float, optional
        Convergence tolerance for threshold change (default: 1e-3).

    Returns
    -------
    float
        Estimated threshold value.
    """
    threshold_old = np.mean(img)

    for _ in range(max_iterations):
        group1 = img[img <= threshold_old]
        group2 = img[img > threshold_old]

        if group1.size == 0 or group2.size == 0:
            break

        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        threshold_new = 0.5 * (mean1 + mean2)

        if abs(threshold_new - threshold_old) < tolerance:
            break

        threshold_old = threshold_new

    return threshold_old

def otsu_threshold(img):
    """
    Computes the optimal global threshold using Otsu's method.

    The method maximizes the between-class variance of the histogram,
    assuming a bimodal intensity distribution.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (uint8).

    Returns
    -------
    int
        Optimal threshold value in the range [0, 255].
    """
    hist = cal_histogram(img)
    total = img.size

    weight_bg = 0.0
    weight_fg = np.sum(hist)

    mean_bg = 0.0
    mean_fg = np.sum(np.arange(256) * hist)

    max_variance = 0.0
    optimal_threshold = 0

    for t in range(1, 256):
        weight_bg += hist[t - 1]
        weight_fg -= hist[t - 1]

        if weight_bg == 0 or weight_fg == 0:
            continue

        mean_bg += (t - 1) * hist[t - 1]
        mean_fg -= (t - 1) * hist[t - 1]

        diff = (mean_bg / weight_bg) - (mean_fg / weight_fg)
        between_class_variance = weight_bg * weight_fg * diff ** 2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = t

    return optimal_threshold

def flood_fill(image, x, y, visited):
    """
    Flood-fill helper used for hysteresis thresholding.

    Propagates connectivity from strong edges to weak edges
    using 8-neighborhood connectivity.

    Parameters
    ----------
    image : np.ndarray
        Binary image containing weak edges (non-zero values).
    x, y : int
        Seed coordinates.
    visited : np.ndarray
        Boolean array indicating visited pixels.
    """
    height, width = image.shape
    stack = [(x, y)]

    while stack:
        px, py = stack.pop()

        if visited[py, px]:
            continue

        visited[py, px] = True
        image[py, px] = 255

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = px + dx, py + dy
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and not visited[ny, nx]
                    and image[ny, nx] != 0
                ):
                    stack.append((nx, ny))

def hysteresis_threshold(image, low_threshold, high_threshold):
    """
    Applies hysteresis thresholding to an image.

    Strong edges (>= high_threshold) are used as seeds to
    propagate connectivity through weak edges
    (between low_threshold and high_threshold).

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    low_threshold : float
        Lower threshold.
    high_threshold : float
        Upper threshold.

    Returns
    -------
    np.ndarray
        Binary image containing the final edges.
    """
    strong = (image >= high_threshold).astype(np.uint8) * 255
    weak = ((image >= low_threshold) & (image < high_threshold)).astype(np.uint8) * 255

    visited = np.zeros_like(image, dtype=bool)
    height, width = image.shape

    for y in range(height):
        for x in range(width):
            if strong[y, x] == 255 and not visited[y, x]:
                flood_fill(weak, x, y, visited)

    return weak

def remove_intensity_range(img, low, high, fill=0, inplace=False):
    """
    Removes (replaces) a given intensity range from an image.

    For grayscale images, pixels within [low, high] are replaced.
    For BGR images, the mask is computed on the grayscale conversion
    and applied to all three channels.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or BGR).
    low, high : int
        Inclusive intensity range to remove.
    fill : int, optional
        Replacement value (default: 0).
    inplace : bool, optional
        If True, modifies the input image; otherwise returns a copy.

    Returns
    -------
    out : np.ndarray
        Image with the intensity range removed.
    mask : np.ndarray
        Boolean mask of replaced pixels.
    """
    out = img if inplace else img.copy()

    low, high = int(low), int(high)
    if low > high:
        low, high = high, low

    if out.ndim == 2:
        mask = (out >= low) & (out <= high)
        out[mask] = fill
        return out, mask

    if out.ndim == 3 and out.shape[2] == 3:
        gray = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
        mask = (gray >= low) & (gray <= high)
        out[mask] = fill
        return out, mask

    raise ValueError("Input image must be grayscale (2D) or BGR (HxWx3).")
