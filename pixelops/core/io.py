"""
Image I/O utilities.

This module provides functions for loading and processing images
from disk with common preprocessing operations.
"""

import cv2 as cv
import numpy as np


def open_image(path: str, mode: str = "bgr") -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str
        Path to the image file.

    mode : {"bgr", "gray"}, optional
        Color mode of the loaded image. Default is "bgr".

    Returns
    -------
    np.ndarray
        Loaded image as uint8. Shape is (H, W, 3) for "bgr"
        or (H, W) for "gray".

    Raises
    ------
    ValueError
        If mode is not "bgr" or "gray".
    IOError
        If the image cannot be loaded.

    Notes
    -----
    - Uses OpenCV's imread internally.
    - Supports common formats: PNG, JPG, BMP, TIFF, etc.
    """
    if mode == "bgr":
        flag = cv.IMREAD_COLOR
    elif mode == "gray":
        flag = cv.IMREAD_GRAYSCALE
    else:
        raise ValueError("mode must be 'bgr' or 'gray'.")

    img = cv.imread(path, flag)
    if img is None:
        raise IOError(f"Could not read image: {path}")

    return img

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an array to uint8 range [0, 255].

    Performs min-max normalization followed by scaling to
    the full uint8 range.

    Parameters
    ----------
    arr : np.ndarray
        Input array of any numeric dtype and shape.

    Returns
    -------
    np.ndarray
        Normalized array of same shape with dtype uint8.
        Values span [0, 255] unless input is uniform.

    Notes
    -----
    - If all values are equal, returns zeros.
    - Useful for visualization of float arrays.
    """
    arr_min = arr.min()
    arr_max = arr.max()
    arr_range = arr_max - arr_min
    
    if arr_range == 0:
        # Avoid division by zero when the array has uniform values.
        # Return a zero-filled uint8 array with the same shape.
        return np.zeros_like(arr, dtype=np.uint8)
    
    arr_norm = (arr - arr_min) / arr_range
    arr_scaled = arr_norm * 255
    return arr_scaled.astype(np.uint8)