import cv2 as cv
import numpy as np


def open_image(path, mode="bgr"):
    """
    Loads an image from disk.

    Parameters
    ----------
    path : str
        Path to the image file.
    mode : {"bgr", "gray"}
        Color mode of the loaded image.

    Returns
    -------
    np.ndarray
        Loaded image.

    Raises
    ------
    IOError
        If the image cannot be loaded.
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
