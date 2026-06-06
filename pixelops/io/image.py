"""
Image I/O utilities.
"""

import cv2 as cv
import numpy as np


def imread(path: str, mode: str = "rgb") -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str
        Path to the image file.

    mode : {"rgb", "gray"}, optional
        Color mode of the loaded image.

    Returns
    -------
    np.ndarray
        Loaded image as uint8.

        - RGB images have shape (H, W, 3)
        - Grayscale images have shape (H, W)

    Raises
    ------
    ValueError
        If mode is invalid.

    IOError
        If the image cannot be loaded.

    Notes
    -----
    Internally uses OpenCV but always returns RGB images.
    """

    if mode == "rgb":

        img = cv.imread(path, cv.IMREAD_COLOR)

        if img is None:
            raise IOError(
                f"Could not read image: {path}"
            )

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return img

    elif mode == "gray":

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if img is None:
            raise IOError(
                f"Could not read image: {path}"
            )

        return img

    else:
        raise ValueError(
            "mode must be 'rgb' or 'gray'."
    )