"""
Validation utilities for images and parameters.
"""

import numpy as np


def validate_image(image: np.ndarray) -> None:
    """
    Validate a generic image.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Raises
    ------
    TypeError
        If input is not a numpy array.

    ValueError
        If image dimensions are invalid.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "image must be a numpy.ndarray."
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            "image must have shape (H, W) or (H, W, C)."
        )


def validate_grayscale(image: np.ndarray) -> None:
    """
    Validate a grayscale image.
    """

    validate_image(image)

    if image.ndim != 2:
        raise ValueError(
            "Expected grayscale image with shape (H, W)."
        )


def validate_rgb(image: np.ndarray) -> None:
    """
    Validate an RGB image.
    """

    validate_image(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected RGB image with shape (H, W, 3)."
        )