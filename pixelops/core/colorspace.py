"""
Color space conversion utilities.

This module provides RGB to Lab and Lab to RGB conversions
using OpenCV as backend. All functions follow BGR convention
compatible with OpenCV.
"""

import cv2 as cv
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to CIE Lab color space.

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image of shape (H, W, 3).
        Any numeric dtype is accepted.

    Returns
    -------
    np.ndarray
        Lab image as float32 in OpenCV Lab ranges:
        - L: [0, 100]
        - a, b: [-128, 127] (stored as [0, 255] with offset)

    Notes
    -----
    - Input is normalized to [0, 1] before conversion.
    - Uses D65 illuminant and sRGB color space.
    """
    rgb = rgb.astype(np.float32) / 255.0
    return cv.cvtColor(rgb, cv.COLOR_RGB2Lab)

def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert a CIE Lab image to RGB color space.

    Parameters
    ----------
    lab : np.ndarray
        Lab image as float32 in OpenCV Lab ranges.

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3) and dtype uint8.
        Values are clipped to [0, 255].

    Notes
    -----
    - Output is scaled to [0, 255] and clipped.
    - Uses D65 illuminant and sRGB color space.
    """
    rgb = cv.cvtColor(lab, cv.COLOR_Lab2RGB)
    rgb = np.clip(rgb * 255.0, 0, 255)
    return rgb.astype(np.uint8)