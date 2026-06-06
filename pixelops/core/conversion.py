"""
Image conversion utilities.
"""

import numpy as np
from .types import ImageArray, ImageFloat, ImageUInt8

def to_float32(image: np.ndarray) -> ImageArray:
    """
    Convert an image to float32 in the range [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    ImageArray
        Float32 normalized image.

    Raises
    ------
    TypeError
        If the image dtype is unsupported.
    """

    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0

    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0

    elif image.dtype == np.float32:
        return np.clip(image, 0.0, 1.0)

    elif image.dtype == np.float64:
        return np.clip(image.astype(np.float32), 0.0, 1.0)

    else:
        raise TypeError(
            f"Unsupported image dtype: {image.dtype}"
        )
    
def rescale_to_uint8(arr: np.ndarray) -> ImageUInt8:
    """
    Rescale an array to uint8 using min-max normalization.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.

    Returns
    -------
    ImageUInt8
        Rescaled uint8 array.

    Notes
    -----
    This operation modifies image contrast by stretching
    the intensity range to [0,255].
    """

    arr_min = arr.min()
    arr_max = arr.max()

    arr_range = arr_max - arr_min

    if arr_range == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    arr_norm = (arr - arr_min) / arr_range

    return (arr_norm * 255).astype(np.uint8)

def to_uint8(image: ImageFloat) -> ImageUInt8:
    """
    Convert a float image in [0,1] to uint8.
    """

    image = np.clip(image, 0.0, 1.0)

    return (image * 255.0).astype(np.uint8)