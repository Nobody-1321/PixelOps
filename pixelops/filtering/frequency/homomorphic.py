import numpy as np
from ..utils import _distance_matrix
from .fft import _fft_pass, _ifft_pass
from pixelops.core import (
    to_float32,
    validate_image
)

def homomorphic(image: np.ndarray, gammaL: float = 0.5, gammaH: float = 1.5, sigma: float = 30.0) -> np.ndarray:
    """
    Apply homomorphic filtering for illumination correction and contrast tuning.

    Processes illumination components in log-frequency transform space.

    Parameters
    ----------
    image : np.ndarray
        Input spatial image of shape (H, W) or (H, W, C).
    gammaL : float, optional
        Low-frequency gain (controls illumination compression). Default is 0.5.
    gammaH : float, optional
        High-frequency gain (controls reflectance sharpening). Default is 1.5.
    sigma : float, optional
        Cutoff metric parameter for the homomorphic curve width. Default is 30.0.

    Returns
    -------
    np.ndarray
        Filtered enhanced image (float32) normalized to the [0.0, 1.0] interval.
    """
    validate_image(image)
    img_f = to_float32(image)
    log_img = np.log1p(img_f)
    
    distance = _distance_matrix(img_f.shape[:2])
    homomorphic_mask = (gammaH - gammaL) * (1.0 - np.exp(-(distance ** 2) / (2.0 * (sigma ** 2)))) + gammaL

    if log_img.ndim == 2:
        dft = _fft_pass(log_img)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * homomorphic_mask
        spatial_filtered = _ifft_pass(np.fft.ifftshift(filtered_dft))
        out = np.expm1(spatial_filtered)
    else:
        out = np.empty_like(log_img)
        for c in range(log_img.shape[2]):
            dft = _fft_pass(log_img[:, :, c])
            dft_shift = np.fft.fftshift(dft)
            filtered_dft = dft_shift * homomorphic_mask
            spatial_filtered = _ifft_pass(np.fft.ifftshift(filtered_dft))
            out[:, :, c] = np.expm1(spatial_filtered)

    return np.clip(out, 0.0, 1.0)