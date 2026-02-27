"""
Reinhard Color Transfer utilities.

This module provides:
- Robust BGR ↔ CIE Lab (real-valued) conversions
- Classic Reinhard color transfer (Reinhard et al., 2001)
- A controlled variant with partial blending and stability controls

All functions operate on BGR images in uint8 format, following OpenCV
conventions.
"""

import cv2
import numpy as np

#============================================================
#Internal validation utilities
#============================================================


def _validate_bgr_image(img: np.ndarray, name: str = "image") -> None:
    """
    Validate that an image is a BGR uint8 image with shape (H, W, 3).

    Parameters
    ----------
    img : np.ndarray
        Input image.
    name : str
        Name used in error messages.

    Raises
    ------
    TypeError
        If img is not a numpy array.
    ValueError
        If shape or dtype are invalid.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3)")

    if img.dtype != np.uint8:
        raise ValueError(f"{name} must be of dtype uint8")

def _validate_lab_real_image(lab: np.ndarray, name: str = "lab") -> None:
    """
    Validate a real-valued Lab image.

    Parameters
    ----------
    lab : np.ndarray
        Lab image.
    name : str
        Name used in error messages.

    Raises
    ------
    TypeError
        If lab is not a numpy array.
    ValueError
        If shape or dtype are invalid.
    """
    if not isinstance(lab, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")

    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3)")

    if lab.dtype not in (np.float32, np.float64):
        raise ValueError(f"{name} must be float32 or float64")

# ============================================================
# Color space conversions
# ============================================================

def bgr_to_lab_real(bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 image to real-valued CIE Lab.

    Parameters
    ----------
    bgr : np.ndarray
        BGR image in uint8 format with shape (H, W, 3).

    Returns
    -------
    lab : np.ndarray
        Real-valued Lab image in float32:
        - L* in [0, 100]
        - a* in [-128, 127]
        - b* in [-128, 127]

    Notes
    -----
    OpenCV represents Lab as uint8 with:
    - L in [0, 255]
    - a, b in [0, 255] with an offset of 128

    This function converts OpenCV Lab to the standard CIE Lab ranges.
    """
    _validate_bgr_image(bgr, "bgr")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0

    return lab

def lab_real_to_bgr(lab: np.ndarray) -> np.ndarray:
    """
    Convert a real-valued CIE Lab image to BGR uint8.
    """
    _validate_lab_real_image(lab, "lab")

    lab_cv = lab.astype(np.float32, copy=True)

    # Clip a* y b* ANTES de sumar el offset
    lab_cv[..., 0] = np.clip(lab_cv[..., 0], 0.0, 100.0)
    lab_cv[..., 1] = np.clip(lab_cv[..., 1], -128.0, 127.0)
    lab_cv[..., 2] = np.clip(lab_cv[..., 2], -128.0, 127.0)

    lab_cv[..., 0] *= 255.0 / 100.0
    lab_cv[..., 1] += 128.0
    lab_cv[..., 2] += 128.0

    # Ahora sí, clip final para uint8
    lab_cv = np.clip(lab_cv, 0, 255).astype(np.uint8)

    return cv2.cvtColor(lab_cv, cv2.COLOR_LAB2BGR)

# ============================================================
# Reinhard Color Transfer
# ============================================================

def reinhard_color_transfer(
    src_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Perform classic Reinhard color transfer.

    The method matches the mean and standard deviation of the source
    image to those of the target image in CIE Lab space.

    Parameters
    ----------
    src_bgr : np.ndarray
        Source image in BGR uint8 format.
    tgt_bgr : np.ndarray
        Target image in BGR uint8 format.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    out_bgr : np.ndarray
        Color-transferred image in BGR uint8 format.

    References
    ----------
    Reinhard et al., "Color Transfer between Images", IEEE CG&A, 2001.
    """
    _validate_bgr_image(src_bgr, "src_bgr")
    _validate_bgr_image(tgt_bgr, "tgt_bgr")

    src_lab = bgr_to_lab_real(src_bgr)
    tgt_lab = bgr_to_lab_real(tgt_bgr)

    out_lab = np.empty_like(src_lab, dtype=np.float32)

    for c in range(3):  # L*, a*, b*
        mu_s = np.mean(src_lab[..., c])
        mu_t = np.mean(tgt_lab[..., c])

        std_s = np.std(src_lab[..., c]) + eps
        std_t = np.std(tgt_lab[..., c]) + eps

        out_lab[..., c] = (src_lab[..., c] - mu_s) * (std_t / std_s) + mu_t

    out_lab[..., 0] = np.clip(out_lab[..., 0], 0.0, 100.0)
    out_lab[..., 1:] = np.clip(out_lab[..., 1:], -128.0, 127.0)

    return lab_real_to_bgr(out_lab)

# ============================================================
# Controlled Reinhard Color Transfer
# ============================================================

def reinhard_color_transfer_controlled(
    src_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    alpha_L: float = 0.3,
    alpha_ab: float = 0.6,
    clip_ratio: float = 3.0,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Perform a controlled variant of Reinhard color transfer.

    This version extends the classic method by:
    - Partial blending between original and transferred colors
    - Independent control for luminance (L*) and chrominance (a*, b*)
    - Clipping of standard deviation ratios to avoid extreme amplification

    Parameters
    ----------
    src_bgr : np.ndarray
        Source image in BGR uint8 format.
    tgt_bgr : np.ndarray
        Target image in BGR uint8 format.
    alpha_L : float, optional
        Blending factor for L* channel (0 = identity, 1 = full transfer).
    alpha_ab : float, optional
        Blending factor for a* and b* channels.
    clip_ratio : float, optional
        Maximum allowed ratio between target and source standard deviations.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    out_bgr : np.ndarray
        Color-transferred image in BGR uint8 format.
    """
    _validate_bgr_image(src_bgr, "src_bgr")
    _validate_bgr_image(tgt_bgr, "tgt_bgr")

    src_lab = bgr_to_lab_real(src_bgr)
    tgt_lab = bgr_to_lab_real(tgt_bgr)

    out_lab = np.empty_like(src_lab, dtype=np.float32)

    for c in range(3):  # L*, a*, b*
        mu_s = np.mean(src_lab[..., c])
        mu_t = np.mean(tgt_lab[..., c])

        std_s = np.std(src_lab[..., c]) + eps
        std_t = np.std(tgt_lab[..., c]) + eps

        ratio = std_t / std_s
        ratio = np.clip(ratio, 1.0 / clip_ratio, clip_ratio)

        transformed = (src_lab[..., c] - mu_s) * ratio + mu_t

        alpha = alpha_L if c == 0 else alpha_ab
        out_lab[..., c] = (1.0 - alpha) * src_lab[..., c] + alpha * transformed

    out_lab[..., 0] = np.clip(out_lab[..., 0], 0.0, 100.0)
    out_lab[..., 1:] = np.clip(out_lab[..., 1:], -127.0, 127.0)

    return lab_real_to_bgr(out_lab)
