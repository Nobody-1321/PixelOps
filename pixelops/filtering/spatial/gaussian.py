import numpy as np
from ..utils import convolve_separable
from ..kernels import create_gaussian_kernel

def gaussian_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    img_f = img.astype(np.float32)

    kernel = create_gaussian_kernel(sigma)

    if img_f.ndim == 2:
        out = convolve_separable(img_f, kernel, kernel)

    elif img_f.ndim == 3:
        out = np.empty_like(img_f)
        for c in range(img_f.shape[2]):
            out[:, :, c] = convolve_separable(
                img_f[:, :, c], kernel, kernel
            )
    else:
        raise ValueError("Invalid image dimensions.")
    
    return out