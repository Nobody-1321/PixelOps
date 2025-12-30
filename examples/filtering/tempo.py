"""
Interactive Gaussian Blur Brush Tool

Allows painting variable blur strength onto an image using a brush.
Uses precomputed blur levels and smooth interpolation for real-time performance.
"""

import cv2
import numpy as np
from numba import njit, prange
from pixelops.filtering import gaussian_filter_bgr, gaussian_filter_lab_luminance

cv2.setUseOptimized(True)


# ======================================================
# Configuration
# ======================================================

class Config:
    """Configuration parameters for the blur brush tool."""
    
    #IMG_PATH = "./data/img/cerezo.png"
    IMG_PATH = "./data/img/mujerIA.webp"
    WINDOW_NAME = "Blur Brush"
    WINDOW_SIZE = (1200, 900)
    
    # Blur levels to precompute
    SIGMA_LEVELS = np.array(
        [0, 1, 2, 3, 4, 5, 6,7,8],
        #[0, 2, 4, 6, 8, 12, 16, 24, 32, 40],
        dtype=np.float32
    )
    MAX_SIGMA = float(SIGMA_LEVELS[-1])
    
    # Precompute inverse intervals for fast interpolation
    INV_INTERVALS = 1.0 / np.diff(SIGMA_LEVELS)
    
    # Brush defaults
    DEFAULT_RADIUS = 40
    DEFAULT_STRENGTH = 0.5
    MAX_RADIUS = 150
    MAX_STRENGTH = 3.0


# ======================================================
# Image Loading and Preprocessing
# ======================================================

def load_and_prepare_image(path: str) -> tuple[np.ndarray, int, int]:
    """
    Load image and convert to float32.
    
    Returns
    -------
    tuple[np.ndarray, int, int]
        Image array, height, width
    """
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Could not load image: {path}")
    
    img = img.astype(np.float32)
    H, W = img.shape[:2]
    
    return img, H, W


def precompute_blur_stack(
    img: np.ndarray,
    sigma_levels: np.ndarray
) -> np.ndarray:
    """
    Precompute Gaussian blurs for all sigma levels using LAB color space.
    
    Applies Gaussian blur to the luminance channel only (L in LAB),
    preserving chrominance information for perceptually better results.
    
    Parameters
    ----------
    img : np.ndarray
        Base image in float32 format, BGR color space
    sigma_levels : np.ndarray
        Array of sigma values to precompute
        
    Returns
    -------
    np.ndarray
        Stack of blurred images with shape (n_levels, H, W, 3) in float32
    """
    H, W = img.shape[:2]
    n_levels = len(sigma_levels)
    
    # Convert to uint8 for gaussian_filter_lab_luminance
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    
    stack = np.ascontiguousarray(
        np.zeros((n_levels, H, W, 3), dtype=np.float32)
    )
    
    for i, sigma in enumerate(sigma_levels):
        if sigma == 0:
            stack[i] = img
        else:
            # Apply LAB luminance blur (returns uint8)
            blurred = gaussian_filter_lab_luminance(img_uint8, sigma=sigma)
            # Convert back to float32
            stack[i] = blurred.astype(np.float32)
    
    # Optional: Display blur levels for debugging
    for i in range(n_levels):
        display = np.clip(stack[i], 0, 255).astype(np.uint8)
        cv2.imshow(f"Blur Level Sigma={sigma_levels[i]}", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return stack

'''
def precompute_blur_stack(
    img: np.ndarray,
    sigma_levels: np.ndarray
) -> np.ndarray:
    """
    Precompute Gaussian blurs for all sigma levels.
    
    Parameters
    ----------
    img : np.ndarray
        Base image in float32 format
    sigma_levels : np.ndarray
        Array of sigma values to precompute
        
    Returns
    -------
    np.ndarray
        Stack of blurred images with shape (n_levels, H, W, 3)
    """
    H, W = img.shape[:2]
    n_levels = len(sigma_levels)
    
    stack = np.ascontiguousarray(
        np.zeros((n_levels, H, W, 3), dtype=np.float32)
    )
    
    for i, sigma in enumerate(sigma_levels):
        if sigma == 0:
            stack[i] = img
        else:
            stack[i] = gaussian_filter_bgr(img, sigma=sigma)

    for i in range(n_levels):
        img = np.clip(stack[i], 0, 255).astype(np.uint8)
        cv2.imshow(f"Blur Level Sigma={Config.SIGMA_LEVELS[i]}", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    return stack
'''



# ======================================================
# Rendering Engine
# ======================================================

@njit(parallel=True, fastmath=True, cache=True)
def render_blurred_image(
    img_out: np.ndarray,
    blur_map: np.ndarray,
    sigma_levels: np.ndarray,
    inv_intervals: np.ndarray,
    blur_stack: np.ndarray
) -> None:
    """
    Render final image by interpolating between precomputed blur levels.
    
    Uses smooth linear interpolation between adjacent blur levels based
    on the blur_map values. Parallelized with Numba for performance.
    
    Parameters
    ----------
    img_out : np.ndarray
        Output image buffer (H, W, 3), modified in-place
    blur_map : np.ndarray
        Per-pixel blur strength map (H, W)
    sigma_levels : np.ndarray
        Sorted array of precomputed sigma values
    inv_intervals : np.ndarray
        Precomputed 1/(sigma[i+1] - sigma[i]) for fast interpolation
    blur_stack : np.ndarray
        Precomputed blur levels (n_levels, H, W, 3)
    """
    H, W = blur_map.shape
    n = len(sigma_levels)

    for y in prange(H):
        for x in range(W):
            s = blur_map[y, x]

            # Clamp to min sigma
            if s <= sigma_levels[0]:
                img_out[y, x] = blur_stack[0, y, x]
                continue

            # Clamp to max sigma
            if s >= sigma_levels[n - 1]:
                img_out[y, x] = blur_stack[n - 1, y, x]
                continue

            # Linear interpolation between levels
            for i in range(n - 1):
                s0 = sigma_levels[i]
                s1 = sigma_levels[i + 1]
                if s0 <= s <= s1:
                    t = (s - s0) * inv_intervals[i]
                    img_out[y, x] = (
                        blur_stack[i, y, x] * (1.0 - t) +
                        blur_stack[i + 1, y, x] * t
                    )
                    break


# ======================================================
# Brush Tool
# ======================================================

class BrushState:
    """Mutable state for the brush tool."""
    
    def __init__(self, H: int, W: int):
        self.blur_map = np.zeros((H, W), dtype=np.float32)
        self.drawing = False
        self.radius = Config.DEFAULT_RADIUS
        self.strength = Config.DEFAULT_STRENGTH


def apply_brush_stroke(
    blur_map: np.ndarray,
    x: int,
    y: int,
    radius: int,
    strength: float,
    max_sigma: float,
    H: int,
    W: int
) -> None:
    """
    Apply a single brush stroke with Gaussian falloff.
    
    Parameters
    ----------
    blur_map : np.ndarray
        Current blur map (H, W), modified in-place
    x, y : int
        Brush center coordinates
    radius : int
        Brush radius in pixels
    strength : float
        Blur strength to add per stroke
    max_sigma : float
        Maximum allowed blur value
    H, W : int
        Image dimensions
    """
    y0, y1 = max(0, y - radius), min(H, y + radius)
    x0, x1 = max(0, x - radius), min(W, x + radius)

    if y1 <= y0 or x1 <= x0:
        return

    # Create distance grid
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dy = yy - y
    dx = xx - x
    d2 = dx * dx + dy * dy
    
    # Circular mask
    mask = d2 <= radius * radius
    if not np.any(mask):
        return

    # Gaussian falloff
    sigma_brush = radius * 0.5
    falloff = np.exp(-d2 / (2.0 * sigma_brush * sigma_brush))

    # Apply with clamping
    local = blur_map[y0:y1, x0:x1]
    local[mask] = np.minimum(
        local[mask] + falloff[mask] * strength,
        max_sigma
    )


def create_mouse_callback(state: BrushState, H: int, W: int):
    """Factory function to create mouse callback with enclosed state."""
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            
        elif event == cv2.EVENT_LBUTTONUP:
            state.drawing = False
            
        elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
            apply_brush_stroke(
                state.blur_map,
                x, y,
                state.radius,
                state.strength,
                Config.MAX_SIGMA,
                H, W
            )
    
    return mouse_callback


# ======================================================
# UI Setup
# ======================================================

def setup_window(state: BrushState) -> None:
    """Create and configure the main window with trackbars."""
    
    cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Config.WINDOW_NAME, *Config.WINDOW_SIZE)
    
    def nothing(x):
        pass
    
    cv2.createTrackbar(
        "Radius",
        Config.WINDOW_NAME,
        Config.DEFAULT_RADIUS,
        Config.MAX_RADIUS,
        nothing
    )
    
    cv2.createTrackbar(
        "Strength x100",
        Config.WINDOW_NAME,
        int(Config.DEFAULT_STRENGTH * 100),
        int(Config.MAX_STRENGTH * 100),
        nothing
    )


def update_brush_parameters(state: BrushState) -> None:
    """Read trackbar values and update brush state."""
    
    state.radius = max(
        1,
        cv2.getTrackbarPos("Radius", Config.WINDOW_NAME)
    )
    state.strength = (
        cv2.getTrackbarPos("Strength x100", Config.WINDOW_NAME) / 100.0
    )


# ======================================================
# Main Application
# ======================================================

def main():
    """Run the interactive blur brush application."""
    
    # Load image
    img_base, H, W = load_and_prepare_image(Config.IMG_PATH)
    
    # Precompute blur stack
    print("Precomputing blur levels...")
    blur_stack = precompute_blur_stack(img_base, Config.SIGMA_LEVELS)
    print("Done!")
    
    # Initialize state
    state = BrushState(H, W)
    img_result = img_base.copy()
    
    # Setup UI
    setup_window(state)
    mouse_callback = create_mouse_callback(state, H, W)
    cv2.setMouseCallback(Config.WINDOW_NAME, mouse_callback)
    
    # Main loop
    print("\nControls:")
    print("  - Left mouse drag: Paint blur")
    print("  - 'r': Reset blur map")
    print("  - ESC: Exit")
    
    while True:
        # Update brush parameters from trackbars
        update_brush_parameters(state)
        
        # Render image with current blur map
        render_blurred_image(
            img_result,
            state.blur_map,
            Config.SIGMA_LEVELS,
            Config.INV_INTERVALS,
            blur_stack
        )
        
        # Display
        display = np.clip(img_result, 0, 255).astype(np.uint8)
        cv2.imshow(Config.WINDOW_NAME, display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset
            state.blur_map[:] = 0.0
            print("Blur map reset")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
