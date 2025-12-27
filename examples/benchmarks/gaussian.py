import time
import timeit
import numpy as np
import cv2 as cv
import cProfile
from scipy.ndimage import convolve1d

def convolve_separable_scipy(
    img: np.ndarray,
    kernel_h: np.ndarray,
    kernel_v: np.ndarray
) -> np.ndarray:
    """
    Apply a separable 2D convolution to a grayscale image using
    1D horizontal and vertical kernels.

    This function performs convolution by first filtering along
    the horizontal axis and then along the vertical axis, using
    `scipy.ndimage.convolve1d` for efficiency.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image of shape (H, W).
        Typically of type float32 or float64.

    kernel_h : np.ndarray
        1D convolution kernel applied along the horizontal axis
        (columns).

    kernel_v : np.ndarray
        1D convolution kernel applied along the vertical axis
        (rows).

    Returns
    -------
    np.ndarray
        Image of shape (H, W) after separable convolution.
        The output dtype follows SciPy's convolution rules.

    Notes
    -----
    - Boundary handling is performed using `mode='reflect'`,
      which mirrors the image at the borders.
    - Separable convolution reduces computational complexity
      from O(N²) to O(2N).
    """
    
    tmp = convolve1d(img, kernel_h, axis=1, mode="reflect")
    return convolve1d(tmp, kernel_v, axis=0, mode="reflect")

from pixelops.filtering import (
    convolve_separable,
    create_gaussian_kernel,
    gaussian_filter_grayscale
)

# ============================
# GENERAR IMAGEN DE PRUEBA
# ============================
def generate_test_image(shape=(512, 512)):
    """
    Generate a synthetic grayscale uint8 image for benchmarking.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)

img = generate_test_image((512, 512))

# ============================
# 1. BENCHMARK CON TIMEIT
# ============================
def run_timeit():
    n_runs = 1

    # warm-up
    kernel = create_gaussian_kernel(sigma=2.5)
    convolve_separable_scipy(img, kernel, kernel)
    convolve_separable(img, kernel, kernel)

    # Pasar kernel en globals
    globals_dict = {
        'img': img,
        'kernel': kernel,
        'convolve_separable': convolve_separable,
        'convolve_separable_scipy': convolve_separable_scipy
    }

    t_custom = timeit.timeit(
        stmt="convolve_separable(img, kernel, kernel)",
        globals=globals_dict,
        number=n_runs
    )
    
    t_scipy = timeit.timeit(
        stmt="convolve_separable_scipy(img, kernel, kernel)",
        globals=globals_dict,
        number=n_runs
    )

    print("=== STEADY STATE BENCHMARK ===")
    print(f"Runs                   : {n_runs}")
    print(f" Numba Convolve average   : {t_custom / n_runs:.6f} s")
    print(f" SciPy Convolve average   : {t_scipy / n_runs:.6f} s")
    print()


# ============================
# 2. PROFILING CON cProfile
# ============================
def run_cprofile():
    
    print("=== cProfile: Custom gaussian_filter_grayscale ===")
    import io
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    gaussian_filter_grayscale(img, sigma=2.5)
    profiler.disable()
    
    # Guardar en archivo
    profiler.dump_stats("profile_output.prof")
    
    # Mostrar en terminal (Top 10 funciones)
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(100)
    
    with open("profile_output.txt", "w") as f:
        f.write(s.getvalue())
    
    print("Resultados guardados en:")
    print("  - profile_output.prof (binario)")
    print("  - profile_output.txt (texto legible)")
    print()

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    run_timeit()
    run_cprofile()