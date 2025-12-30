import time
import timeit
import numpy as np
import cv2 as cv
import cProfile

def MedianFilterGrayscale(image, window_size):
    """
    Applies a median filter to a grayscale image using efficient NumPy operations.

    Parameters:
    - image: Grayscale image (numpy array).
    - window_size: Size of the square window to compute the median (must be odd).

    Returns:
    - Filtered image with the median filter applied.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    pad = window_size // 2
    # Pad the image to handle borders
    padded = np.pad(image, pad, mode='edge')
    height, width = image.shape
    filtered = np.zeros_like(image, dtype=np.uint8)

    # Use a sliding window and compute the median for each region
    for y in range(height):
        for x in range(width):
            window = padded[y:y+window_size, x:x+window_size]
            filtered[y, x] = np.median(window)
    return filtered


from pixelops.filtering import (
    median_filter_grayscale
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

img = generate_test_image((1024, 512))

# ============================
# 1. BENCHMARK CON TIMEIT
# ============================
def run_timeit():
    n_runs = 10
    window_size = 9

    # warm-up
    median_filter_grayscale(img, window_size=3)
    cv.medianBlur(img, window_size)

    # Pasar kernel en globals
    globals_dict = {
        'img': img,
        "window_size": window_size,
        'median_filter_grayscale': median_filter_grayscale,
        'MedianFilterGrayscale': MedianFilterGrayscale,
        'cv': cv,
    }

    t_custom = timeit.timeit(
        stmt="median_filter_grayscale(img, window_size=window_size)",
        globals=globals_dict,
        number=n_runs
    )

    t_opencv = timeit.timeit(
        stmt="cv.medianBlur(img, window_size)",
        globals=globals_dict,
        number=n_runs
    )

    t_naive = timeit.timeit(
        stmt='MedianFilterGrayscale(img, window_size=window_size)',
        globals=globals_dict,
        number=n_runs
    )

    print("=== STEADY STATE BENCHMARK ===")
    print(f"Window size            : {window_size}")
    print(f"Image shape            : {img.shape}")
    print(f"Runs                   : {n_runs}")
    print(f" Custom Median average : {t_custom / n_runs:.6f} s")
    print(f" OpenCV Median average : {t_opencv / n_runs:.6f} s")
    print(f" Naive Median average  : {t_naive / n_runs:.6f} s")
    print()
    print(f"Speedup Custom vs Naive  : {t_naive / t_custom:.2f}x")
    print(f"Speedup Custom vs OpenCV : {t_opencv / t_custom:.2f}x")
    print()


# ============================
# 2. PROFILING CON cProfile
# ============================
def run_cprofile():
    
    print("=== cProfile: median_filter_grayscale ===")
    import io
    import pstats
    
    window_size = 9
    
    # Warm-up para compilar JIT (si usa numba)
    _ = median_filter_grayscale(img, window_size=window_size)
    
    profiler = cProfile.Profile()
    profiler.enable()
    median_filter_grayscale(img, window_size=window_size)
    profiler.disable()
    
    # Guardar en archivo binario
    profiler.dump_stats("profile_median_output.prof")
    
    # Mostrar en terminal (Top 100 funciones)
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(100)
    
    with open("profile_median_output.txt", "w") as f:
        f.write(s.getvalue())
    
    print("Resultados guardados en:")
    print("  - profile_median_output.prof (binario)")
    print("  - profile_median_output.txt (texto legible)")
    print()


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    run_timeit()
    run_cprofile()