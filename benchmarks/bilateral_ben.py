import numpy as np
from pixelops.filtering.spatial.bilateral import (
    bilateral_filter_grayscale,
    bilateral_filter_core,
    binomial_coeffs
)

from pixelops.filtering.kernels import create_gaussian_kernel_radius
import timeit
import cProfile
import cv2 as cv 

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
    n_runs = 50
    
    print("=== TIMEIT BENCHMARK ===")
    # warm-up
    bilateral_filter_grayscale(img, 1.2, 0.7, 8, 5)
    cv.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)

    # Pasar kernel en globals
    globals_dict = {
        'img': img,
        'bilateral_filter_grayscale': bilateral_filter_grayscale,
        'cv': cv,
    }

    t_custom = timeit.timeit(
        stmt="bilateral_filter_grayscale(img, 1.2, 0.7, 8, 5)",
        globals=globals_dict,
        number=n_runs
    )

    t_cv = timeit.timeit(
        stmt="cv.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)",
        globals=globals_dict,
        number=n_runs
    )

    print("=== STEADY STATE BENCHMARK ===")
    print(f"Runs                       : {n_runs}")
    print(f" Custom bilateral          : {t_custom / n_runs:.6f} s")
    print(f" OpenCV bilateral          : {t_cv / n_runs:.6f} s")
    print (f"Speedup (Custom vs OpenCV): {t_custom / t_cv:.2f}x")
    print()

def run_timeit_core():
    """Benchmark solo del core"""
    n_runs = 50
    ss, sr, n_iter, n = 1.2, 0.7, 8, 5
    
    img_f = img.astype(np.float32) / 255.0
    binom = binomial_coeffs(n, img_f.dtype)
    gauss_kernel = create_gaussian_kernel_radius(ss)
    
    # Warm-up
    _ = bilateral_filter_core(img_f.copy(), binom, gauss_kernel, n_iter, sr)
    
    globals_dict = {
        'img_f': img_f,
        'binom': binom,
        'gauss_kernel': gauss_kernel,
        'bilateral_filter_core': bilateral_filter_core,
        'np': np,
    }
    
    t = timeit.timeit(
        stmt="bilateral_filter_core(img_f.copy(), binom, gauss_kernel, 8, 0.7)",
        globals=globals_dict,
        number=n_runs
    )
    
    print(f"=== bilateral_filter_core ===")
    print(f"Tiempo promedio      : {t / n_runs:.3f} ms")
    print(f"Total ({n_runs} runs): {t:.3f} s")


# ============================
# 2. PROFILING CON cProfile
# ============================
def run_cprofile():
    
    print("=== cProfile: bilateral_filter_core ===")
    import io
    import pstats
    
    # Preparar datos para bilateral_filter_core
    ss = 1.2
    sr = 0.7
    n_iter = 8
    n = 5
    
    img_f = img.astype(np.float32) / 255.0
    binom = binomial_coeffs(n, img_f.dtype)
    gauss_kernel = create_gaussian_kernel_radius(ss)
    
    # Warm-up para compilar JIT
    _ = bilateral_filter_core(img_f.copy(), binom, gauss_kernel, n_iter, sr)
    
    # Profiling solo de la función core
    img_copy = img_f.copy()
    profiler = cProfile.Profile()
    profiler.enable()
    bilateral_filter_core(img_copy, binom, gauss_kernel, n_iter, sr)
    profiler.disable()
    
    # Mostrar en terminal (Top 100 funciones)
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(100)
    
    with open("profile_bilateral_output.txt", "w") as f:
        f.write(s.getvalue())
    
    print("Resultados guardados en:")
    print("  - profile_bilateral_output.txt (texto legible)")
    print()

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    run_timeit()
    run_timeit_core()
    #run_cprofile()