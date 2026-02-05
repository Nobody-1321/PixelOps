import time
import timeit
import numpy as np
import cv2 as cv
import cProfile
from numba import njit

# ============================
# VERSIÓN NO OPTIMIZADA (ORIGINAL)
# ============================
@njit
def MeanShiftFilterGrayscale_old(image, hs, hr, max_iter=5, eps=1.0):
    """
    Original non-optimized mean shift filter.
    Uses window slicing and intermediate arrays.
    """
    h, w = image.shape
    image_f = image.astype(np.float32)
    output = np.zeros_like(image_f)

    kernel_size = 2 * hs + 1
    spatial_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            dy = i - hs
            dx = j - hs
            spatial_kernel[i, j] = np.exp(-(dx**2 + dy**2) / (2 * hs**2))

    for i in range(h):
        for j in range(w):
            xc, yc = j, i
            vc = image_f[i, j]
            for _ in range(max_iter):
                x_min = int(max(xc - hs, 0))
                x_max = int(min(xc + hs + 1, w))
                y_min = int(max(yc - hs, 0))
                y_max = int(min(yc + hs + 1, h))
                window = image_f[y_min:y_max, x_min:x_max]

                sk_y0 = int(hs - (yc - y_min))
                sk_y1 = sk_y0 + (y_max - y_min)
                sk_x0 = int(hs - (xc - x_min))
                sk_x1 = sk_x0 + (x_max - x_min)
                sk = spatial_kernel[sk_y0:sk_y1, sk_x0:sk_x1]

                weights = np.empty_like(window)
                for wi in range(window.shape[0]):
                    for wj in range(window.shape[1]):
                        dv = window[wi, wj] - vc
                        weights[wi, wj] = sk[wi, wj] * np.exp(- (dv * dv) / (2 * hr * hr))

                total_weight = np.sum(weights)
                if total_weight < 1e-5:
                    break

                mean_x = 0.0
                mean_y = 0.0
                mean_v = 0.0
                for wi in range(window.shape[0]):
                    for wj in range(window.shape[1]):
                        px = x_min + wj
                        py = y_min + wi
                        w_ = weights[wi, wj]
                        mean_x += w_ * px
                        mean_y += w_ * py
                        mean_v += w_ * window[wi, wj]
                mean_x /= total_weight
                mean_y /= total_weight
                mean_v /= total_weight

                shift = np.sqrt((mean_x - xc)**2 + (mean_y - yc)**2 + (mean_v - vc)**2)
                xc, yc, vc = mean_x, mean_y, mean_v

                if shift < eps:
                    break

            output[i, j] = vc

    return np.clip(output, 0, 255).astype(np.uint8)


from pixelops.filtering import mean_shift_filter_grayscale

# ============================
# GENERAR IMAGEN DE PRUEBA
# ============================
def generate_test_image(shape=(256, 256)):
    """
    Generate a synthetic grayscale uint8 image for benchmarking.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)

img = generate_test_image((1024, 1024))

# ============================
# 1. BENCHMARK CON TIMEIT
# ============================
def run_timeit():
    n_runs = 5
    hs = 5
    hr = 30.0
    max_iter = 5
    eps = 1.0

    print("=== TIMEIT BENCHMARK: Mean Shift Filter ===")
    print(f"Image shape  : {img.shape}")
    print(f"hs (spatial) : {hs}")
    print(f"hr (range)   : {hr}")
    print(f"max_iter     : {max_iter}")
    print(f"eps          : {eps}")
    print()
    
    # warm-up (JIT compilation)
    print("Warming up JIT compilation...")
    _ = mean_shift_filter_grayscale(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)
    _ = MeanShiftFilterGrayscale_old(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)
    _ = cv.pyrMeanShiftFiltering(cv.cvtColor(img, cv.COLOR_GRAY2BGR), sp=hs, sr=hr)
    print("Warm-up complete.\n")

    # Pasar variables en globals
    globals_dict = {
        'img': img,
        'hs': hs,
        'hr': hr,
        'max_iter': max_iter,
        'eps': eps,
        'mean_shift_filter_grayscale': mean_shift_filter_grayscale,
        'MeanShiftFilterGrayscale_old': MeanShiftFilterGrayscale_old,
        'cv': cv,
    }

    # Optimized version
    t_optimized = timeit.timeit(
        stmt="mean_shift_filter_grayscale(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)",
        globals=globals_dict,
        number=n_runs
    )

    # Old version
    t_old = timeit.timeit(
        stmt="MeanShiftFilterGrayscale_old(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)",
        globals=globals_dict,
        number=n_runs
    )

    # OpenCV (note: pyrMeanShiftFiltering works on BGR, so we convert)
    t_opencv = timeit.timeit(
        stmt="cv.pyrMeanShiftFiltering(cv.cvtColor(img, cv.COLOR_GRAY2BGR), sp=hs, sr=hr)",
        globals=globals_dict,
        number=n_runs
    )

    print("=== RESULTS ===")
    print(f"Runs                      : {n_runs}")
    print(f" Optimized (parallel)     : {t_optimized / n_runs:.6f} s")
    print(f" Old (sequential)         : {t_old / n_runs:.6f} s")
    print(f" OpenCV pyrMeanShift      : {t_opencv / n_runs:.6f} s")
    print()
    print(f"Speedup Optimized vs Old    : {t_old / t_optimized:.2f}x")
    print(f"Speedup OpenCV vs Optimized : {t_optimized / t_opencv:.2f}x")
    print()


# ============================
# 2. BENCHMARK MANUAL (más detallado)
# ============================
def run_manual_timing():
    print("=== MANUAL TIMING: Mean Shift Filter ===")
    
    hs = 5
    hr = 30.0
    max_iter = 5
    eps = 1.0
    n_runs = 5

    # Warm-up
    _ = mean_shift_filter_grayscale(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)
    _ = MeanShiftFilterGrayscale_old(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)

    # Optimized version
    times_opt = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = mean_shift_filter_grayscale(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)
        end = time.perf_counter()
        times_opt.append(end - start)

    # Old version
    times_old = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = MeanShiftFilterGrayscale_old(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)
        end = time.perf_counter()
        times_old.append(end - start)

    times_opt = np.array(times_opt) * 1000  # ms
    times_old = np.array(times_old) * 1000  # ms

    print(f"=== Optimized (parallel) ===")
    print(f"  Mean : {times_opt.mean():.3f} ms")
    print(f"  Std  : {times_opt.std():.3f} ms")
    print(f"  Min  : {times_opt.min():.3f} ms")
    print(f"  Max  : {times_opt.max():.3f} ms")
    print()
    print(f"=== Old (sequential) ===")
    print(f"  Mean : {times_old.mean():.3f} ms")
    print(f"  Std  : {times_old.std():.3f} ms")
    print(f"  Min  : {times_old.min():.3f} ms")
    print(f"  Max  : {times_old.max():.3f} ms")
    print()
    print(f"Speedup: {times_old.mean() / times_opt.mean():.2f}x")
    print()


# ============================
# 3. VERIFY OUTPUT CORRECTNESS
# ============================
def verify_output():
    print("=== VERIFICATION: Output Comparison ===")
    
    hs = 5
    hr = 30.0
    max_iter = 5
    eps = 1.0

    out_opt = mean_shift_filter_grayscale(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)
    out_old = MeanShiftFilterGrayscale_old(img, hs=hs, hr=hr, max_iter=max_iter, eps=eps)

    diff = np.abs(out_opt.astype(np.float32) - out_old.astype(np.float32))
    
    print(f"Max absolute difference : {diff.max():.2f}")
    print(f"Mean absolute difference: {diff.mean():.4f}")
    print(f"Pixels with diff > 1    : {np.sum(diff > 1)} / {diff.size}")
    
    if diff.max() < 2:
        print("✓ Outputs are equivalent (small numerical differences expected)")
    else:
        print("✗ Warning: Significant differences detected")
    print()


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    verify_output()
    run_timeit()
    run_manual_timing()
