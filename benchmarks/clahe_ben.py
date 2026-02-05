import timeit
import numpy as np
import cv2 as cv
import cProfile

#temp

def CalHistogram(channel):
    
    """
    Compute the histogram of one-channel image.

    Parameters:
        channel (numpy.ndarray): The input channel image.

    Returns:
        numpy.ndarray: The computed histogram with 256 bins.
    """
    # Compute the histogram

    hist, bins = np.histogram(channel.flatten(), bins=256, range=[0, 256])
    
    return hist

def ClipHistogram(hist, clip_limit):
    """
    Clips the histogram by limiting each bin to the clip_limit and redistributes
    the clipped amount uniformly among all the bins.

    Parameters:
        hist (numpy.ndarray): The histogram to be clipped.
        clip_limit (int): The maximum allowed value for each histogram bin.
    
    Returns:
        numpy.ndarray: The clipped histogram with redistributed excess.
    """
    excess = hist - clip_limit
    excess[excess < 0] = 0
    total_excess = np.sum(excess)
    
    # Clip the histogram
    hist = np.minimum(hist, clip_limit)
    
    # Redistribute the excess uniformly
    redist = total_excess // 256
    hist = hist + redist
    
    # Distribute the remainder sequentially
    remainder = total_excess % 256
    for i in range(256):
        if remainder <= 0:
            break
        hist[i] += 1
        remainder -= 1
        
    return hist

def CreateMapping(hist, block_size):
    """
    Calculates the mapping function (lookup table) from the clipped histogram
    using the cumulative distribution function (CDF).

    Parameters:
        hist (numpy.ndarray): The clipped histogram.
        block_size (int): The total number of pixels in the block.

    Returns:
        numpy.ndarray: The mapping function that maps pixel intensities to new values.
    """
    cdf = np.cumsum(hist)
    # Avoid division by zero: find the first non-zero value
    cdf_min = cdf[np.nonzero(cdf)][0]
    
    # Normalize the CDF to [0, 255]
    mapping = np.round((cdf - cdf_min) / float(block_size - cdf_min) * 255).astype(np.uint8)
    return mapping

def ComputeMappings(image, n_rows, n_cols, cell_h, cell_w, clip_limit):
    """Computes histogram equalization mappings for each block."""
    mappings = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for i in range(n_rows):
        for j in range(n_cols):
            r0, r1 = i * cell_h, min((i + 1) * cell_h, image.shape[0])
            c0, c1 = j * cell_w, min((j + 1) * cell_w, image.shape[1])

            block = image[r0:r1, c0:c1]
            hist = CalHistogram(block)
            hist_clipped = ClipHistogram(hist, clip_limit)
            mappings[i][j] = CreateMapping(hist_clipped, block.size)

    return mappings

def HistogramEqualizationClaheGrayscale(image, clip_limit=10, grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image 
    or just a single channel.

    Parameters:
        image: uint8 numpy array of shape (height, width).
        clip_limit: maximum allowed value for each histogram bin.
        grid_size: tuple (n_rows, n_cols) indicating how many blocks the image is divided into.

    Returns:
        equalized_image: uint8 numpy array of the resulting image with enhanced contrast.
    """
    height, width = image.shape
    n_rows, n_cols = grid_size
    cell_h, cell_w = height // n_rows, width // n_cols

    # Compute histogram mappings for each block
    mappings = ComputeMappings(image, n_rows, n_cols, cell_h, cell_w, clip_limit)

    # Apply bilinear interpolation to construct the output image
    return ApplyInterpolation(image, mappings, n_rows, n_cols, cell_h, cell_w)

def ApplyInterpolation(image, mappings, n_rows, n_cols, cell_h, cell_w):
    """Applies bilinear interpolation to map pixel intensities using CLAHE mappings."""
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.uint8)

    for y in range(height):
        i0, i1, y_weight = InterpolationIndices(y, cell_h, n_rows)
        
        for x in range(width):
            j0, j1, x_weight = InterpolationIndices(x, cell_w, n_cols)
            
            intensity = image[y, x]
            val_tl, val_tr = mappings[i0][j0][intensity], mappings[i0][j1][intensity]
            val_bl, val_br = mappings[i1][j0][intensity], mappings[i1][j1][intensity]

            top = val_tl * (1 - x_weight) + val_tr * x_weight
            bottom = val_bl * (1 - x_weight) + val_br * x_weight
            output[y, x] = int(np.round(top * (1 - y_weight) + bottom * y_weight))

    return output

def InterpolationIndices(coord, cell_size, max_index):
    """Computes interpolation indices and weights for bilinear interpolation."""
    f = (coord + 0.5) / cell_size - 0.5
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, max_index - 1)
    weight = f - i0 if 0 <= i0 < max_index - 1 else 0
    return max(0, i0), i1, weight

# ============================
# IMPORTA TU FUNCIÓN
# ============================
from pixelops.histogram import clahe_grayscale

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
# CLAHE OPENCV (baseline)
# ============================
clahe_cv = cv.createCLAHE(
    clipLimit=10,
    tileGridSize=(8, 8)
)

def clahe_opencv(img):
    return clahe_cv.apply(img)

# ============================
# 1. BENCHMARK CON TIMEIT
# ============================
def run_timeit():
    n_runs = 10

    # warm-up
    clahe_grayscale(img)
    clahe_opencv(img)

    t_custom = timeit.timeit(
        stmt="clahe_grayscale(img)",
        globals=globals(),
        number=n_runs
    )
    # histogram equalization with CLAHE normal HistogramEqualizationClaheGrayscale
    t_clahe = timeit.timeit(
        stmt="HistogramEqualizationClaheGrayscale(img, clip_limit=10, grid_size=(8, 8))",
        globals=globals(),
        number=n_runs
    )
    
    t_opencv = timeit.timeit(
        stmt="clahe_opencv(img)",
        globals=globals(),
        number=n_runs
    )

    print("=== STEADY STATE BENCHMARK ===")
    print(f"Runs                   : {n_runs}")
    print(f"Numba CLAHE average   : {t_custom / n_runs:.6f} s")
    print(f"OpenCV CLAHE average   : {t_opencv / n_runs:.6f} s")
    print(f"Normal CLAHE average   : {t_clahe / n_runs:.6f} s")
    print(f"Speedup (Opencv / normal) : {t_clahe / t_opencv:.2f}x")
    print(f"Speedup (OpenCV / numba ) : {t_custom / t_opencv:.2f}x")

# ============================
# 2. PROFILING CON cProfile
# ============================
def run_cprofile():
    print("=== cProfile: Custom CLAHE ===")
    cProfile.run("clahe_grayscale(img)")
    print()

    print("=== cProfile: OpenCV CLAHE ===")
    cProfile.run("clahe_opencv(img)")
    print()

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    run_timeit()
    run_cprofile()
