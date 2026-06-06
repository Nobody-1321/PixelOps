import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import importlib.util

# Configuration (no CLI args as preferred)
INPUT_PATH = "./data/img/marsNO.jpg"
OUTPUT_PATH = "img_data/temp/cartoon_line_combo_result.png"
K = 32
MEANSHIFT_SPATIAL = 15
MEANSHIFT_COLOR = 15
MEANSHIFT_MAX_ITER = 10
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Load the coherent line drawing module (safe: it uses if __name__ == '__main__')
LINE_DRAWING_PATH = os.path.join(os.path.dirname(__file__), "line_drawing.py")
_spec = importlib.util.spec_from_file_location("line_module", LINE_DRAWING_PATH)
line_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(line_module)

# Try to import the custom 'lip' module used by the cartoon script; fall back gracefully
try:
    import basura.tempo.src.lip as lip
    _HAS_LIP = True
except Exception:
    lip = None
    _HAS_LIP = False


def bilateral_filter(image, sigma_d=15, sigma_r=0.1):
    return cv2.bilateralFilter(image, d=-1,
                               sigmaColor=sigma_r * 255,
                               sigmaSpace=sigma_d)


def compute_detail_layer(img, sigma_d=30, sigma_r=0.4, eps=1e-3):
    base = bilateral_filter(img, sigma_d, sigma_r)
    detail = (img + eps) / (base + eps)
    return np.clip(detail, 0.8, 1.2)


def cartoonize_image(img_bgr):
    """Return a cartoonized version of the input BGR image.

    Steps (simplified from examples/wip/cartoonizingNO.py):
    - Mean-shift denoising (if `lip` available) or bilateral fallback
    - Color quantization in Lab + spatial coordinates using KMeans
    - Detail transfer using bilateral-based detail layer
    - Optional unsharp masking via `lip`, otherwise simple sharpening
    """
    img = img_bgr.copy()

    # 1) Denoise: try custom mean-shift if available
    if _HAS_LIP and hasattr(lip, 'MeanShiftFilterBGR'):
        img_denoised = lip.MeanShiftFilterBGR(img, MEANSHIFT_SPATIAL, MEANSHIFT_COLOR, MEANSHIFT_MAX_ITER)
    else:
        # fallback: apply a sequence of bilateral filters
        img_denoised = cv2.bilateralFilter(img, d=BILATERAL_D, sigmaColor=BILATERAL_SIGMA_COLOR, sigmaSpace=BILATERAL_SIGMA_SPACE)

    # small additional bilateral to smooth further
    img_denoised = cv2.bilateralFilter(img_denoised, d=9, sigmaColor=75, sigmaSpace=75)

    # 2) Quantize color in Lab + coordinates
    h, w = img_denoised.shape[:2]
    img_lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2Lab)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    features = np.concatenate((
        img_lab.reshape((-1, 3)).astype(np.float32),
        0.05 * np.stack((X.ravel(), Y.ravel()), axis=1).astype(np.float32)
    ), axis=1)

    kmeans = MiniBatchKMeans(n_clusters=K, init='k-means++', batch_size=10000, n_init=3)
    labels = kmeans.fit_predict(features)

    centers_lab = np.uint8(kmeans.cluster_centers_[:, :3])
    quantized = centers_lab[labels]
    img_lab_quant = quantized.reshape((h, w, 3))
    img_quant = cv2.cvtColor(img_lab_quant, cv2.COLOR_Lab2BGR)

    # 3) detail transfer
    img_float = img.astype(np.float32) / 255.0
    detail_layer = compute_detail_layer(img_float, sigma_d=10, sigma_r=1.0)
    alpha = 0.6
    detail_layer = np.power(detail_layer, alpha)

    cartoon = np.clip((img_quant.astype(np.float32) / 255.0) * detail_layer, 0, 1)
    cartoon = (cartoon * 255).astype(np.uint8)

    # 4) optional sharpening
    if _HAS_LIP and hasattr(lip, 'CreateUnsharpMaskingFilter') and hasattr(lip, 'ApplyFrequencyDomainFilterLabL'):
        h0, w0 = cartoon.shape[:2]
        um_kernel = lip.CreateUnsharpMaskingFilter((h0, w0), 80, alpha=0.9, method='butterworth')
        cartoon_sharp = lip.ApplyFrequencyDomainFilterLabL(cartoon, um_kernel)
    else:
        # simple unsharp mask fallback
        blurred = cv2.GaussianBlur(cartoon, (0, 0), 3)
        cartoon_sharp = cv2.addWeighted(cartoon, 1.3, blurred, -0.3, 0)

    return cartoon_sharp


def combine_cartoon_and_lines(img_bgr):
    # Cartoonize
    cartoon = cartoonize_image(img_bgr)

    # Compute coherent line drawing using the imported module
    # The line_module expects BGR images and returns a binary map (0 for line)
    line_map = line_module.coherent_line_drawing(img_bgr, etf_r=1, fdog_iter=1, sigma_m=3.0, sigma_c=0.7, tau=0.5)

    # Overlay lines: set cartoon pixels to black where lines present
    overlay = cartoon.copy()
    if line_map.ndim == 3:
        # ensure single channel
        line_gray = cv2.cvtColor(line_map, cv2.COLOR_BGR2GRAY)
    else:
        line_gray = line_map

    mask = (line_gray == 0)
    overlay[mask] = 0

    # Also produce a composited visualization: cartoon + subtle lines
    lines_colored = cv2.cvtColor(line_gray, cv2.COLOR_GRAY2BGR)
    lines_colored = (lines_colored // 255 * np.array([0, 0, 0], dtype=np.uint8))

    composite = overlay

    return cartoon, line_gray, composite


def main():
    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {INPUT_PATH}")
    
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)

    cartoon, line_map, composite = combine_cartoon_and_lines(img)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, composite)

    cv2.imshow("Original", img)
    cv2.imshow("Cartoon", cartoon)
    cv2.imshow("Lines", line_map)
    cv2.imshow("Composite", composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
