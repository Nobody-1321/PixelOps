import cv2
import numpy as np
from numba import njit

# ======================================================
# CONFIGURACIÓN
# ======================================================

IMG_PATH = "./data/img/mujerIA.webp"

MAX_SIGMA = 40.0
SIGMA_LEVELS = np.array(
    [0, 2, 4, 6, 8, 12, 16, 24, 32, 40],
    dtype=np.float32
)

# ======================================================
# CARGA DE IMAGEN
# ======================================================

img_base = cv2.imread(IMG_PATH)
if img_base is None:
    raise IOError("No se pudo cargar la imagen")

img_base = cv2.resize(img_base, (1200, 1200))
img_base = img_base.astype(np.float32)

H, W = img_base.shape[:2]

# ======================================================
# MAPA DE BLUR
# ======================================================

blur_map = np.zeros((H, W), dtype=np.float32)

# ======================================================
# BLUR PERCEPTUAL (solo luminancia)
# ======================================================

def gaussian_blur_luminance(img_bgr, sigma):
    lab = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    L, A, B = cv2.split(lab)

    if sigma > 0:
        L = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)

    lab_blur = cv2.merge([L, A, B])
    return cv2.cvtColor(lab_blur.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

# ======================================================
# PRECALCULAR BLURS
# ======================================================

blurred_stack = np.zeros(
    (len(SIGMA_LEVELS), H, W, 3),
    dtype=np.float32
)

for i, s in enumerate(SIGMA_LEVELS):
    if s == 0:
        blurred_stack[i] = img_base
    else:
        blurred_stack[i] = gaussian_blur_luminance(img_base, s)

# ======================================================
# RENDER CONTINUO (interpolado)
# ======================================================

@njit
def render(img_out, blur_map, sigmas, stack):
    H, W = blur_map.shape
    n = len(sigmas)

    for y in range(H):
        for x in range(W):
            s = blur_map[y, x]

            if s <= sigmas[0]:
                img_out[y, x] = stack[0, y, x]
                continue

            if s >= sigmas[n - 1]:
                img_out[y, x] = stack[n - 1, y, x]
                continue

            for i in range(n - 1):
                s0 = sigmas[i]
                s1 = sigmas[i + 1]
                if s0 <= s <= s1:
                    t = (s - s0) / (s1 - s0)
                    img_out[y, x] = (
                        stack[i, y, x] * (1.0 - t) +
                        stack[i + 1, y, x] * t
                    )
                    break

# ======================================================
# PINCEL (falloff gaussiano)
# ======================================================

drawing = False

def paint_blur(event, x, y, flags, param):
    global drawing, blur_map

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        r = BRUSH_RADIUS
        strength = BRUSH_STRENGTH

        y0 = max(0, y - r)
        y1 = min(H, y + r)
        x0 = max(0, x - r)
        x1 = min(W, x + r)

        sigma_brush = r * 0.5

        for yy in range(y0, y1):
            for xx in range(x0, x1):
                dx = xx - x
                dy = yy - y
                d2 = dx * dx + dy * dy

                if d2 < r * r:
                    falloff = np.exp(-d2 / (2.0 * sigma_brush * sigma_brush))
                    blur_map[yy, xx] += falloff * strength

                    if blur_map[yy, xx] > MAX_SIGMA:
                        blur_map[yy, xx] = MAX_SIGMA

# ======================================================
# UI
# ======================================================

def nothing(x):
    pass

cv2.namedWindow("Blur Brush", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blur Brush", 1200, 900)

BRUSH_RADIUS = 40
BRUSH_STRENGTH = 0.5   # sigma por pasada (realista)

cv2.createTrackbar("Radius", "Blur Brush", BRUSH_RADIUS, 150, nothing)
cv2.createTrackbar("Strength x100", "Blur Brush", int(BRUSH_STRENGTH * 100), 200, nothing)

cv2.setMouseCallback("Blur Brush", paint_blur)

# ======================================================
# LOOP PRINCIPAL
# ======================================================

img_result = img_base.copy()

while True:
    BRUSH_RADIUS = max(1, cv2.getTrackbarPos("Radius", "Blur Brush"))
    BRUSH_STRENGTH = cv2.getTrackbarPos("Strength x100", "Blur Brush") / 100.0

    render(img_result, blur_map, SIGMA_LEVELS, blurred_stack)

    display = np.clip(img_result, 0, 255).astype(np.uint8)
    cv2.imshow("Blur Brush", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):
        blur_map[:] = 0.0

cv2.destroyAllWindows()
