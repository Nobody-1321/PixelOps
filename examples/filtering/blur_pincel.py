import cv2
import numpy as np
from numba import jit
from pixelops.filtering import gaussian_filter_bgr

# --------------------------------
# Cargar imagen
# --------------------------------

img = cv2.imread("./data/img/mujerIA.webp")
if img is None:
    raise IOError("No se pudo cargar la imagen.")

img = cv2.resize(img, (1200, 1200))

img = img.astype(np.float32)
img_result = img.copy()

H, W = img.shape[:2]

# --------------------------------
# Valores iniciales
# --------------------------------

BLUR_SIGMA = 15
BRUSH_RADIUS = 30
BRUSH_STRENGTH = 50  # se maneja como entero [0,100]

# Pre-calcular blurs para sigmas comunes
blur_cache = {}
def get_blur(img, sigma):
    if sigma not in blur_cache:
        #blur_cache[sigma] = cv2.GaussianBlur(img, (0, 0), sigma)
        blur_cache[sigma] = gaussian_filter_bgr(img_result, sigma=sigma)
    return blur_cache[sigma]

# Compilar con Numba para máxima velocidad
@jit(nopython=True)
def apply_brush_diffusion(result, x, y, radius, sigma, H, W):
    x_min = max(0, x - radius)
    x_max = min(W, x + radius)
    y_min = max(0, y - radius)
    y_max = min(H, y + radius)

    for yy in range(y_min, y_max):
        for xx in range(x_min, x_max):
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)
            if dist <= radius:
                # difusión simple (aprox blur)
                result[yy, xx] = (
                    result[yy, xx] * 0.9 +
                    (
                        result[max(yy-1,0), xx] +
                        result[min(yy+1,H-1), xx] +
                        result[yy, max(xx-1,0)] +
                        result[yy, min(xx+1,W-1)]
                    ) * 0.025
                )

drawing = False

# --------------------------------
# Trackbar callbacks (vacíos)
# --------------------------------

def nothing(x):
    pass

# --------------------------------
# Mouse callback (pincel)
# --------------------------------
def paint_blur(event, x, y, flags, param):
    global drawing, img_result, BLUR_SIGMA, BRUSH_RADIUS, BRUSH_STRENGTH

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        strength = BRUSH_STRENGTH / 100.0
        blurred = get_blur(img, BLUR_SIGMA)
        
        # Usar Numba para pincel rápido
        apply_brush_diffusion(img_result, x, y, BRUSH_RADIUS, BLUR_SIGMA, H, W)


# --------------------------------
# Ventana y trackbars
# --------------------------------

cv2.namedWindow("Blur Brush", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blur Brush", 1200, 900)  # ancho, alto

cv2.createTrackbar("Blur Sigma", "Blur Brush", BLUR_SIGMA, 50, nothing)
cv2.createTrackbar("Brush Radius", "Blur Brush", BRUSH_RADIUS, 100, nothing)
cv2.createTrackbar("Brush Strength", "Blur Brush", BRUSH_STRENGTH, 100, nothing)

cv2.setMouseCallback("Blur Brush", paint_blur)

prev_sigma = BLUR_SIGMA

# --------------------------------
# Loop principal
# --------------------------------

while True:
    # Leer sliders
    BLUR_SIGMA = cv2.getTrackbarPos("Blur Sigma", "Blur Brush")
    BRUSH_RADIUS = max(1, cv2.getTrackbarPos("Brush Radius", "Blur Brush"))
    BRUSH_STRENGTH = cv2.getTrackbarPos("Brush Strength", "Blur Brush")

    # Mostrar resultado
    result_u8 = np.clip(img_result, 0, 255).astype(np.uint8)
    cv2.imshow("Blur Brush", result_u8)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:        # ESC
        break
    elif key == ord('r'):  # reset
        img_result[:] = img.copy()


cv2.destroyAllWindows()
