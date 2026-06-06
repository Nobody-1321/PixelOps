import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import basura.tempo.src.lip as lip   # tu librería personalizada


# ============================================================
# 1. LECTURA Y REDUCCIÓN DE RUIDO (MEAN SHIFT)
# ------------------------------------------------------------
# Se suavizan las regiones de color preservando bordes gruesos.
# ============================================================

#img = cv2.imread("img_data/joker.webp")
#img = cv2.imread("img_data/cat_noise.png")
#img = cv2.imread("img_data/RowImageSlider19.webp")
#img = cv2.imread("img_data/Rose_2.jpg")
#img = cv2.imread("img_data/mujerIA.webp")
#img = cv2.imread("img_data/mantis.png")
#img = cv2.imread("img_data/bosque_lago.jpg")
#img = cv2.imread("img_data/paisaje.jpeg")
#img = cv2.imread("img_data/spain-7.jpg")
#img  = cv2.imread("img_data/scarlett_2.webp")
img  = cv2.imread("img_data/cerezo_.jpg")
#img = cv2.imread("img_data/scarlett_2.webp")
#img = cv2.imread("img_data/camellos.jpg")
#img = cv2.imread("img_data/nature-5.jpg")
#img = cv2.imread("img_data/mary-and-jesus.jpg")
#img = cv2.imread("img_data/coffee.jpg")
#img = cv2.imread("img_data/superman.jpg")
#img = cv2.imread("img_data/tazas.jpg")
#img = cv2.imread("img_data/monarch.png")
#img = cv2.imread("img_data/train_images/baby.png")
#img = cv2.imread("img_data/flowers.png")


if img is None:
    raise FileNotFoundError("La imagen no se pudo cargar. Verifica la ruta.")

img_name = "flores"

#img = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
img_denoised = lip.MeanShiftFilterBGR(img, 15, 15, 10)
img_denoisedMeanTemp = img_denoised.copy()
img_denoised = cv2.bilateralFilter(img_denoised, d=9, sigmaColor=75, sigmaSpace=75)
img_denoisedBilateralTemp = img_denoised.copy()

# ============================================================
# 2. CONVERSIÓN A LAB + POSICIÓN ESPACIAL
# ------------------------------------------------------------
# Se agrupa en espacio perceptual (Lab) y se añade (x, y) para
# evitar fusiones entre regiones alejadas pero similares en color.
# ============================================================

h, w = img_denoised.shape[:2]
img_lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2Lab)
X, Y = np.meshgrid(np.arange(w), np.arange(h))

# Combinar color + coordenadas (factor espacial ajustable)
features = np.concatenate((
    img_lab.reshape((-1, 3)),
    0.05 * np.stack((X.ravel(), Y.ravel()), axis=1)
), axis=1)
Z = np.float32(features)

# ============================================================
# 3. CUANTIZACIÓN DE COLOR (K-MEANS)
# ------------------------------------------------------------
# Reduce la paleta a K colores, generando zonas planas.
# ============================================================

K = 25
kmeans = MiniBatchKMeans(n_clusters=K, init='k-means++', batch_size=10000, n_init=5)
labels = kmeans.fit_predict(Z)

# Tomar solo los canales Lab (sin x,y)
centers_lab = np.uint8(kmeans.cluster_centers_[:, :3])
quantized = centers_lab[labels]

# Reconstruir imagen en Lab y convertir a BGR
img_lab_quant = quantized.reshape(img_denoised.shape)
img_quant = cv2.cvtColor(img_lab_quant, cv2.COLOR_Lab2BGR)

cv2.imshow("Quantized", img_quant)


# ============================================================
# 4. TRANSFERENCIA DE DETALLE (EN LUGAR DE CANNY)
# ------------------------------------------------------------
# Extrae los detalles finos del original mediante un filtro bilateral,
# y los transfiere multiplicativamente a la imagen cuantizada.
# Esto reemplaza las líneas negras por detalles realistas de textura.
# ============================================================

def bilateral_filter(image, sigma_d=15, sigma_r=0.1):
    return cv2.bilateralFilter(image, d=-1,
                               sigmaColor=sigma_r * 255,
                               sigmaSpace=sigma_d)

def compute_detail_layer(img, sigma_d=30, sigma_r=0.4, eps=1e-3):
    base = bilateral_filter(img, sigma_d, sigma_r)
    detail = (img + eps) / (base + eps)
    return np.clip(detail, 0.8, 1.2)  # limitar amplificación extrema

# --- Calcular capa de detalle desde la imagen original ---

detail_layer = compute_detail_layer(img.astype(np.float32)/255.0, sigma_d=10, sigma_r=1.0)

'''
# Convertir la capa de detalle a magnitud (intensidad)
detail_gray = np.mean(detail_layer, axis=2)  # promedio de los 3 canales
detail_gray = np.clip(detail_gray, 0, 2)  # limitar valores extremos

# Normalizar para visualización
detail_vis = ((detail_gray - detail_gray.min()) / 
              (detail_gray.max() - detail_gray.min() + 1e-6) * 255).astype(np.uint8)

cv2.imshow("Detail Layer (Gray Magnitude)", detail_vis)
'''
'''
# Centrar alrededor de 1.0
detail_diff = detail_layer - 1.0
detail_mag = np.mean(detail_diff, axis=2)

# Normalizar alrededor de 0
detail_norm = cv2.normalize(detail_mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
detail_norm = detail_norm.astype(np.uint8)

# Aplicar un mapa de color para ver contrastes
detail_colormap = cv2.applyColorMap(detail_norm, cv2.COLORMAP_JET)
cv2.imshow("Detail Layer (Heatmap)", detail_colormap)
'''
# Escalar logarítmicamente para ver detalle multiplicativo sin saturar
detail_vis = np.log(detail_layer + 1e-3)
detail_vis = (detail_vis - detail_vis.min()) / (detail_vis.max() - detail_vis.min() + 1e-6)
detail_vis = np.clip(detail_vis * 255, 0, 255).astype(np.uint8)
cv2.imshow("Detail Layer (Log Scaled RGB)", detail_vis)

alpha = 0.6
detail_layer = np.power(detail_layer, alpha)

# --- Aplicar el detalle sobre la imagen cuantizada ---
cartoon = np.clip((img_quant.astype(np.float32)/255.0) * detail_layer, 0, 1)
cartoon = (cartoon * 255).astype(np.uint8)

cv2.imshow("Cartoon + Detail Transfer", cartoon)

# ============================================================
# 5. REALCE FINAL (UNSHARP MASK EN DOMINIO DE FRECUENCIA)
# ------------------------------------------------------------
# Mejora la microtextura y el contraste sin perder estilo plano.
# ============================================================

height, width = cartoon.shape[:2]
um_kernel = lip.CreateUnsharpMaskingFilter((height, width), 80, alpha=0.9, method='butterworth')
cartoon_sharp = lip.ApplyFrequencyDomainFilterLabL(cartoon, um_kernel)


# ============================================================
# 6. SALIDA FINAL Y COMPARACIÓN
# ------------------------------------------------------------
# ============================================================

final = np.hstack((img, img_denoisedMeanTemp, img_denoisedBilateralTemp, detail_vis, img_quant, cartoon_sharp))
cv2.imwrite(f"img_data/temp/img_cartoonized_transfer_{img_name}.png", final)

cv2.imshow("Cartoonized (Detail Transfer)", cartoon_sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()
