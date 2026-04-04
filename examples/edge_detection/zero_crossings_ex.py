import numpy as np
import matplotlib.pyplot as plt
from pixelops.filtering.spatial.gradient import log_gradient
import cv2 as cv

# Crear imagen con blob circular
img = cv.imread("./data/img/woman_ai.webp", cv.IMREAD_GRAYSCALE)

# Aplicar LoG
log_response = log_gradient(img, sigma_s=3.0, sigma_d=3.0)

# Zero-crossings están donde cambia de signo
zero_crossings = np.diff(np.sign(log_response), axis=0) != 0

plt.imshow(log_response, cmap='RdBu')  # Rojo=negativo, Azul=positivo
plt.colorbar()
plt.title("LoG: Zero-crossings = bordes")
plt.show()