import numpy as np
import pixelops as pix
from pixelops.filtering import gaussian_gradient, sobel_gradient, log_gradient

def normalize_to_uint8(arr):
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())  # Escala a [0, 1]
    arr_scaled = arr_norm * 255                             # Escala a [0, 255]
    return arr_scaled.astype(np.uint8)

img = pix.open_image("./data/img/mujerIANO.webp", mode="gray")
Gx, Gy, Gmag, Gphase = gaussian_gradient(img, sigma_s=0.5, sigma_d=0.5)

Gx = normalize_to_uint8(Gx)
Gy = normalize_to_uint8(Gy)
Gmag = normalize_to_uint8(Gmag)
Gphase = normalize_to_uint8(Gphase)

pix.show_images(
    [Gx, Gy, Gmag, Gphase],
    titles=["Gradient X", "Gradient Y", "Gradient Magnitude", "Gradient Phase"]
)   

GX,GY,GMAG,GPHASE = sobel_gradient(img)

GX = normalize_to_uint8(GX)
GY = normalize_to_uint8(GY)
GMAG = normalize_to_uint8(GMAG)
GPHASE = normalize_to_uint8(GPHASE)

pix.show_images(
    [GX, GY, GMAG, GPHASE],
    titles=["Gradient X", "Gradient Y", "Gradient Magnitude", "Gradient Phase"]
) 


LOG = log_gradient(img, sigma_s=0.5, sigma_d=0.5)
LOG = normalize_to_uint8(LOG)
pix.show_images(
    [LOG,LOG],
    titles=["Laplacian of Gaussian Gradient", "Laplacian of Gaussian Gradient"]
)