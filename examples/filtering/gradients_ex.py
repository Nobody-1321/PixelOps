import numpy as np
import pixelops as pix
from pixelops.filtering import gaussian_gradient, sobel_gradient, log_gradient

img = pix.open_image("./data/img/woman_ai.webp", mode="gray")

# Gaussian Gradient

Gx, Gy, Gmag, Gphase = gaussian_gradient(img, sigma_s=0.5, sigma_d=0.5)
Gx = pix.normalize_to_uint8(Gx)
Gy = pix.normalize_to_uint8(Gy)
Gmag = pix.normalize_to_uint8(Gmag)
Gphase = pix.normalize_to_uint8(Gphase)
pix.show_images(
    [Gx, Gy, Gmag, Gphase],
    titles=["Gradient X", "Gradient Y", "Gradient Magnitude", "Gradient Phase"]
)   

# Sobel Gradient

GX,GY,GMAG,GPHASE = sobel_gradient(img)
GX = pix.normalize_to_uint8(GX)
GY = pix.normalize_to_uint8(GY)
GMAG = pix.normalize_to_uint8(GMAG)
GPHASE = pix.normalize_to_uint8(GPHASE)
pix.show_images(
    [GX, GY, GMAG, GPHASE],
    titles=["Gradient X", "Gradient Y", "Gradient Magnitude", "Gradient Phase"]
) 

# Laplacian of Gaussian Gradient

LOG = log_gradient(img, sigma_s=0.5, sigma_d=0.5)
LOG = pix.normalize_to_uint8(LOG)
pix.show_images(
    [LOG,LOG],
    titles=["Laplacian of Gaussian Gradient", "Laplacian of Gaussian Gradient"]
)