import numpy as np
import matplotlib.pyplot as plt
from pixelops.filtering import gaussian_gradient, sobel_gradient 
from pixelops.core import rescale_to_uint8
from pixelops.io import imread
from pixelops.visualization import show_image

img = imread("./data/img/woman_ai.webp", mode="gray")

# Gaussian Gradient

Gx, Gy, Gmag, Gphase = gaussian_gradient(img, sigma_s=0.5, sigma_d=0.5)
Gx = rescale_to_uint8(Gx)
Gy = rescale_to_uint8(Gy)
Gmag = rescale_to_uint8(Gmag)
Gphase = rescale_to_uint8(Gphase)

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
show_image(ax[0, 0], Gx, title="GX")
show_image(ax[0, 1], Gy, title="GY")
show_image(ax[1, 0], Gmag, title="Mag")
show_image(ax[1, 1], Gphase, title="Phase")

#plt.tight_layout()

fig.subplots_adjust(
    wspace=-0.5,
    hspace=0.25
)

plt.show()

# Sobel Gradient

GX,GY,GMAG,GPHASE = sobel_gradient(img)
GX = rescale_to_uint8(GX)
GY = rescale_to_uint8(GY)
GMAG = rescale_to_uint8(GMAG)
GPHASE = rescale_to_uint8(GPHASE)

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
show_image(ax[0, 0], GX, title="GX")
show_image(ax[0, 1], GY, title="GY")
show_image(ax[1, 0], GMAG, title="Mag")
show_image(ax[1, 1], GPHASE, title="Phase")

fig.subplots_adjust(
    wspace=-0.5,
    hspace=0.25
)
plt.show()
