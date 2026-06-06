from pixelops.filtering.spatial.isotropic_diffusion import isotropic_diffusion
import pixelops as pix
import numpy as np
import matplotlib.pyplot as plt
from pixelops.core import rescale_to_uint8
from pixelops.io import imread
from pixelops.visualization import show_image


img_path = "./data/img/botticelli-primavera.jpg"

iterations = 35
gamma_s = 0.06

img = imread(img_path, mode="gray")
out_iso = isotropic_diffusion(img, n_iter=iterations, gamma=gamma_s)
out_iso = pix.normalize_to_uint8(out_iso)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img, title="Original")
show_image(ax[1], out_iso, title="Isotropic Diffusion")
plt.show()  

img_rgb = imread(img_path, mode="rgb")
out_iso_bgr = isotropic_diffusion(img_rgb, n_iter=iterations, gamma=gamma_s)
out_iso_bgr = pix.normalize_to_uint8(out_iso_bgr)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img_rgb, title="Original")
show_image(ax[1], out_iso_bgr, title="Isotropic Diffusion")
plt.show()