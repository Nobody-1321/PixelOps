import numpy as np
import matplotlib.pyplot as plt
from pixelops.filtering import log
from pixelops.core import rescale_to_uint8
from pixelops.io import imread
from pixelops.visualization import show_image

img = imread("./data/img/woman_ai.webp", mode="gray")

LOG = log(img, sigma_s=0.5, sigma_d=0.5)
LOG = rescale_to_uint8(LOG)
fig, ax = plt.subplots(1, figsize=(12, 6))
show_image(ax, LOG, title="Laplacian of Gaussian")
plt.show()