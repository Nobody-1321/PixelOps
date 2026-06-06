import numpy as np
import pixelops as pix
from pixelops.filtering import gaussian
from pixelops.core import rescale_to_uint8
from pixelops.io import imread
from pixelops.visualization import show_image
import matplotlib.pyplot as plt

img_rgb = imread("./data/img/woman_ai.webp", mode="rgb")
img_gaussian_rgb = gaussian(img_rgb, sigma=2.5)
img_gaussian_rgb = rescale_to_uint8(img_gaussian_rgb)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[0], img_rgb, title="Original RGB Image")
show_image(ax[1], img_gaussian_rgb, title="Gaussian Blurred RGB Image")
plt.show()


img = imread("./data/img/woman_ai.webp", mode="gray")
img_gaussian = gaussian(img, sigma=8.5)
img_gaussian = rescale_to_uint8(img_gaussian)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[0], img, title="Original Grayscale Image")
show_image(ax[1], img_gaussian, title="Gaussian Blurred Grayscale Image")
plt.show()
