from pixelops.histogram.equalization import clahe
import matplotlib.pyplot as plt
from pixelops.core import rescale_to_uint8
from pixelops.io import imread
from pixelops.visualization import show_image

img = imread("./data/img/Moises.jpg", mode="gray")
img_eq = clahe(img, clip_limit=20, grid_size=(9,9))
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img, title="Original Grayscale Image")
show_image(ax[1], img_eq, title="CLAHE Grayscale Image")
plt.show()


img_color = imread("./data/img/Moises.jpg", mode="rgb")
img_color_eq = clahe(img_color, clip_limit=20, grid_size=(9,9))
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img_color, title="Original Color Image")
show_image(ax[1], img_color_eq, title="CLAHE Color Image")
plt.show()