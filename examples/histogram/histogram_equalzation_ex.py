from pixelops.histogram import histogram_equalization
import matplotlib.pyplot as plt
from pixelops.core import rescale_to_uint8
from pixelops.io import imread
from pixelops.visualization import show_image

img = imread("./data/img/Moises.jpg", mode="gray")

img_eq = histogram_equalization(img)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img, title="Original Grayscale Image")
show_image(ax[1], img_eq, title="Equalized Grayscale Image")
plt.show()



img_color = imread("./data/img/Moises.jpg", mode="rgb")
img_eq = histogram_equalization(img_color)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img_color, title="Original RGB Image")
show_image(ax[1], img_eq, title="Equalized RGB Image")
plt.show()