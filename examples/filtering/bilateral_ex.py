from pixelops.io import imread
from pixelops.visualization import show_image
from pixelops.core import rescale_to_uint8
import matplotlib.pyplot as plt
from pixelops.filtering import bilateral

img_gray = imread("./data/img/desert.jpg", mode="gray")
filtered_img = bilateral(img_gray, 1.2, 0.7, 8, 5)
filtered_img =rescale_to_uint8(filtered_img)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[0], img_gray, title="Original Grayscale")
show_image(ax[1], filtered_img, title="Bilateral Filtered Grayscale")
plt.show()


img_bgr = imread("./data/img/desert.jpg", mode="rgb")
filtered_img_bgr = bilateral(img_bgr, 2.0, 0.8, 8, 3)
filtered_img_bgr = rescale_to_uint8(filtered_img_bgr)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[0], img_bgr, title="Original RGB")
show_image(ax[1], filtered_img_bgr, title="Bilateral Filtered RGB")
plt.show()

