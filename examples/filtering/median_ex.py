from pixelops.filtering import median
import pixelops as pix
from pixelops.visualization import show_image
from pixelops.io import imread
import matplotlib.pyplot as plt


img = imread("./data/img/botticelli-primavera.jpg", mode="gray")
out = median(img, window_size=3)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img, title="Original Grayscale")  
show_image(ax[1], out, title="Median Filtered Grayscale")
plt.show()

img_rgb = imread("./data/img/botticelli-primavera.jpg", mode="rgb")  # Load as RGB  
out_rgb = median(img_rgb, window_size=3)  # Apply median filter to RGB image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img_rgb, title="Original RGB")
show_image(ax[1], out_rgb, title="Median Filtered RGB")
plt.show()