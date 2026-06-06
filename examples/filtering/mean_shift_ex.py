from pixelops.filtering import mean_shift
import matplotlib.pyplot as plt
from pixelops.core import rescale_to_uint8
from pixelops.io import imread
from pixelops.visualization import show_image


img = imread("./data/img/Rabbit_Flowers.png", mode="gray")

out = mean_shift(img, hs=3, hr=0.10, max_iter=5, eps=1.0)
out = rescale_to_uint8(out)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img, title="Original")
show_image(ax[1], out, title="Mean Shift Filtered")
plt.show()

img_rgb = imread("./data/img/Rabbit_Flowers.png", mode="rgb")
out_rgb = mean_shift(img_rgb, hs=20, hr=0.10, max_iter=10, eps=1.0)
out_rgb = rescale_to_uint8(out_rgb)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img_rgb, title="Original")
show_image(ax[1], out_rgb, title="Mean Shift Filtered")
plt.show()
