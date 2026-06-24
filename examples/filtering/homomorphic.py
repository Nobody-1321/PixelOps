from pixelops.io import imread
from pixelops.visualization import show_image
import matplotlib.pyplot as plt
from pixelops.filtering import homomorphic

img_gray = imread("./data/img/Julian_Onderdonk.jpeg", mode="gray")
filtered = homomorphic(img_gray, gammaL=1.5, gammaH=1.5, sigma=30)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[1], filtered, title="Filtered Image")
show_image(ax[0], img_gray, title="Original Image")
plt.show()

img_rgb = imread("./data/img/Julian_Onderdonk.jpeg", mode="rgb")
filtered = homomorphic(img_rgb, gammaL=1.5, gammaH=1.5, sigma=30)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[1], filtered, title="Filtered Image")
show_image(ax[0], img_rgb, title="Original Image")
plt.show()