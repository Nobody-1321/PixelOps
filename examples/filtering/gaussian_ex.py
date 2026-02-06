import numpy as np
import pixelops as pix
from pixelops.filtering import gaussian_filter


img_bgr = pix.open_image("./data/img/mujerIANO.webp", mode="bgr")

img_gaussian_bgr = gaussian_filter(img_bgr, sigma=2.5)

pix.show_side_by_side(
    img_bgr,
    img_gaussian_bgr.astype(np.uint8),
    title1="Original BGR Image", 
    title2="Gaussian Smoothed BGR Image"
)

img = pix.open_image("./data/img/mujerIANO.webp", mode="gray")

img_gaussian = gaussian_filter(img, sigma=8.5)
img_gaussian = img_gaussian.astype(np.uint8)

pix.show_side_by_side(
    img,
    img_gaussian.astype(np.uint8), 
    title1="Original Image", 
    title2="Gaussian Smoothed Image"
)