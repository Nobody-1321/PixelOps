import pixelops as pix
from pixelops.filtering import convolve_separable


img_bgr = pix.open_image("./data/img/mujerIA.webp", mode="bgr")

img_gaussian_bgr = pix.gaussian_filter_bgr(img_bgr, sigma=6.5)

pix.show_side_by_side(
    img_bgr,
    img_gaussian_bgr, 
    title1="Original BGR Image", 
    title2="Gaussian Smoothed BGR Image"
)

img = pix.open_image("./data/img/mujerIA.webp", mode="gray")

img_gaussian = pix.gaussian_filter_grayscale(img, sigma=6.5)

pix.show_side_by_side(
    img,
    img_gaussian, 
    title1="Original Image", 
    title2="Gaussian Smoothed Image"
)