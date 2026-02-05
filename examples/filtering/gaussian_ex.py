import pixelops as pix
from pixelops.filtering import gaussian_filter_lab_luminance, gaussian_filter_bgr, gaussian_filter_grayscale


img_bgr = pix.open_image("./data/img/mujerIA.webp", mode="bgr")

img_gaussian_bgr = gaussian_filter_bgr(img_bgr, sigma=2.5)

pix.show_side_by_side(
    img_bgr,
    img_gaussian_bgr, 
    title1="Original BGR Image", 
    title2="Gaussian Smoothed BGR Image"
)

img = pix.open_image("./data/img/mujerIA.webp", mode="gray")

img_gaussian = gaussian_filter_grayscale(img, sigma=8.5)

pix.show_side_by_side(
    img,
    img_gaussian, 
    title1="Original Image", 
    title2="Gaussian Smoothed Image"
)

img_lab_l = gaussian_filter_lab_luminance(img_bgr, sigma=6.5)

pix.show_side_by_side(
    img_gaussian_bgr,
    img_lab_l, 
    title1="Gaussian BGR Image", 
    title2="Gaussian Smoothed LAB Luminance Image"
)
