import pixelops as pix
from pixelops.filtering import bilateral_filter_grayscale, bilateral_filter_bgr

img_gray = pix.open_image("./data/img/desert.jpg", mode="gray")

filtered_img = bilateral_filter_grayscale(img_gray, 1.2, 0.7, 8, 5)

pix.show_side_by_side(
    img_gray,
    filtered_img, 
    title1="Original Grayscale Image", 
    title2="Bilateral Filtered Grayscale Image"
)

img_bgr = pix.open_image("./data/img/desert.jpg", mode="bgr")

filtered_img_bgr = bilateral_filter_bgr(img_bgr, 2.0, 0.8, 8, 3)

pix.show_side_by_side(
    img_bgr,
    filtered_img_bgr,
    title1="Original BGR Image",
    title2="Bilateral Filtered BGR Image"
)

