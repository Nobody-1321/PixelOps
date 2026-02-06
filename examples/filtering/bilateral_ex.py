import numpy as np
import pixelops as pix
from pixelops.filtering import bilateral_filter

img_gray = pix.open_image("./data/img/desertNO.jpg", mode="gray")

filtered_img = bilateral_filter(img_gray, 1.2, 0.7, 8, 5)
filtered_img = np.clip(filtered_img*255.0, 0, 255).astype(np.uint8)

pix.show_side_by_side(
    img_gray,
    filtered_img, 
    title1="Original Grayscale Image", 
    title2="Bilateral Filtered Grayscale Image"
)

img_bgr = pix.open_image("./data/img/desertNO.jpg", mode="bgr")

filtered_img_bgr = bilateral_filter(img_bgr, 2.0, 0.8, 8, 3)
filtered_img_bgr = np.clip(filtered_img_bgr*255.0, 0, 255).astype(np.uint8)

pix.show_side_by_side(
    img_bgr,
    filtered_img_bgr,
    title1="Original BGR Image",
    title2="Bilateral Filtered BGR Image"
)

