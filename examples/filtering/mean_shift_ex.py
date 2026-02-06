from pixelops.filtering import mean_shift_filter
import pixelops as pix
import numpy as np

def normalize_to_uint8(arr):
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())  # Escala a [0, 1]
    arr_scaled = arr_norm * 255                             # Escala a [0, 255]
    return arr_scaled.astype(np.uint8)

img = pix.open_image("./data/img/cerezoNO.png", mode="gray")

out = mean_shift_filter(img, hs=3, hr=30.0, max_iter=5, eps=1.0)
out = normalize_to_uint8(out)

pix.show_side_by_side(img, out, title1="Original", title2="Mean Shift Filtered")

img_bgr = pix.open_image("./data/img/mujerIANO.webp", mode="bgr")
out_bgr = mean_shift_filter(img_bgr, hs=15, hr=15.0, max_iter=25, eps=1.0)
out_bgr = normalize_to_uint8(out_bgr)
pix.show_side_by_side(img_bgr, out_bgr, title1="Original", title2="Mean Shift Filtered")
