from pixelops.filtering import mean_shift_filter
import pixelops as pix
import numpy as np



img = pix.open_image("./data/img/cerezo.png", mode="gray")

out = mean_shift_filter(img, hs=3, hr=30.0, max_iter=5, eps=1.0)
out = pix.normalize_to_uint8(out)

pix.show_side_by_side(img, out, title1="Original", title2="Mean Shift Filtered")

img_bgr = pix.open_image("./data/img/cerezo.png", mode="bgr")
out_bgr = mean_shift_filter(img_bgr, hs=15, hr=15.0, max_iter=25, eps=1.0)
out_bgr = pix.normalize_to_uint8(out_bgr)
pix.show_side_by_side(img_bgr, out_bgr, title1="Original", title2="Mean Shift Filtered")
