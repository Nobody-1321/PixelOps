from pixelops.filtering import median_filter_grayscale, median_filter_bgr
import pixelops as pix


img = pix.open_image("./data/img/lena_salt.jpg", mode="gray")

out = median_filter_grayscale(img, window_size=13)

pix.show_side_by_side(img, out, title1="Original", title2="Median Filtered")

img_bgr = pix.open_image("./data/img/lena_salt.jpg", mode="bgr")

out_bgr = median_filter_bgr(img_bgr, window_size=13)

pix.show_side_by_side(img_bgr, out_bgr, title1="Original BGR", title2="Median Filtered BGR")