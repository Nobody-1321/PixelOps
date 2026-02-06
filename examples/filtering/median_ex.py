from pixelops.filtering import median_filter
import pixelops as pix


img = pix.open_image("./data/img/botticelli-primaveraNO.jpg", mode="gray")
out = median_filter(img, window_size=3)
pix.show_side_by_side(img, out, title1="Original", title2="Median Filtered")

img_bgr = pix.open_image("./data/img/botticelli-primaveraNO.jpg", mode="bgr")
out_bgr = median_filter(img_bgr, window_size=7)
pix.show_side_by_side(img_bgr, out_bgr, title1="Original BGR", title2="Median Filtered BGR")