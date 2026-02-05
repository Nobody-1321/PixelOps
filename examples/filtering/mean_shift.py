from pixelops.filtering import mean_shift_filter_grayscale, mean_shift_filter_bgr
import pixelops as pix

'''
img = pix.open_image("./data/img/cerezo.png", mode="gray")
out = mean_shift_filter_grayscale(img, hs=3, hr=30.0, max_iter=5, eps=1.0)
pix.show_side_by_side(img, out, title1="Original", title2="Mean Shift Filtered")
'''

img_bgr = pix.open_image("./data/img/mujerIA.webp", mode="bgr")
out_bgr = mean_shift_filter_bgr(img_bgr, hs=15, hr=15.0, max_iter=25, eps=1.0)
pix.show_side_by_side(img_bgr, out_bgr, title1="Original", title2="Mean Shift Filtered")