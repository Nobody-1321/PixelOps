from pixelops.filtering.spatial.anisotropic_diffusion import (
    anisotropic_diffusion
)
import pixelops as pix

img_path = "./data/img/cat_1.png"
iterations = 40
kappa_s = 15.0

import cv2 as cv

img = pix.open_image(img_path, mode="gray")
out_ani_1 = anisotropic_diffusion(img, n_iter=iterations, kappa=kappa_s, option=1)
out_ani_1 = pix.normalize_to_uint8(out_ani_1)
out_ani_2 = anisotropic_diffusion(img, n_iter=iterations, kappa=kappa_s, option=2)
out_ani_2 = pix.normalize_to_uint8(out_ani_2)

cv.imshow("Anisotropic Diffusion Grayscale", out_ani_1)
cv.imshow("Anisotropic Diffusion Grayscale Option 2", out_ani_2)
cv.waitKey(0)
cv.destroyAllWindows()

img_rgb = pix.open_image(img_path, mode="bgr")
out_ani_bgr_1 = anisotropic_diffusion(img_rgb, n_iter=iterations, kappa=kappa_s,option=1)
out_ani_bgr_1 = pix.normalize_to_uint8(out_ani_bgr_1)
out_ani_bgr_2 = anisotropic_diffusion(img_rgb, n_iter=iterations, kappa=kappa_s, option=2)
out_ani_bgr_2 = pix.normalize_to_uint8(out_ani_bgr_2)

cv.imshow("Anisotropic Diffusion BGR", out_ani_bgr_1)
cv.imshow("Anisotropic Diffusion BGR Option 2", out_ani_bgr_2)
cv.waitKey(0)
cv.destroyAllWindows()
