from pixelops.filtering.spatial.anisotropic_diffusion import (
    anisotropic_diffusion
)
import pixelops as pix

img_path = "./data/img/catNO.png"
iterations = 40
kappa_s = 15.0

import cv2 as cv

img = pix.open_image(img_path, mode="gray")
out_ani = anisotropic_diffusion(img, n_iter=iterations, kappa=kappa_s)
out_ani = pix.normalize_to_uint8(out_ani)
cv.imshow("Anisotropic Diffusion Grayscale", out_ani)
cv.waitKey(0)
cv.destroyAllWindows()

img_rgb = pix.open_image(img_path, mode="bgr")
out_ani_bgr = anisotropic_diffusion(img_rgb, n_iter=iterations, kappa=kappa_s)
out_ani_bgr = pix.normalize_to_uint8(out_ani_bgr)
cv.imshow("Anisotropic Diffusion BGR", out_ani_bgr)
cv.waitKey(0)
cv.destroyAllWindows()
