from pixelops.filtering.spatial.isotropic_diffusion import isotropic_diffusion
import pixelops as pix
import cv2 as cv
import numpy as np

img_path = "./data/img/botticelli-primavera.jpg"

iterations = 35
gamma_s = 0.06

img = pix.open_image(img_path, mode="gray")
out_iso = isotropic_diffusion(img, n_iter=iterations, gamma=gamma_s)
out_iso = pix.normalize_to_uint8(out_iso)

cv.imshow("Isotropic Diffusion Grayscale", out_iso)
cv.waitKey(0)
cv.destroyAllWindows()

img_rgb = pix.open_image(img_path, mode="bgr")
out_iso_bgr = isotropic_diffusion(img_rgb, n_iter=iterations, gamma=gamma_s)
out_iso_bgr = pix.normalize_to_uint8(out_iso_bgr)

cv.imshow("Isotropic Diffusion BGR", out_iso_bgr)
cv.waitKey(0)
cv.destroyAllWindows()
