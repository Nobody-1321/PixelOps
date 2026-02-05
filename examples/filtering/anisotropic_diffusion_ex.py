from pixelops.filtering.spatial.anisotropic_diffusion import (
    anisotropic_diffusion_grayscale,
    anisotropic_diffusion_bgr,
)

from pixelops.filtering.spatial.isotropic_diffusion import (
    isotropic_diffusion_grayscale,
    isotropic_diffusion_bgr,
)

import pixelops as pix

img_path = "./data/img/botticelli-primavera.jpg"
iterations = 30
gamma_s = 0.25
kappa_s = 20.0

img = pix.open_image(img_path, mode="gray")
out_iso = isotropic_diffusion_grayscale(img, n_iter=iterations, gamma=gamma_s)
out_ani = anisotropic_diffusion_grayscale(img, n_iter=iterations, kappa=kappa_s)
pix.show_side_by_side(out_iso, out_ani, title1="Isotropic Diffusion", title2="Anisotropic Diffusion")

img_rgb = pix.open_image(img_path, mode="bgr")
out_iso_bgr = isotropic_diffusion_bgr(img_rgb, n_iter=iterations, gamma=gamma_s)
pix.show_side_by_side(img_rgb, out_iso_bgr, title1="Original BGR", title2="Isotropic Diffusion BGR")

out_ani_bgr = anisotropic_diffusion_bgr(img_rgb, n_iter=iterations, kappa=kappa_s)
pix.show_side_by_side(img_rgb, out_ani_bgr, title1="Original BGR", title2="Anisotropic Diffusion BGR")

pix.show_side_by_side(out_iso_bgr, out_ani_bgr, title1="Isotropic Diffusion BGR", title2="Anisotropic Diffusion BGR")


