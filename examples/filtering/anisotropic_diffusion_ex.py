from pixelops.filtering import anisotropic_diffusion
from pixelops.io import imread
from pixelops.visualization import show_image
from pixelops.core import rescale_to_uint8
import matplotlib.pyplot as plt

img_path = "./data/img/cat_1.png"
iterations = 40
kappa_s = 15.0

img = imread(img_path, mode="gray")

out_ani_1 = anisotropic_diffusion(img, n_iter=iterations, kappa=kappa_s, method="exponential")
out_ani_1 = rescale_to_uint8(out_ani_1)

out_ani_2 = anisotropic_diffusion(img, n_iter=iterations, kappa=kappa_s, method="inverse" )
out_ani_2 = rescale_to_uint8(out_ani_2)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[0], out_ani_1,title="Anisotropic Diffusion Option 1")
show_image(ax[1], out_ani_2, title="Anisotropic Diffusion Option 2")
plt.show()

img_rgb = imread(img_path, mode="rgb")

out_ani_bgr_1 = anisotropic_diffusion(img_rgb, n_iter=iterations, kappa=kappa_s, method="exponential")
out_ani_bgr_1 = rescale_to_uint8(out_ani_bgr_1)

out_ani_bgr_2 = anisotropic_diffusion(img_rgb, n_iter=iterations, kappa=kappa_s, method="inverse")
out_ani_bgr_2 = rescale_to_uint8(out_ani_bgr_2)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[0], out_ani_bgr_1,title="Anisotropic Diffusion RGB Option 1")
show_image(ax[1], out_ani_bgr_2, title="Anisotropic Diffusion RGB Option 2")
plt.show()