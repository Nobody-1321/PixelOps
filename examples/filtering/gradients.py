import pixelops as pix
import matplotlib.pyplot as plt

img = pix.open_image("./data/img/mujerIA.webp", mode="gray")

GX,GY,GMAG,GPHASE = pix.compute_sobel_image_gradient_vis(img)

pix.show_images(
    [GX, GY, GMAG, GPHASE],
    titles=["Gradient X", "Gradient Y", "Gradient Magnitude", "Gradient Phase"]
) 
