import pixelops as pix
from pixelops.filtering import sobel_gradient

img = pix.open_image("./data/img/mujerIA.webp", mode="gray")

GX,GY,GMAG,GPHASE = sobel_gradient(img)
pix.show_images(
    [GX, GY, GMAG, GPHASE],
    titles=["Gradient X", "Gradient Y", "Gradient Magnitude", "Gradient Phase"]
) 
