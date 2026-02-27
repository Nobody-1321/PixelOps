import pixelops as pix
from pixelops.histogram.equalization import clahe
#img = pix.open_image("./data/img/Moises.jpg", mode="gray")
img = pix.open_image("./data/img/ciervoNO.jpg", mode="gray")

img_eq = clahe(img, clip_limit=20, grid_size=(9,9))

pix.show_side_by_side(img, img_eq, "Original", "Equalized Grayscale Image")

#img_color = pix.open_image("./data/img/Moises.jpg", mode="bgr")
img_color = pix.open_image("./data/img/MoisesNO.jpg", mode="bgr")
img_color_eq = clahe(img_color, clip_limit=15, grid_size=(13,13))
pix.show_side_by_side(img_color, img_color_eq, "Original", "Equalized BGR Image")