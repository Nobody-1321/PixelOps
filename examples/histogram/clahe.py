import pixelops as pix

#img = pix.open_image("./data/img/Moises.jpg", mode="gray")
img = pix.open_image("./data/img/ciervo.jpg", mode="gray")

img_eq = pix.clahe_grayscale(img, clip_limit=20, grid_size=(9,9))

pix.show_side_by_side(img, img_eq, "Original", "Equalized Grayscale Image")

#img_color = pix.open_image("./data/img/Moises.jpg", mode="bgr")
img_color = pix.open_image("./data/img/ciervo.jpg", mode="bgr")
img_color_eq = pix.clahe_bgr(img_color, clip_limit=20, grid_size=(9,9))
pix.show_side_by_side(img_color, img_color_eq, "Original", "Equalized BGR Image")