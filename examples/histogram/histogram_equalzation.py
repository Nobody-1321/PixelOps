import pixelops as pix

img = pix.open_image("./data/img/Moises.jpg", mode="gray")

img_eq = pix.histogram_equalization_gray(img)

pix.show_side_by_side(img, img_eq, "Original", "Equalized Grayscale Image")

img_color = pix.open_image("./data/img/Moises.jpg", mode="bgr")

img_eq = pix.histogram_equalization_bgr(img_color)

pix.show_side_by_side(img_color, img_eq, "Original", "Equalized Color Image")