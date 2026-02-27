from pixelops.histogram import histogram_equalization
import pixelops as pix

img = pix.open_image("./data/img/MoisesNO.jpg", mode="gray")

img_eq = histogram_equalization(img)
pix.show_side_by_side(img, img_eq, "Original", "Equalized Grayscale Image")

img_color = pix.open_image("./data/img/MoisesNO.jpg", mode="bgr")

img_eq = histogram_equalization(img_color)
pix.show_side_by_side(img_color, img_eq, "Original", "Equalized Color Image")