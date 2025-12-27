import pixelops as pix
import cv2 as cv

img = pix.open_image("./data/img/uvas.jpg", mode="gray")

threshold = pix.otsu_threshold(img)

print(f"Otsu's optimal threshold: {threshold}")

_, binary_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

pix.show_side_by_side(
    img,
    binary_img, 
    title1="Original Image", 
    title2="Otsu's Thresholding"
)
