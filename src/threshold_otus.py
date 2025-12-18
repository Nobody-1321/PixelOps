import cv2 as cv 
import lip

#img = cv.imread('img_data/uvas.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread('img_data/coins.webp', cv.IMREAD_GRAYSCALE)
#img = cv.imread('img_data/naturalezaM.jpg', cv.IMREAD_GRAYSCALE)

if img is None:
    raise Exception("Image not found")

img = cv.resize(img, (668, 956))

cv.imshow('Grayscale Image', img)
cv.waitKey(0)
cv.destroyAllWindows()

lip.show_image_with_histogram(img, title='Grayscale Image with Histogram')

thr_ots = lip.OtsuThreshold(img)
print(f"Otsu Threshold: {thr_ots}")

_, binary_img = cv.threshold(img, thr_ots, 255, cv.THRESH_BINARY)

lip.show_images_together([img, binary_img], titles=['Original Image', 'Otsu Thresholding'])

img_rg, mask_rg = lip.RemoveIntensityRange(img, low=0, high=245, fill=255)

lip.show_images_together([img, mask_rg.astype('uint8')*255], titles=['', ''])
