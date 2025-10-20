import cv2
import numpy as np
import lip   # tu librería con las funciones implementadas

if __name__ == "__main__":

    img = cv2.imread("img_data/bosque_lago.jpg")

    if img is None:
        print("Error")
        exit(1)

    img_name = "bosque_lago"

    new_w = img.shape[1] * 2
    new_h = img.shape[0] * 2

    print(f"new size: {new_w}x{new_h}")
    
    nn   = lip.resize_nearest_neighbor(img, new_w, new_h)
    cv2.imwrite(f"img_data/resize/{img_name}_resize_nn.jpg", nn)

    bil  = lip.resize_bilinear(img, new_w, new_h)
    cv2.imwrite(f"img_data/resize/{img_name}_resize_bil.jpg", bil)    

    bic  = lip.resize_bicubic(img, new_w, new_h)
    cv2.imwrite(f"img_data/resize/{img_name}_resize_bic.jpg", bic)

    lanc = lip.resize_lanczos_fast(img, new_w, new_h, a=20)
    cv2.imwrite(f"img_data/resize/{img_name}_resize_lanc.jpg", lanc)


