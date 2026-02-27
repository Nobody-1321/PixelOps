import pixelops as pix
from pixelops.color import reinhard_color_transfer_controlled, reinhard_color_transfer
import cv2
import numpy as np

img_src = pix.open_image("./data/img/scarlettNO.webp", mode = 'bgr')
img_tgt = pix.open_image("./data/img/sunNO.jpg", mode = 'bgr')


out_1 = reinhard_color_transfer_controlled(img_src, img_tgt, alpha_L=0.0, alpha_ab=0.5)
out_2 = reinhard_color_transfer(img_src, img_tgt)

pix.show_side_by_side(out_1, out_2, 'Reinhard Controlled', 'Reinhard Classic')
