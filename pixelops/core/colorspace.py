import cv2 as cv
import numpy as np

def rgb_to_lab(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    return cv.cvtColor(rgb, cv.COLOR_RGB2Lab)

def lab_to_rgb(lab):
    rgb = cv.cvtColor(lab, cv.COLOR_Lab2RGB)
    rgb = np.clip(rgb * 255.0, 0, 255)
    return rgb.astype(np.uint8)