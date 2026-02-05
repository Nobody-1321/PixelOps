import numpy as np
from numba import njit, prange


@njit
def floyd_steinberg_dithering(image: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg: difusión 7/16, 3/16, 5/16, 1/16"""
    img = image.astype(np.float32)
    h, w = img.shape

    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255.0 if old_pixel > 127.0 else 0.0
            img[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < w:
                img[y, x + 1] += error * 0.4375      # 7/16
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += error * 0.1875  # 3/16
                img[y + 1, x] += error * 0.3125          # 5/16
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 0.0625  # 1/16

    for y in range(h):
        for x in range(w):
            img[y, x] = min(max(img[y, x], 0.0), 255.0)

    return img.astype(np.uint8)

@njit
def atkinson_dithering(image: np.ndarray) -> np.ndarray:
    """Atkinson: difunde solo 6/8 del error (más contraste)"""
    img = image.astype(np.float32)
    h, w = img.shape

    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255.0 if old_pixel > 127.0 else 0.0
            img[y, x] = new_pixel
            error = (old_pixel - new_pixel) / 8.0

            if x + 1 < w:
                img[y, x + 1] += error
            if x + 2 < w:
                img[y, x + 2] += error
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += error
                img[y + 1, x] += error
                if x + 1 < w:
                    img[y + 1, x + 1] += error
            if y + 2 < h:
                img[y + 2, x] += error

    for y in range(h):
        for x in range(w):
            img[y, x] = min(max(img[y, x], 0.0), 255.0)

    return img.astype(np.uint8)

@njit(parallel=True)
def bayer_dithering(image: np.ndarray, matrix_size: int = 4) -> np.ndarray:
    """Ordered dithering con matriz Bayer (paralelizable)"""
    # Matriz Bayer 4x4 normalizada
    bayer_4x4 = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ], dtype=np.float32) / 16.0

    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)

    for y in prange(h):
        for x in range(w):
            threshold = bayer_4x4[y % 4, x % 4] * 255.0
            output[y, x] = 255 if image[y, x] > threshold else 0

    return output

@njit
def uniform_quantize(image: np.ndarray, levels: int = 8) -> np.ndarray:
    """Cuantización uniforme a N niveles"""
    step = 256.0 / levels
    output = np.zeros_like(image)
    h, w = image.shape

    for y in range(h):
        for x in range(w):
            output[y, x] = np.uint8((image[y, x] // step) * step + step / 2)

    return output

def floyd_steinberg_serpentine(image: np.ndarray) -> np.ndarray:
    """
    Floyd–Steinberg error diffusion with serpentine scanning.
    """

    img = image.astype(np.float32)
    h, w = img.shape

    for y in range(h):

        if y % 2 == 0:
            x_range = range(w)
            direction = 1
        else:
            x_range = range(w - 1, -1, -1)
            direction = -1

        for x in x_range:
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            img[y, x] = new_pixel

            error = old_pixel - new_pixel

            nx = x + direction
            if 0 <= nx < w:
                img[y, nx] += error * 7 / 16

            if y + 1 < h:
                img[y + 1, x] += error * 5 / 16

                nx = x - direction
                if 0 <= nx < w:
                    img[y + 1, nx] += error * 3 / 16

                nx = x + direction
                if 0 <= nx < w:
                    img[y + 1, nx] += error * 1 / 16

    return np.clip(img, 0, 255).astype(np.uint8)
