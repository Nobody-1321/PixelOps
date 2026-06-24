from pixelops.io import imread
from pixelops.visualization import show_image
import matplotlib.pyplot as plt
from pixelops.filtering import fourier_spectra

img_gray = imread("./data/img/desert.jpg", mode="gray")
magnitude_spectrum, phase_spectrum = fourier_spectra(img_gray)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
show_image(ax[0], magnitude_spectrum, title="Magnitude Spectrum")
show_image(ax[1], phase_spectrum, title="Phase Spectrum")
plt.show()


