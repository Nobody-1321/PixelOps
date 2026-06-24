from pixelops.io import imread
from pixelops.visualization import show_image
from pixelops.core import rescale_to_uint8
import matplotlib.pyplot as plt
from pixelops.filtering import (
    gaussian_highpass_mask,
    apply_frequency_filter,
    ideal_highpass_mask,
    butterworth_highpass_mask,
)

img = imread("./data/img/cafeterrasse-bei-Nacht.jpg", mode="gray")
mask = gaussian_highpass_mask(img.shape, cutoff_frequency=20)
filtered = apply_frequency_filter(img, mask)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
show_image(axes[0], img, title="Original Grayscale Image")
show_image(axes[1], rescale_to_uint8(filtered), title="Gaussian Highpass Filtered Image")
plt.show() 

mask = ideal_highpass_mask(img.shape, cutoff_frequency=20)
filtered = apply_frequency_filter(img, mask)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
show_image(axes[0], img, title="Original Grayscale Image")
show_image(axes[1], rescale_to_uint8(filtered), title="Ideal Highpass Filtered Image")
plt.show()

mask = butterworth_highpass_mask(img.shape, cutoff_frequency=20, order=2)
filtered = apply_frequency_filter(img, mask)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
show_image(axes[0], img, title="Original Grayscale Image")
show_image(axes[1], rescale_to_uint8(filtered), title="Butterworth Highpass Filtered Image")
plt.show()
