from pixelops.io import imread
from pixelops.visualization import show_image
from pixelops.core import rescale_to_uint8
import matplotlib.pyplot as plt
from pixelops.filtering import (
    gaussian_lowpass_mask,
    apply_frequency_filter,
    ideal_lowpass_mask,
    butterworth_lowpass_mask,
    lanczos_lowpass_mask
)

img = imread("./data/img/desert.jpg", mode="gray")
mask = gaussian_lowpass_mask(img.shape, cutoff_frequency=20)
filtered = apply_frequency_filter(img, mask)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
show_image(axes[0], img, title="Original Grayscale Image")
show_image(axes[1], rescale_to_uint8(filtered), title="Gaussian Lowpass Filtered Image")
plt.show() 

mask = ideal_lowpass_mask(img.shape, cutoff_frequency=20)
filtered = apply_frequency_filter(img, mask)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
show_image(axes[0], img, title="Original Grayscale Image")
show_image(axes[1], rescale_to_uint8(filtered), title="Ideal Lowpass Filtered Image")
plt.show()

mask = butterworth_lowpass_mask(img.shape, cutoff_frequency=20, order=2)
filtered = apply_frequency_filter(img, mask)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
show_image(axes[0], img, title="Original Grayscale Image")
show_image(axes[1], rescale_to_uint8(filtered), title="Butterworth Lowpass Filtered Image")
plt.show()

mask = lanczos_lowpass_mask(img.shape, cutoff_frequency=20, a=5)
filtered = apply_frequency_filter(img, mask)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
show_image(axes[0], img, title="Original Grayscale Image")
show_image(axes[1], rescale_to_uint8(filtered), title="Lanczos Lowpass Filtered Image")
plt.show()