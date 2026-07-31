from pixelops.edges import coherent_line_drawing
import matplotlib.pyplot as plt
from pixelops.io import imread
from pixelops.visualization import show_image

# Load an RGB image from the existing data directory[cite: 6]
img_rgb = imread("./data/img/media_NO.jpg", mode="rgb")

if img_rgb is None:
    raise FileNotFoundError("Image not found. Please check the path.")

# Apply the Coherent Line Drawing filter
# Using the parameters optimized for this algorithm
out_lines = coherent_line_drawing(
    img_rgb, 
    etf_r=3, 
    fdog_iter=2, 
    sigma_m=1.9, 
    sigma_c=0.6, 
    rho=0.99, 
    tau=0.9
)

# Plot the original image and the line drawing side by side[cite: 6]
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img_rgb, title="Original")
show_image(ax[1], out_lines, title="Coherent Line Drawing")
plt.show()

# If you also want to test it on a grayscale image directly[cite: 6]
img_gray = imread("./data/img/media_NO.jpg", mode="gray")

out_lines_gray = coherent_line_drawing(
    img_gray, 
    etf_r=5, 
    fdog_iter=3, 
    sigma_m=3.0, 
    sigma_c=1.0, 
    rho=0.99, 
    tau=0.5
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
show_image(ax[0], img_gray, title="Original Grayscale")
show_image(ax[1], out_lines_gray, title="Coherent Line Drawing (Standard Params)")
plt.show()