"""
Visualization utilities.

This module provides functions for displaying images using
Matplotlib and OpenCV backends.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def _imshow(ax, img: np.ndarray, title: str) -> None:
    """
    Helper to display an image on a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to display the image on.

    img : np.ndarray
        Image of shape (H, W) or (H, W, 3).

    title : str
        Title to display above the image.

    Raises
    ------
    TypeError
        If img is not a numpy array.
    ValueError
        If image format is not supported.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Image must be a numpy array.")

    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    elif img.ndim == 3 and img.shape[2] == 3:
        # BGR → RGB for matplotlib
        img_rgb = img[..., ::-1]
        ax.imshow(img_rgb)
    else:
        raise ValueError(
            "Unsupported image format. "
            "Expected H×W (grayscale) or H×W×3 (RGB)."
        )

    ax.set_title(title)
    ax.axis("off")

def show_side_by_side(
    img1,
    img2,
    title1="Image 1",
    title2="Image 2"
):
    """
    Display two images side by side using matplotlib.

    Supports both grayscale (H×W) and color images (H×W×3 in RGB).

    Parameters
    ----------
    img1 : np.ndarray
        First image (grayscale or RGB).
    img2 : np.ndarray
        Second image (grayscale or RGB).
    title1 : str, optional
        Title for the first image.
    title2 : str, optional
        Title for the second image.

    Raises
    ------
    ValueError
        If image dimensions are not supported.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    _imshow(axes[0], img1, title1)
    _imshow(axes[1], img2, title2)

    plt.tight_layout()
    plt.show()

def show_images(images, titles=None):
    """
    Display multiple images together in a single OpenCV window.

    This utility function arranges 2, 3, or 4 images into a single
    composite image and displays it using OpenCV. Grayscale images
    are automatically converted to BGR for visualization.

    Parameters
    ----------
    images : list of np.ndarray
        List of input images to display. The list must contain
        exactly 2, 3, or 4 images. Each image must have the same
        spatial dimensions. Images can be either grayscale
        (H, W) or color (H, W, 3).

    titles : list of str, optional
        List of titles corresponding to each image. If provided,
        its length should match the number of images. Titles are
        rendered on top of each image region.

    Raises
    ------
    ValueError
        If the number of images is not 2, 3, or 4.

    Notes
    -----
    - All images are assumed to have the same height and width.
    - Grayscale images are internally converted to BGR.
    - The layout is:
        * 2 images: 1 row × 2 columns
        * 3 images: 2 × 2 grid (bottom-right empty)
        * 4 images: 2 × 2 grid
    - This function is intended for visualization and debugging
      purposes only.
    """

    num_images = len(images)
    if num_images not in [2, 3, 4]:
        raise ValueError("The images list must contain 2, 3, or 4 images.")

    height, width = images[0].shape[:2]

    images = [
        cv.cvtColor(img, cv.COLOR_GRAY2BGR) if img.ndim == 2 else img
        for img in images
    ]

    if num_images == 2:
        combined_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
    else:
        combined_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

    combined_image[:height, :width] = images[0]
    combined_image[:height, width:width * 2] = images[1]

    if num_images > 2:
        combined_image[height:height * 2, :width] = images[2]
    if num_images == 4:
        combined_image[height:height * 2, width:width * 2] = images[3]

    if titles:
        for i, title in enumerate(titles):
            if i == 0:
                pos = (10, 30)
            elif i == 1:
                pos = (width + 10, 30)
            elif i == 2:
                pos = (10, height + 30)
            else:
                pos = (width + 10, height + 30)

            cv.putText(
                combined_image,
                title,
                pos,
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

    cv.imshow("Combined Image", combined_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
