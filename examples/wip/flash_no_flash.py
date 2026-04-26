import cv2
import numpy as np

def bilateral_filter(image, sigma_d=1, sigma_r=0.1):
    """
    Applies a bilateral filter to an image.

    Parameters:
        image (numpy array): Input image.
        sigma_d (float): Spatial sigma (controls the range of spatial smoothing).
        sigma_r (float): Range sigma (controls the range of intensity smoothing).

    Returns:
        numpy array: Filtered image.
    """
    return cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_r * 255, sigmaSpace=sigma_d)

def joint_bilateral_filter(ambient, flash, sigma_d=15, sigma_r=0.1):
    """
    Applies a joint bilateral filter using the flash image as a guide.

    Parameters:
        ambient (numpy array): Ambient image (to be filtered).
        flash (numpy array): Flash image (used as a guide).
        sigma_d (float): Spatial sigma (controls the range of spatial smoothing).
        sigma_r (float): Range sigma (controls the range of intensity smoothing).

    Returns:
        numpy array: Filtered ambient image.
    """
    import cv2.ximgproc  # Ensure opencv-contrib-python is installed

    ambient_uint8 = ambient.astype(np.uint8)
    flash_uint8 = flash.astype(np.uint8)

    filtered = np.zeros_like(ambient_uint8)
    
    for i in range(3):  # Process each channel (BGR)
        filtered[..., i] = cv2.ximgproc.jointBilateralFilter(
            joint=flash_uint8[..., i],    # Guide image (flash)
            src=ambient_uint8[..., i],    # Image to be filtered (ambient)
            d=-1,
            sigmaColor=sigma_r * 255,
            sigmaSpace=sigma_d
        )

    return filtered.astype(np.float32)

def compute_detail_layer(flash, sigma_d=10, sigma_r=0.7, epsilon=0.5):
    """
    Computes the detail layer of the flash image.

    Parameters:
        flash (numpy array): Flash image.
        sigma_d (float): Spatial sigma for the bilateral filter.
        sigma_r (float): Range sigma for the bilateral filter.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        numpy array: Detail layer.
    """
    base = bilateral_filter(flash, sigma_d, sigma_r)
    detail = (flash + epsilon) / (base + epsilon)
    return detail

def apply_masked_merge(a, b, mask):
    """
    Merges two images using a mask.

    Parameters:
        a (numpy array): First image.
        b (numpy array): Second image.
        mask (numpy array): Mask to control blending.

    Returns:
        numpy array: Merged image.
    """
    return (1 - mask) * a + mask * b

def detect_flash_shadows(flash_lin, ambient_lin, tau=0.09):
    """
    Detects shadows caused by the flash.

    Parameters:
        flash_lin (numpy array): Flash image (linearized).
        ambient_lin (numpy array): Ambient image (linearized).
        tau (float): Threshold for shadow detection.

    Returns:
        numpy array: Shadow mask.
    """
    diff = flash_lin - ambient_lin
    shadow_mask = np.all(diff < tau, axis=2).astype(np.float32)
    return cv2.dilate(shadow_mask, None)

def detect_flash_specularities(flash_lin, threshold=0.95):
    """
    Detects specular highlights caused by the flash.

    Parameters:
        flash_lin (numpy array): Flash image (linearized).
        threshold (float): Threshold for specular highlight detection.

    Returns:
        numpy array: Specular highlight mask.
    """
    luminance = 0.2126 * flash_lin[..., 2] + 0.7152 * flash_lin[..., 1] + 0.0722 * flash_lin[..., 0]
    specular_mask = (luminance >= threshold).astype(np.float32)
    return cv2.dilate(specular_mask, None)

def white_balance_by_ambient_color(ambient_lin, ambient_color):
    """
    Applies white balance to the ambient image using the ambient color.

    Parameters:
        ambient_lin (numpy array): Ambient image (linearized).
        ambient_color (numpy array): Ambient color.

    Returns:
        numpy array: White-balanced image.
    """
    wb_image = ambient_lin / (ambient_color[None, None, :] + 1e-6)
    return np.clip(wb_image, 0, 1)

def enhance_ambient_with_flash(ambient, flash):
    """
    Enhances the ambient image using the flash image.

    Parameters:
        ambient (numpy array): Ambient image.
        flash (numpy array): Flash image.

    Returns:
        numpy array: Enhanced ambient image.
    """
    ambient_lin = ambient.astype(np.float32) / 255
    flash_lin = flash.astype(np.float32) / 255
    
    # Compute ambient color and denoise
    denoised_ambient = joint_bilateral_filter(ambient, flash, sigma_d=10, sigma_r=0.2)

    # Compute detail layer
    detail_layer = compute_detail_layer(flash, sigma_d=30, sigma_r=0.9, epsilon=0.01)
    
    # Detect shadows and specular highlights
    specular_mask = detect_flash_specularities(flash_lin, threshold=0.95)
    mask = detect_flash_shadows(flash_lin, ambient_lin, tau=0.01)

    # Combine shadow and specular masks
    full_mask = np.clip(mask + specular_mask, 0, 1)
    full_mask = cv2.GaussianBlur(full_mask, (5, 5), 5)
    full_mask = np.repeat(full_mask[..., np.newaxis], 3, axis=2)

    # Final merge
    transferred = denoised_ambient * detail_layer
    final_image = apply_masked_merge(transferred, denoised_ambient, full_mask)

    return np.clip(final_image, 0, 255).astype(np.uint8)

# Load images
ambient = cv2.imread('img_data/flash_ambient/3_ambient.jpeg').astype(np.float32)
flash = cv2.imread('img_data/flash_ambient/3_flash.jpeg').astype(np.float32)

# Resize if not aligned
ambient = cv2.resize(ambient, (flash.shape[1], flash.shape[0]))

# Enhance ambient image
enhanced = enhance_ambient_with_flash(ambient, flash)

# Display and save the result
cv2.imshow('Enhanced Image', enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('enhanced_image.png', enhanced)
print("Enhanced image saved as 'enhanced_image.png'.")