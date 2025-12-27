"""PixelOps - Image processing library."""

__version__ = '0.1.0'

# Solo submódulos - NO cargan todo al inicio
from . import core
from . import histogram
from . import filtering
from . import edges
from . import segmentation
from . import enhancement
from . import visualization

# Solo las 5-10 funciones MÁS usadas
from .core import open_image
from .visualization import show_side_by_side, show_images
from .filtering.spatial.gaussian import gaussian_filter_bgr, gaussian_filter_grayscale

from .filtering.spatial.gradiend import (
    compute_gaussian_image_gradient_vis,
    compute_sobel_image_gradient_vis,
    compute_log_image_vis,
)

__all__ = [
    
    # Submódulos
    'core', 'histogram', 'filtering', 'edges', 
    'segmentation', 'enhancement', 'visualization',
    
    # Solo funciones críticas
    'open_image',
    'show_side_by_side',
    'show_images',
]