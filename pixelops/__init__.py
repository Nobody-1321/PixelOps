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
from . import quantization

# Solo las 5-10 funciones MÁS usadas
from .core import open_image, normalize_to_uint8
from .visualization import show_side_by_side, show_images

__all__ = [
    
    # Submódulos
    'core', 
    'histogram', 
    'filtering', 
    'edges', 
    'segmentation', 
    'enhancement', 
    'visualization',
    'quantization',
    
    # Solo funciones críticas
    'open_image',
    'show_side_by_side',
    'show_images',
    'normalize_to_uint8',
]