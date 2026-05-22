"""
Thin API for global-upgrade plan — delegates to utils.image_processing.
"""

from utils.image_processing import preprocess_image, enhance_image, detect_edges

from preprocessing.enhancer import FingerprintEnhancer

__all__ = [
    "preprocess_image",
    "enhance_image",
    "detect_edges",
    "FingerprintEnhancer",
]
