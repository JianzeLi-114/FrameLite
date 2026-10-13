"""FrameLite: single-image aesthetic crop inference."""

from .geometry import InvalidCropError, crop_image, parse_crop, raster_crop_box, to_pixel_box
from .model import FrameLite

__version__ = "0.1.0"
__all__ = [
    "FrameLite", "InvalidCropError", "crop_image", "parse_crop", "raster_crop_box", "to_pixel_box"
]
