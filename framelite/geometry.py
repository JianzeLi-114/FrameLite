"""The released one-box output contract and original-image coordinate system."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from PIL import Image

Box = tuple[float, float, float, float]


class InvalidCropError(ValueError):
    """The generated answer is not a legal, single normalized crop."""


def _reject_constant(value: str) -> None:
    raise InvalidCropError(f"non-standard JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidCropError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def validate_box(value: object) -> Box:
    """Check normalized xyxy without clipping, reordering, or quantization."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise InvalidCropError("crop must contain exactly 4 numeric values")
    coordinates = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise InvalidCropError("crop values must be numeric")
        number = float(coordinate)
        if not math.isfinite(number):
            raise InvalidCropError("crop values must be finite")
        coordinates.append(number)
    x1, y1, x2, y2 = coordinates
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise InvalidCropError("crop must be a legal normalized xyxy box")
    return x1, y1, x2, y2


def parse_crop(text: str) -> Box:
    """Parse the exact crop-only JSON contract used for reported predictions.

    Leading/trailing whitespace, Markdown, extra fields, duplicate keys and
    incomplete answers are invalid. Decimal places are not imposed by parsing.
    """
    if not isinstance(text, str) or not text or text != text.strip():
        raise InvalidCropError("output must be non-empty JSON without outer whitespace")
    try:
        value = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except InvalidCropError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise InvalidCropError(f"invalid JSON output: {error}") from error
    if not isinstance(value, dict):
        raise InvalidCropError("output must be one JSON object")
    if tuple(value) != ("crop",):
        raise InvalidCropError("crop JSON requires exactly one crop field")
    return validate_box(value["crop"])


def to_pixel_box(box: Sequence[float], image_size: tuple[int, int]) -> Box:
    """Map normalized coordinates to floating-point original-image pixels."""
    x1, y1, x2, y2 = validate_box(list(box))
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return x1 * width, y1 * height, x2 * width, y2 * height


def raster_crop_box(
    box: Sequence[float], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Rendering only: floor the upper-left and ceil the lower-right corner.

    The half-open pixel crop preserves the unrounded prediction returned by
    ``FrameLite.predict``; this operation is never fed back to the model.
    """
    x1, y1, x2, y2 = to_pixel_box(box, image_size)
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def load_image(image: str | Path | Image.Image) -> Image.Image:
    """Return an independent RGB image on the original decoded pixel canvas.

    The benchmark protocol uses PIL's ``convert('RGB')`` without applying EXIF
    orientation. File inputs and PIL inputs follow that same convention.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.convert("RGB")
    raise TypeError("image must be a file path or PIL.Image.Image")


def crop_image(image: str | Path | Image.Image, box: Sequence[float]) -> Image.Image:
    """Render a valid predicted crop at its native aspect ratio and resolution."""
    original = load_image(image)
    try:
        return original.crop(raster_crop_box(box, original.size))
    finally:
        original.close()
