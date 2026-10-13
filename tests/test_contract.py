"""Focused tests for the released parser and image-coordinate convention."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from framelite.geometry import (
    InvalidCropError, crop_image, load_image, parse_crop, raster_crop_box, to_pixel_box,
)
from framelite.model import DEFAULT_PROMPT, _generation_prompt


class ContractTests(unittest.TestCase):
    def test_valid_numeric_crop_preserves_precision(self):
        self.assertEqual(parse_crop('{"crop":[0.12345,0,0.9,1]}'), (0.12345, 0.0, 0.9, 1.0))

    def test_invalid_answers_are_not_repaired(self):
        invalid = [
            ' {"crop":[0,0,1,1]}', '{"crop":[0,0,1,1]}\n',
            '```json\n{"crop":[0,0,1,1]}\n```',
            '{"crop":[0,0,1,1],"bbox":[0,0,1,1]}',
            '{"crop":[0,0,1,1],"crop":[0,0,1,1]}',
            '{"crop":[-0.01,0,1,1]}', '{"crop":[0.8,0,0.2,1]}',
            '{"crop":[0,0,0,1]}', '{"crop":[false,0,1,1]}',
            '{"crop":["0",0,1,1]}', '{"crop":[NaN,0,1,1]}',
            '{"crop":[0,0,Infinity,1]}', '{"crop":[0,0,1,1]',
            '{"bbox":[0,0,1,1]}', '{"crop":[[0,0,1,1],[0,0,1,1]]}',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(InvalidCropError):
                parse_crop(text)

    def test_original_coordinates_and_rendering_rounding(self):
        box = (0.125, 0.25, 0.875, 0.75)
        self.assertEqual(to_pixel_box(box, (101, 51)), (12.625, 12.75, 88.375, 38.25))
        self.assertEqual(raster_crop_box(box, (101, 51)), (12, 12, 89, 39))
        with Image.new("RGB", (101, 51)) as image:
            with crop_image(image, box) as crop:
                self.assertEqual(crop.size, (77, 27))
            self.assertEqual(image.size, (101, 51))

    def test_exif_orientation_does_not_rotate_the_evaluation_canvas(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oriented.jpg"
            with Image.new("RGB", (19, 11)) as image:
                exif = Image.Exif()
                exif[274] = 6
                image.save(path, exif=exif)
            with load_image(path) as loaded:
                self.assertEqual(loaded.size, (19, 11))
                self.assertEqual(loaded.mode, "RGB")

    def test_exact_chat_prompt_has_thinking_disabled(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "rendered"

        processor = Processor()
        self.assertEqual(_generation_prompt(processor, DEFAULT_PROMPT), "rendered")
        self.assertEqual(processor.messages, [{
            "role": "user", "content": [{"type": "image"}, {"type": "text", "text":
                "Select one aesthetically pleasing crop for this image. "
                "Return only the requested short structured output."}],
        }])
        self.assertEqual(processor.kwargs, {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        })


if __name__ == "__main__":
    unittest.main()
