#!/usr/bin/env python3
"""Run FrameLite on one image or a directory, with one greedy answer per image."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="An image file or image directory.")
    parser.add_argument("--weights", type=Path, default=Path(__file__).parent / "weights/framelite")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="Reference: bfloat16 on CUDA. float32 is available for CPU/debug use.")
    parser.add_argument("--recursive", action="store_true", help="Include images in subdirectories.")
    parser.add_argument("--no-save-crops", action="store_true", help="Write predictions without crop PNGs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing predictions.jsonl and generated crops.")
    return parser.parse_args()


def collect_images(source: Path, recursive: bool) -> tuple[list[Path], Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        return [source], source.parent
    if not source.is_dir():
        raise ValueError(f"Input does not exist: {source}")
    candidates = source.rglob("*") if recursive else source.iterdir()
    paths = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images found in: {source}")
    return paths, source


def main() -> int:
    args = parse_args()
    try:
        images, source_root = collect_images(args.image, args.recursive)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output_dir = args.output_dir.expanduser().resolve()
    if any(output_dir == image or output_dir in image.parents for image in images):
        print("Choose an output directory that does not already contain input images.", file=sys.stderr)
        return 2
    prediction_path = output_dir / "predictions.jsonl"
    if prediction_path.exists() and not args.overwrite:
        print(f"{prediction_path} already exists; use a new output directory or --overwrite.", file=sys.stderr)
        return 2

    # Keep --help usable before the GPU environment is installed.
    from PIL import Image
    from framelite import FrameLite

    model = FrameLite.from_pretrained(args.weights.expanduser(), device=args.device, dtype=args.dtype)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_count = 0
    execution_failures = 0
    with prediction_path.open("w", encoding="utf-8") as output:
        for index, image_path in enumerate(images, 1):
            relative = image_path.relative_to(source_root)
            image = None
            try:
                # Match the model's decoded RGB canvas; do not silently apply EXIF rotation.
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                result = model.predict(image)
                result["image"] = relative.as_posix()
                result["crop_path"] = None
                if result["valid"]:
                    if not args.no_save_crops:
                        x1, y1, x2, y2 = result["bbox_pixels"]
                        bounds = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
                        crop_path = output_dir / "crops" / relative.parent / (relative.name + ".png")
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        image.crop(bounds).save(crop_path, format="PNG")
                        result["crop_path"] = crop_path.relative_to(output_dir).as_posix()
                    valid_count += 1
            except (OSError, ValueError) as exc:
                execution_failures += 1
                result = {"image": relative.as_posix(), "valid": False,
                          "error": f"{type(exc).__name__}: {exc}", "raw_text": None,
                          "bbox_normalized": None, "bbox_pixels": None,
                          "image_size": None, "crop_path": None}
            finally:
                if image is not None:
                    image.close()
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            print(f"[{index}/{len(images)}] {relative}: {'valid' if result['valid'] else result['error']}")
    print(f"Saved {prediction_path} ({valid_count}/{len(images)} valid crops).")
    return 1 if execution_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
