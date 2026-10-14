# Inference Details

[Back to README](../README.md)

## Model Loading

`FrameLite.from_pretrained` reads a local bundle containing the Qwen3.5-0.8B base model, tokenizer and processor, the FrameLite task weights, and `inference_config.json`. It does not automatically download a model from Hugging Face. No API key is required for local inference.

Download the FrameLite task weights from [zlab-sjtu/FrameLite](https://huggingface.co/zlab-sjtu/FrameLite) and the backbone separately from [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), following the two download commands in the README.

The task weights contain the trained vision merger and unmerged language LoRA. They are not a complete Hugging Face model or a standard PEFT adapter that can be loaded independently. Keep the bundle structure described in the [README](../README.md#pretrained-model).

## Inference Settings

The reference setting is CUDA BF16 with the versions pinned in `requirements.txt`. The processor preserves image aspect ratio, subject to its native size rounding, with a pixel budget of 65,536–589,824. The model uses a fixed prompt, thinking disabled, and greedy decoding with at most 32 new tokens. Directory inference processes one image at a time.

For CPU debugging, use `--device cpu --dtype float32`. This changes the numerical setting. Different precision, kernels, or dependency versions can change autoregressive outputs; keep the reference settings when comparing with the paper.

## Outputs and Coordinates

The CLI writes `predictions.jsonl` and, unless `--no-save-crops` is set, valid crops under `crops/`. Records include the source image, original dimensions, raw answer, validity, normalized coordinates, and pixel coordinates. Directory inputs retain relative paths. Existing output files are protected unless `--overwrite` is specified.

Coordinates follow `[x1, y1, x2, y2]` on the original decoded RGB image. The top-left corner is the origin. Valid boxes satisfy `0 ≤ x1 < x2 ≤ 1` and `0 ≤ y1 < y2 ≤ 1`. Multiply x coordinates by image width and y coordinates by image height to obtain pixel coordinates.

EXIF orientation is not applied automatically. Pixel coordinates remain floating point in the prediction record. PNG export uses floor for top-left edges and ceil for bottom-right edges with half-open pixel bounds. This rounding only affects the saved image, not the recorded prediction.

The parser expects exactly one JSON object with one `crop` field and four finite, legal coordinates. Extra text or fields, multiple boxes, malformed JSON, reversed edges, and out-of-bounds boxes are rejected. Invalid outputs are reported rather than repaired or replaced with a full-image crop. The raw answer is retained for inspection.

## Reported Evaluation

The README table reproduces the paper's results, not a new evaluation of this package. The final FrameLite checkpoint uses one greedy crop per image on all three benchmarks. GAICv2 uses 500 test images, FLMS uses 500 images with the corrected references, and FCDB uses its 345-image official test set. FLMS uses the maximum IoU over multiple references. No reference aspect ratio or dataset-specific checkpoint selection is supplied at inference.

Development used public benchmark feedback; the reported scores are not an independent unseen-test confirmation. The complete evaluation protocol is described in the paper.

## Existing Implementation Checks

The initial standalone implementation was tested offline with the pinned dependencies on an RTX 4090. On three fixed training images, its generated token sequences matched the original inference implementation. This was a portability check, not a new benchmark evaluation.

To run the focused parser and coordinate checks:

```bash
python -m unittest discover -s tests -v
```
