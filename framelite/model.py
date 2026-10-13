"""Standalone inference for the fixed FrameLite QOS1 → GRO1 model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .geometry import InvalidCropError, load_image, parse_crop, to_pixel_box

if TYPE_CHECKING:
    from PIL import Image

DEFAULT_PROMPT = (
    "Select one aesthetically pleasing crop for this image. "
    "Return only the requested short structured output."
)

LORA_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _generation_prompt(processor: Any, prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _lora_targets(model: Any, suffixes: tuple[str, ...]) -> list[str]:
    targets = [
        name for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and name.rsplit(".", 1)[-1] in suffixes
        and module.__class__.__name__ == "Linear"
    ]
    missing = set(suffixes) - {name.rsplit(".", 1)[-1] for name in targets}
    if not targets or missing:
        raise RuntimeError(f"incompatible Qwen language modules; missing families: {sorted(missing)}")
    return targets


def _configure_merger(merger: Any, storage_dtype: str) -> None:
    import torch

    if storage_dtype != "float32":
        raise ValueError("the released FrameLite checkpoint requires float32 merger storage")
    # Cast BEFORE loading: copying through BF16 would discard learned updates.
    merger.float()
    original_forward = merger.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        inputs = args[0] if args else kwargs["x"]
        with torch.autocast(
            inputs.device.type, dtype=torch.bfloat16,
            enabled=inputs.dtype == torch.bfloat16,
        ):
            return original_forward(*args, **kwargs)

    merger.forward = forward


class FrameLite:
    """One original image → one greedy normalized crop, with no search or repair.

    Use ``from_pretrained`` with the released local weights directory. The
    base model, language LoRA and full vision merger remain separate so loading
    preserves the numerical behavior of the reported model.
    """

    def __init__(self, model: Any, processor: Any, config: dict[str, Any], device: str) -> None:
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path = "weights/framelite",
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ) -> FrameLite:
        """Load local base weights and the exact FrameLite LoRA/merger tensors.

        The reported setting is CUDA BF16 with SDPA. ``dtype='float32'`` is
        available for CPU/debug use and is a different numerical setting.
        This method does not download weights or merge LoRA into the backbone.
        """
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        directory = Path(path).expanduser().resolve()
        config = json.loads((directory / "inference_config.json").read_text(encoding="utf-8"))
        if config.get("format") != "framelite-inference/v1":
            raise ValueError("unsupported FrameLite inference configuration")
        if dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; install a compatible PyTorch CUDA build")
        base_path = directory / config["base_model"]
        if not base_path.is_dir():
            raise FileNotFoundError(f"local Qwen base weights are missing: {base_path}")
        processor = AutoProcessor.from_pretrained(
            base_path, local_files_only=True,
            min_pixels=int(config["image_processing"]["min_pixels"]),
            max_pixels=int(config["image_processing"]["max_pixels"]),
        )
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_path, local_files_only=True, dtype=torch_dtype,
            attn_implementation=config["model"]["attn_implementation"],
        ).to(device)
        lora = config["lora"]
        targets = _lora_targets(base, tuple(lora["target_suffixes"]))
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        model = get_peft_model(base, LoraConfig(
            r=int(lora["rank"]), lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]), bias="none",
            task_type=TaskType.CAUSAL_LM, target_modules=targets,
        ))
        _configure_merger(model.get_base_model().model.visual.merger, config["merger_storage_dtype"])
        state = load_file(str(directory / config["adapter_file"]), device="cpu")
        parameters = {
            name: parameter for name, parameter in model.named_parameters()
            if "lora_A" in name or "lora_B" in name or ".visual.merger." in name
        }
        if set(state) != set(parameters):
            missing = sorted(set(parameters) - set(state))[:5]
            unexpected = sorted(set(state) - set(parameters))[:5]
            raise RuntimeError(f"incompatible FrameLite weights; missing={missing}, unexpected={unexpected}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name]
                if value.shape != parameter.shape:
                    raise RuntimeError(f"incompatible FrameLite tensor shape: {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        del state
        model.requires_grad_(False)
        model.gradient_checkpointing_disable()
        model.eval()
        return cls(model, processor, config, str(device))

    def predict(self, image: str | Path | Image.Image) -> dict[str, Any]:
        """Predict one image using the fixed singleton greedy protocol.

        ``bbox_pixels`` contains floating-point coordinates on the decoded
        original image. Invalid answers return null boxes and an error string;
        no full-image fallback, clamping or coordinate swapping is applied.
        """
        import torch

        rgb = load_image(image)
        width, height = rgb.size
        try:
            text = _generation_prompt(self.processor, self.config["prompt"])
            tokenizer = self.processor.tokenizer
            previous_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            try:
                encoded = dict(self.processor(
                    text=[text], images=[rgb], padding=True, truncation=False, return_tensors="pt"
                ))
            finally:
                tokenizer.padding_side = previous_padding_side
            encoded = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            prompt_width = int(encoded["input_ids"].shape[1])
            generation = self.config["generation"]
            self.model.eval()
            with torch.inference_mode():
                sequences = self.model.generate(
                    **encoded,
                    max_new_tokens=int(generation["max_new_tokens"]),
                    do_sample=bool(generation["do_sample"]),
                    num_beams=int(generation["num_beams"]),
                    num_return_sequences=int(generation["num_return_sequences"]),
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            rgb.close()

        completion = sequences[0, prompt_width:]
        eos = torch.nonzero(completion == int(tokenizer.eos_token_id), as_tuple=False).flatten()
        eos_generated = bool(eos.numel())
        if eos_generated:
            completion = completion[:int(eos[0]) + 1]
        else:
            non_pad = torch.nonzero(completion != int(tokenizer.pad_token_id), as_tuple=False).flatten()
            completion = completion[:int(non_pad[-1]) + 1 if non_pad.numel() else 0]
        raw_text = self.processor.batch_decode(
            [completion], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        error = None
        try:
            bbox = parse_crop(raw_text)
        except InvalidCropError as invalid:
            bbox = None
            error = str(invalid)
        return {
            "raw_text": raw_text,
            "completion_token_ids": [int(item) for item in completion.tolist()],
            "bbox_normalized": list(bbox) if bbox is not None else None,
            "bbox_pixels": list(to_pixel_box(bbox, (width, height))) if bbox is not None else None,
            "valid": bbox is not None,
            "error": error,
            "image_size": [width, height],
            "eos_generated": eos_generated,
        }
