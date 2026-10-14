<div align="center">

# FrameLite: Decision-Oriented Aesthetic Cropping<br>with a Compact Vision–Language Model

**Jianze Li**<sup>1</sup>, **Kai Liu**<sup>1</sup>, **Ziqing Zhang**<sup>1</sup>, **Weihang Zhang**<sup>2</sup>,<br>
**Yong Guo**<sup>1</sup>, **Linghe Kong**<sup>1</sup>, **Yulun Zhang**<sup>1,†</sup>

<sup>1</sup> Shanghai Jiao Tong University<br>
<sup>2</sup> Institute of Media Technology and Experience Design, Huawei<br>
<sup>†</sup> Corresponding author

[![Paper](https://img.shields.io/badge/Paper-526878?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.17639)
[![Supplementary Material](https://img.shields.io/badge/Supplementary_Material-697d6c?style=flat-square)](https://github.com/JianzeLi-114/FrameLite/releases/download/v1.0/FrameLite_supplementary.pdf)
[![Hugging Face Models](https://img.shields.io/badge/Hugging_Face-Models-d7ab45?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/mon3tr/FrameLite)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-7a7a7a?style=flat-square)](LICENSE)

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=JianzeLi-114/FrameLite)
[![GitHub Stars](https://img.shields.io/github/stars/JianzeLi-114/FrameLite?style=social&logo=github&label=Stars)](https://github.com/JianzeLi-114/FrameLite/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/JianzeLi-114/FrameLite?style=social&logo=github&label=Forks)](https://github.com/JianzeLi-114/FrameLite/forks)
[![HF downloads/month](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fmon3tr%2FFrameLite&query=%24.downloads&label=HF+downloads%2Fmonth&color=FFD21E&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/mon3tr/FrameLite)

[Quick Start](#quick-start) · [中文](README_zh.md)

</div>

<table>
  <tr>
    <td align="center" width="33%"><img src="assets/examples/curved_architecture/input.jpg" height="180" alt="Input: curved architecture"></td>
    <td align="center" width="33%"><img src="assets/examples/mountain_reflection/input.jpg" height="180" alt="Input: mountain reflection"></td>
    <td align="center" width="33%"><img src="assets/examples/arcade_pedestrian/input.jpg" height="180" alt="Input: pedestrian in an arcade"></td>
  </tr>
  <tr>
    <td align="center" width="33%"><img src="assets/examples/curved_architecture/framelite.jpg" height="180" alt="FrameLite crop: curved architecture"></td>
    <td align="center" width="33%"><img src="assets/examples/mountain_reflection/framelite.jpg" height="180" alt="FrameLite crop: mountain reflection"></td>
    <td align="center" width="33%"><img src="assets/examples/arcade_pedestrian/framelite.jpg" height="180" alt="FrameLite crop: pedestrian in an arcade"></td>
  </tr>
</table>
<p align="center"><em>Examples from the paper. Top: input images. Bottom: crops predicted by FrameLite.</em></p>

## Abstract

> Aesthetic image cropping improves composition by balancing subjects, context, and image boundaries. Large vision–language models bring semantic knowledge to this task, but their scale limits lightweight photography assistance. We propose **FrameLite**, a compact cropping model built on Qwen3.5-0.8B. Our approach is motivated by the observation that a small model can generate better crops than its first answer suggests. We develop three-stage training to turn this potential into stronger direct predictions. Spatial and composition training establishes localization and framing preferences. Quality-Guided Online Supervision (QOS) then learns the highest-quality complete answers among the current policy's generated candidates. Greedy-Referenced Optimization (GRO) further refines the policy using positive and negative quality differences relative to its current greedy answer. Both online stages use annotation-derived feedback, while inference returns one coordinate answer without search or an external scorer. Experiments show that FrameLite outperforms the compared large-VLM croppers across public benchmarks with a substantially smaller backbone. Controlled studies confirm that selecting high-quality training answers improves the final model beyond online sampling alone.

## Method

<p align="center">
  <img src="assets/method_overview.png" width="100%" alt="FrameLite's three-stage training: spatial and composition training, Quality-Guided Online Supervision, and Greedy-Referenced Optimization">
</p>

FrameLite strengthens the **same compact cropping policy** through three stages:

1. **Stage I — Spatial and composition training.** Learn where important content lies, how alternative frames compare, and how to predict complete reference frames.
2. **Stage II — Quality-Guided Online Supervision (QOS).** Generate greedy and sampled answers, evaluate their complete-frame quality, and imitate the best answer.
3. **Stage III — Greedy-Referenced Optimization (GRO).** Refine fresh samples using signed quality differences relative to the current greedy answer.

At inference, the model directly predicts normalized crop coordinates with one greedy generation. It needs no candidate search, external scorer, retrieval database, or external API.

<details>
<summary><strong>QOS and GRO in detail</strong></summary>
<br>
<img src="assets/online_training.png" width="100%" alt="QOS selects the best generated crop for imitation; GRO refines sampled crops relative to the greedy crop">

QOS turns better generated candidates into learning targets. GRO then uses both positive and negative quality differences to refine the policy. Human crop annotations provide training feedback; they are not required at inference.

</details>

## Pretrained Model

| Model | Backbone | Checkpoint | Download |
|:---|:---|:---|:---|
| FrameLite | Qwen3.5-0.8B | Final model after Stage I → QOS → GRO | [Hugging Face](https://huggingface.co/mon3tr/FrameLite) |

The FrameLite task weights (**224 MB**) contain the trained language LoRA and vision merger. Download them from [Hugging Face](https://huggingface.co/mon3tr/FrameLite), and the backbone from [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B). The downloads use this local layout:

```text
weights/framelite/
├── base/                    # Downloaded separately from Qwen/Qwen3.5-0.8B
├── framelite.safetensors    # FrameLite language LoRA and trained vision merger
└── inference_config.json
```

Download both components before loading the model with the `FrameLite` API.

## Quick Start

### Installation

Use **Python 3.10** and a CUDA GPU with BF16 support:

```bash
git clone https://github.com/JianzeLi-114/FrameLite.git
cd FrameLite

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

### Download the Model

From the repository root, download the FrameLite task weights and the Qwen backbone:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mon3tr/FrameLite",
    local_dir="weights/framelite",
    allow_patterns=[
        "framelite.safetensors", "inference_config.json", "LICENSE", "NOTICE",
    ],
)

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    local_dir="weights/framelite/base",
)
PY
```

### Crop an Image

Try one of the images included in the repository:

```bash
python infer.py \
  --image assets/examples/curved_architecture/input.jpg \
  --output-dir outputs \
  --device cuda:0
```

To process a directory, pass its path to `--image`:

```bash
python infer.py --image path/to/images --output-dir outputs --device cuda:0
```

Predicted coordinates are saved in `outputs/predictions.jsonl`, and valid crops in `outputs/crops/`. Add `--recursive` to include subdirectories or `--no-save-crops` to save coordinates only. Use `python infer.py --help` for all options.

### Python API

```python
from framelite import FrameLite, crop_image

model = FrameLite.from_pretrained("weights/framelite", device="cuda:0")
image_path = "assets/examples/curved_architecture/input.jpg"
result = model.predict(image_path)

if result["valid"]:
    print(result["bbox_normalized"])  # [x1, y1, x2, y2] in [0, 1]
    print(result["bbox_pixels"])      # Coordinates on the original image
    crop_image(image_path, result["bbox_normalized"]).save("crop.png")
else:
    print(result["error"])
```

The output contains normalized `[x1, y1, x2, y2]` crop coordinates on the original image, with the origin at the top left.

See [Inference Details](docs/inference.md) for output fields, coordinate conventions, and runtime notes.

## Results

Selected comparisons from the paper, evaluated with one predicted crop per image. FrameLite uses the **same final model across all three benchmarks**.

| Method | GAICv2 Acc .85 ↑ | GAICv2 Acc .90 ↑ | FLMS IoU ↑ | FLMS BDE ↓ | FCDB IoU ↑ | FCDB BDE ↓ |
|:---|---:|---:|---:|---:|---:|---:|
| GAICD | 0.8300 | 0.7640 | 0.8025 | 0.0470 | 0.6759 | 0.0819 |
| CACNet | 0.5100 | 0.2360 | 0.8401 | 0.0368 | 0.6967 | 0.0751 |
| Cropper | 0.6900 | 0.6460 | 0.7845 | 0.0829 | 0.6519 | 0.1089 |
| InstructCrop | 0.4740 | 0.1680 | 0.8145 | 0.0422 | 0.6848 | 0.0774 |
| Venus | 0.8380 | 0.7200 | 0.8638 | 0.0304 | 0.6836 | 0.0794 |
| **FrameLite** | **0.9140** | **0.8340** | **0.8679** | **0.0297** | **0.7124** | **0.0704** |

Acc denotes GAICv2 $\mathrm{Acc}_{1/5}$ at the indicated IoU threshold. Higher Acc and IoU and lower BDE are better. See the [paper](https://arxiv.org/abs/2610.17639) for the complete comparison and evaluation protocol, and the [supplementary material](https://github.com/JianzeLi-114/FrameLite/releases/download/v1.0/FrameLite_supplementary.pdf) for additional ablations, the user study, and qualitative comparisons.

## Citation

If you find FrameLite useful, please cite our work:

```bibtex
@misc{li2026framelite,
  title={FrameLite: Decision-Oriented Aesthetic Cropping with a Compact Vision--Language Model},
  author={Li, Jianze and Liu, Kai and Zhang, Ziqing and Zhang, Weihang and Guo, Yong and Kong, Linghe and Zhang, Yulun},
  year={2026},
  eprint={2610.17639},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.17639}
}
```

## License and Acknowledgments

FrameLite code and task weights use the [Apache License 2.0](LICENSE). The separately downloaded Qwen3.5-0.8B backbone retains its upstream [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE); see [NOTICE](NOTICE).

We thank the developers of [Qwen](https://github.com/QwenLM), [Transformers](https://github.com/huggingface/transformers), and [PEFT](https://github.com/huggingface/peft).

For questions, please open an issue. Correspondence: [Yulun Zhang](mailto:yulun100@gmail.com).
