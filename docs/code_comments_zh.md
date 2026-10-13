# FrameLite 代码注释中英对照

整理日期：2026-10-08。范围为独立推理包 `FrameLite/` 中的 Python 行注释、文档字符串，以及 `requirements.txt` 的注释，共 **19 项**；另说明 1 项脚本解释器声明。

按文件和当前起始行号定位。英文原文仅合并排版折行，保留段落和技术标识。模型提示词、命令行帮助、报错字符串、README 正文及底座附带的第三方文件不属于此次注释翻译范围。本文仅提供对照，未修改源代码。

## 1. `infer.py`：命令行推理入口

| 位置 | 英文原文 | 中文翻译 |
| --- | --- | --- |
| 第 2 行，模块文档字符串 | Run FrameLite on one image or a directory, with one greedy answer per image. | 对单张图片或图片目录运行 FrameLite，每张图片通过贪心解码生成一个答案。 |
| 第 58 行，行注释 | Keep --help usable before the GPU environment is installed. | 即使尚未安装 GPU 运行环境，也应能使用 `--help` 查看帮助。 |
| 第 71 行，行注释 | Match the model's decoded RGB canvas; do not silently apply EXIF rotation. | 与模型使用的解码后 RGB 画布保持一致；不要隐式应用 EXIF 方向旋转。 |

第 1 行的 `#!/usr/bin/env python3` 是解释器声明（shebang），不是需要翻译的自然语言注释。其含义是：通过 `env` 查找当前环境中的 `python3` 来运行脚本。

## 2. `framelite/__init__.py`：包入口

| 位置 | 英文原文 | 中文翻译 |
| --- | --- | --- |
| 第 1 行，模块文档字符串 | FrameLite: single-image aesthetic crop inference. | FrameLite：单图美学裁剪推理。 |

## 3. `framelite/model.py`：模型加载与预测

| 位置 | 英文原文 | 中文翻译 |
| --- | --- | --- |
| 第 1 行，模块文档字符串 | Standalone inference for the fixed FrameLite QOS1 → GRO1 model. | 为固定的 FrameLite QOS1 → GRO1 模型提供独立推理。 |
| 第 54 行，`_configure_merger` 内的行注释 | Cast BEFORE loading: copying through BF16 would discard learned updates. | 必须在加载权重之前转换数据类型：若先通过 BF16 进行拷贝，会丢失已学习到的参数更新。 |
| 第 70 行，`FrameLite` 类文档字符串 | One original image → one greedy normalized crop, with no search or repair.<br><br>Use ``from_pretrained`` with the released local weights directory. The base model, language LoRA and full vision merger remain separate so loading preserves the numerical behavior of the reported model. | 输入一张原图，通过贪心解码得到一个归一化裁剪框，不进行搜索或修框。<br><br>调用 `from_pretrained` 并传入发布的本地权重目录。底座模型、语言 LoRA 和完整视觉 merger 保持分离，使加载后的数值行为与报告中的模型一致。 |
| 第 91 行，`from_pretrained` 方法文档字符串 | Load local base weights and the exact FrameLite LoRA/merger tensors.<br><br>The reported setting is CUDA BF16 with SDPA. ``dtype='float32'`` is available for CPU/debug use and is a different numerical setting. This method does not download weights or merge LoRA into the backbone. | 加载本地底座权重及原样保存的 FrameLite LoRA/merger 张量。<br><br>报告采用的配置是在 CUDA 上使用 BF16 和 SDPA。`dtype='float32'` 可用于 CPU 运行或调试，但属于不同的数值设置。此方法不会下载权重，也不会将 LoRA 合并进主干模型。 |
| 第 155 行，`predict` 方法文档字符串 | Predict one image using the fixed singleton greedy protocol.<br><br>``bbox_pixels`` contains floating-point coordinates on the decoded original image. Invalid answers return null boxes and an error string; no full-image fallback, clamping or coordinate swapping is applied. | 使用固定的单图贪心解码协议进行预测。<br><br>`bbox_pixels` 保存解码后原图上的浮点坐标。非法答案返回空框值及错误字符串；不使用整图作为兜底，不将坐标截断到合法范围，也不交换坐标顺序。 |

## 4. `framelite/geometry.py`：输出解析与图像几何

| 位置 | 英文原文 | 中文翻译 |
| --- | --- | --- |
| 第 1 行，模块文档字符串 | The released one-box output contract and original-image coordinate system. | 发布版的单框输出格式约定与原图坐标系。 |
| 第 17 行，`InvalidCropError` 类文档字符串 | The generated answer is not a legal, single normalized crop. | 生成的回答不是合法的单个归一化裁剪框。 |
| 第 34 行，`validate_box` 函数文档字符串 | Check normalized xyxy without clipping, reordering, or quantization. | 检查归一化的 `xyxy` 坐标，不进行坐标截断、重排或量化。 |
| 第 52 行，`parse_crop` 函数文档字符串 | Parse the exact crop-only JSON contract used for reported predictions.<br><br>Leading/trailing whitespace, Markdown, extra fields, duplicate keys and incomplete answers are invalid. Decimal places are not imposed by parsing. | 严格按所报告预测结果采用的输出格式约定，解析仅包含 `crop` 字段的 JSON。<br><br>首尾空白、Markdown、额外字段、重复键和不完整回答均不合法。解析时不限定小数位数。 |
| 第 75 行，`to_pixel_box` 函数文档字符串 | Map normalized coordinates to floating-point original-image pixels. | 将归一化坐标映射为原图上的浮点像素坐标。 |
| 第 86 行，`raster_crop_box` 函数文档字符串 | Rendering only: floor the upper-left and ceil the lower-right corner.<br><br>The half-open pixel crop preserves the unrounded prediction returned by ``FrameLite.predict``; this operation is never fed back to the model. | 仅用于渲染：左上角坐标向下取整，右下角坐标向上取整。<br><br>像素裁剪采用半开区间，同时保留 `FrameLite.predict` 返回的未取整预测值；此操作不会反馈给模型。 |
| 第 96 行，`load_image` 函数文档字符串 | Return an independent RGB image on the original decoded pixel canvas.<br><br>The benchmark protocol uses PIL's ``convert('RGB')`` without applying EXIF orientation. File inputs and PIL inputs follow that same convention. | 返回独立的 RGB 图像，保留解码后的原始像素画布。<br><br>基准评测协议使用 PIL 的 `convert('RGB')`，不应用 EXIF 方向信息。文件输入与 PIL 图像输入均遵循这一约定。 |
| 第 112 行，`crop_image` 函数文档字符串 | Render a valid predicted crop at its native aspect ratio and resolution. | 按裁剪区域原有的宽高比和分辨率，渲染合法的预测裁剪结果。 |

## 5. `tests/test_contract.py`：输出格式与坐标测试

| 位置 | 英文原文 | 中文翻译 |
| --- | --- | --- |
| 第 1 行，模块文档字符串 | Focused tests for the released parser and image-coordinate convention. | 针对发布版解析器及图像坐标约定的专项测试。 |

## 6. `requirements.txt`：依赖清单

| 位置 | 英文原文 | 中文翻译 |
| --- | --- | --- |
| 第 1 行，行注释 | Reference inference environment. Install the CUDA PyTorch wheels first (README). | 参考推理环境。请先安装 CUDA 版 PyTorch 的 wheel 安装包，具体步骤见 README。 |

## 术语说明

| 术语 | 本文含义 |
| --- | --- |
| output contract | 输出格式约定，即允许的字段、结构和坐标合法条件。 |
| singleton greedy | 每次只处理一张图片，并采用贪心解码。 |
| merger | 视觉特征合并模块；保留代码中的名称，便于对应实现。 |
| BF16 / FP32 | 分别指 bfloat16 和 float32 数值精度。 |
| SDPA | 缩放点积注意力（scaled dot-product attention）。 |
| half-open pixel crop | 半开像素裁剪区间：包含左、上边界，不包含右、下边界。 |
| null boxes | 没有有效裁剪框；Python 返回 `None`，JSON 序列化后为 `null`。 |
