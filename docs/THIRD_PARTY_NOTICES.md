# 第三方依赖与模型声明

核实日期：2026-11-24。本项目自身新增源码使用根目录 `LICENSE` 中的 MIT 许可证。第三方依赖、模型、字体、编解码器及 Gemini 服务保留各自的许可证与条款；本项目的 MIT 许可证不替代它们。

此源码仓库/ZIP **不包含虚拟环境、第三方 wheel、FFmpeg 二进制、字体、OCR 权重或 LaMa 权重**。`setup.sh` 从包源安装依赖；依赖发行物自带的许可文件应予保留。下表依据本次实际安装包的 metadata 与许可文本核实，不能自动外推到其他版本或其他平台构建。

## 直接运行依赖

| 组件与实测版本 | 核实许可 | 说明与原始出处 |
|---|---|---|
| NumPy 2.5.5 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`（该 wheel 的 License-Expression） | NumPy 主体为 BSD-3-Clause，wheel 还含其他上游组件及 OpenBLAS 等通知；以发行物 `LICENSE.txt` 和所列 license files 为准。[NumPy](https://github.com/numpy/numpy) |
| opencv-python-headless 4.14.0.94 | OpenCV 本体 Apache-2.0；wheel 包装代码 MIT；随 wheel 的 FFmpeg 有 LGPL-2.1 通知 | 已读该 wheel 的 `LICENSE.txt` 和 `LICENSE-3RD-PARTY.txt`，不能笼统把整个二进制包归为单一 Apache 许可证。[OpenCV 许可](https://opencv.org/license/)、[Python 包装项目](https://github.com/opencv/opencv-python) |
| Pillow 13.0.0 | MIT-CMU | 依据该版本 License-Expression 与发行物 `LICENSE`。[Pillow](https://github.com/python-pillow/Pillow) |
| PySceneDetect / scenedetect 0.6.7 | BSD-3-Clause | 已核实实际 `LICENSE`，版权归 Brandon Castellano。该包 metadata 中存在 MIT classifier，但真实许可正文及 License 字段为 BSD-3-Clause，以许可正文为准。[PySceneDetect](https://github.com/Breakthrough/PySceneDetect) |
| RapidOCR 3.10.0 | Apache-2.0；随 wheel 的三份 OCR 模型有独立来源/归属声明 | 实际发行物含 `LICENSE`、`MODEL_LICENSES.md`。模型详情见下节。[RapidOCR](https://github.com/RapidAI/RapidOCR) |
| ONNX Runtime 1.32.0 | MIT | 核实实际 `onnxruntime/LICENSE`，版权归 Microsoft；二进制依赖与上游 notices 也应保留。[ONNX Runtime](https://github.com/microsoft/onnxruntime) |
| Google Gen AI SDK / google-genai 2.34.0 | Apache-2.0 | SDK 代码许可与 Gemini API 服务条款是不同事项；模型使用、计费及输出适用 Google 的相关服务条款。[官方 SDK](https://github.com/googleapis/python-genai)、[Gemini API 条款](https://ai.google.dev/gemini-api/terms) |

开发测试使用 pytest 9.2.0（MIT）。传递依赖由包管理器安装，保留各自发行物中的许可证；例如本次环境中的 Shapely 2.2.0 为 BSD-3-Clause，并自带 GEOS 许可通知，Pyclipper 1.4.0 为 MIT，OmegaConf 2.4.0 自带 BSD 与 ANTLR 归属文件。此表是直接依赖说明，不是对所有平台构建的完整 SBOM。

## RapidOCR wheel 中的默认 OCR 模型

本次实际安装的 RapidOCR 3.10.0 wheel 含以下文件，共约 **30.28 MiB**。这三个文件保留在依赖安装位置，不进入本项目源码包。适配器显式使用这些已安装文件；缺失时报告错误，不在处理用户视频时悄悄下载替代模型。

| 文件 | 源头/用途 | 实际 SHA-256 |
|---|---|---|
| `PP-OCRv6_det_small.onnx` | PaddleOCR `PP-OCRv6_small_det`，文字检测 | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `PP-OCRv6_rec_small.onnx` | PaddleOCR `PP-OCRv6_small_rec`，文字识别 | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | PaddleOCR `ch_ppocr_mobile_v2.0_cls`，文字方向分类 | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |

三个实际文件的哈希均与该 wheel 的 `MODEL_LICENSES.md` 清单相符。该声明将 RapidOCR 工程代码与 OCR 模型分为不同版权作品：模型权重归 Baidu 和/或适用的 PaddleOCR 权利人，清单覆盖的模型及其转换版本按 **Apache-2.0** 分发。声明列出了 PaddleOCR 官方模型来源及 PP-OCRv6 官方模型卡；旧方向分类器通过 PaddleOCR 官方模型列表和下载服务溯源。这里采用上游发行物的许可/来源声明，并未独立审计训练数据授权。

来源：[RapidOCR 模型许可声明](https://github.com/RapidAI/RapidOCR/blob/main/python/MODEL_LICENSES.md)、[默认模型登记](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml)、[PaddleOCR LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)、[检测模型卡](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx)、[识别模型卡](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx)。

更换为其他 OCR 模型、其他转换权重或不同 RapidOCR 版本时，应重新检查模型来源与许可证，不能仅凭文件名套用此结论。若后续重新分发上述模型或包含依赖的镜像/安装包，应保留相应版权、Apache 2.0 许可证、上游 NOTICE（如有）及转换归属通知；本源码包没有重新分发这些模型。

## FFmpeg、字体与未引入项目

- 系统 `ffmpeg` / `ffprobe` 是外部可执行程序。FFmpeg 主体为 LGPL-2.1-or-later，启用 GPL 组件的构建会适用 GPL；具体以使用的构建及 `ffmpeg -buildconf` 为准。本项目不分发该二进制。[官方说明](https://ffmpeg.org/legal.html)
- 合成测试视频仅由本项目生成几何图形与文字，没有下载第三方影片。生成器使用机器已安装的字体，**不打包字体文件**；Noto 字体具有自己的 OFL 许可。传入自定义字体时，其授权由该字体的来源决定。[Noto CJK](https://github.com/notofonts/noto-cjk)
- 调研了 GPL-3.0 的 `ComfyUI-Inpaint-CropAndStitch`，但**没有复制其实现、引入其依赖或分发其代码**。局部裁剪、NumPy 掩码赋值与有限羽化为本项目独立实现。
- 没有复制外部字幕去除应用、smart-thumbnailer、lighthouse、IOPaint 或 LaMa 的源码，仅通过公开 API 使用 OpenCV、PySceneDetect、RapidOCR 等依赖。
- 默认环境不引入 PyTorch、PaddlePaddle、ComfyUI、Gradio、FastAPI 或网页服务。Gemini 是显式启用的远程 API，未把服务输出误称为开源模型权重。

若将本项目改成包含全部依赖/模型的 Docker 镜像、独立可执行文件或商业安装包，需以实际分发的版本和二进制重新收集上游许可与 notices；本文件不代替那些完整许可文本。
