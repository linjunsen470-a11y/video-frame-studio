# OCR、Mask 与 CPU 修复选型核查

核查日期：2026-11-24。以下为官方仓库、官方安装文档、代表源码的桌面审查；不是完整安全审计。实际运行证据以 `TEST_RESULTS.md` 为准。没有复制以下项目的实现代码。

## 结论

默认使用 **RapidOCR 3.10 的小型中文/英文 ONNX 模型 + ONNX Runtime CPU + OpenCV headless**。不安装 PaddlePaddle、PyTorch、IOPaint、Stable Diffusion、Web 服务或 GPU 运行时。模型随 RapidOCR wheel 安装，不进本项目仓库；适配层显式引用 bundled model 路径，缺失时明确失败，避免运行中隐式下载。不同项目的 `opencv-python`、`opencv-python-headless`、`opencv-contrib-python` 共用 `cv2` 命名空间，必须只安装本项目选定的一个 wheel。

| 项目 | 核实功能与 CPU 能力 | 维护与代码观察 | 许可证 / 本项目决定 |
|---|---|---|---|
| [RapidOCR](https://github.com/RapidAI/RapidOCR) | 检测、方向分类、识别；中文/英文；ONNX Runtime CPU。新版输出 `boxes/txts/scores`，不是旧包 `(result, elapsed)` 元组 | 当前统一 Python 包有多后端、Docker CPU 示例、测试与模型许可说明。3.10 起依赖 headless OpenCV。适配层同时识别旧 API，但默认测试新包 | Apache-2.0；官方另列模型许可。本项目通过公开 API 调用，不复制源码，不重新分发权重 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 移动端 PP-OCR 可在 CPU 运行；不等于必须使用较重的 PaddleOCR-VL | 活跃项目、独立推理与版本文档；安装 PaddlePaddle 带来额外运行时。新版与旧版调用接口有差异 | Apache-2.0；默认通过 RapidOCR 的 ONNX 版本使用其 OCR 技术，不引入完整 Paddle 生态 |
| [OpenCV](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html) | CPU `cv2.inpaint` 提供 Telea 与 Navier-Stokes；适于小缺损/细笔画，无法可靠重建被覆盖的复杂语义 | 成熟 API；本项目实际调用形态学、角点/光流、仿射 RANSAC、有限核高斯及无损 PNG | OpenCV 4.5+ 为 Apache-2.0；默认修复 Level 1 使用它 |
| [IOPaint](https://github.com/Sanster/IOPaint) | README 明确提供 `--model=lama --device=cpu`；含多种修复模型、UI 和服务 | **仓库在 2025-08-13 归档，只读**。requirements 包含 torch、diffusers、transformers、FastAPI、Gradio，且多项固定旧版本。功能广但不适合轻量 CLI 直接依赖 | Apache-2.0 工具代码；不同模型的权重许可须分别审查。仅参考设计，不依赖 |
| [LaMa](https://github.com/advimman/lama) | Fourier convolution 图像修复；原研究仓库基于 PyTorch，旧环境依赖较重。CPU 可以执行但不意味着适合低内存大图批量任务 | 研究代码规模较小；README 提醒旧模型下载链接失效。未将第三方 ONNX 转换视为官方稳定接口 | [原仓库 LICENSE](https://github.com/advimman/lama/blob/main/LICENSE) 为 Apache-2.0。只提供用户自备 ONNX 的可选适配，不下载、不捆绑未核实模型 |

## 已落实的实现约束

- OCR 缓存键包含图像内容、形状、配置及安装包版本；推理失败不等于“没有字幕”。Mock 会明确标记，`off` 抛出不可验证异常。
- 坐标由 OCR 返回并限制在原始图像边界。Gemini 不承担精确像素定位。
- `auto` 仅将同位置文字变化或显式 overlay 标记作为擦除证据；仅凭底部位置不足以区分包装、招牌和字幕。`subtitle` 是调用者明确授权底部字幕带，`all` 明确授权全部识别文字。保持词/保护矩形优先于擦除模式。
- Fine Mask 使用局部明/暗形态学帽及颜色残差，再按有限像素范围扩张覆盖描边、抗锯齿和阴影。低对比度失败会记录 polygon fallback。它不是对任意字体/纹理都准确的分割模型，仍需重新 OCR 与质检。
- 相邻帧修复要求 donor 帧提供完整 OCR 排除掩码、几何配准通过、全局及局部环带误差受限，并且所有目标 mask 像素都有无文字 donor。没有完整覆盖时不冒充成功。
- Level 3 ONNX 协议固定为 `image` RGB float32 NCHW `[0,1]`、`mask` float32 N1HW `[0,1]`、RGB float32 NCHW `[0,1]` 输出，兼容导出需要自行确认；不是通吃任意名为 LaMa 的 ONNX。默认禁用。
- 合成先复制原图，再只写入最终支持掩码。羽化使用有限核，核心 mask 内 alpha 为 1，避免原字幕重新混入。所有最终修改支持像素计入 mask，掩码外逐像素保持。

## 复核来源

1. [RapidOCR 安装指南](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/install/)：包整合、ONNX Runtime 分离依赖、3.10 headless 变更。
2. [RapidOCR CPU Docker 与测试](https://github.com/RapidAI/RapidOCR/blob/main/docker/README.md)。
3. [RapidOCR 推理配置](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/config.yaml)、[ONNX provider 选择](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/inference_engine/onnxruntime/provider_config.py)、[输出类型](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/utils/typings.py)。
4. [RapidOCR 模型许可清单](https://github.com/RapidAI/RapidOCR/blob/main/python/MODEL_LICENSES.md)。
5. [PaddleOCR CPU 安装说明](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/paddlepaddle_installation.en.md)、[LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)。
6. [IOPaint requirements](https://github.com/Sanster/IOPaint/blob/main/requirements.txt)、[LaMa 实现](https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py)。
7. [OpenCV 许可证说明](https://opencv.org/license/)。

固定时长字幕、极小文字、低对比度字幕、与主体同色的花字仍可能漏检。图像缺损的真实内容可能不可从视频中恢复；修复自然度属于启发式判断。保留原像素不代表可以规避平台 AI 内容标注。
