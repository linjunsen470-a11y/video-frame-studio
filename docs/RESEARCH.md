# 开源实现调研与取舍

本文保留早期技术调研。当前正式修复仅使用 Gemini 整帧编辑，OpenCV 用于解码、蒙版、配准与像素质检。见 [当前流程](../README.md)。

核实日期：**2026-11-24（北京时间）**。资料来自原始 GitHub 仓库、GitHub REST 返回及官方文档。范围包括 README、许可证、依赖清单和下列代表实现的抽查；**没有逐个安装、运行或审计全部上游项目**。表中的维护时间是此次 API 查询返回的仓库 `pushed_at`，不能等同于最新正式版、默认分支最后提交时间或质量保证。

## 选择结论

主路径采用 FFmpeg/ffprobe、OpenCV、NumPy 与 CPU OCR，将镜头候选、文字坐标、有限范围 Mask、修复与质量门槛固化到代码。Gemini 为人工选中的帧提供整帧编辑提案。已有框架中可借鉴的是算法组合与明确的质量门槛，没必要引入视频修复 GUI、ComfyUI 或训练框架。

本项目不打包任何上游完整应用、预训练修复模型或视频素材。上游源码许可证和模型权重许可证应分别核查；某个项目允许商用，并不自动覆盖它引用的全部模型、数据、编解码器和第三方素材。

## 指定项目逐项核实

| 项目 | 已核实功能与代表实现 | 许可与维护记录 | CPU、依赖与本项目取舍 |
|---|---|---|---|
| [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) | 处理视频/图片硬字幕；可限定字幕区域或检测整幅文字。`backend/main.py` 包含镜头分段、字幕连续区间、预读与 ProPainter/STTN/LaMa 路径。 | Apache-2.0；`pushed_at` 2026-06-30，未归档。 | 存在 CPU 路径，但 PaddleOCR、PyTorch、GUI、视频修复模型使默认安装较重；当前 requirements 明列 PaddleOCR 3.4、NumPy 2.2.5、AV 17、PySide 相关包。借鉴字幕区间/镜头边界思路，不作为依赖。CPU 可运行不代表 CPU 速度适合本项目。 |
| [SysAdminDoc/VideoSubtitleRemover](https://github.com/SysAdminDoc/VideoSubtitleRemover) | Windows 主应用；OCR 轨迹、区域/Mask 审查、背景参考、修复和报告。抽查 `backend/reference_fill.py`：有时间偏移、配准置信度、缩放矩阵和局部合成。 | MIT；`pushed_at` 2026-11-19，未归档。查询默认分支 commit 端点另返回 2026-09-26，故不把仓库 push 日期宣称为最新默认分支 commit。 | 明确区分 CPU/NVIDIA 依赖配置，存在 ONNX/OpenCV DNN 路径。当前包同时包含大量桌面与工作流模块，不适合作为轻量库整体导入。参考配准质量门槛、无文字 donor、Mask 审查。 |
| [line/lighthouse](https://github.com/line/lighthouse) | 视频片段检索、高光定位及音频检索；是研究模型的统一推理/复现框架，不是轻量逐帧美学筛选器。`lighthouse/models.py` 加载多个 DETR 家族模型及视觉/文字/音频编码器。 | Apache-2.0；`pushed_at` 2026-11-01，未归档；源码另保留第三方 MIT 通知。 | 接受 device 参数，但需要 Torch、CLIP 等模型。`setup.py` 限制 `numpy<=1.23.5`、`transformers<=4.51.3`，还引入 pandas、Gradio、音频组件，与现代轻量 ONNX/OpenCV 环境不宜硬拼。其 checkpoint 路径使用 `torch.load(..., weights_only=False)`，不导入、不下载权重。 |
| [fralapo/smart-thumbnailer](https://github.com/fralapo/smart-thumbnailer) | 以清晰度、色彩、显著性、人脸位置、稳定性和 dHash 选缩略图；按时间区间选优。已读 `thumbnailer.py` 与 requirements。 | MIT；`pushed_at` 2026-06-17，未归档；仓库体量很小、测试证据有限，不能称作已充分验证的大型成熟框架。 | 核心只需 OpenCV+NumPy；人脸模型会联网下载，PySceneDetect 可选。适合参考两阶段/分区选优设计；候选整帧常驻内存、自动模型下载等细节不照搬。 |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | 专门处理镜头切分；`ContentDetector` 计算 HSV 差异，`AdaptiveDetector` 使用局部变化基线；可接收已有解码帧，避免独立完整解码。抽查 ContentDetector 与 pyproject。 | BSD-3-Clause；`pushed_at` 2026-11-24，未归档；当前官方文档版本 0.7.3。 | CPU 友好、无需深度学习权重。当前 packaging 区分 GUI/headless OpenCV 变体，不能同时安装多个拥有 `cv2` 的 wheel；实际使用版本需与当前 API 配合。可选集成或采用 OpenCV 变化度作为默认轻量切分。 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 中文/英文等文字检测与识别，覆盖四边形坐标；字体、旋转、遮挡等仍会影响召回。 | Apache-2.0；活跃上游，代表源码/CPU 官方文档见 [OCR 调研](research-ocr.md)。需要区分代码与模型许可。 | 有 CPU 路径，但完整 Paddle 推理栈和可选文档组件较重。作为对比/替代方案，不把整个应用栈加入默认安装。 |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) | 基于成熟 OCR 模型的轻量推理封装，可返回文字四边形、内容和置信度。 | Apache-2.0；当前统一包 3.10 支持 headless OpenCV，模型许可另列，见 [OCR 调研](research-ocr.md)。 | 选择 CPU 推理路径；初始化一次、同图缓存。识别置信度不代表文字一定是后期字幕，应另用位置、时序与保护规则判断。 |
| [Sanster/IOPaint](https://github.com/Sanster/IOPaint) | 图像修复应用，支持多种模型与 UI。 | Apache-2.0；官方仓库 2025-08-13 归档，只读，见 [OCR 调研](research-ocr.md)。 | 适合人工修图或多模型试验；完整 UI/服务和模型生态超出本 CLI 需求。不作为运行时依赖。 |
| [LaMa](https://github.com/advimman/lama) | 大掩码图像修复模型，处理纹理与背景；不自带字幕/真实文字分类能力。 | 原代码 Apache-2.0；旧模型链接失效提醒见 [OCR 调研](research-ocr.md)；导出的 ONNX 权重还应核查来源、许可和哈希。 | Torch 原版不适合默认轻量环境；保留为调研对照。 |
| [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) | 按 Mask 外接区域裁剪，推理后拼回原图；支持尺寸适配。抽查 `inpaint_cropandstitch.py` 和 pyproject，依赖 Torch、Torchvision、SciPy、ComfyUI 生态。 | GPL-3.0；`pushed_at` 2026-10-21，未归档。 | 节点工具在 CPU 上有代码路径，但整个生成图像工作流并不轻量。本项目仅使用公开通用的裁剪/有限 Mask 合成思路，**不复制 GPL 节点实现、不引入 ComfyUI**。 |
| [FFmpeg](https://ffmpeg.org/) | ffprobe 读取流信息；ffmpeg 解码、采样、无损帧输出。 | 主要 LGPL-2.1-or-later；启用 GPL 组件会改变对应 FFmpeg 构建许可。官方最新稳定页显示 9.1 于 2026-11-20 发布。 | CPU 解码成熟；系统独立命令调用可省 Python 封装。项目不分发 FFmpeg 二进制；实际构建可用 `ffmpeg -buildconf` 检查。原始分辨率不等于保留原始 HDR/10-bit 色深，应明确解码后的 BGR8 工作空间。 |
| [OpenCV](https://opencv.org/) | 图像变换、边缘/形态学、配准、Telea/Navier–Stokes 修复。官方 inpaint 使用单通道 8-bit 非零 Mask 指定修复区域。 | 4.5 及以后 Apache-2.0；更早版本 BSD-3-Clause。上游 2026-11-24 仍有提交。 | 主要算法可在 CPU 运行；细线、小洞和平滑背景合适，无法保证大块遮挡下的人脸/结构恢复。本项目直接调用成熟算法，最终通过 NumPy 严格限制修改区域。 |

## 代码抽查与兼容性判断

- **本次实际采用的兼容组合。** Python **3.12+**；测试环境为 Python 3.12，NumPy **2.5.5**、OpenCV headless **4.14.0.94**、Pillow **13.0.0**、RapidOCR **3.10.0**、ONNX Runtime **1.32.0**、google-genai **2.34.0**、PySceneDetect **0.6.7**。使用 0.6.7 的已验证逐帧 detector 接口；当前上游 0.7.3 的默认包会引入 `opencv-python`，本项目避免让它与 headless 包同时拥有 `cv2`。默认环境只安装一个 OpenCV wheel。固定版本依据实际安装与测试选择，并非把所有上游“最新版”直接堆在一起。第三方许可与 OCR 模型哈希见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- **上游设计参考，不等同于逐项复现。** 已查看了 smart-thumbnailer 的指标和筛选实现、YaoFANGUK 的字幕连续区间与场景划分、SysAdminDoc 的参考帧配准、lighthouse 的模型加载、PySceneDetect 的 HSV 内容差及 CropAndStitch 的依赖/张量处理。项目源码中的独立实现由本项目测试验证；没有用“GitHub 星数”替代质量判断。
- **一个环境只应有一个 OpenCV 包。** `opencv-python`、`opencv-python-headless`、contrib 变体都会提供 `cv2`，必须检查解析结果；CPU/服务器优先 headless，但最终以 OCR 包实际依赖为准。默认路径不混装 Paddle、Torch、ComfyUI。
- **输入与输出的像素合同。** 精确不变基准是保存到 `originals/` 的解码帧，最终 PNG 重新读取后检查 Mask 外差分；不能将有损视频编码前不存在的“原始 RGB”作为基准。若羽化扩张，则羽化支持域计入最终 Mask。
- **质量能力边界。** OCR 检出“文字”不等于检出“字幕”；几何和时序只能提供启发式证据。OCR 未检出不证明绝无残留。人工查看实际修复区域并记录可见问题。
- **费用与版本。** Gemini 的模型与价格单独配置，详细官方资料见 [Gemini 调研](research-gemini.md)。本调研没有据此声称已调用真实 Gemini API；实测范围以交付测试记录为准。

## 可追溯来源

### 仓库与代表文件

1. [YaoFANGUK requirements](https://github.com/YaoFANGUK/video-subtitle-remover/blob/main/requirements.txt)、[主处理器](https://github.com/YaoFANGUK/video-subtitle-remover/blob/main/backend/main.py)、[许可](https://github.com/YaoFANGUK/video-subtitle-remover/blob/main/LICENSE)。
2. [SysAdminDoc requirements](https://github.com/SysAdminDoc/VideoSubtitleRemover/blob/main/requirements.txt)、[参考帧恢复](https://github.com/SysAdminDoc/VideoSubtitleRemover/blob/main/backend/reference_fill.py)、[许可](https://github.com/SysAdminDoc/VideoSubtitleRemover/blob/main/LICENSE)。
3. [lighthouse 安装依赖](https://github.com/line/lighthouse/blob/main/setup.py)、[模型实现](https://github.com/line/lighthouse/blob/main/lighthouse/models.py)、[许可](https://github.com/line/lighthouse/blob/main/LICENSE)。
4. [smart-thumbnailer 实现](https://github.com/fralapo/smart-thumbnailer/blob/master/thumbnailer.py)、[依赖](https://github.com/fralapo/smart-thumbnailer/blob/master/requirements.txt)、[许可](https://github.com/fralapo/smart-thumbnailer/blob/master/LICENSE)。
5. [PySceneDetect pyproject](https://github.com/Breakthrough/PySceneDetect/blob/main/pyproject.toml)、[ContentDetector](https://github.com/Breakthrough/PySceneDetect/blob/main/scenedetect/detectors/content_detector.py)、[官方检测器文档](https://www.scenedetect.com/docs/latest/api/detectors.html)。
6. [CropAndStitch 实现](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch/blob/main/inpaint_cropandstitch.py)、[pyproject](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch/blob/main/pyproject.toml)、[许可](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch/blob/main/LICENSE)。
7. [FFmpeg 官方许可说明](https://ffmpeg.org/legal.html)、[官方发布页](https://ffmpeg.org/download.html)、[ffprobe 文档](https://ffmpeg.org/ffprobe.html)。
8. [OpenCV 许可](https://opencv.org/license/)、[inpaint API](https://docs.opencv.org/4.x/d7/d8b/group__photo__inpaint.html)。

维护字段通过各仓库 `https://api.github.com/repos/OWNER/REPO` 返回的 `archived`、`pushed_at`、`license` 核对；对代表仓库另查 `/commits?per_page=1`。这些远端路径会随维护而更新，本文件记录的是核实当日观察。
