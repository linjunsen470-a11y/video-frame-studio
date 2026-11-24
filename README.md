# Video Frame Studio

轻量的 Python CLI：从视频中筛选优质帧，定位并擦除字幕，导出原始分辨率 PNG。
默认面向 Linux CPU / Codex Cloud。无需 GPU、Torch、Web 服务、数据库或 Agent 框架。

**可以严格验证的承诺：最终 PNG 在最终掩码以外，与对应原始解码帧逐像素完全相同。**
字幕是否全部检出、被遮挡背景是否真实、人物/产品是否自然，仍受 OCR、启发式和生成模型能力限制。
不承诺规避任何平台的 AI 内容检测、SynthID 或标注机制。

## 快速开始

需要 Python **3.12+**、Linux x86-64、FFmpeg/ffprobe。实际验证环境为 Python 3.12 / CPU。
默认仅 7 个直接 Python 依赖；完整解析结果固定在 `requirements.lock`。

```bash
cd video-frame-studio
bash setup.sh
source .venv/bin/activate
cp .env.example .env
```

在 `.env` 填入自己的 Google Gemini API Key：

```dotenv
GEMINI_API_KEY=your_key_here
```

或通过 Codex 环境的 secret / 环境变量注入 `GEMINI_API_KEY`，不用创建 `.env`。
不要把 Key 写入命令参数、聊天记录、配置仓库或日志。程序只读取它，不输出它。

```bash
python pipeline.py --input video.mp4 --count 20 --output ./output
```

没有 Key 时，必须明确选择离线处理；不会假装完成 Gemini 语义评分：

```bash
python pipeline.py --input video.mp4 --count 20 --output ./output --offline
```

离线模式可以真实执行解码、OCR、传统/时序修复及确定性质检。复杂背景或来源不明的文字会进入审核，
不会为了凑足数量降低标准。

## 常用命令

| 任务 | 命令 / 选项 |
|---|---|
| 限制候选及 API 消耗 | `--candidates 60 --budget-calls 30 --budget-usd 1.5` |
| 断点续跑 | 原命令追加 `--resume` |
| 只审计已有 PNG | `python pipeline.py --output ./output --qc-only` |
| 指定字幕模式 | `--subtitle-mode auto` / `subtitle` / `all` / `manual` |
| 保留候选和邻帧 | `--keep-intermediates` |
| 加载自定义参数 | `--config configs/my-video.toml` |
| 无真实 API 的接口测试 | `--mock-gemini`，报告明确标记 Mock |
| 顺序批量处理 | 重复 `--input a.mp4 --input b.mp4 --output ./batch` |

批处理在独立子目录保存每个视频；一次命令共享调用数/预留费用预算，顺序执行，不同时装载多个视频。
`--resume` 不允许更换输入内容、模型、OCR/掩码/质检语义配置或 count；改变这些设置应使用新输出目录。
可以调整调用预算、超时、限流、线程数和临时文件保留选项。

直接视频 URL：

```bash
python pipeline.py --input 'https://example.com/authorized-video.mp4' \
  --count 10 --output ./url-output --budget-calls 20
```

仅支持可直接下载的 HTTP(S) 视频。网页、需要登录/Cookie 的平台视频、HLS/DASH 播放列表不支持。
不集成 yt-dlp 或浏览器。URL 下载有大小、超时、重定向和私网地址限制，仍只应处理自己信任且有权使用的媒体。

## 处理流程

1. **单次顺序解码。** 用 FFmpeg 探测格式、OpenCV 解码；PySceneDetect 在采样帧上检测镜头切换。
   先检查清晰度、曝光、对比度，再按视频时间段选优并用 64-bit DCT pHash 去重。
2. **少量语义评分。** 缩小后的候选按批交给 Gemini，评估吸引力、主体、构图、产品展示、人物表现和素材价值。
   不上传整个视频。无预算时记录缺失评分，使用技术分；需要模型才能修复/辨别的帧不因此自动获准通过。
3. **真实 CPU OCR。** RapidOCR + ONNX Runtime 检出中英文多边形框，坐标始终映射到原分辨率。
   同一像素图/参数/版本的 OCR 结果按哈希缓存。包内三个 OCR 模型约 30.28 MiB，不额外在线下载模型。
4. **判定文字性质。** 使用同镜头邻帧文字变化、显式保护词和区域判断。仍不确定时可用 Gemini 只判断
   “后期覆盖文字 / 真实场景文字 / 不确定”；不采用大模型生成的新坐标。低置信度保留为审核。
5. **精细 Mask。** 形态学笔画、颜色残差、平滑背景颜色平面，加有限描边/阴影扩展；在水平框两端搜索相邻笔画，
   补偿 OCR 框截断首字/尾字。遇到可靠性不足的边缘，记录并拒绝通过。不是逐帧完美分割的承诺。
6. **分级局部修复。** 简单背景用 OpenCV；再尝试同镜头邻帧配准和无文字 donor；可选用户提供的 ONNX；
   最后才调用 Gemini 局部 crop 编辑。每张图最多 3 次有效图像尝试，网络重试也纳入 API 调用账本。
7. **局部合成与审核。** 复制原图，只改最终掩码范围；羽化支持也进入最终掩码，保护区再次恢复原像素。
   PNG 保存后重读，检查尺寸、掩码外完全一致、OCR 残留、疑似孤立描边、清晰度、融合边缘和重复图。
   生成式修复还必须通过 Gemini 局部视觉复核；模型复核不能推翻确定性失败。
8. **只交付通过图片。** 不足数时输出 partial 和实际数量。失败或不确定的图片不进入 `images/`。

采样默认 0.5 秒，长视频会自动放宽，最多 600 个采样点；镜头检测是采样级近似，可能漏掉极短镜头。
可以在配置中减小 `sample_seconds`、提高 `max_scan_samples`。保留候选上限默认 96，
自动数量为 `min(max_candidates, max(3*count, ceil(duration/4)))`。

## 字幕模式与真实文字保护

| 模式 | 行为 |
|---|---|
| `auto`（默认） | 保守使用时序变化和可选语义辅助；不确定文字进入 review。最适合完整 API 工作流。 |
| `subtitle` | 明确允许擦除底部字幕带内的横行文字；其他文字仍须判断/保护。 |
| `all` | 擦除所有 OCR 检出的文字，保护词/区域优先；可能包含真实包装/招牌，只在你明确需要时使用。 |
| `manual` | 使用给定像素矩形或多边形；无需让模型猜框。 |

示例 `configs/my-video.toml`（只写需要覆盖的字段）：

```toml
[ocr]
mode = "auto"
keep_words = ["SANMUZUO", "BRAND"]
protect_regions = [[30, 20, 180, 95]]
bottom_start = 0.60

[video]
candidates = 60
max_cache_mb = 1536

[gemini]
max_calls = 30
max_cost_usd = 1.5
```

矩形格式为原始像素 `[x0,y0,x1,y1]`，保护矩形右/下边界包含在内。
手工区域示例：

```toml
[ocr]
mode = "manual"
manual_regions = [[180, 800, 900, 930]]
protect_regions = [[500, 100, 650, 180]]
```

手工模式也支持 `manual_regions = [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]]` 多边形。
所有模式仍启用真实 OCR 复核。没有 OCR 依赖或识别失败时不会把图片判为“无字幕”。

## Gemini 模型、费用和失败处理

模型由 `configs/default.toml` 配置；环境变量 `GEMINI_ANALYSIS_MODEL` / `GEMINI_IMAGE_MODEL` 可覆盖。
当前调研采用 `gemini-3.1-flash-lite` 做分析，`gemini-nano-banana-2.1` 做局部图像编辑。
这两个模型 ID 不嵌入业务代码；可按账号可用性修改。价格、官方来源和接口限制见 `docs/research-gemini.md`。

**本次没有真实 API Key：SDK 请求类型、预算、缓存、重试、结构化输出和 crop/edit 路径通过 Mock 测试；
真实模型可用性、输出质量、实际账单未实测。**

- `max_calls` 是所有实际请求尝试的硬上限，包括失败与重试。
- `max_cost_usd` 限制本地保守预留金额，**不是 Google 账单硬上限**。请求前预留，失败或进程崩溃也不退回，
  避免“服务端已计费但客户端未收到响应”导致低估。实际账单以 Google 控制台为准。
- 报告分别给出 reserved cost、按配置价格/返回 usage 估算的 cost、估算是否完整、缓存命中及各模型调用数。
- 更换模型时同步提供 `[gemini.prices."model-id"]`；启用美元预算而没有价格时停止该调用。
- SDK 内部重试、自动 continuation、自动函数调用关闭；程序只使用一层有界重试，默认一次请求最多 3 次尝试。
- 同一输入、模型、提示词、配置和图像的结果可复用；API 请求账本在进程间加锁并持久保存。
- 429/暂时服务错误按退避重试；预算不足或网络失败的候选保存为可恢复状态。提高已授权预算后 `--resume`。
- Mock 与真实请求使用隔离账本和缓存；报告不会把 Mock 次数计作真实 API。

原生 Gemini 编辑不是精确像素 Mask API。本项目发送局部原图及区域提示，返回仅作 proposal；
最终由 NumPy/OpenCV 合成。原尺寸不匹配且无法可靠映射的返回会被拒绝，不会直接覆盖整帧。

## 可选 ONNX 修复

默认不带 LaMa/Torch。只有提供模型路径才尝试 Level 3：

```toml
[repair]
onnx_model = "/absolute/path/to/licensed-model.onnx"
```

支持限定协议：图像 RGB NCHW float32 `[0,1]`、单通道 NCHW Mask `[0,1]`，输出 RGB NCHW `[0,1]`；
输入名称/形状必须符合适配器要求，详见 `vfs/repair.py`。模型可为固定/动态大小，按适配器做局部尺度调整。
不保证任意 LaMa 导出都兼容。**真实 LaMa 权重未实测**，只验证了适配接口和异常降级；权重授权由提供者确认。
生成式 ONNX 修复默认还要求 Gemini 视觉复核，因此完全离线时可能进入 review。

## 输出、缓存与退出状态

输出包含 `images/`、`originals/`、`masks/`、`previews/`、`report.json`、`report.md`。
`masks` 为二值最终修改支持区，包含羽化范围。`originals` 和通过图对应，不能为节省空间删掉后仍宣称可复核。

额外 `.checkpoint.json`、`.cache/ocr/`、`.cache/gemini/` 用于续跑和计费缓存；输出目录不会提交到 GitHub。
完成后默认删除候选、邻帧、临时 PNG 和下载视频；审核预览、成品与对应原图/掩码保留。
可恢复 API 故障暂时保留候选缓存，供下次续跑。`--keep-intermediates` 可保留完整候选。

首次扫描中途断开时会重新做该次顺序扫描；完成扫描后的中断复用候选，不重复解码。
已完成任务续跑只复核已有结果。质检淘汰的图片不会因为重复执行而无限重试。
`--qc-only` 生成 `qc_report.json/md`，不调用 Gemini、不解码视频，不修改原报告和图像。
空成品集合不能判为质检通过。

| 退出码 | 含义 |
|---:|---|
| 0 | 正常任务达到数量，或已有 PNG 审计通过 |
| 2 | 合格数量不足，或仅质检发现失败 |
| 1 | 配置、输入、依赖、OCR 或运行错误 |
| 130 | 用户中断；可使用 `--resume` |

默认 2 个 CPU 线程、最多 4 个同镜头 donor、顺序 API 调用；内存不会随视频帧数线性增长。
候选缓存默认硬限制 1536 MiB；高分辨率视频可减少候选或显式增加限额。
当前仅支持 SDR 8-bit 视频。HDR/>8-bit 输入会拒绝，避免把转成 BGR8 误称为保留原始色深。
输出采用编码像素方向，手机旋转元数据不自动应用；可先明确旋转转码，再将转码视频作为像素基准。

## 测试与可复现验证

```bash
bash setup.sh --dev
source .venv/bin/activate
python -m pytest -q
python scripts/smoke_test.py --workdir /tmp/vfs-smoke-new
python scripts/smoke_test.py --workdir /tmp/vfs-smoke-mock-new --mock-gemini
python -m pip_audit
```

Smoke 测试自行生成视频，不下载第三方素材；中英文示例需要系统 Noto CJK 字体，Debian/Ubuntu 可安装
`fonts-noto-cjk`。普通处理不需要这个字体。测试目录必须不存在，以免覆盖既有证据。

测试覆盖真实 CPU OCR、细 Mask、缺首/尾字回归、邻帧恢复/污染拒绝、100% 掩码外保护、PNG 审计、
异常 OCR、重复帧、预算/缓存/重试、单次解码、断点续跑、危险输入限制，以及真实视频端到端流程。
实测结果及未实测范围见 `docs/VALIDATION.md`。

## 给 Codex Agent

`AGENTS.md` 固化执行规则；仓库技能在 `.agents/skills/video-frame-studio/SKILL.md`，并提供 UI 元数据。
它随仓库交付，没有安装到你的全局个人技能目录。日常模型只需执行 CLI、阅读短报告和交付结果。
不要为每个视频重写提示词或编排，不要自动放宽阈值来补足图片数量。

## 开源调研与许可

源码采用 MIT；依赖及模型通知见 `docs/THIRD_PARTY_NOTICES.md`。
已研究用户指定的字幕擦除、精彩帧、OCR、inpainting、Crop-and-Stitch 项目，比较与来源见 `docs/RESEARCH.md`。
复用成熟依赖 API 与算法思想，没有复制 GPL 项目源码，也没有把庞大项目直接嵌入 CLI。

源码包不包含视频、真实 Key、模型权重、虚拟环境、字体或大型二进制。
首次在 GitHub 建库可在网站创建私有空仓库后上传，或在已认证的 `gh` 环境执行：

```bash
git init
git add .
git commit -m "Initial Video Frame Studio implementation"
gh repo create video-frame-studio --private --source=. --remote=origin --push
```

## 已知边界

固定不变字幕、极小字、艺术字、低对比度、透明花字可能漏检；整行没被 OCR 检出不能靠边缘扩展补全。
时序变化也不是严格的字幕证明：画面中电子屏幕或内容变化的包装可能需要保护区。
邻帧必须真的露出背景、配准可信且无文字污染；持续覆盖同一区域的字幕无法凭空恢复真实背景。
生成式修复与视觉复核存在误判，严格像素约束只保证掩码外，不能保证掩码内事实正确。
本次只验证了自制视频和合成图，不把它外推为任意真实影片的成功率或速度保证。
