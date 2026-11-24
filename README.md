# Video Frame Studio

轻量的 Python CLI：从视频中筛选优质帧，定位并擦除字幕，导出原始分辨率 PNG。
默认面向 Linux CPU / Codex Cloud。无需 GPU、Torch、Web 服务、数据库或 Agent 框架。

**可以严格验证的承诺：最终 PNG 在最终掩码以外，与对应原始解码帧逐像素完全相同。**
字幕是否全部检出、被遮挡背景是否真实、人物/产品是否自然，仍受 OCR、启发式和生成模型能力限制。
不承诺规避任何平台的 AI 内容检测、SynthID 或标注机制。

## 小红书团餐图文

在 Codex Cloud 打开这个仓库后，用 `$xiaohongshu-catering-editorial` 调用
[团餐图文技能](.agents/skills/xiaohongshu-catering-editorial/SKILL.md)。提供实拍照片、通过质检的抽帧目录或现有图集，说明这次要做食堂承包、员工餐配送或后厨记录。技能可输出排好的图集和发布文案，也可按要求只改其中一项。

```text
$xiaohongshu-catering-editorial 用我附的饭堂菜盘和餐线照片做一篇现场记录。
挑封面、排图，写一版可以直接发的小红书文案。
```

图文工作从已有照片开始。云端已配置小红书站内访问时可补查原帖，其余环境可直接用仓库内的研究附录。私有原图、待发文案和渲染成品放在 `runs/` 或本地 `catering-xhs/`，两个目录都已忽略。

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

### 1. 本地准备候选

```bash
python pipeline.py --prepare --input video.mp4 --output runs/job-001 --candidates 48
```

这一步无需 API Key。FFmpeg/OpenCV 解码，PySceneDetect 辅助检测切镜，再做本地清晰度、曝光、对比度和去重筛选。
按时间顺序查看 `selection/sheet-*.png`，或从 `selection/index.json` 打开候选原图。
Codex 对照用户本次 prompt 判断主体、动作时机和构图，确认入选 ID 并记录理由。素材类型不限。
若目标动作不在候选里，可在新目录增加 `--candidates`，并通过 TOML 缩短 `video.sample_seconds` 或增大 `video.max_scan_samples`。
默认采样间隔为 0.5 秒，长视频会自动放宽。默认最多保留 96 个候选。

### 2. 仅重绘明确选中的帧

```bash
python pipeline.py --input video.mp4 --output runs/job-001 \
  --select-frame f00000060 --select-frame f00000120 \
  --subtitle-mode subtitle --budget-calls 8 --budget-usd 2
```

把示例 ID 换成候选索引中的实际 ID，按交付优先级排列。程序核对视频指纹、候选原图和帧 ID，
处理数量由选中的 ID 决定。帧不通过时记录缺额。没有目标字幕的帧不会计为去字成品。

RapidOCR 在本地定位文字。程序用邻帧、字幕位置和保护区判断目标文字，文字来源不确定则进入待审核。
Gemini 只收到选定帧的局部裁剪及遮罩参考图，只负责重绘。选帧、文字判定、视觉复核都由本地流程或 Codex 处理。
此入口的实际修复只用 Gemini。程序在遮罩内合成候选结果，重读 PNG 检查尺寸、遮罩外像素、OCR 残留、
疑似笔画、清晰度、融合边缘及重复图。交付前由 Codex 目视核对字幕区域、主体和背景边缘。

### 3. 审计与续跑

```bash
python pipeline.py --output runs/job-001 --qc-only
```

可恢复故障后将原编辑命令追加 `--resume`，保留原输入、输出目录、帧 ID 顺序和语义配置。
两个阶段各接收一个视频，`--input` 支持本地文件或可直接下载的 HTTP(S) 视频。
登录页、HLS/DASH 播放列表和私网地址不可用。下载受大小和超时限制。

`--offline` 和 `--mock-gemini` 仍可用于隔离的本地验证及接口测试。带 `--select-frame` 时可用 Mock 测试选帧到改绘的边界。
生产流程按上述两个阶段执行。

## 字幕模式与真实文字保护

| 模式 | 行为 |
|---|---|
| `auto`（默认） | 保守使用时序变化；不确定文字进入 review。 |
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

选帧工作流使用 `configs/default.toml` 中的 `gemini-nano-banana-2.1`，默认输出 1K 裁剪图，
最终 PNG 采用原帧分辨率。`GEMINI_IMAGE_MODEL` 可覆盖图像模型。价格和接口限制见 `docs/research-gemini.md`。

- `max_calls` 是所有实际请求尝试的硬上限，包括失败与重试。
- `max_cost_usd` 限制本地保守预留金额，**不是 Google 账单硬上限**。请求前预留，失败或进程崩溃也不退回，
  避免“服务端已计费但客户端未收到响应”导致低估。实际账单以 Google 控制台为准。
- 报告分别给出 reserved cost、按配置价格/返回 usage 估算的 cost、估算是否完整、缓存命中及各模型调用数。
- 更换模型时同步提供 `[gemini.prices."model-id"]`；启用美元预算而没有价格时停止该调用。
- SDK 内部重试、自动 continuation、自动函数调用关闭。网络请求默认零自动重试；单帧的质检改绘尝试仍受任务预算约束。
- 同一输入、模型、提示词、配置和图像的结果可复用；API 请求账本在进程间加锁并持久保存。
- 工作流默认串行。Banana 2.1 本地上限为 10 RPM、160,000 估算输入 TPM、800 RPD，按太平洋时间午夜重置。
  额度账本在本仓库各输出目录间共享。预算或网络故障保存为可恢复状态，可在问题解除后 `--resume`。
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
选帧工作流保留候选索引、联系表和原始帧缓存，以便对照与续跑。离线验证完成后默认清理扫描缓存。
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
按用户当次 prompt 查看候选和原帧，记录入选理由后调用改绘。交付前查看成品，并报告任何缺额和可见修复痕迹。

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
