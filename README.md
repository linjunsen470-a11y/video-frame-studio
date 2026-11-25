# Video Frame Studio

轻量的 Python CLI：从视频中筛选优质帧，定位并擦除字幕，导出原始分辨率 PNG。
默认面向 Linux CPU / Codex Cloud。无需 GPU、Torch、Web 服务、数据库或 Agent 框架。

**可以严格验证的承诺：最终 PNG 在最终掩码以外，与对应原始解码帧逐像素完全相同。**
字幕是否全部检出、被遮挡背景是否真实、人物/产品是否自然，仍受 OCR、启发式和生成模型能力限制。
不承诺规避任何平台的 AI 内容检测、SynthID 或标注机制。

## 运行环境

请在该仓库的 Codex Cloud 环境运行 `python pipeline.py` 和 `python -m catering_visual`。
处理命令会在加载配置、读取素材或写输出前检查 Linux 托管工作区。普通本机和无法确认的环境会提示并以状态码 `1` 退出。
识别依据是已有云环境提供的 `/workspace/library-files`、`/workspace/scratch` 和 `/workspace/shared`。`--help` 仍可查询命令用法。

## 小红书团餐图文与实拍图集

用 `$xiaohongshu-catering-editorial` 调用
[团餐图文技能](.agents/skills/xiaohongshu-catering-editorial/SKILL.md)。提供实拍照片、通过质检的抽帧或现有图集，
说明要做食堂承包、员工餐配送或后厨记录。Codex 按本次需求挑封面、排图和写发布文案，也可以只改其中一项。

```text
$xiaohongshu-catering-editorial 用我附的饭堂菜盘和餐线照片做一篇现场记录。
挑封面、排图，写一版可以直接发的小红书文案。
```

需要生成成品时，用 `python -m catering_visual` 将选好的照片和文案写成 JSON 设计，再生成 3:4 JPEG，按单张 300 KB 目标审计，并输出笔记文案。
图集支持竖向单图和横向双帧上下拼接、单页重渲染及文案审计。文案默认 80–150 字，广州业务在标题和话题写明城市。
查看 [餐饮图集指南](docs/CATERING-GALLERY.md)。

云端已配置小红书站内访问时可补查原帖，也可参考技能附带的研究记录。
私有原图、待发文案、设计 JSON 和成品放在根目录的 `runs/` 或 `catering-xhs/` 下，这两个目录均已忽略。

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

也可通过 Codex 的环境变量注入 `GEMINI_API_KEY`。
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
处理数量由选中的 ID 决定。帧不通过时记录缺额。经逐帧核对确认没有目标叠字时，可用空区域列表导出原帧。

RapidOCR 在本地定位文字。程序用邻帧、字幕位置和保护区判断目标文字，文字来源不确定则进入待审核。
每轮向 Gemini 提交完整选定帧和一张合并所有目标框的蒙版。选帧、文字判定、对齐和视觉复核由本地流程或 Codex 处理。
此入口的实际修复只用 Gemini。程序在遮罩内合成候选结果，重读 PNG 检查尺寸、遮罩外像素、OCR 残留、
疑似笔画、清晰度、融合边缘及重复图。交付前由 Codex 目视核对字幕区域、主体和背景边缘。

### 3. 审计与续跑

```bash
python pipeline.py --output runs/job-001 --qc-only
```

可恢复故障后将原编辑命令追加 `--resume`，保留原输入、输出目录、帧 ID 顺序和语义配置。
两个阶段各接收一个视频，`--input` 支持本地文件或可直接下载的 HTTP(S) 视频。
登录页、HLS/DASH 播放列表和私网地址不可用。下载受大小和超时限制。

`--mock-gemini` 用于显式选帧的无网络测试。正式编辑使用 Gemini。

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

小红书的定稿选帧使用 `[ocr.reviewed_frames]`。为每个 `--select-frame` ID 提供一项，用原始像素框标记所有经 OCR 和目视确认的字幕或独立「广告」标记。OCR 漏检时补上目视确认的框。显式空列表代表确认无目标叠字，可直接进行无损质检；若全部选定帧均为空列表，无需 `GEMINI_API_KEY`。

```toml
[ocr.reviewed_frames]
f00000240 = [
  {kind = "ad_label", text = "广告", box = [72, 138, 160, 185]},
  {kind = "subtitle", text = "看着香吗？", box = [247, 975, 462, 1031]},
]
f00000255 = []
```

## Gemini 模型、费用和失败处理

选帧工作流使用 `configs/default.toml` 中的 `gemini-nano-banana-2.1`，默认返回 1K 整帧提案，
最终 PNG 采用原帧分辨率。`GEMINI_IMAGE_MODEL` 可覆盖图像模型。价格和接口限制见 `docs/research-gemini.md`。

- `max_calls` 是所有实际请求尝试的硬上限，包括失败与重试。
- `max_cost_usd` 限制本地保守预留金额，**不是 Google 账单硬上限**。请求前预留，失败或进程崩溃也不退回，
  避免“服务端已计费但客户端未收到响应”导致低估。实际账单以 Google 控制台为准。
- 报告分别给出 reserved cost、按配置价格/返回 usage 估算的 cost、估算是否完整、缓存命中及各模型调用数。
- 更换模型时同步提供 `[gemini.prices."model-id"]`；启用美元预算而没有价格时停止该调用。
- SDK 内部重试、自动 continuation、自动函数调用关闭。网络请求默认零自动重试；单帧的质检改绘尝试仍受任务预算约束。
- 同一输入、模型、提示词、配置和图像的结果可复用；几何或配准失败的提案不会写入有效缓存。API 请求账本在进程间加锁并持久保存。
- 工作流默认串行。Banana 2.1 本地上限为 10 RPM、160,000 估算输入 TPM、800 RPD，按太平洋时间午夜重置。
  额度账本在本仓库各输出目录间共享。预算或网络故障保存为可恢复状态，可在问题解除后 `--resume`。
- Mock 与真实请求使用隔离账本和缓存；报告不会把 Mock 次数计作真实 API。

Gemini 蒙版作为编辑参考。返回图经宽高比和场景配准验证后，由 NumPy/OpenCV 只把目标蒙版内的像素合回原图。无法可靠对齐的提案进入审核。

## 输出、缓存与退出状态

输出包含 `images/`、`originals/`、`masks/`、`previews/`、`report.json`、`report.md`。
`masks` 为二值最终修改支持区，包含羽化范围。`originals` 和通过图对应，不能为节省空间删掉后仍宣称可复核。

额外 `.checkpoint.json`、`.cache/ocr/`、`.cache/gemini/` 用于续跑和计费缓存；输出目录不会提交到 GitHub。
选帧工作流保留候选索引、联系表和原始帧缓存。历史运行可以用 `--qc-only` 审计。旧配置里的 `[repair]` 和旧自动修复入口已经移除。

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

默认 2 个 CPU 线程，顺序调用 API。邻帧只为本地 OCR 提供时序证据。
候选缓存默认硬限制 1536 MiB；高分辨率视频可减少候选或显式增加限额。
当前仅支持 SDR 8-bit 视频。HDR/>8-bit 输入会拒绝，避免把转成 BGR8 误称为保留原始色深。
输出采用编码像素方向，手机旋转元数据不自动应用，扫描时会在报告提示角度；可先明确旋转转码，再将转码视频作为像素基准。

## 测试与可复现验证

```bash
bash setup.sh --dev
source .venv/bin/activate
python -m pytest -q
python scripts/smoke_test.py --workdir /tmp/vfs-smoke-new
python -m pip_audit
```

Smoke 测试自行生成视频，不下载第三方素材；中英文示例需要系统 Noto CJK 字体，Debian/Ubuntu 可安装
`fonts-noto-cjk`。普通处理不需要这个字体。测试目录必须不存在，以免覆盖既有证据。

测试覆盖真实 CPU OCR、细 Mask、缺首/尾字回归、Gemini 整帧请求与配准、100% 掩码外保护、PNG 审计、
异常 OCR、重复帧、预算/缓存/重试、单次解码、断点续跑、危险输入限制，以及真实视频端到端流程。
实测结果及未实测范围见 `docs/VALIDATION.md`。

## 给 Codex Agent

`AGENTS.md` 固化执行规则；仓库技能在 `.agents/skills/video-frame-studio/SKILL.md`，并提供 UI 元数据。
按用户当次 prompt 查看候选和原帧，记录入选理由后调用改绘。交付前查看成品，并报告任何缺额和可见修复痕迹。

## 许可与边界

源码采用 MIT；直接依赖、模型哈希及第三方许可声明见 [第三方声明](docs/THIRD_PARTY_NOTICES.md)。
源视频、密钥、模型、缓存和交付产物应留在本地忽略目录。

固定不变字幕、极小字、艺术字、低对比度、透明花字可能漏检；整行没被 OCR 检出不能靠边缘扩展补全。
时序变化也不是严格的字幕证明：画面中电子屏幕或内容变化的包装可能需要保护区。
生成式修复与视觉复核存在误判，严格像素约束只保证掩码外，不能保证掩码内事实正确。
验证范围与限制见 [验证记录](docs/VALIDATION.md)。
