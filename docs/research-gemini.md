# Gemini 接口选型与验证范围

核查日期：2026-11-24。仅参考 Google 官方文档和 `googleapis/python-genai` 原仓库。

## 选型

| 部分 | 项目采用 | 依据与边界 |
| --- | --- | --- |
| SDK | 官方 `google-genai`，已在 CPU Python 环境验证 2.34.0 的请求类型 | SDK 为 Apache-2.0；依赖许可不等同于云服务条款。没有引入 Agent 框架。 |
| 常规分析 | 配置默认 `gemini-3.1-flash-lite`；可通过环境变量替换 | 图像理解、结构化评分、OCR 区域语义分类和局部质检；仅提交传统算法筛选后的少量图片。更近的 `gemini-3.5-flash-lite` 可另行配置价格与模型，不自动升级。 |
| 困难修复 | 配置默认 `gemini-nano-banana-2.1` | 官方当前推荐的高效率图像编辑模型，支持多参考图。`gemini-3.1-flash-lite-image` 可作为低成本选项，但其官方说明不以多参考图一致性为主要优势。 |
| 原生 Mask | 不假设存在精确 Mask 请求参数 | 官方的语义 inpainting 通过提示描述修改位置；项目把 OCR mask 当作普通参考图片，输出始终只是裁剪区域修复提案。 |
| 最终像素保护 | 由本地 mask 合成和 PNG 差分完成 | 这是确定性代码属性，不能靠模型提示保证；不承诺恢复被遮挡内容的真实原貌，也不承诺规避 AI 内容标识。 |

官方来源：[SDK 与调用示例](https://github.com/googleapis/python-genai)、[SDK 许可证](https://github.com/googleapis/python-genai/blob/main/LICENSE)、[结构化输出](https://aistudio.google.com/docs/structured-output)、[图像理解](https://aistudio.google.com/docs/image-understanding)、[图像生成和编辑](https://aistudio.google.com/docs/image-generation)、[Nano Banana 2.1 模型卡](https://aistudio.google.com/docs/models/gemini-nano-banana-2.1)、[Flash-Lite 3.5 模型卡](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite)。

## 费用与重试

核查当日标准付费价格（美元 / 一百万 tokens）：

| 模型 | 输入 | 文本和思考输出 | 图像输出 |
| --- | ---: | ---: | ---: |
| `gemini-3.1-flash-lite` | 0.25 | 1.50 | 不适用 |
| `gemini-3.5-flash-lite` | 0.30 | 2.50 | 不适用 |
| `gemini-nano-banana-2.1` | 1.50 | 7.50 | 30.00 |
| `gemini-3.1-flash-lite-image` | 0.25 | 1.50 | 30.00 |

来源：[Google 标准价格表](https://aistudio.google.com/docs/pricing)。Nano Banana 2.1 的 1K 图像输出约 $0.0336，仅是图像输出部分；输入和思考另计。价格可能变化，模型与价格均配置在外部文件，程序不会根据模型名猜价。

`max_calls` 是跨断点续跑的请求尝试硬上限，失败、429、超时及重试均占用。`max_cost_usd` 限制的是**本地保守预留金额**，不是 Google 账单上限。每个请求先原子写入预留账本，再发起调用；中断和错误不退预留，因此不会因进程崩溃重复获得预算。未知模型价格且指定美元预算时拒绝付费调用。真实 token usage 用于另一项估算费用统计；拿不到 usage 时明确标记估算不完整。SDK 内部重试设为一次，自动函数调用和自动 continuation 均禁用，避免隐藏调用。独立进程通过文件锁串行使用同一账本。

API 超时采用 SDK 的毫秒参数；包装层对临时网络错误、限流及服务端错误退避重试，对认证等确定性失败直接停止。错误日志不写 API Key、完整请求或供应商响应正文。SDK 依据：[重试实现](https://github.com/googleapis/python-genai/blob/main/google/genai/_api_client.py)、[SDK 文档](https://googleapis.github.io/python-genai/)。

推荐配置 `analysis_thinking_level: "minimal"` 与 `image_thinking_level: "minimal"`。当前所选模型支持此设置；更换模型时按其文档调整或删除该配置。Nano Banana 2.1 默认 medium 思考，使用 minimal 可减少简单局部擦字的思考消耗，但不能保证每次都在输出 token 限额内完成。来源：[图像思考级别](https://aistudio.google.com/docs/image-generation#thinking-levels)、[思考参数](https://aistudio.google.com/docs/thinking)。

## 局部编辑与语义分类

按 Mask 连通域合并相邻字幕，再扩出有限上下文。裁剪图补成方形，提交原裁剪和黑白参考 Mask；请求 1K 方形输出。返回尺寸先做安全检查，宽高比偏离方形超过 3% 则拒绝，合法提案再缩放回原裁剪尺寸。无论 API 返回什么，最终交付必须经过本地掩码合成、原分辨率 PNG 保存及非 Mask 区域逐像素验证。

`classify_regions()` 仅协助判断 OCR 已检出的文字是否后期叠加。模型只能返回输入区域索引、类别和置信度；所有模型生成坐标均被忽略。模糊文字来源仍须保留为待复核，不能把单帧视觉判断当作确定事实。

## 已验证与尚未验证

`tests/test_gemini.py` 使用真实 SDK 2.34.0 的请求类型和离线响应替身，覆盖缓存与恢复、call/美元预算、失败重试、缺 Key、结构化输出、错误脱敏、并发预算、裁剪、异常宽高比、视觉质检阈值和文字分类。Mock 修复仅执行 OpenCV inpaint，用于验证接口编排；Mock 评分、语义分类和视觉复核均明确带 `mock` 字段，不代表模型质量。

**开发环境没有 Gemini API Key，未向 Google 发起真实付费请求。** 账户模型可用性、实际延迟、实际账单、复杂素材上的编辑保真和视觉判断质量尚待真实 API 验证。格式校验和离线测试不能替代这些验证。
