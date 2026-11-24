# 餐饮图集渲染选型

在目标 Linux 环境浅克隆了简报中的五个候选，检查提交、README、依赖、许可证声明和主要渲染入口。对比基于实际代码阅读，实跑验证集中在本期采用的 Pillow 方案。

| 候选与检查版本 | 代码中确认的能力与取舍 |
|---|---|
| gimpish `426e6c8` | MIT。具备场景描述、CLI 和可编辑图层。`packages/cli/src/browser-render.ts` 启动 Electron，并要求预先构建前端渲染包。适合后续交互式精修候选。 [代码](https://github.com/jvanderberg/gimpish/blob/426e6c89375a45930590adc40f46ed5d907459d0/packages/cli/src/browser-render.ts) |
| picture-it `5a3247d` | CLI、Sharp 合成、Satori/resvg 字体渲染及 JSON 链路。`src/fal.ts` 单列了外部 AI 处理。该提交未发现可确认的仓库许可证，未引入其源码或包。 [代码](https://github.com/geongeorge/picture-it/tree/5a3247d90298d8fbd21d675d02c293c1d6a13d52) |
| Photopea MCP `7dd1806` | MIT。已有文件、图层、文字和导出工具，实际执行通过 WebSocket 连接浏览器中的 Photopea。更适用于人工审图后的精修。 [代码](https://github.com/attalla1/photopea-mcp-server/tree/7dd18067e7fbfc15c81904e035787902c2dd7793/src) |
| Ultra Collage `cf4d84f` | React/Konva 画布和 Tauri 桌面入口，`src/layout` 有网格树和排布，图片导出位于客户端。README 声明 MIT，当次检出没有独立许可证文件，未复用源码。 [代码](https://github.com/viceduong/ultra-collage/tree/cf4d84fddcad942f2724a38784c3e96e429553bb) |
| OpenTempl `015064f` | MIT。具备编辑器、MCP、草稿与预览，仓库包含 Chromium/Puppeteer 渲染以及 Vercel/Cloudflare 服务分层。可用于模板研究。 [代码](https://github.com/noah-thing/opentempl/tree/015064f1005e63563f73ebe0dfbeb84d90439bcc) |

选用仓库已锁定的 **Pillow + fontconfig 系统字体**。Pillow 提供裁切、缩放、图层合成、模糊投影、FreeType 文字度量和 PNG 导出。小量网格坐标与 JSON 配置足以表达首批版式。整条生产链可在冷启动的 Python 环境串行运行，避免引入浏览器、数据库和云端生成调用。 [Pillow ImageOps 与 ImageDraw 文档](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html)

本期使用 16 张有来源记录的 Wikimedia Commons 实拍照片，以相同的内容和分组生成 A/B/C 三套图集，每套 6 页。验收覆盖中文字体、布局、单页重渲染、确定性 PNG 和源文件哈希。公开素材及渲染物在本地验收包中保留来源和许可信息，仓库仅包含源码与说明。没有实跑其他候选引擎来对比渲染时间，也没有进行平台曝光或转化测试。
