# 餐饮实拍图集

从已准备好的 JPEG、PNG、WebP 静态素材开始。Agent 查看照片，按本次目标分类、选择、分组和填写业务已确认的文案。程序在 CPU 上逐页渲染 PNG，保留每页的来源哈希和照片区域坐标。

小红书团餐项目可先用 [团餐图文技能](../.agents/skills/xiaohongshu-catering-editorial/SKILL.md) 确定首图、翻图顺序和发布文案，再把选好的照片和文字写入图集设计。

## 安装

在仓库根目录运行。现有 `.venv` 已安装 Pillow，可直接复用。单独运行图集只需 Python 3.12、仓库锁定的 Pillow 版本和字体。Debian/Ubuntu 上可安装 `fontconfig` 与 `fonts-noto-cjk`。无需 API Key。

## 从素材到图集

例如，把私有原图放在 `runs/catering/photos/`。原图、设计、待发文案和生成文件均留在已忽略的 `runs/` 或 `catering-xhs/` 下。

```bash
python -m catering_visual analyze runs/catering/photos --out runs/catering/inventory.json

python -m catering_visual plan runs/catering/photos --out runs/catering/design.json \
  --style editorial \
  --page canteen-hero hall.jpg \
  --page hero-1plus3 line.jpg dish-1.jpg dish-2.jpg dish-3.jpg

python -m catering_visual render runs/catering/design.json --out runs/catering/gallery
python -m catering_visual inspect runs/catering/gallery --plan runs/catering/design.json
```

`analyze` 输出经 EXIF 方向校正的尺寸、内容哈希及待标注的类别。Agent 根据实际图像填写分组和文案。程序只渲染设计文件里的文字，因此合作案例、供餐量、认证和配送范围等事实需要事先核实。素材不够时减少页数。

编辑 `design.json`，为页面添加 `label`、`headline`、`caption`，为照片添加可选的 `label`、`fit`、`focal` 或调色参数。文字位于照片区域之外。补充标题或调整一页后，用 `--page` 重渲染该页：

```bash
python -m catering_visual render runs/catering/design.json \
  --out runs/catering/gallery --page page-02
python -m catering_visual inspect runs/catering/gallery --plan runs/catering/design.json
```

看图集用 `preview.html`，发布图片在 `pages/`，以 `gallery-report.json` 中状态为 `rendered` 的记录为索引。图片或设计改动后先重新审计，避免发布旧文件。退出码 0 表示目标页成功，2 表示报告有失败或缺页，1 表示输入或配置错误。

## 预设和风格

| 预设 | 布局 | 照片数 |
|---|---|---:|
| `canteen-hero` | `single-hero` | 1 |
| `food-hero` | `single-hero` | 1 |
| `food-grid-4` | `grid-2x2` | 4 |
| `hero-1plus3` | `hero-1plus3` | 4 |
| `kitchen-process` | `three-step-story` | 3 |
| `ingredient-mosaic` | `asymmetric-mosaic` | 3 |

- **`real-photo`** 使用自然米白底、细边框和大图。
- **`editorial`** 使用暖色背景、较大的中文衬线标题和分隔线。
- **`corporate-clean`** 使用冷绿色调、圆角及轻投影。

预设引用布局。页面也可直接指定 `layout`。照片顺序对应槽位，主图总在前。整个图集指定 `style`，单页可覆盖 `style` 和 `theme`。新布局可在 `catering_visual/presets.py` 添加网格槽，新风格在同文件添加视觉参数。

### 设计文件示例

```json
{
  "version": 1,
  "asset_root": "photos",
  "canvas": {"width": 1080, "height": 1440},
  "style": "real-photo",
  "pages": [
    {
      "id": "cover",
      "preset": "canteen-hero",
      "style": "editorial",
      "theme": {"accent": "#804032"},
      "label": "食堂空间",
      "headline": "走进食堂",
      "photos": [{"src": "hall.jpg", "fit": "cover", "focal": [0.58, 0.45]}]
    }
  ]
}
```

- `asset_root` 相对设计文件，照片路径相对素材根目录。照片不得通过路径或符号链接逃出根目录。
- `cover` 填满区域并按 `focal` 分配各方向的裁切量，`[0.5, 0.5]` 居中。`contain` 完整保留画面，必要时留边。三种轻度调色项为 `brightness`、`contrast`、`saturation`，范围均为 0.85–1.15。
- `theme` 可覆盖 `background`、`foreground`、`accent`、`line`、`margin`、`gap`、`radius`、`border`、`shadow`、`headline_size`。尺寸参数以 1080 × 1440 为基准。页级覆盖优先于全局设置。
- 可用 `"fonts": {"regular": {"path": "fonts/body.otf"}, "bold": {"path": "fonts/title.ttc", "index": 2}}` 指定设计目录内具备授权的字体。程序检查实际字符覆盖，必要时回退到系统中文字体。缺字或无法在规定区域内排版时，该页失败。固定字体文件、Pillow 版本、设计和照片可复现输出。

## 质检边界

程序验证选定页面尺寸、PNG/RGB、空白、字体覆盖、文字边界、输入输出哈希和素材重复。照片和文字使用独立的布局区域。来源分辨率低、文字对比度低和风格混用会出现在报告中。

裁切是否保住食物主体、跨页叙事是否成立、文案是否真实仍需 Agent 目视和核实来源。公开演示素材仅用于对照版式，不代表合作伙伴、项目案例或小红书运营验证。发布前使用获授权的业务照片。第一阶段只导出 PNG，保留其余风格、自由式图层、更多输出格式和运营反馈供后续验证。
