# 小红书实拍图集

选用照片或通过抽帧质检的 PNG，以翻图顺序准备素材。每次交付含 3:4 JPEG 和可复制的笔记文案。命令在 Codex Cloud 仓库根目录运行。

## 生成和复核

```bash
python -m catering_visual analyze runs/catering/photos --out runs/catering/inventory.json

python -m catering_visual plan runs/catering/photos --out runs/catering/design.json \
  --copy runs/catering/copy.json \
  --page open-box.png \
  --page packing-wide.png loading-wide.png

python -m catering_visual render runs/catering/design.json --out runs/catering/gallery
python -m catering_visual inspect runs/catering/gallery --plan runs/catering/design.json
```

文案 JSON 包含 `title`、`body` 和 `tags`。话题数组中的名称省略井号。

```json
{
  "title": "员工餐怎么装盒？",
  "body": "打开餐盒，米饭和配菜分格摆放。",
  "tags": ["员工餐", "团餐配送"]
}
```

每个 `--page` 收一张竖图或两张不同的横图。竖图默认满版。若主体会被裁掉，在版本 2 的设计文件中调整该页 `focal`，或指定 `"fit": "contain"` 保留全图。双横帧依照参数顺序上下等比拼接，默认 900 宽时中缝 12 像素。缺少第二张横帧时该页无法生成。

成品在 `pages/*.jpg`，文案在 `post.md`，可用 `preview.html` 检查顺序。审计检查比例、尺寸、文件字节数、文案和源图哈希。带 ICC 色彩配置（如 Display P3）的实拍图自动转到 sRGB；JPEG 采用渐进式编码，从 900 × 1200 的 4:4:4 质量 90 依次降级至 4:2:0 质量 90/85/80/75，必要时逐级等比缩小，最低 600 × 800。单张超 300,000 字节、标题超小红书 20 字上限或正文超 150 字会在报告提示。原始修复 PNG 留在抽帧任务目录。

加 `--page page-02` 可只重渲染指定页。最终交付前审计全套。旧版设计请用新版 `plan` 命令从源图重建，历史输出文件仍可直接查看。
