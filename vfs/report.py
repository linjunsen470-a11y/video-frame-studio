from __future__ import annotations

from collections import Counter
from pathlib import Path

from .util import atomic_json


def write_report(output: Path, report: dict, name: str = "report") -> None:
    atomic_json(output / f"{name}.json", report)
    counts = Counter(item["status"] for item in report.get("items", []))
    api = report.get("api", {})
    lines = ["# Video Frame Studio 质检报告", "",
             f"- 状态：**{report['status']}**",
             f"- 目标 {report.get('requested_count', 0)} 张；通过 {counts['passed']} 张；"
             f"待人工审核 {counts['review']} 张；淘汰 {counts['rejected']} 张。",
             f"- 模型模式：{report.get('mode', 'unknown')}。",
             f"- 实际 API 尝试：{api.get('api_calls', 0)}；Mock：{api.get('mock_calls', 0)}；"
             f"缓存命中：{api.get('cache_hits', 0)}。",
             "- 原始像素基准：FFmpeg/OpenCV 解码的 8-bit BGR 帧；PNG 无损复核。",
             "- 掩码外一致性要求：100%；羽化范围包含于最终掩码。",
             "- OCR/自然程度检查有漏检可能；本报告不承诺重建被遮挡的真实内容，"
             "也不承诺规避任何 AI 内容识别或标注。", ""]
    if report.get("warnings"):
        lines += ["## 提示", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    lines += ["## 帧结果", "", "| 帧 | 时间/s | 结果 | 修复方法 | 掩码外改动像素 | 原因 |",
              "|---|---:|---|---|---:|---|"]
    for item in report.get("items", []):
        qc = item.get("qc", {})
        issues = ", ".join(qc.get("issues", item.get("issues", [])))
        # OCR/provider text can contain Markdown; do not turn it into active markup.
        issues = issues.replace("|", "/").replace("\n", " ")
        lines.append(f"| {item['id']} | {item.get('time', 0):.3f} | {item['status']} | "
                     f"{item.get('method', '—')} | {qc.get('outside_changed_pixels', '—')} | {issues} |")
    lines += ["", "完整坐标、掩码元数据、重试、文件哈希、预算与检查结果见同目录 JSON。", ""]
    (output / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
