#!/usr/bin/env python3
"""Real CPU/OCR end-to-end test; no external media and no real Gemini calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vfs.util import atomic_json, file_sha256  # noqa: E402


def execute(argv):
    subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True, help="A new, non-existing directory")
    p.add_argument("--mock-gemini", action="store_true", help="Exercise Mock API orchestration too")
    args = p.parse_args()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    video, truth, output = root / "demo.mkv", root / "truth", root / "output"
    execute(["scripts/make_demo.py", "--output", video, "--truth-dir", truth])
    command = ["pipeline.py", "--input", video, "--count", "3", "--output", output,
               "--config", "configs/demo.toml", "--mock-gemini" if args.mock_gemini else "--offline"]
    before = time.monotonic()
    execute(command)
    elapsed = time.monotonic() - before
    report = json.loads((output / "report.json").read_text())
    ground_truth = json.loads((truth / "ground_truth.json").read_text())
    results = []
    for item in report["items"]:
        if item["status"] != "passed":
            continue
        original = cv2.imread(str(output / item["files"]["original"]))
        final = cv2.imread(str(output / item["files"]["image"]))
        mask = cv2.imread(str(output / item["files"]["mask"]), 0) > 0
        gt = ground_truth["records"][item["frame_index"]]
        changed = cv2.imread(str(truth / gt["changed"]), 0) > 0
        coverage = float(np.mean(mask[changed]))
        assert coverage == 1.0, (item["id"], "missed synthetic subtitle pixel", coverage)
        assert original.shape == final.shape == (360, 640, 3)
        assert np.array_equal(original[~mask], final[~mask])
        for region in gt["protect"]:
            x0, y0, x1, y1 = region["bbox"]
            assert np.array_equal(original[y0:y1, x0:x1], final[y0:y1, x0:x1])
        assert item["qc"]["residual_text"] == [] and item["qc"]["passed"]
        results.append({"id": item["id"], "subtitle_truth_coverage": coverage,
                        "outside_pixel_equality": 1.0, "protected_pixels_equal": True,
                        "dimensions": [640, 360], "sha256": file_sha256(output / item["files"]["image"])})
    assert len(results) == 3
    assert report["api"]["api_calls"] == 0
    assert not (output / ".cache/frames").exists(), "Finished scan cache was not cleaned"
    execute([*command, "--resume"])
    assert not (output / ".cache/frames").exists(), "Completed resume unexpectedly decoded frames"
    for item in results:
        assert file_sha256(output / "images" / (item["id"] + ".png")) == item["sha256"]
    execute(["pipeline.py", "--output", output, "--qc-only"])
    assert json.loads((output / "qc_report.json").read_text())["status"] == "qc_passed"
    summary = {"passed": True, "real_ocr": True, "real_video_decode": True,
               "real_gemini_calls": 0, "mock_gemini": args.mock_gemini,
               "pipeline_wall_seconds": round(elapsed, 3),
               "scan": report["scan"], "api": report["api"], "ocr": report["ocr"],
               "items": results, "resume_no_redecode": True, "qc_only_passed": True}
    atomic_json(root / "smoke_result.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
