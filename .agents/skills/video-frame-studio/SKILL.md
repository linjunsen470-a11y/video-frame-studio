---
name: video-frame-studio
description: Extract high-quality PNG stills from a video and remove subtitles using this repository's CPU CLI, cached OCR, conservative masks and optional Gemini analysis or local editing. Use when asked to process videos, resume a run, or audit frame exports; do not redesign the workflow.
---

# Video Frame Studio

Work from the repository root containing `pipeline.py`. Read `AGENTS.md` once.

1. Initialize once: `bash setup.sh`; then `source .venv/bin/activate`.
2. Use the requested count and budget:
   `python pipeline.py --input INPUT --count N --output OUTPUT --budget-calls 40 --budget-usd 2`.
3. Supply `GEMINI_API_KEY` through the environment or local `.env`. Never print it.
   Without a key, explicitly choose `--offline`; use `--mock-gemini` only for simulated API testing.
4. Resume an interrupted run with the same arguments plus `--resume`. Preserve
   the budget ledger; do not delete caches to evade a limit. Increase limits only
   within the user's authorized budget. Final failed quality attempts stay failed.
5. Read `OUTPUT/report.md`. Inspect only relevant JSON items if needed. Exit 0
   meets the requested count; exit 2 is a quality shortfall; exit 1 is an error.
6. Audit restored/existing output:
   `python pipeline.py --output OUTPUT --qc-only`. This calls no Gemini API.
7. Deliver passed `images/*.png`, their originals/masks and the reports. Exclude
   review previews from final results. State any shortfall and whether Gemini was
   real, disabled or mocked.

Preserve true text through `ocr.keep_words` or `ocr.protect_regions` in a TOML
config. Use `--subtitle-mode all` only when the user intends all detected text to
be erased subject to protections. Do not lower quality gates to meet a count.
Never export a Gemini full-image response directly. Require 100% pixel equality
outside the saved final mask, including feather support. An uncertain OCR origin
or failed quality check must remain review/rejected.

For syntax or restrictions, read the relevant section of `README.md`. For a
reproducible CPU check, run:
`python scripts/smoke_test.py --workdir /tmp/vfs-smoke-new`.
Do not run tests repeatedly during ordinary video execution.
