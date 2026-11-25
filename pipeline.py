#!/usr/bin/env python3
"""Unified CLI. Exit codes: 0 success, 2 quality shortfall, 1 error, 130 interrupt."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from vfs.runtime import cloud_startup_allowed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CPU video stills with exact outside-mask pixel preservation")
    parser.add_argument("--input", help="Local video or direct public HTTP(S) media URL")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--budget-calls", type=int)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--subtitle-mode", choices=("auto", "subtitle", "all", "manual"))
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--qc-only", action="store_true", help="Audit existing PNGs without decoding video")
    mode.add_argument("--prepare", action="store_true", help="Generate candidate sheets without any Gemini calls")
    mode.add_argument("--select-frame", action="append", help="Edit a prepared frame ID; repeat in priority order")
    parser.add_argument("--mock-gemini", action="store_true", help="Test only; explicitly simulated AI results")
    args = parser.parse_args(argv)
    if not cloud_startup_allowed():
        return 1

    try:
        from vfs.config import ROOT, load_config, load_dotenv, validate_config
        from vfs.pipeline import prepare_selection, quality_only, run
        from vfs.util import output_lock

        load_dotenv(ROOT / ".env")
        cfg = load_config(args.config)
        mapping = (("count", "run", "count"), ("candidates", "video", "candidates"),
                   ("budget_calls", "gemini", "max_calls"), ("budget_usd", "gemini", "max_cost_usd"),
                   ("subtitle_mode", "ocr", "mode"))
        for argument, section, key in mapping:
            value = getattr(args, argument)
            if value is not None:
                cfg[section][key] = value
        if args.mock_gemini:
            cfg["gemini"]["mock"] = True
        validate_config(cfg)
        if args.qc_only:
            with output_lock(args.output):
                result = quality_only(args.output, cfg if args.config else None)
            print(f"{result['status']}: {args.output / 'qc_report.md'}")
            return 0 if result["status"] == "qc_passed" else 2
        if not args.input:
            parser.error("--input is required unless --qc-only")
        if args.prepare:
            if args.resume:
                parser.error("--prepare needs a fresh output directory")
            index = prepare_selection(args.input, args.output, cfg)
            print(f"prepared: {len(index['candidates'])} candidates; {args.output / 'selection/index.json'}")
            return 0
        if not args.select_frame:
            parser.error("Prepare candidates locally, then pass explicit --select-frame IDs")
        if not cfg["gemini"]["mock"] and not os.getenv(cfg["gemini"]["api_key_env"]):
            raise ValueError("GEMINI_API_KEY is missing. Configure .env for editing, or use --mock-gemini in tests")
        report = run(args.input, args.output, cfg, args.resume, selected=args.select_frame)
        passed = sum(item["status"] == "passed" for item in report["items"])
        print(f"{report['status']}: {passed}/{report['requested_count']} images; {args.output / 'report.md'}")
        return 0 if report["status"] == "complete" else 2
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Provider errors can contain credential-bearing URLs; never print their message.
        message = str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
