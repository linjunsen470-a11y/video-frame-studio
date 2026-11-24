#!/usr/bin/env python3
"""Unified CLI. Exit codes: 0 success, 2 quality shortfall, 1 error, 130 interrupt."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import re
import sys

from vfs.config import load_config, load_dotenv, validate_config, ROOT
from vfs.util import digest, output_lock
from vfs.pipeline import run, quality_only


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CPU video stills with exact outside-mask pixel preservation")
    parser.add_argument("--input", action="append", help="Local video or direct public HTTP(S) media URL; repeat for a batch")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--budget-calls", type=int)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--subtitle-mode", choices=("auto", "subtitle", "all", "manual"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qc-only", action="store_true", help="Audit existing report/files; no Gemini calls or video decode")
    provider = parser.add_mutually_exclusive_group()
    provider.add_argument("--offline", action="store_true", help="Skip all Gemini work; keep strict deterministic gates")
    provider.add_argument("--mock-gemini", action="store_true", help="Test only; explicitly simulated AI results")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args(argv)
    try:
        load_dotenv(ROOT / ".env")
        cfg = load_config(args.config)
        mapping = (("count", "run", "count"), ("candidates", "video", "candidates"),
                   ("budget_calls", "gemini", "max_calls"), ("budget_usd", "gemini", "max_cost_usd"),
                   ("subtitle_mode", "ocr", "mode"))
        for argument, section, key in mapping:
            value = getattr(args, argument)
            if value is not None:
                cfg[section][key] = value
        if args.offline:
            cfg["gemini"]["enabled"], cfg["gemini"]["mock"] = False, False
        if args.mock_gemini:
            cfg["gemini"]["enabled"], cfg["gemini"]["mock"] = True, True
        if args.keep_intermediates:
            cfg["run"]["keep_intermediates"] = True
        validate_config(cfg)
        if args.qc_only:
            with output_lock(args.output):
                result = quality_only(args.output, cfg if args.config else None)
            print(f"{result['status']}: {args.output / 'qc_report.md'}")
            return 0 if result["status"] == "qc_passed" else 2
        if not args.input:
            parser.error("--input is required unless --qc-only")
        if cfg["gemini"]["enabled"] and not cfg["gemini"]["mock"] and not os.getenv(cfg["gemini"]["api_key_env"]):
            raise ValueError("GEMINI_API_KEY is missing. Configure .env, or use --offline / --mock-gemini explicitly")
        exit_code = 0
        batch_calls_used, batch_reserved_used = 0, 0.0
        for source in args.input:
            output = args.output
            if len(args.input) > 1:
                stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(source.split("?", 1)[0]).stem)[:40] or "video"
                output = output / f"{stem}-{digest(source)[:8]}"
            job_config = copy.deepcopy(cfg)
            # A repeated --input invocation shares the stated budget. Each
            # video's persistent ledger still includes its own previous attempts.
            job_config["gemini"]["max_calls"] = max(0, cfg["gemini"]["max_calls"] - batch_calls_used)
            job_config["gemini"]["max_cost_usd"] = max(0., cfg["gemini"]["max_cost_usd"] - batch_reserved_used)
            report = run(source, output, job_config, args.resume)
            batch_calls_used += report["api"].get("calls", 0)
            batch_reserved_used += report["api"].get("reserved_cost_usd", 0.)
            passed = sum(item["status"] == "passed" for item in report["items"])
            print(f"{report['status']}: {passed}/{report['requested_count']} images; {output / 'report.md'}")
            exit_code = max(exit_code, 0 if report["status"] == "complete" else 2)
        return exit_code
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
