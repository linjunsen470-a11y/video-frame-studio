"""Run with python -m catering_visual."""

import argparse
import json
import os
from pathlib import Path
import sys

from vfs.runtime import cloud_startup_allowed

from .presets import PRESETS, STYLES


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare, render and inspect photo galleries in Codex Cloud")
    modes = parser.add_subparsers(dest="command")
    analyze = modes.add_parser("analyze", help="List local photo metadata for agent review")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--out", type=Path)
    plan = modes.add_parser("plan", help="Write a design from explicitly chosen page groups")
    plan.add_argument("input", type=Path)
    plan.add_argument("--out", required=True, type=Path)
    plan.add_argument("--style", choices=STYLES, default="real-photo")
    plan.add_argument("--page", nargs="+", action="append", required=True, metavar=("PRESET", "PHOTO"),
                      help="Repeat with one preset followed by its photo paths, in visual order")
    render = modes.add_parser("render", help="Render all pages or repeat --page for a subset")
    render.add_argument("plan", type=Path)
    render.add_argument("--out", required=True, type=Path)
    render.add_argument("--page", action="append", metavar="ID")
    inspect = modes.add_parser("inspect", help="Audit PNGs and optionally check the current design and inputs")
    inspect.add_argument("output", type=Path)
    inspect.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)
    if not cloud_startup_allowed():
        return 1
    if args.command is None:
        parser.error("the following arguments are required: command")

    try:
        from .design import asset_path, inventory, load_plan, validate_plan
        from .render import inspect_gallery, render_gallery, write_json

        if args.command in {"analyze", "plan"} and args.out and args.out.suffix.lower() != ".json":
            raise ValueError("--out must be a .json file for analyze and plan")
        if args.command == "analyze":
            result = inventory(args.input.resolve())
            if args.out:
                write_json(args.out, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["assets"] and not result["errors"] else 2
        if args.command == "plan":
            root = args.input.resolve()
            pages = []
            for i, (preset, *photos) in enumerate(args.page, 1):
                if preset not in PRESETS:
                    raise ValueError(f"Unknown preset {preset}. Choices: {', '.join(PRESETS)}")
                for name in photos:
                    asset_path(root, name)
                pages.append({"id": f"page-{i:02d}", "preset": preset,
                              "photos": [{"src": name} for name in photos]})
            value = {"version": 1, "asset_root": os.path.relpath(root, args.out.resolve().parent),
                     "style": args.style, "canvas": {"width": 1080, "height": 1440}, "pages": pages}
            write_json(args.out, validate_plan(value))
            print(args.out)
            return 0
        path = args.plan.resolve() if args.plan else None
        value = load_plan(path) if path else None
        if args.command == "render":
            result = render_gallery(value, path, args.out.resolve(), args.page)
            active = set(args.page or [row["id"] for row in result["pages"]])
            passed = all(row["status"] == "rendered" for row in result["pages"] if row["id"] in active)
        else:
            result = inspect_gallery(args.output.resolve(), value, path)
            passed = bool(result["pages"]) and all(row["status"] == "passed" for row in result["pages"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if passed else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"catering-visual: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
