"""Strict TOML configuration and a deliberately small, non-executable .env loader."""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("Invalid .env line: expected NAME=value")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("Invalid environment variable name in .env")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_config(path: Path | None = None) -> dict:
    with (ROOT / "configs/default.toml").open("rb") as f:
        config = tomllib.load(f)
    if path:
        with path.open("rb") as f:
            custom = tomllib.load(f)
        for section, values in custom.items():
            if section not in config or not isinstance(values, dict):
                raise ValueError(f"Unknown configuration section: {section}")
            # Provider settings can grow without a pipeline release. Other sections strict.
            if section not in {"gemini", "ocr", "mask", "repair"}:
                unknown = set(values) - set(config[section])
                if unknown:
                    raise ValueError(f"Unknown settings in {section}: {sorted(unknown)}")
            config[section].update(values)
    for key, env in (("analysis_model", "GEMINI_ANALYSIS_MODEL"),
                     ("image_model", "GEMINI_IMAGE_MODEL")):
        if os.getenv(env):
            config["gemini"][key] = os.environ[env]
    return config


def validate_config(cfg: dict) -> None:
    def finite_tree(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                finite_tree(child, f"{prefix}.{key}")
        elif isinstance(value, list):
            for child in value:
                finite_tree(child, prefix)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite number in configuration: {prefix}")
    finite_tree(cfg)
    integer_fields = {"run": ("count", "max_attempts", "threads"),
                      "video": ("candidates", "max_candidates", "max_scan_samples", "neighbors", "max_pixels"),
                      "gemini": ("max_calls", "max_retries", "rank_batch_size")}
    for section, keys in integer_fields.items():
        for key in keys:
            if type(cfg[section][key]) is not int:
                raise ValueError(f"{section}.{key} must be an integer")
    if not 1 <= cfg["run"]["count"] <= 500:
        raise ValueError("count must be between 1 and 500")
    if not 1 <= cfg["run"]["max_attempts"] <= 3:
        raise ValueError("max_attempts must be between 1 and 3")
    for sec, key in (("video", "max_candidates"), ("video", "max_scan_samples"),
                     ("video", "max_input_mb"), ("video", "max_cache_mb"),
                     ("video", "max_duration_seconds"), ("video", "max_pixels"),
                     ("video", "sample_seconds"), ("run", "threads")):
        if cfg[sec][key] <= 0:
            raise ValueError(f"{sec}.{key} must be positive")
    if cfg["ocr"]["mode"] not in {"auto", "subtitle", "all", "manual"}:
        raise ValueError("Unknown subtitle mode")
    if cfg["gemini"]["max_calls"] < 0 or cfg["gemini"]["max_cost_usd"] < 0:
        raise ValueError("API budgets cannot be negative")
    if cfg["video"]["candidates"] < 0:
        raise ValueError("candidates cannot be negative")
    if not 1 <= cfg["gemini"]["rank_batch_size"] <= 16:
        raise ValueError("rank_batch_size must be between 1 and 16")
    for key in ("max_retries", "crop_padding"):
        if cfg["gemini"][key] < 0:
            raise ValueError(f"gemini.{key} cannot be negative")
    for key in ("calls_per_minute", "timeout_seconds"):
        if cfg["gemini"][key] <= 0:
            raise ValueError(f"gemini.{key} must be positive")
    if not 0 <= cfg["selection"]["semantic_weight"] <= 1:
        raise ValueError("semantic_weight must be between 0 and 1")
    for section, key in (("qc", "max_mask_fraction"), ("mask", "max_area_fraction"),
                         ("ocr", "confidence"), ("ocr", "bottom_start"),
                         ("qc", "residual_overlap"), ("qc", "min_sharpness_ratio")):
        if not isinstance(cfg[section][key], (int, float)) or not 0 <= cfg[section][key] <= 1:
            raise ValueError(f"{section}.{key} must be between 0 and 1")
    if type(cfg["selection"]["phash_distance"]) is not int or not 0 <= cfg["selection"]["phash_distance"] <= 64:
        raise ValueError("selection.phash_distance must be an integer between 0 and 64")
    for key in ("feather_px", "dilate_px", "shadow_px", "max_padding"):
        if type(cfg["mask"][key]) is not int or not 0 <= cfg["mask"][key] <= 64:
            raise ValueError(f"mask.{key} must be an integer between 0 and 64")


def semantic_config(cfg: dict) -> dict:
    """Resume permits larger budgets/threads, never silently different image semantics."""
    result = copy.deepcopy(cfg)
    for key in ("threads", "keep_intermediates", "previews"):
        result["run"].pop(key, None)
    for key in ("max_calls", "max_cost_usd", "calls_per_minute", "timeout_seconds"):
        result["gemini"].pop(key, None)
    return result
