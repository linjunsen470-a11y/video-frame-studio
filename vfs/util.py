"""Small deterministic I/O helpers; no shell interpolation or pickle caches."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_image(path: Path, gray: bool = False) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"Cannot encode PNG: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data.tobytes())
    tmp.replace(path)


def resize_long(image: np.ndarray, limit: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= limit:
        return image
    scale = limit / max(h, w)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


@contextlib.contextmanager
def output_lock(directory: Path):
    """An OS lock is released on crash; concurrent writers never share a ledger."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".vfs.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is writing this output directory.") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def safe_output_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Report path escapes the output directory")
    return path
