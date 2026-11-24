#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ required"'
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  if command -v apt-get >/dev/null && [ "$(id -u)" -eq 0 ]; then
    apt-get update -qq
    apt-get install -y --no-install-recommends ffmpeg
  elif command -v apt-get >/dev/null && command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    sudo -n apt-get update -qq
    sudo -n apt-get install -y --no-install-recommends ffmpeg
  else
    echo 'Install ffmpeg (including ffprobe) with your system package manager, then rerun setup.sh.' >&2
    exit 1
  fi
fi
if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check 'pip>=26.3,<27'
requirements_file=requirements.lock
if [ "${1:-}" = "--dev" ]; then
  requirements_file=requirements-dev.txt
elif [ "$#" -gt 0 ]; then
  echo 'Usage: bash setup.sh [--dev]' >&2
  exit 1
fi
.venv/bin/python -m pip install --disable-pip-version-check --only-binary=:all: -r "$requirements_file"
.venv/bin/python -m pip check
.venv/bin/python -c 'import cv2, rapidocr, onnxruntime; from google import genai; print("CPU runtime ready")'
echo 'Ready: source .venv/bin/activate'
echo 'Then configure .env from .env.example, or use --offline.'
