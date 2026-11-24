"""Deterministic typography with coverage checks before any text is drawn."""

from functools import lru_cache
import subprocess
import unicodedata

from PIL import ImageFont

from .design import asset_path


@lru_cache(maxsize=32)
def system_font(pattern):
    try:
        value = subprocess.check_output(
            ["fc-match", "-f", "%{file}\n%{index}", pattern], text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).splitlines()
        return value[0], int(value[1])
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        raise ValueError("Install fontconfig and a Simplified Chinese font such as fonts-noto-cjk") from None


@lru_cache(maxsize=32)
def coverage(path, index):
    try:
        ranges = subprocess.check_output(
            ["fc-query", "-i", str(index), "-f", "%{charset}", path], text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).split()
    except (OSError, subprocess.SubprocessError):
        raise ValueError(f"Could not read font coverage: {path}") from None
    spans = []
    for block in ranges:
        start, _, end = block.partition("-")
        spans.append((int(start, 16), int(end or start, 16)))
    return spans


def select_font(text, custom, base, *, bold=False, serif=False):
    weight = "bold" if bold else "regular"
    choices = []
    if weight in custom:
        spec = custom[weight]
        choices.append((str(asset_path(base, spec["path"])), spec.get("index", 0)))
    family = "Noto Serif CJK SC" if serif else "Noto Sans CJK SC"
    choices.extend((system_font(f"{family}:style={weight}:lang=zh-cn"),
                    system_font(f"Noto Sans CJK SC:style={weight}:lang=zh-cn")))
    codepoints = {ord(c) for c in unicodedata.normalize("NFC", text) if not c.isspace()}
    for path, index in dict.fromkeys(choices):
        spans = coverage(path, index)
        if all(any(a <= code <= b for a, b in spans) for code in codepoints):
            return path, index
    raise ValueError(f"No font covers all characters in {text[:32]!r}")


def wrap(text, draw, font, width):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for char in paragraph:
            trial = line + char
            if draw.textlength(trial, font=font) > width and line:
                # Preserve Latin word breaks when possible. Chinese may wrap between glyphs.
                boundary = line.rfind(" ")
                if boundary > 0 and char != " ":
                    lines.append(line[:boundary])
                    line = line[boundary+1:] + char
                else:
                    lines.append(line.rstrip())
                    line = char.lstrip()
            else:
                line = trial
        lines.append(line.rstrip())
    return "\n".join(lines)


def fitted_text(draw, text, width, height, size, minimum, font_spec, max_lines):
    text = unicodedata.normalize("NFC", text).strip()
    for current in range(size, minimum - 1, -2):
        font = ImageFont.truetype(font_spec[0], current, index=font_spec[1])
        spacing = max(2, round(current * .16))
        lines = wrap(text, draw, font, width)
        box = draw.multiline_textbbox((0, 0), lines, font=font, spacing=spacing)
        if (lines.count("\n") < max_lines and box[2] - box[0] <= width
                and box[3] - box[1] <= height):
            return lines, font, spacing, box
    raise ValueError(f"Text overflow: {text[:40]!r}")
