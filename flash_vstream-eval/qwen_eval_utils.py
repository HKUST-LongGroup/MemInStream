"""Standalone data, time and video metadata helpers for local Qwen evaluation."""
import json
import math
import os
import re
import subprocess
from pathlib import Path


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("rows", value.get("samples", value.get("data")))
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected JSON array or rows/samples/data array")
    else:
        value = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: every sample must be an object")
    return value


def atomic_write(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if jsonl:
            for row in value:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def seconds(value):
    if value is None or isinstance(value, bool):
        raise ValueError(f"invalid time: {value!r}")
    parts = str(value).strip().replace("：", ":").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"invalid time: {value!r}")
    number = sum(float(x) * (60 ** i) for i, x in enumerate(reversed(parts)))
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid time: {value!r}")
    return float(number)


def options(row):
    return [re.sub(r"^[A-Z][.)]\s*", "", str(x)).strip() for x in row.get("options") or []]


def probe_video(path, timeout):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_streams", "-show_format", "-of", "json", path],
                            capture_output=True, text=True, check=True, timeout=timeout)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return width, height, seconds(data.get("format", {}).get("duration", stream.get("duration")))
