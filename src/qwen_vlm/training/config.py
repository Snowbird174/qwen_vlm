"""读取 configs/train 下的 YAML。实验文件深度覆盖 base.yaml。"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_train_config(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if path.name != "base.yaml":
        base_path = path.parent / "base.yaml"
        if base_path.is_file():
            base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
            raw = deep_merge(base, raw)
    raw["_config_path"] = str(path)
    return raw
