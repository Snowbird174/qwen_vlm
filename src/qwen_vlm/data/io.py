"""JSONL 读写，以及把样本里的 image_path 解析到 data/ 下的绝对路径。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qwen_vlm import DATA_DIR


def load_jsonl(path: Path | str) -> list[dict]:
    """读取 JSONL。空行跳过；某一行不是 JSON 对象时抛出 ValueError。"""
    path = Path(path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: JSON 解析失败: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: 每行必须是 JSON 对象")
            rows.append(obj)
    return rows


def write_jsonl(path: Path | str, samples: list[dict]) -> None:
    """把样本写成 JSONL。键排序、UTF-8、每行一个对象，便于重复生成时比对哈希。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def resolve_image_path(image_path: str, data_dir: Path | str | None = None) -> Path:
    """把相对 data/ 的 image_path 解析为绝对路径。

    拒绝绝对路径和 ``..``，避免样本指向 data/ 以外的文件。
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    root = root.resolve()
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(f"image_path 为空或不是字符串: {image_path!r}")
    if "\\" in image_path:
        raise ValueError(f"image_path 请使用正斜杠: {image_path}")
    rel = Path(image_path)
    if rel.is_absolute():
        raise ValueError(f"image_path 必须是相对 data/ 的路径: {image_path}")
    if any(part == ".." for part in rel.parts):
        raise ValueError(f"image_path 不能包含 .. : {image_path}")
    full = (root / rel).resolve()
    if full != root and root not in full.parents:
        raise ValueError(f"image_path 逃出 data 目录: {image_path}")
    return full


def sha256_file(path: Path | str) -> str:
    """文件内容的 SHA256 十六进制摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
