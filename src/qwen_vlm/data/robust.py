"""Deterministic, evaluation-only robustness variants for ``real_v1``.

Every condition, including clean, uses the same RGB -> transform -> JPEG q95
pipeline.  Outputs live below ``data/robust/`` and are never training files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from qwen_vlm import DATA_DIR
from qwen_vlm.data.io import load_jsonl, resolve_image_path, sha256_file, write_jsonl
from qwen_vlm.data.validate import validate_samples
from qwen_vlm.inference.base import load_image

JPEG_QUALITY = 95
OUTPUT_VERSION = "robust-v1"
OUTPUT_MARKER = ".robust-v1.json"
DEFAULT_SEED = 0


class RobustError(ValueError):
    """Raised when robustness generation violates its evaluation-only contract."""


@dataclass(frozen=True)
class RobustCondition:
    name: str
    id_suffix: str
    kind: str
    parameters: dict[str, object]


_CONDITION_SPECS = (
    ("clean", "r00", "identity", {}),
    ("dark_l1", "r01", "brightness", {"factor": 0.5, "output_clip": [0, 255]}),
    ("dark_l2", "r02", "brightness", {"factor": 0.3, "output_clip": [0, 255]}),
    ("bright_l1", "r03", "brightness", {"factor": 1.5, "output_clip": [0, 255]}),
    ("bright_l2", "r04", "brightness", {"factor": 2.0, "output_clip": [0, 255]}),
    ("blur_l1", "r05", "gaussian_blur", {"sigma": 2.0, "backend": "Pillow ImageFilter.GaussianBlur"}),
    ("blur_l2", "r06", "gaussian_blur", {"sigma": 4.0, "backend": "Pillow ImageFilter.GaussianBlur"}),
    ("noise_l1", "r07", "gaussian_noise", {"mean": 0.0, "sigma": 15.0, "output_clip": [0, 255]}),
    ("noise_l2", "r08", "gaussian_noise", {"mean": 0.0, "sigma": 30.0, "output_clip": [0, 255]}),
)
CONDITIONS = tuple(RobustCondition(*spec) for spec in _CONDITION_SPECS)
CONDITION_NAMES: tuple[str, ...] = tuple(condition.name for condition in CONDITIONS)
CONDITIONS_BY_NAME = {condition.name: condition for condition in CONDITIONS}


@dataclass(frozen=True)
class SourceImage:
    image_path: str
    image_id: str
    rows: tuple[dict, ...]
    sha256: str


@dataclass(frozen=True)
class RobustReport:
    output_dir: Path
    manifest_path: Path
    contact_sheet_path: Path
    source_images: int
    source_samples: int
    generated_images: int
    generated_samples: int
    condition_counts: dict[str, int]


def condition_by_name(name: str) -> RobustCondition:
    if name not in CONDITIONS_BY_NAME:
        raise RobustError(
            f"未知条件 {name!r}，可用: {', '.join(CONDITION_NAMES)}"
        )
    return CONDITIONS_BY_NAME[name]


def _split_sample_id(sample_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"([A-Za-z0-9_]+)_q(\d{2,})", sample_id)
    if match is None:
        raise RobustError(f"sample_id 必须形如 <prefix>_qNN: {sample_id!r}")
    return match.group(1), match.group(2)


def sample_id_for_condition(sample_id: str, id_suffix: str) -> str:
    """Append the suffix inside the image/variant segment, before ``_qNN``."""
    prefix, seq = _split_sample_id(sample_id)
    return f"{prefix}{id_suffix}_q{seq}"


def pair_group_id_for_condition(pair_group_id: str | None, id_suffix: str) -> str | None:
    """Append the suffix to the final pair segment; preserve null."""
    return None if pair_group_id is None else f"{pair_group_id}{id_suffix}"


def deterministic_transform_seed(seed: int, source_sha256: str, condition_name: str) -> int:
    payload = f"{seed}\0{source_sha256}\0{condition_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def apply_condition(image: Image.Image, condition: RobustCondition, *, transform_seed: int) -> Image.Image:
    """Apply one condition, preserving RGB mode and dimensions."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    parameters = condition.parameters
    if condition.kind == "identity":
        transformed = rgb.copy()
    elif condition.kind == "brightness":
        transformed = np.clip(rgb.astype(np.float32) * float(parameters["factor"]), 0, 255)
    elif condition.kind == "gaussian_blur":
        blurred = Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(float(parameters["sigma"])))
        transformed = np.asarray(blurred, dtype=np.uint8)
    elif condition.kind == "gaussian_noise":
        rng = np.random.Generator(np.random.PCG64(transform_seed))
        noise = rng.normal(float(parameters["mean"]), float(parameters["sigma"]), rgb.shape)
        transformed = np.clip(rgb.astype(np.float32) + noise.astype(np.float32), 0, 255)
    else:
        raise RobustError(f"未知变换类型: {condition.kind!r}")
    return Image.fromarray(transformed.astype(np.uint8))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RobustError(f"路径必须位于 {root}: {path}") from exc


def _assert_evaluation_output(out_dir: Path, data_dir: Path) -> None:
    relative = Path(_relative(out_dir, data_dir))
    forbidden = {"train", "val", "test", "splits"} & set(relative.parts)
    if relative.parts[:1] != ("robust",) or forbidden:
        raise RobustError("鲁棒数据仅供评测，输出必须位于 data/robust/ 下")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, text.encode("utf-8"))


def _atomic_jpeg(path: Path, image: Image.Image, *, quality: int = JPEG_QUALITY) -> None:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=quality, subsampling=0,
        optimize=False, progressive=False,
    )
    _atomic_write(path, buffer.getvalue())


def _prepare_output(out_dir: Path) -> None:
    """Require our marker when reusing a non-empty output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / OUTPUT_MARKER
    if any(out_dir.iterdir()) and not marker.is_file():
        raise RobustError(f"{out_dir} 已非空且缺少 {OUTPUT_MARKER}；拒绝覆盖")
    if marker.is_file() and json.loads(marker.read_text()).get("version") != OUTPUT_VERSION:
        raise RobustError(f"{marker} 的版本不是 {OUTPUT_VERSION}")
    marker_data = {"version": OUTPUT_VERSION, "training_eligible": False, "conditions": list(CONDITION_NAMES)}
    _atomic_json(marker, marker_data)


def _load_sources(samples: Sequence[dict], data_dir: Path) -> list[SourceImage]:
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["image_path"]), []).append(sample)

    sources: list[SourceImage] = []
    seen_ids: set[str] = set()
    for image_path, rows in grouped.items():
        ids = {_split_sample_id(row["sample_id"])[0] for row in rows}
        if len(ids) != 1:
            raise RobustError(f"{image_path}: 同图 sample_id 前缀不一致")
        image_id = ids.pop()
        if image_id in seen_ids:
            raise RobustError(f"多张原图使用同一个 ID: {image_id}")
        seen_ids.add(image_id)
        source_path = resolve_image_path(image_path, data_dir)
        sources.append(SourceImage(image_path, image_id, tuple(rows), sha256_file(source_path)))
    if not sources:
        raise RobustError("输入真实样本为空")
    return sources


def _image_path(out_dir: Path, condition: RobustCondition, source: SourceImage) -> Path:
    return out_dir / condition.name / "images" / f"{source.image_id}{condition.id_suffix}.jpg"


def _samples_for_condition(
    sources: Sequence[SourceImage],
    condition: RobustCondition,
    image_paths: dict[str, str],
) -> list[dict]:
    result: list[dict] = []
    for source in sources:
        for original in source.rows:
            sample = dict(original)
            sample["sample_id"] = sample_id_for_condition(
                original["sample_id"], condition.id_suffix
            )
            sample["image_path"] = image_paths[source.image_path]
            sample["pair_group_id"] = pair_group_id_for_condition(
                original.get("pair_group_id"), condition.id_suffix
            )
            tags = list(original.get("condition_tags") or [])
            robust_tag = f"robust_{condition.name}"
            if robust_tag not in tags:
                tags.append(robust_tag)
            sample["condition_tags"] = tags
            result.append(sample)
    return result


def write_contact_sheet(
    *,
    sources: Sequence[SourceImage],
    condition_images: dict[str, dict[str, str]],
    data_dir: Path,
    out_dir: Path,
    cell_size: tuple[int, int] = (240, 135),
) -> Path:
    """Write a full-review sheet with original, clean and eight conditions."""
    width, height = cell_size
    label_height = 30
    columns = 1 + len(CONDITIONS)
    canvas = Image.new("RGB", (columns * width, len(sources) * (height + label_height)), (246, 246, 242))
    draw = ImageDraw.Draw(canvas)
    for row_index, source in enumerate(sources):
        entries = [("original", source.image_path)]
        entries.extend((c.name, condition_images[c.name][source.image_path]) for c in CONDITIONS)
        for column, (label, image_path) in enumerate(entries):
            with load_image(resolve_image_path(image_path, data_dir)) as image:
                thumbnail = image.copy()
            thumbnail.thumbnail((width, height), Image.Resampling.LANCZOS)
            left = column * width
            top = row_index * (height + label_height)
            position = (left + (width - thumbnail.width) // 2, top + (height - thumbnail.height) // 2)
            canvas.paste(thumbnail, position)
            draw.rectangle((left, top + height, left + width, top + height + label_height), fill=(54, 67, 74))
            draw.text((left + 5, top + height + 8), f"{source.image_id} | {label}", fill="white")
    output = out_dir / "contact_sheet.jpg"
    _atomic_jpeg(output, canvas, quality=90)
    return output


def generate_robust_dataset(
    *,
    real_jsonl: Path | str = DATA_DIR / "annotations" / "real_v1.jsonl",
    out_dir: Path | str = DATA_DIR / "robust" / "v1",
    data_dir: Path | str = DATA_DIR,
    seed: int = DEFAULT_SEED,
    contact_sheet_cell_size: tuple[int, int] = (240, 135),
) -> RobustReport:
    """Generate, validate and publish all nine evaluation conditions."""
    data_root = Path(data_dir).resolve()
    source_path = Path(real_jsonl)
    output_root = Path(out_dir)
    _assert_evaluation_output(output_root, data_root)
    relative_source = _relative(source_path, data_root)
    source_samples = load_jsonl(source_path)
    if errors := validate_samples(source_samples, data_dir=data_root, check_images=True):
        raise RobustError("真实输入校验失败:\n" + "\n".join(errors[:20]))
    if any(
        row.get("source") != "real_lab"
        or any(tag.startswith("robust_") for tag in row.get("condition_tags", []))
        for row in source_samples
    ):
        raise RobustError("鲁棒扰动只能由 source=real_lab 的样本生成")
    sources = _load_sources(source_samples, data_root)
    _prepare_output(output_root)

    condition_images: dict[str, dict[str, str]] = {}
    condition_manifest: dict[str, dict] = {}
    for condition in CONDITIONS:
        image_paths: dict[str, str] = {}
        image_records: list[dict] = []
        for source in sources:
            image_file = _image_path(output_root, condition, source)
            _relative(image_file, output_root)
            transform_seed = deterministic_transform_seed(seed, source.sha256, condition.name)
            with load_image(resolve_image_path(source.image_path, data_root)) as image:
                transformed = apply_condition(image, condition, transform_seed=transform_seed)
            _atomic_jpeg(image_file, transformed)
            image_paths[source.image_path] = _relative(image_file, data_root)
            image_records.append({"image_path": image_paths[source.image_path], "sha256": sha256_file(image_file)})

        samples = _samples_for_condition(sources, condition, image_paths)
        samples_path = output_root / condition.name / "samples.jsonl"
        write_jsonl(samples_path, samples)
        condition_images[condition.name] = image_paths
        condition_manifest[condition.name] = {
            "kind": condition.kind,
            "parameters": condition.parameters,
            "id_suffix": condition.id_suffix,
            "samples_path": _relative(samples_path, data_root),
            "samples_sha256": sha256_file(samples_path),
            "sample_count": len(samples),
            "image_count": len(image_records),
            "images": image_records,
        }
    contact_sheet = write_contact_sheet(
        sources=sources, condition_images=condition_images, data_dir=data_root,
        out_dir=output_root, cell_size=contact_sheet_cell_size,
    )
    manifest = {
        "version": OUTPUT_VERSION,
        "training_eligible": False,
        "source_jsonl": relative_source,
        "source_jsonl_sha256": sha256_file(source_path),
        "seed": seed,
        "jpeg_quality": JPEG_QUALITY,
        "conditions": condition_manifest,
        "contact_sheet": _relative(contact_sheet, data_root),
    }
    manifest_path = output_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return RobustReport(
        output_dir=output_root,
        manifest_path=manifest_path,
        contact_sheet_path=contact_sheet,
        source_images=len(sources),
        source_samples=len(source_samples),
        generated_images=sum(record["image_count"] for record in condition_manifest.values()),
        generated_samples=sum(record["sample_count"] for record in condition_manifest.values()),
        condition_counts={name: record["sample_count"] for name, record in condition_manifest.items()},
    )
