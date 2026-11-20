"""Deterministic box-guided paired data augmentation.

Real images are never emitted as training samples.  Each source image produces
one or more pairs where a bait edit and a true edit use the same visual
operation.  The pair differs in exactly one item state, and every generated
sample inherits the source image's session/scene ids.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from qwen_vlm import DATA_DIR, PROJECT_ROOT
from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATES
from qwen_vlm.data.convert import load_question_templates
from qwen_vlm.data.io import load_jsonl, resolve_image_path, sha256_file
from qwen_vlm.data.validate import validate_samples

ANNOTATION_TO_CATEGORY = {
    "screwdriver": "screwdriver",
    "pliers": "pliers",
    "cable": "cable_bundle",
    "cable_bundle": "cable_bundle",
    "multimeter": "multimeter",
    "hammer": "hammer",
}
SYNTHETIC_SOURCE = "synthetic_edit"
DEFAULT_OUTPUT_SIZE = (1280, 720)
DEFAULT_JPEG_QUALITY = 95
DEFAULT_PAIRS_PER_IMAGE = 2
DEFAULT_REMOVAL_PADDING = 0.16
DEFAULT_OCCLUSION_SCALE = 0.82
MIN_REMOVAL_COVERAGE = 0.78


class AugmentError(ValueError):
    """Raised when inputs or generated pairs violate the augmentation contract."""


@dataclass(frozen=True)
class BoxAnnotation:
    item: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class AnnotationImage:
    image: str
    width: int
    height: int
    boxes: tuple[BoxAnnotation, ...]


@dataclass(frozen=True)
class SourceImage:
    image_path: str
    sample_prefix: str
    session_id: str
    scene_group_id: str
    variant_id: str
    item_states: dict[str, str]
    unknown_reason: dict[str, str]
    condition_tags: tuple[str, ...]
    hard_negative_tags: tuple[str, ...]
    source_pair_ids: tuple[str, ...]
    width: int
    height: int
    boxes: tuple[BoxAnnotation, ...]


@dataclass(frozen=True)
class PairPlan:
    source: SourceImage
    pair_group_id: str
    pair_index: int
    edit_kind: str
    target_item: str
    bait_image_id: str
    true_image_id: str
    bait_image_path: str
    true_image_path: str
    bait_states: dict[str, str]
    true_states: dict[str, str]
    bait_reasons: dict[str, str]
    true_reasons: dict[str, str]
    true_rect: tuple[int, int, int, int]
    bait_rect: tuple[int, int, int, int]
    edit_mask: np.ndarray | None
    hard_negative_tags: tuple[str, ...]


@dataclass(frozen=True)
class AugmentReport:
    source_images: int
    pairs: int
    synthetic_images: int
    samples: int
    pair_types: dict[str, int]
    hard_negative_pairs: int
    annotation_checks: dict


def _relative_data_path(path: Path, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError as exc:
        raise AugmentError(f"输出图片必须位于 data_dir 内: {path}") from exc


def _parse_sample_prefix(sample_id: str) -> str:
    match = re.fullmatch(r"(.+)_q(\d{2,})", sample_id)
    if not match:
        raise AugmentError(f"sample_id 必须形如 <prefix>_qNN: {sample_id!r}")
    return match.group(1)


def load_annotation_images(path: Path | str) -> list[AnnotationImage]:
    """Load annotations.json and normalize ``cable`` to ``cable_bundle``."""
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    raw_images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(raw_images, list):
        raise AugmentError(f"{path}: annotations.json 缺少 images 列表")

    images: list[AnnotationImage] = []
    for index, raw in enumerate(raw_images, start=1):
        if not isinstance(raw, dict):
            raise AugmentError(f"{path}: 第 {index} 个 image 不是对象")
        image = raw.get("image")
        width = raw.get("width")
        height = raw.get("height")
        raw_boxes = raw.get("annotations")
        if not isinstance(image, str) or not image:
            raise AugmentError(f"{path}: 第 {index} 个 image 的名称无效")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise AugmentError(f"{path}: {image} 的 width/height 无效")
        if not isinstance(raw_boxes, list):
            raise AugmentError(f"{path}: {image} 缺少 annotations 列表")

        boxes: list[BoxAnnotation] = []
        seen: set[str] = set()
        for box_index, raw_box in enumerate(raw_boxes, start=1):
            if not isinstance(raw_box, dict):
                raise AugmentError(f"{path}: {image} 的第 {box_index} 个框不是对象")
            raw_class = raw_box.get("class")
            item = ANNOTATION_TO_CATEGORY.get(raw_class)
            if item is None:
                raise AugmentError(f"{path}: {image} 有未知框标签 {raw_class!r}")
            if item in seen:
                raise AugmentError(f"{path}: {image} 的类别 {item} 有重复框")
            seen.add(item)
            xyxy = raw_box.get("box")
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                raise AugmentError(f"{path}: {image}/{item} 的 box 必须是四元列表")
            try:
                coords = tuple(float(value) for value in xyxy)
            except (TypeError, ValueError) as exc:
                raise AugmentError(f"{path}: {image}/{item} 的 box 含非数值") from exc
            boxes.append(BoxAnnotation(item=item, xyxy=coords))  # type: ignore[arg-type]
        images.append(
            AnnotationImage(
                image=image,
                width=width,
                height=height,
                boxes=tuple(sorted(boxes, key=lambda box: CATEGORY_IDS.index(box.item))),
            )
        )
    return images


def prepare_source_images(samples: Sequence[dict], *, checklists_per_image: int = 2) -> list[SourceImage]:
    """Collapse one-image/multiple-question rows into source image records.

    The real input must contain exactly two default checklist questions per
    image.  No original image is later copied into the synthetic pair dataset.
    """
    by_image: dict[str, list[dict]] = {}
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise AugmentError(f"真实样本第 {index} 行不是对象")
        image_path = sample.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise AugmentError(f"真实样本第 {index} 行 image_path 无效")
        by_image.setdefault(image_path, []).append(sample)

    prepared: list[SourceImage] = []
    for image_path, rows in by_image.items():
        for field in ("session_id", "scene_group_id", "item_states", "unknown_reason"):
            values = [row.get(field) for row in rows]
            if any(value != values[0] for value in values[1:]):
                raise AugmentError(f"{image_path}: 同图 {field} 不一致")
        checklist = [row for row in rows if row.get("task") == "checklist"]
        if len(checklist) != checklists_per_image:
            raise AugmentError(
                f"{image_path}: 期望 {checklists_per_image} 条 checklist，实际 {len(checklist)} 条"
            )
        if any(row.get("source") != "real_lab" for row in checklist):
            raise AugmentError(f"{image_path}: 输入样本 source 必须是 real_lab")
        source_pair_ids = tuple(
            sorted(
                {
                    str(row["pair_group_id"])
                    for row in checklist
                    if row.get("pair_group_id")
                }
            )
        )

        prefixes = {_parse_sample_prefix(str(row.get("sample_id", ""))) for row in checklist}
        if len(prefixes) != 1:
            raise AugmentError(f"{image_path}: 同图 sample_id 前缀不一致")
        sample_prefix = prefixes.pop()
        if "_" not in sample_prefix:
            raise AugmentError(f"{image_path}: sample_id 前缀格式错误: {sample_prefix}")
        variant_id = sample_prefix.rsplit("_", 1)[-1]
        if not variant_id:
            raise AugmentError(f"{image_path}: variant_id 为空")

        states = dict(checklist[0]["item_states"])
        if set(states) != set(CATEGORY_IDS) or any(value not in STATES for value in states.values()):
            raise AugmentError(f"{image_path}: item_states 类别或状态无效")
        reasons = dict(checklist[0]["unknown_reason"])
        if set(reasons) != {item for item, state in states.items() if state == "unknown"}:
            raise AugmentError(f"{image_path}: unknown_reason 与 unknown 类别不一致")

        session_id = str(checklist[0]["session_id"])
        scene_group_id = str(checklist[0]["scene_group_id"])
        if session_id not in sample_prefix or scene_group_id not in sample_prefix:
            raise AugmentError(f"{image_path}: sample_id 前缀没有继承 session/scene id")
        prepared.append(
            SourceImage(
                image_path=image_path,
                sample_prefix=sample_prefix,
                session_id=session_id,
                scene_group_id=scene_group_id,
                variant_id=variant_id,
                item_states={item: states[item] for item in CATEGORY_IDS},
                unknown_reason={item: reasons[item] for item in CATEGORY_IDS if item in reasons},
                condition_tags=tuple(checklist[0].get("condition_tags") or []),
                hard_negative_tags=tuple(checklist[0].get("hard_negative_tags") or []),
                source_pair_ids=source_pair_ids,
                width=0,
                height=0,
                boxes=(),
            )
        )
    if not prepared:
        raise AugmentError("真实样本为空")
    return prepared


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _box_intersection(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = _box_intersection(left, right)
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def attach_annotations(
    source_images: Sequence[SourceImage],
    annotation_images: Sequence[AnnotationImage],
    *,
    data_dir: Path | str = DATA_DIR,
) -> tuple[list[SourceImage], dict]:
    """Validate full-frame boxes/labels and attach them to source images."""
    root = Path(data_dir)
    by_name: dict[str, AnnotationImage] = {}
    for annotation in annotation_images:
        if annotation.image in by_name:
            raise AugmentError(f"annotations.json 中图片重复: {annotation.image}")
        by_name[annotation.image] = annotation

    source_names = {Path(source.image_path).name for source in source_images}
    missing_annotations = sorted(source_names - set(by_name))
    extra_annotations = sorted(set(by_name) - source_names)
    if missing_annotations:
        raise AugmentError(f"真实图缺少框标注: {missing_annotations[:5]}")
    if extra_annotations:
        raise AugmentError(f"框标注含未使用图片: {extra_annotations[:5]}")

    attached: list[SourceImage] = []
    errors: list[str] = []
    box_count = 0
    unknown_without_box = 0
    max_iou = 0.0
    overlapping_pairs = 0
    for source in source_images:
        annotation = by_name[Path(source.image_path).name]
        image_file = resolve_image_path(source.image_path, root)
        if not image_file.is_file():
            errors.append(f"{source.image_path}: 原图不存在")
            continue
        try:
            with Image.open(image_file) as image:
                actual_size = image.size
        except OSError as exc:
            errors.append(f"{source.image_path}: 原图无法打开: {exc}")
            continue
        if actual_size != (annotation.width, annotation.height):
            errors.append(
                f"{source.image_path}: 原图尺寸 {actual_size} != annotations {(annotation.width, annotation.height)}"
            )

        boxes_by_item = {box.item: box for box in annotation.boxes}
        for box in annotation.boxes:
            box_count += 1
            x1, y1, x2, y2 = box.xyxy
            if not all(math.isfinite(value) for value in box.xyxy):
                errors.append(f"{source.image_path}/{box.item}: 框含 NaN/Inf")
            elif not (0 <= x1 < x2 <= annotation.width and 0 <= y1 < y2 <= annotation.height):
                errors.append(
                    f"{source.image_path}/{box.item}: 框越出全图 "
                    f"{annotation.width}x{annotation.height}: {box.xyxy}"
                )
        for item, state in source.item_states.items():
            has_box = item in boxes_by_item
            if state == "present" and not has_box:
                errors.append(f"{source.image_path}/{item}: present 但缺少框")
            elif state == "unknown" and not has_box:
                unknown_without_box += 1
            elif state == "absent" and has_box:
                errors.append(f"{source.image_path}/{item}: absent 但有框")

        box_values = [box.xyxy for box in annotation.boxes]
        for left_index, left in enumerate(box_values):
            for right in box_values[left_index + 1 :]:
                iou = _box_iou(left, right)
                max_iou = max(max_iou, iou)
                if _box_intersection(left, right) > 0:
                    overlapping_pairs += 1
                if iou > 0.75:
                    errors.append(f"{source.image_path}: 两个器材框 IoU={iou:.3f} 过高")
        attached.append(
            replace(
                source,
                width=annotation.width,
                height=annotation.height,
                boxes=annotation.boxes,
            )
        )

    if errors:
        raise AugmentError("输入框/标签检查失败:\n" + "\n".join(errors))
    report = {
        "source_images": len(attached),
        "boxes": box_count,
        "present_boxes": sum(
            1
            for source in attached
            for item, state in source.item_states.items()
            if state == "present" and any(box.item == item for box in source.boxes)
        ),
        "unknown_boxes": sum(
            1
            for source in attached
            for item, state in source.item_states.items()
            if state == "unknown" and any(box.item == item for box in source.boxes)
        ),
        "unknown_without_box": unknown_without_box,
        "overlapping_box_pairs": overlapping_pairs,
        "max_box_iou": round(max_iou, 6),
        "out_of_bounds_boxes": 0,
        "label_mismatches": 0,
    }
    return attached, report


def _scale_xyxy(
    box: Sequence[float], source_size: tuple[int, int], output_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    output_width, output_height = output_size
    sx = output_width / source_width
    sy = output_height / source_height
    x1 = max(0, min(output_width, int(round(box[0] * sx))))
    y1 = max(0, min(output_height, int(round(box[1] * sy))))
    x2 = max(0, min(output_width, int(round(box[2] * sx))))
    y2 = max(0, min(output_height, int(round(box[3] * sy))))
    if x2 <= x1 or y2 <= y1:
        raise AugmentError(f"缩放后框退化: {box} -> {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def _expand_rect(
    rect: tuple[int, int, int, int],
    *,
    padding: float,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    pad_x = max(2, int(round(width * padding)))
    pad_y = max(2, int(round(height * padding)))
    frame_width, frame_height = frame_size
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_width, x2 + pad_x),
        min(frame_height, y2 + pad_y),
    )


def _inflate_rect(
    rect: tuple[int, int, int, int],
    amount: int,
    *,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = frame_size
    return (
        max(0, rect[0] - amount),
        max(0, rect[1] - amount),
        min(width, rect[2] + amount),
        min(height, rect[3] + amount),
    )


def _rect_intersection(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> int:
    return int(_box_intersection(left, right))


def _removal_mask(
    target_rect: tuple[int, int, int, int],
    protected_boxes: Iterable[tuple[int, int, int, int]],
    *,
    frame_size: tuple[int, int],
    image_rgb: np.ndarray | None = None,
    item: str | None = None,
) -> tuple[np.ndarray, float]:
    mask = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)
    x1, y1, x2, y2 = target_rect
    mask[y1:y2, x1:x2] = 255
    for box in protected_boxes:
        protected = _inflate_rect(box, 2, frame_size=frame_size)
        px1, py1, px2, py2 = protected
        mask[py1:py2, px1:px2] = 0
    if image_rgb is not None and item == "multimeter":
        roi = image_rgb[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        red = (
            ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168))
            & (hsv[:, :, 1] >= 80)
            & (hsv[:, :, 2] >= 35)
        ).astype(np.uint8) * 255
        red = cv2.morphologyEx(
            red,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        red = cv2.dilate(red, np.ones((3, 3), dtype=np.uint8), iterations=1)
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], red)
    target_area = max(1, _box_area(target_rect))
    unmasked_target = np.count_nonzero(mask[y1:y2, x1:x2] == 0)
    coverage = 1.0 - unmasked_target / target_area
    return mask, coverage


def _integral_sum(integral: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = rect
    return float(
        integral[y2, x2]
        - integral[y1, x2]
        - integral[y2, x1]
        + integral[y1, x1]
    )


def _find_blank_rect(
    image_rgb: np.ndarray,
    protected_boxes: Sequence[tuple[int, int, int, int]],
    width: int,
    height: int,
    *,
    clearance: int = 10,
) -> tuple[tuple[int, int, int, int], float]:
    """Find a bright, low-variance rectangle outside every equipment box."""
    frame_height, frame_width = image_rgb.shape[:2]
    if width > frame_width or height > frame_height:
        raise AugmentError(f"诱饵块 {width}x{height} 大于输出图")
    protected = [
        _inflate_rect(box, clearance, frame_size=(frame_width, frame_height))
        for box in protected_boxes
    ]
    max_x = frame_width - width
    max_y = frame_height - height
    if max_x < 0 or max_y < 0:
        raise AugmentError("诱饵块无法放入输出图")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    integral = np.pad(gray.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    integral_sq = np.pad((gray * gray).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    step = max(8, min(28, min(width, height) // 8))
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for y in range(0, max_y + 1, step):
        for x in range(0, max_x + 1, step):
            rect = (x, y, x + width, y + height)
            if any(_rect_intersection(rect, box) > 0 for box in protected):
                continue
            total = _integral_sum(integral, rect)
            total_sq = _integral_sum(integral_sq, rect)
            area = width * height
            mean = total / area
            variance = max(0.0, total_sq / area - mean * mean)
            score = mean - 1.8 * math.sqrt(variance)
            candidate = (score, rect)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise AugmentError(f"找不到 {width}x{height} 的无器材诱饵区域")
    return best[1], best[0]


def _find_blank_rect_near_size(
    image_rgb: np.ndarray,
    protected_boxes: Sequence[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], float]:
    """Prefer an equal-sized bait trace, shrinking only when the board is full."""
    last_error: AugmentError | None = None
    for scale in (1.0, 0.9, 0.8, 0.7):
        candidate_width = max(24, int(round(width * scale)))
        candidate_height = max(24, int(round(height * scale)))
        try:
            return _find_blank_rect(
                image_rgb,
                protected_boxes,
                candidate_width,
                candidate_height,
            )
        except AugmentError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise AugmentError("找不到诱饵区域")


def _occlusion_candidates(
    rect: tuple[int, int, int, int],
    other_boxes: Sequence[tuple[int, int, int, int]],
    *,
    preferred_scale: float,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    x1, y1, x2, y2 = rect
    target_width = x2 - x1
    target_height = y2 - y1
    candidates: list[tuple[tuple[int, int, int, int], float, int]] = []
    for scale in (preferred_scale, 0.78, 0.74, 0.70, 0.66):
        patch_width = max(24, int(round(target_width * scale)))
        patch_height = max(24, int(round(target_height * scale)))
        if patch_width <= 0 or patch_height <= 0:
            continue
        max_x = x2 - patch_width
        max_y = y2 - patch_height
        if max_x < x1 or max_y < y1:
            continue
        xs = sorted(
            {
                x1 + int((max_x - x1) * fraction)
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
            }
        )
        ys = sorted(
            {
                y1 + int((max_y - y1) * fraction)
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
            }
        )
        for patch_y in ys:
            for patch_x in xs:
                patch = (patch_x, patch_y, patch_x + patch_width, patch_y + patch_height)
                overlap = sum(_rect_intersection(patch, box) for box in other_boxes)
                coverage = _box_area(patch) / max(1, _box_area(rect))
                if coverage >= 0.54:
                    candidates.append((patch, coverage, overlap))
    return candidates


def _choose_removal_candidate(
    source: SourceImage,
    image_rgb: np.ndarray,
    scaled_boxes: dict[str, tuple[int, int, int, int]],
    *,
    removal_padding: float,
) -> tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]:
    candidates: list[
        tuple[tuple[float, ...], str, tuple[int, int, int, int], tuple[int, int, int, int]]
    ] = []
    frame_size = (image_rgb.shape[1], image_rgb.shape[0])
    for item in CATEGORY_IDS:
        if source.item_states[item] != "present" or item not in scaled_boxes:
            continue
        true_rect = _expand_rect(
            scaled_boxes[item],
            padding=removal_padding,
            frame_size=frame_size,
        )
        other_boxes = [box for name, box in scaled_boxes.items() if name != item]
        mask, coverage = _removal_mask(
            true_rect,
            other_boxes,
            frame_size=frame_size,
            image_rgb=image_rgb,
            item=item,
        )
        if coverage < MIN_REMOVAL_COVERAGE:
            continue
        try:
            bait_rect, blank_score = _find_blank_rect_near_size(
                image_rgb,
                list(scaled_boxes.values()),
                true_rect[2] - true_rect[0],
                true_rect[3] - true_rect[1],
            )
        except AugmentError:
            continue
        hard = item == "multimeter" and source.item_states["cable_bundle"] == "present"
        relative_area = _box_area(true_rect) / (frame_size[0] * frame_size[1])
        priority = (2 if hard else (1 if item == "multimeter" else 0), coverage, -relative_area, blank_score)
        candidates.append((priority, item, true_rect, bait_rect))
    if not candidates:
        raise AugmentError(f"{source.image_path}: 找不到可安全抹除的 present 器材")
    _, item, true_rect, bait_rect = max(candidates, key=lambda value: (value[0], value[1]))
    return item, true_rect, bait_rect


def _choose_occlusion_candidate(
    source: SourceImage,
    image_rgb: np.ndarray,
    scaled_boxes: dict[str, tuple[int, int, int, int]],
    *,
    occlusion_scale: float,
) -> tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]:
    preference = {
        "screwdriver": 3,
        "pliers": 2,
        "hammer": 2,
        "multimeter": 1,
        "cable_bundle": 1,
    }
    candidates: list[
        tuple[tuple[float, ...], str, tuple[int, int, int, int]]
    ] = []
    frame_size = (image_rgb.shape[1], image_rgb.shape[0])
    for item in CATEGORY_IDS:
        if source.item_states[item] != "present" or item not in scaled_boxes:
            continue
        other_boxes = [box for name, box in scaled_boxes.items() if name != item]
        for patch, coverage, overlap in _occlusion_candidates(
            scaled_boxes[item],
            other_boxes,
            preferred_scale=occlusion_scale,
        ):
            if overlap > 0:
                continue
            priority = (preference[item], coverage, -_box_area(patch))
            candidates.append((priority, item, patch))
    if not candidates:
        raise AugmentError(f"{source.image_path}: 找不到可安全合成遮挡的 present 器材")
    tried: set[tuple[int, int, int, int]] = set()
    for _, item, patch in sorted(
        candidates,
        key=lambda value: (value[0], value[1]),
        reverse=True,
    ):
        if patch in tried:
            continue
        tried.add(patch)
        try:
            bait_rect, _ = _find_blank_rect_near_size(
                image_rgb,
                list(scaled_boxes.values()),
                patch[2] - patch[0],
                patch[3] - patch[1],
            )
        except AugmentError:
            continue
        return item, patch, bait_rect
    raise AugmentError(f"{source.image_path}: 遮挡候选都找不到无器材诱饵区域")


def _rng_for(seed: int, *parts: str) -> Random:
    digest = hashlib.sha256(("|".join((str(seed), *parts))).encode("utf-8")).digest()
    return Random(int.from_bytes(digest[:8], "big"))


def plan_pairs(
    source_images: Sequence[SourceImage],
    *,
    data_dir: Path | str = DATA_DIR,
    out_dir: Path | str,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    pairs_per_image: int = DEFAULT_PAIRS_PER_IMAGE,
    removal_padding: float = DEFAULT_REMOVAL_PADDING,
    occlusion_scale: float = DEFAULT_OCCLUSION_SCALE,
    seed: int = 0,
) -> list[PairPlan]:
    """Plan bait/true edits before rendering; all decisions are deterministic."""
    if pairs_per_image not in (1, 2):
        raise AugmentError("pairs_per_image 只支持 1 或 2")
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise AugmentError("output_size 必须为正")
    root = Path(data_dir)
    out_root = Path(out_dir).resolve()
    _ = _relative_data_path(out_root / "dummy.jpg", root)

    plans: list[PairPlan] = []
    for source in source_images:
        image_file = resolve_image_path(source.image_path, root)
        with Image.open(image_file) as loaded:
            rgb = np.asarray(loaded.convert("RGB").resize(output_size, Image.Resampling.LANCZOS))
        scaled_boxes = {
            box.item: _scale_xyxy(box.xyxy, (source.width, source.height), output_size)
            for box in source.boxes
        }
        image_rel_root = _relative_data_path(out_root / "images", root)
        scene_dir = source.scene_group_id

        removal_item, removal_rect, removal_bait_rect = _choose_removal_candidate(
            source,
            rgb,
            scaled_boxes,
            removal_padding=removal_padding,
        )
        paired_specs: list[dict] = []
        hard_tags = (
            ("cable_without_multimeter",)
            if removal_item == "multimeter" and source.item_states["cable_bundle"] == "present"
            else ()
        )
        paired_specs.append(
            {
                "kind": "remove",
                "target": removal_item,
                "true_states": {
                    item: ("absent" if item == removal_item else state)
                    for item, state in source.item_states.items()
                },
                "true_reasons": dict(source.unknown_reason),
                "true_rect": removal_rect,
                "bait_rect": removal_bait_rect,
                "edit_mask": None,
                "hard_tags": hard_tags,
            }
        )
        if pairs_per_image == 2:
            occlusion_item, occlusion_rect, occlusion_bait_rect = _choose_occlusion_candidate(
                source,
                rgb,
                scaled_boxes,
                occlusion_scale=occlusion_scale,
            )
            true_reasons = dict(source.unknown_reason)
            true_reasons[occlusion_item] = "合成遮挡"
            paired_specs.append(
                {
                    "kind": "occlude",
                    "target": occlusion_item,
                    "true_states": {
                        item: ("unknown" if item == occlusion_item else state)
                        for item, state in source.item_states.items()
                    },
                    "true_reasons": true_reasons,
                    "true_rect": occlusion_rect,
                    "bait_rect": occlusion_bait_rect,
                    "edit_mask": None,
                    "hard_tags": (),
                }
            )

        for pair_index, spec in enumerate(paired_specs, start=1):
            bait_id = f"{source.sample_prefix}e{(pair_index - 1) * 2 + 1:02d}"
            true_id = f"{source.sample_prefix}e{pair_index * 2:02d}"
            pair_id = f"{source.sample_prefix}p{pair_index:02d}"
            plans.append(
                PairPlan(
                    source=source,
                    pair_group_id=pair_id,
                    pair_index=pair_index,
                    edit_kind=str(spec["kind"]),
                    target_item=str(spec["target"]),
                    bait_image_id=bait_id,
                    true_image_id=true_id,
                    bait_image_path=(
                        f"{image_rel_root}/{scene_dir}/{bait_id}_bait.jpg"
                    ),
                    true_image_path=(
                        f"{image_rel_root}/{scene_dir}/{true_id}_true.jpg"
                    ),
                    bait_states=dict(source.item_states),
                    true_states=dict(spec["true_states"]),
                    bait_reasons=dict(source.unknown_reason),
                    true_reasons=dict(spec["true_reasons"]),
                    true_rect=tuple(spec["true_rect"]),  # type: ignore[arg-type]
                    bait_rect=tuple(spec["bait_rect"]),  # type: ignore[arg-type]
                    edit_mask=spec["edit_mask"],
                    hard_negative_tags=tuple(spec["hard_tags"]),
                )
            )
    return plans


def _paper_texture(
    width: int,
    height: int,
    *,
    seed: int,
    base_color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_gradient = np.linspace(-7, 7, height, dtype=np.float32)[:, None, None]
    x_gradient = np.linspace(4, -4, width, dtype=np.float32)[None, :, None]
    noise = rng.normal(0.0, 4.0, size=(height, width, 1)).astype(np.float32)
    color = np.asarray(base_color, dtype=np.float32)[None, None, :]
    patch = np.clip(color + y_gradient + x_gradient + noise, 0, 255)
    return (patch * alpha + color * (1 - alpha)).astype(np.uint8)


def _overlay_rect_texture(
    image_rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    mask: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    x1, y1, x2, y2 = rect
    patch = _paper_texture(
        x2 - x1,
        y2 - y1,
        seed=seed,
        base_color=(244, 243, 238),
        alpha=1.0,
    )
    result = image_rgb.copy()
    roi = result[y1:y2, x1:x2].astype(np.float32)
    blended = roi * 0.04 + patch.astype(np.float32) * 0.96
    if mask is not None:
        local_mask = mask[y1:y2, x1:x2].astype(np.float32)[..., None] / 255.0
        blended = roi * (1.0 - local_mask) + blended * local_mask
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _apply_removal(
    image_rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    radius = max(5, min(13, min(rect[2] - rect[0], rect[3] - rect[1]) // 22))
    inpainted = cv2.inpaint(image_rgb, mask, radius, cv2.INPAINT_TELEA)
    return _overlay_rect_texture(inpainted, rect, mask=mask, seed=seed)


def _apply_occlusion(
    image_rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    seed: int,
) -> np.ndarray:
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    source = Image.fromarray(image_rgb, mode="RGB")
    shadow = Image.new("RGBA", source.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    radius = max(4, min(width, height) // 12)
    shadow_draw.rounded_rectangle(
        (x1 + 5, y1 + 7, x2 + 5, y2 + 7),
        radius=radius,
        fill=(35, 32, 27, 45),
    )
    source = Image.alpha_composite(source.convert("RGBA"), shadow).convert("RGB")

    patch = Image.fromarray(
        _paper_texture(
            width,
            height,
            seed=seed,
            base_color=(192, 176, 143),
            alpha=0.35,
        ),
        mode="RGB",
    ).convert("RGBA")
    patch_mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(patch_mask)
    mask_draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=radius,
        fill=246,
    )
    rng = Random(seed ^ 0x5A17)
    for _ in range(max(3, width * height // 2800)):
        line_y = rng.randrange(max(1, height - 2))
        line_x = rng.randrange(max(1, width // 5))
        mask_draw.line(
            (line_x, line_y, min(width - 1, line_x + rng.randrange(8, max(9, width // 3))), line_y),
            fill=235,
            width=1,
        )
    patch.putalpha(patch_mask)
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    overlay.alpha_composite(patch, dest=(x1, y1))
    return np.asarray(Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB"))


def render_pair_images(
    plan: PairPlan,
    *,
    data_dir: Path | str = DATA_DIR,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Render one bait/true pair and return their absolute paths."""
    root = Path(data_dir)
    source_file = resolve_image_path(plan.source.image_path, root)
    with Image.open(source_file) as loaded:
        base = np.asarray(
            loaded.convert("RGB").resize(output_size, Image.Resampling.LANCZOS)
        )
    frame_size = (output_size[0], output_size[1])
    other_boxes = [
        _scale_xyxy(box.xyxy, (plan.source.width, plan.source.height), output_size)
        for box in plan.source.boxes
        if box.item != plan.target_item
    ]

    pair_rng = _rng_for(seed, plan.pair_group_id)
    texture_seed = pair_rng.randrange(2**32)
    if plan.edit_kind == "remove":
        bait_mask, _ = _removal_mask(plan.bait_rect, (), frame_size=frame_size)
        bait = _apply_removal(
            base,
            plan.bait_rect,
            bait_mask,
            seed=texture_seed,
        )
        true_mask, _ = _removal_mask(
            plan.true_rect,
            other_boxes,
            frame_size=frame_size,
            image_rgb=base,
            item=plan.target_item,
        )
        true = _apply_removal(
            base,
            plan.true_rect,
            true_mask,
            seed=texture_seed,
        )
    elif plan.edit_kind == "occlude":
        bait = _apply_occlusion(base, plan.bait_rect, seed=texture_seed)
        true = _apply_occlusion(base, plan.true_rect, seed=texture_seed)
    else:
        raise AugmentError(f"未知编辑类型: {plan.edit_kind}")

    bait_path = resolve_image_path(plan.bait_image_path, root)
    true_path = resolve_image_path(plan.true_image_path, root)
    for path, image in ((bait_path, bait), (true_path, true)):
        _atomic_save_jpeg(
            path,
            Image.fromarray(image, mode="RGB"),
            quality=jpeg_quality,
        )
    return bait_path, true_path


def _prepare_tags(
    source_tags: Sequence[str],
    extra: Sequence[str],
) -> list[str]:
    result: list[str] = []
    for tag in (*source_tags, *extra):
        if tag and tag not in result:
            result.append(str(tag))
    return result


def _applicable_source_hard_tags(
    source: SourceImage,
    states: dict[str, str],
) -> list[str]:
    """Preserve source hard tags unless they mention an item whose state changed."""
    applicable: list[str] = []
    for tag in source.hard_negative_tags:
        referenced = [
            item
            for item in CATEGORY_IDS
            if item in tag or NAME_ZH[item] in tag
        ]
        if any(states[item] != source.item_states[item] for item in referenced):
            continue
        applicable.append(tag)
    return applicable


def _samples_for_image(
    *,
    image_id: str,
    image_path: str,
    source: SourceImage,
    pair_group_id: str,
    item_states: dict[str, str],
    unknown_reason: dict[str, str],
    condition_tags: Sequence[str],
    hard_negative_tags: Sequence[str],
    templates: dict,
    seed: int,
) -> list[dict]:
    defaults = templates["defaults"]
    group = templates["question_sets"][defaults["question_set"]]
    rng = _rng_for(seed, pair_group_id)
    pools = group["checklist_pools"]
    count = int(defaults["checklists_per_image"])
    chosen = rng.sample(pools, count) if count <= len(pools) else [
        rng.choice(pools) for _ in range(count)
    ]
    questions = [rng.choice(group["checklist_questions"]) for _ in chosen]
    ordered_states = {item: item_states[item] for item in CATEGORY_IDS}
    ordered_reasons = {
        item: unknown_reason[item] for item in CATEGORY_IDS if item in unknown_reason
    }
    if set(ordered_reasons) != {
        item for item, state in ordered_states.items() if state == "unknown"
    }:
        raise AugmentError(f"{image_id}: unknown_reason 与 item_states 不一致")

    samples: list[dict] = []
    for seq, (required_items, question) in enumerate(zip(chosen, questions), start=1):
        samples.append(
            {
                "sample_id": f"{image_id}_q{seq:02d}",
                "image_path": image_path,
                "session_id": source.session_id,
                "scene_group_id": source.scene_group_id,
                "pair_group_id": pair_group_id,
                "task": "checklist",
                "required_items": list(required_items),
                "question_zh": question,
                "item_states": ordered_states,
                "condition_tags": list(condition_tags),
                "hard_negative_tags": list(hard_negative_tags),
                "unknown_reason": ordered_reasons,
                "source": SYNTHETIC_SOURCE,
                "annotation_verified": False,
            }
        )
    return samples


def build_samples(plans: Sequence[PairPlan], *, templates: dict, seed: int = 0) -> list[dict]:
    """Build two default checklist questions for every synthetic image."""
    samples: list[dict] = []
    for plan in plans:
        kind_label = "remove" if plan.edit_kind == "remove" else "occlude"
        true_change_tag = (
            f"removed_{plan.target_item}"
            if plan.edit_kind == "remove"
            else f"occluded_{plan.target_item}"
        )
        bait_tags = _prepare_tags(
            plan.source.condition_tags,
            (
                "synthetic_edit",
                f"{kind_label}_pair",
                "bait_edit",
                f"synthetic_{kind_label}_bait",
            ),
        )
        true_tags = _prepare_tags(
            plan.source.condition_tags,
            (
                "synthetic_edit",
                f"{kind_label}_pair",
                true_change_tag,
                f"synthetic_{kind_label}",
            ),
        )
        bait_hard = _applicable_source_hard_tags(plan.source, plan.bait_states)
        true_hard = _prepare_tags(
            _applicable_source_hard_tags(plan.source, plan.true_states),
            plan.hard_negative_tags,
        )
        samples.extend(
            _samples_for_image(
                image_id=plan.bait_image_id,
                image_path=plan.bait_image_path,
                source=plan.source,
                pair_group_id=plan.pair_group_id,
                item_states=plan.bait_states,
                unknown_reason=plan.bait_reasons,
                condition_tags=bait_tags,
                hard_negative_tags=bait_hard,
                templates=templates,
                seed=seed,
            )
        )
        samples.extend(
            _samples_for_image(
                image_id=plan.true_image_id,
                image_path=plan.true_image_path,
                source=plan.source,
                pair_group_id=plan.pair_group_id,
                item_states=plan.true_states,
                unknown_reason=plan.true_reasons,
                condition_tags=true_tags,
                hard_negative_tags=true_hard,
                templates=templates,
                seed=seed,
            )
        )
    return samples


def validate_pair_integrity(
    samples: Sequence[dict],
    *,
    source_images: Sequence[SourceImage],
) -> list[str]:
    """Check inheritance, bait-only changes, one-state pairs and no source reuse."""
    errors: list[str] = []
    source_by_scene = {source.scene_group_id: source for source in source_images}
    source_by_id = {source.sample_prefix: source for source in source_images}
    source_paths = {source.image_path for source in source_images}
    source_pair_ids = {
        pair_id for source in source_images for pair_id in source.source_pair_ids
    }

    by_image: dict[str, list[dict]] = defaultdict(list)
    by_pair: dict[str, set[str]] = defaultdict(set)
    pair_source_prefixes: set[str] = set()
    for sample in samples:
        image_path = sample.get("image_path")
        pair_id = sample.get("pair_group_id")
        hard_tags = list(sample.get("hard_negative_tags") or [])
        diagnostics_in_hard = [
            tag
            for tag in hard_tags
            if str(tag).startswith(("synthetic_", "removed_", "occluded_"))
        ]
        if diagnostics_in_hard:
            errors.append(
                f"{sample.get('sample_id')}: 诊断标签不应进入 hard_negative_tags: "
                f"{diagnostics_in_hard}"
            )
        states = sample.get("item_states") or {}
        has_cable_without_mm = "cable_without_multimeter" in hard_tags
        construct_holds = (
            states.get("cable_bundle") == "present"
            and states.get("multimeter") == "absent"
        )
        if has_cable_without_mm != construct_holds and (
            has_cable_without_mm or "removed_multimeter" in sample.get("condition_tags", [])
        ):
            errors.append(
                f"{sample.get('sample_id')}: cable_without_multimeter 标签与"
                " cable present + multimeter absent 不一致"
            )
        if sample.get("source") != SYNTHETIC_SOURCE:
            errors.append(f"{sample.get('sample_id')}: source 不是 {SYNTHETIC_SOURCE}")
        if sample.get("annotation_verified") is not False:
            errors.append(f"{sample.get('sample_id')}: annotation_verified 必须为 false")
        if image_path in source_paths:
            errors.append(f"{sample.get('sample_id')}: 原图不得进入合成数据集")
        if not isinstance(pair_id, str) or not pair_id:
            errors.append(f"{sample.get('sample_id')}: 缺少 pair_group_id")
            continue
        by_image[str(image_path)].append(sample)
        by_pair[pair_id].add(str(image_path))
        image_id = str(sample.get("sample_id", "")).rsplit("_q", 1)[0]
        prefix = re.sub(r"e\d{2}$", "", image_id)
        source = source_by_id.get(prefix)
        if source is None:
            errors.append(f"{sample.get('sample_id')}: 无法从 image id 还原原图")
            continue
        if sample.get("session_id") != source.session_id:
            errors.append(f"{sample.get('sample_id')}: session_id 没有继承")
        if sample.get("scene_group_id") != source.scene_group_id:
            errors.append(f"{sample.get('sample_id')}: scene_group_id 没有继承")
        pair_source_prefixes.add(prefix)

    for image_path, rows in by_image.items():
        if len(rows) != 2:
            errors.append(f"{image_path}: 期望 2 条 checklist，实际 {len(rows)} 条")
        if any(row.get("task") != "checklist" for row in rows):
            errors.append(f"{image_path}: 合成图只允许 checklist")
        if len(rows) >= 2 and rows[0]["item_states"] != rows[1]["item_states"]:
            errors.append(f"{image_path}: 同图两条 checklist 的 item_states 不一致")
        bait_rows = [
            row for row in rows if "bait_edit" in row.get("condition_tags", [])
        ]
        if len(bait_rows) not in (0, 2):
            errors.append(
                f"{image_path}: bait_edit 标记应为 0 或 2 行，实际 {len(bait_rows)}"
            )
        elif len(bait_rows) == 2:
            bait_prefix = re.sub(
                r"e\d{2}$",
                "",
                str(bait_rows[0]["sample_id"]).rsplit("_q", 1)[0],
            )
            bait_source = source_by_id.get(bait_prefix)
            if bait_source is None:
                errors.append(f"{image_path}: 无法还原诱饵图原图")
            elif bait_rows[0]["item_states"] != bait_source.item_states:
                errors.append(f"{image_path}: 诱饵图状态与原图不一致")

    for pair_id, image_paths in sorted(by_pair.items()):
        if pair_id in source_pair_ids:
            errors.append(f"{pair_id}: 合成 pair_group_id 与真实 pair 冲突")
        if len(image_paths) != 2:
            errors.append(f"{pair_id}: 期望恰好 2 张图，实际 {len(image_paths)} 张")
            continue
        left_path, right_path = sorted(image_paths)
        left = by_image[left_path][0]
        right = by_image[right_path][0]
        diff = [
            item
            for item in CATEGORY_IDS
            if left["item_states"].get(item) != right["item_states"].get(item)
        ]
        if len(diff) != 1:
            errors.append(f"{pair_id}: 期望恰差 1 类，实际 {diff}")
        scenes = {
            sample.get("scene_group_id")
            for sample in (*by_image[left_path], *by_image[right_path])
        }
        if len(scenes) != 1:
            errors.append(f"{pair_id}: pair 跨 scene_group: {sorted(str(x) for x in scenes)}")
        elif next(iter(scenes)) not in source_by_scene:
            errors.append(f"{pair_id}: scene_group 不属于任何真实原图")
        if not any(
            "bait_edit" in sample.get("condition_tags", [])
            for sample in (*by_image[left_path], *by_image[right_path])
        ):
            errors.append(f"{pair_id}: 没有诱饵图")
        left_signatures = [
            (
                tuple(row.get("required_items") or []),
                row.get("question_zh"),
            )
            for row in sorted(by_image[left_path], key=lambda row: str(row.get("sample_id")))
        ]
        right_signatures = [
            (
                tuple(row.get("required_items") or []),
                row.get("question_zh"),
            )
            for row in sorted(by_image[right_path], key=lambda row: str(row.get("sample_id")))
        ]
        if left_signatures != right_signatures:
            errors.append(f"{pair_id}: 诱饵与真编辑的 checklist 问法/清单不一致")
        bait_images = {
            image
            for image in image_paths
            if "bait_edit" in by_image[image][0].get("condition_tags", [])
        }
        if len(bait_images) != 1:
            errors.append(f"{pair_id}: 期望恰好 1 张诱饵图，实际 {len(bait_images)} 张")

    missing_pair_sources = sorted(set(source_by_id) - pair_source_prefixes)
    if missing_pair_sources:
        errors.append(f"以下原图没有合成 pair: {missing_pair_sources[:10]}")
    return errors


def _bait_state_errors(plans: Sequence[PairPlan]) -> list[str]:
    errors: list[str] = []
    for plan in plans:
        if plan.bait_states != plan.source.item_states:
            errors.append(f"{plan.pair_group_id}: 诱饵图改变了器材状态")
        changed = [
            item
            for item in CATEGORY_IDS
            if plan.bait_states[item] != plan.true_states[item]
        ]
        if changed != [plan.target_item]:
            errors.append(
                f"{plan.pair_group_id}: 真编辑应只改变 {plan.target_item}，实际 {changed}"
            )
    return errors


def _choose_contact_sheet_plans(plans: Sequence[PairPlan], count: int) -> list[PairPlan]:
    removals = [plan for plan in plans if plan.edit_kind == "remove"]
    occlusions = [plan for plan in plans if plan.edit_kind == "occlude"]
    chosen: list[PairPlan] = []
    if count <= 0:
        return chosen
    if removals:
        n_remove = min(len(removals), (count + 1) // 2)
        indices = np.linspace(0, len(removals) - 1, n_remove, dtype=int)
        chosen.extend(removals[int(index)] for index in indices)
    remaining = count - len(chosen)
    if occlusions and remaining > 0:
        n_occlude = min(len(occlusions), remaining)
        indices = np.linspace(0, len(occlusions) - 1, n_occlude, dtype=int)
        chosen.extend(occlusions[int(index)] for index in indices)
    return chosen[:count]


def write_contact_sheet(
    plans: Sequence[PairPlan],
    *,
    data_dir: Path | str,
    output_path: Path | str,
    pair_count: int = 10,
    cell_size: tuple[int, int] = (320, 180),
    seed: int = 0,
) -> list[dict]:
    """Write a 4-column contact sheet with bait/true pairs adjacent."""
    selected = _choose_contact_sheet_plans(plans, pair_count)
    if not selected:
        return []
    root = Path(data_dir)
    cell_width, cell_height = cell_size
    label_height = 30
    columns = 4
    rows = math.ceil(len(selected) * 2 / columns)
    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        (246, 246, 243),
    )
    draw = ImageDraw.Draw(canvas)
    entries: list[dict] = []
    for pair_index, plan in enumerate(selected):
        for role, image_path, state_text in (
            ("bait", plan.bait_image_path, "unchanged"),
            (
                "true",
                plan.true_image_path,
                "absent" if plan.edit_kind == "remove" else "unknown",
            ),
        ):
            logical_index = pair_index * 2 + (0 if role == "bait" else 1)
            column = logical_index % columns
            row = logical_index // columns
            left = column * cell_width
            top = row * (cell_height + label_height)
            image = Image.open(resolve_image_path(image_path, root)).convert("RGB")
            canvas.paste(image.resize((cell_width, cell_height), Image.Resampling.LANCZOS), (left, top))
            color = (35, 100, 55) if role == "bait" else (145, 45, 38)
            draw.rectangle((left, top + cell_height, left + cell_width, top + cell_height + label_height), fill=color)
            label = (
                f"{plan.pair_group_id} {role.upper()} "
                f"{plan.edit_kind}/{plan.target_item}={state_text}"
            )
            draw.text((left + 6, top + cell_height + 7), label, fill=(255, 255, 255))
            entries.append(
                {
                    "pair_group_id": plan.pair_group_id,
                    "role": role,
                    "image_path": image_path,
                    "edit_kind": plan.edit_kind,
                    "target_item": plan.target_item,
                    "target_state": state_text,
                }
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_jpeg(Path(output_path), canvas, quality=90)
    return entries


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_jsonl(path: Path, samples: Sequence[dict]) -> None:
    text = "".join(
        json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
        for sample in samples
    )
    _atomic_write_text(path, text)


def _atomic_save_jpeg(path: Path, image: Image.Image, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    try:
        image.save(
            temporary,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _prepare_managed_images(
    out_dir: Path,
    generated_image_paths: Sequence[str],
    *,
    data_dir: Path,
) -> None:
    """Register generated paths while keeping existing images readable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = (out_dir / "images").resolve()
    marker = images_dir / ".augment-v1.json"
    old_paths = []
    if images_dir.exists():
        if not marker.is_file():
            raise AugmentError(
                f"{images_dir} 已存在但缺少 .augment-v1.json 所有权标记；"
                "拒绝清理或覆盖该目录"
            )
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AugmentError(f"无法读取 {marker} 的所有权标记: {exc}") from exc
        if previous.get("version") != "aug-v1":
            raise AugmentError(f"{marker} 的版本不是 aug-v1；拒绝清理")
        old_paths = previous.get("generated_image_paths")
        if not isinstance(old_paths, list) or any(
            not isinstance(path, str) for path in old_paths
        ):
            raise AugmentError(f"{marker} 缺少 generated_image_paths 白名单")
    else:
        images_dir.mkdir(parents=True)
    managed = set(old_paths)
    for image_path in generated_image_paths:
        full = resolve_image_path(image_path, data_dir)
        if images_dir not in full.parents:
            raise AugmentError(f"生成路径逃出 images 目录: {image_path}")
        if image_path not in managed and full.exists():
            raise AugmentError(f"拒绝覆盖未登记的图片: {image_path}")
    _atomic_write_json(
        marker,
        {
            "version": "aug-v1",
            "generated_image_paths": sorted(managed.union(generated_image_paths)),
        },
    )


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def generate_augmented_dataset(
    *,
    real_jsonl: Path | str,
    annotations_json: Path | str,
    templates_path: Path | str,
    out_dir: Path | str,
    data_dir: Path | str = DATA_DIR,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    pairs_per_image: int = DEFAULT_PAIRS_PER_IMAGE,
    seed: int = 0,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    contact_sheet_pairs: int = 10,
) -> AugmentReport:
    """Generate, write and validate the complete deterministic dataset."""
    root = Path(data_dir)
    real_jsonl = Path(real_jsonl)
    annotations_json = Path(annotations_json)
    templates_path = Path(templates_path)
    out_dir = Path(out_dir)
    _relative_data_path(out_dir / "dummy.jpg", root)

    source_images = prepare_source_images(load_jsonl(real_jsonl))
    annotation_images = load_annotation_images(annotations_json)
    source_images, annotation_checks = attach_annotations(
        source_images,
        annotation_images,
        data_dir=root,
    )
    plans = plan_pairs(
        source_images,
        data_dir=root,
        out_dir=out_dir,
        output_size=output_size,
        pairs_per_image=pairs_per_image,
        seed=seed,
    )
    expected_pairs = len(source_images) * pairs_per_image
    if len(plans) != expected_pairs:
        raise AugmentError(f"计划 pair 数 {len(plans)} != {expected_pairs}")
    plan_errors = _bait_state_errors(plans)
    if plan_errors:
        raise AugmentError("计划检查失败:\n" + "\n".join(plan_errors))

    templates = load_question_templates(templates_path)
    if templates["defaults"]["question_set"] != "default":
        raise AugmentError("question_templates 默认集合必须是 default")
    if int(templates["defaults"]["checklists_per_image"]) != 2:
        raise AugmentError("default 集合必须配置每图 2 条 checklist")
    samples = build_samples(plans, templates=templates, seed=seed)

    generated_image_paths = sorted(
        path
        for plan in plans
        for path in (plan.bait_image_path, plan.true_image_path)
    )
    _prepare_managed_images(
        out_dir,
        generated_image_paths,
        data_dir=root,
    )
    for plan in plans:
        render_pair_images(
            plan,
            data_dir=root,
            output_size=output_size,
            jpeg_quality=jpeg_quality,
            seed=seed,
        )
    _atomic_write_jsonl(out_dir / "samples.jsonl", samples)

    errors = validate_samples(samples, data_dir=root, check_images=True)
    errors.extend(validate_pair_integrity(samples, source_images=source_images))
    if errors:
        raise AugmentError("合成数据检查失败:\n" + "\n".join(errors[:50]))

    contact_entries = write_contact_sheet(
        plans,
        data_dir=root,
        output_path=out_dir / "contact_sheet.jpg",
        pair_count=contact_sheet_pairs,
    )
    _atomic_write_json(out_dir / "contact_sheet.json", contact_entries)

    pair_types = dict(Counter(plan.edit_kind for plan in plans))
    hard_negative_pairs = sum(
        "cable_without_multimeter" in plan.hard_negative_tags for plan in plans
    )
    report = AugmentReport(
        source_images=len(source_images),
        pairs=len(plans),
        synthetic_images=len(plans) * 2,
        samples=len(samples),
        pair_types=pair_types,
        hard_negative_pairs=hard_negative_pairs,
        annotation_checks=annotation_checks,
    )
    manifest = {
        "version": "aug-v1",
        "seed": seed,
        "source_jsonl": _path_label(real_jsonl),
        "annotations_json": _path_label(annotations_json),
        "templates_yaml": _path_label(templates_path),
        "source_jsonl_sha256": sha256_file(real_jsonl),
        "annotations_json_sha256": sha256_file(annotations_json),
        "templates_yaml_sha256": sha256_file(templates_path),
        "samples_jsonl_sha256": sha256_file(out_dir / "samples.jsonl"),
        "output_size": list(output_size),
        "jpeg_quality": jpeg_quality,
        "pairs_per_source_image": pairs_per_image,
        "checklists_per_image": 2,
        "source_images": report.source_images,
        "pairs": report.pairs,
        "synthetic_images": report.synthetic_images,
        "samples": report.samples,
        "pair_types": report.pair_types,
        "hard_negative_cable_without_multimeter_pairs": report.hard_negative_pairs,
        "annotation_checks": report.annotation_checks,
        "contact_sheet": "contact_sheet.jpg",
        "contact_sheet_pairs": len(contact_entries) // 2,
        "generated_image_paths": generated_image_paths,
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)
    return report
