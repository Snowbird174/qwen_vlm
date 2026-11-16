"""逐条样本校验：冻结 schema，再加上图片、唯一性和成对约束。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.common.schema import validate_sample
from qwen_vlm.data.io import resolve_image_path


def validate_samples(
    samples: list[dict],
    *,
    data_dir: Path | str | None = None,
    check_images: bool = True,
) -> list[str]:
    """返回人类可读的错误列表。空列表表示通过。

    在 ``validate_sample`` 之外检查：

    - ``sample_id`` 在文件内唯一；
    - 同一 ``image_path`` 的 ``item_states``、``session_id``、``scene_group_id``、
      ``pair_group_id`` 一致；
    - presence 样本的 ``target_item`` 状态与同图 ``item_states`` 一致
      （schema 没有单独的状态字段，真值就是 ``item_states[target_item]``）；
    - 图片存在且能用 PIL 打开；
    - 每个非空 ``pair_group_id`` 恰好对应 2 张不同的图，且只有 1 个类别的真值不同。
    """
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    by_image: dict[str, list[tuple[int, dict]]] = {}

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            errors.append(f"第 {index} 行: 不是 JSON 对象")
            continue
        sid = sample.get("sample_id") or f"<无 sample_id>"
        prefix = f"第 {index} 行 sample_id={sid}"
        schema_errors = validate_sample(sample)
        errors.extend(f"{prefix}: {msg}" for msg in schema_errors)
        if isinstance(sid, str) and sid:
            if sid in seen_ids:
                errors.append(f"{prefix}: sample_id 重复，首次出现于第 {seen_ids[sid]} 行")
            else:
                seen_ids[sid] = index
        if schema_errors:
            continue
        by_image.setdefault(sample["image_path"], []).append((index, sample))

    errors.extend(_consistency_errors(by_image))
    if check_images:
        errors.extend(_image_errors(by_image, data_dir))
    errors.extend(_pair_errors(by_image))
    return errors


def _consistency_errors(by_image: dict[str, list[tuple[int, dict]]]) -> list[str]:
    errors: list[str] = []
    for image, rows in by_image.items():
        _, base = rows[0]
        for _, sample in rows[1:]:
            if sample["item_states"] != base["item_states"]:
                errors.append(
                    f"image_path={image}: item_states 不一致"
                    f"（sample_id {base['sample_id']} 与 {sample['sample_id']}）"
                )
                errors.extend(_presence_state_errors(image, base, sample))
            for field in ("session_id", "scene_group_id", "pair_group_id"):
                if sample[field] != base[field]:
                    errors.append(
                        f"image_path={image}: {field} 不一致"
                        f"（sample_id {base['sample_id']} 与 {sample['sample_id']}）"
                    )
    return errors


def _presence_state_errors(image: str, left: dict, right: dict) -> list[str]:
    """同图真值冲突时，点名 presence 样本的 target_item 状态。"""
    found: list[str] = []
    for sample, other in ((left, right), (right, left)):
        if sample.get("task") != "presence":
            continue
        target = sample["target_item"]
        got = sample["item_states"].get(target)
        expected = other["item_states"].get(target)
        if got != expected:
            found.append(
                f"image_path={image}: presence 样本 {sample['sample_id']} 的"
                f" target_item={target} 状态为 {got}，与同图 item_states 中的 {expected} 不一致"
            )
    return found


def _image_errors(by_image: dict[str, list[tuple[int, dict]]], data_dir: Path | str | None) -> list[str]:
    errors: list[str] = []
    for image in sorted(by_image):
        try:
            full = resolve_image_path(image, data_dir)
        except ValueError as exc:
            errors.append(f"image_path={image}: {exc}")
            continue
        if not full.is_file():
            errors.append(f"image_path={image}: 图片不存在 ({full})")
            continue
        try:
            with Image.open(full) as im:
                im.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            errors.append(f"image_path={image}: 图片无法打开 ({full}): {exc}")
    return errors


def _pair_errors(by_image: dict[str, list[tuple[int, dict]]]) -> list[str]:
    """按图片聚合后再看 pair：同一张图的多条问法只算一张图。"""
    groups: dict[str, dict[str, dict]] = {}
    for image, rows in by_image.items():
        sample = rows[0][1]
        pair_id = sample.get("pair_group_id")
        if not pair_id:
            continue
        groups.setdefault(pair_id, {})[image] = sample["item_states"]

    errors: list[str] = []
    for pair_id, images in sorted(groups.items()):
        paths = list(images)
        if len(paths) != 2:
            errors.append(f"pair_group_id={pair_id}: 期望恰好 2 张不同的图，实际 {len(paths)} 张")
            continue
        left, right = images[paths[0]], images[paths[1]]
        diff = [cat for cat in CATEGORY_IDS if left.get(cat) != right.get(cat)]
        if len(diff) != 1:
            shown = "、".join(diff) if diff else "无"
            errors.append(
                f"pair_group_id={pair_id}: 期望恰好 1 个类别真值不同，实际 {len(diff)} 个（{shown}）；"
                f"图片 {paths[0]} 与 {paths[1]}"
            )
    return errors
