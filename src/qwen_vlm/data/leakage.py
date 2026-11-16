"""检查划分之间的场景泄漏、成对泄漏、精确重复和感知近重复。"""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image

from qwen_vlm.data.io import resolve_image_path, sha256_file


def check_leakage(
    splits: dict[str, list[dict]],
    *,
    data_dir: Path | str | None = None,
    phash_threshold: int = 8,
) -> dict:
    """对比多个 split，返回报告字典。

    ``ok`` 为 False 表示：场景组、成对组或图片路径跨了 split，不同路径的文件
    sha256 相同，或跨 split 的感知哈希距离不超过阈值。
    同一 split 内的近重复只记在 ``within_split_near_duplicates``，不因此失败：
    成对拍摄本来就会很像，而且它们必须待在同一个 split。
    """
    if phash_threshold < 0:
        raise ValueError(f"phash_threshold 不能为负: {phash_threshold}")

    scene_to_splits: dict[str, set[str]] = {}
    pair_to_splits: dict[str, set[str]] = {}
    path_to_splits: dict[str, set[str]] = {}
    for split_name, samples in splits.items():
        for sample in samples:
            scene_to_splits.setdefault(sample["scene_group_id"], set()).add(split_name)
            pair_id = sample.get("pair_group_id")
            if pair_id:
                pair_to_splits.setdefault(pair_id, set()).add(split_name)
            path_to_splits.setdefault(sample["image_path"], set()).add(split_name)

    scene_group_leaks = [
        {"scene_group_id": scene, "splits": sorted(names)}
        for scene, names in sorted(scene_to_splits.items())
        if len(names) > 1
    ]
    pair_group_leaks = [
        {"pair_group_id": pair_id, "splits": sorted(names)}
        for pair_id, names in sorted(pair_to_splits.items())
        if len(names) > 1
    ]
    image_path_leaks = [
        {"image_path": path, "splits": sorted(names)}
        for path, names in sorted(path_to_splits.items())
        if len(names) > 1
    ]

    hashed: list[dict] = []
    unreadable: list[dict] = []
    for path, names in sorted(path_to_splits.items()):
        try:
            full = resolve_image_path(path, data_dir)
        except ValueError as exc:
            unreadable.append({"image_path": path, "error": str(exc)})
            continue
        if not full.is_file():
            unreadable.append({"image_path": path, "error": f"图片不存在 ({full})"})
            continue
        try:
            with Image.open(full) as im:
                perceptual = imagehash.phash(im.convert("RGB"))
        except OSError as exc:
            unreadable.append({"image_path": path, "error": f"图片无法打开: {exc}"})
            continue
        hashed.append(
            {
                "image_path": path,
                "splits": sorted(names),
                "sha256": sha256_file(full),
                "phash": perceptual,
            }
        )

    by_sha: dict[str, list[dict]] = {}
    for item in hashed:
        by_sha.setdefault(item["sha256"], []).append(item)
    exact_duplicates = []
    for digest, items in sorted(by_sha.items()):
        paths = [item["image_path"] for item in items]
        if len(set(paths)) <= 1:
            continue
        split_names = sorted({name for item in items for name in item["splits"]})
        exact_duplicates.append(
            {
                "sha256": digest,
                "cross_split": len(split_names) > 1,
                "images": [{"image_path": item["image_path"], "splits": item["splits"]} for item in items],
            }
        )

    near_duplicates = []
    within_split_near = []
    for i, left in enumerate(hashed):
        for right in hashed[i + 1 :]:
            if left["image_path"] == right["image_path"]:
                continue
            distance = int(left["phash"] - right["phash"])
            if distance > phash_threshold:
                continue
            left_splits = set(left["splits"])
            right_splits = set(right["splits"])
            record = {
                "image_a": left["image_path"],
                "splits_a": left["splits"],
                "image_b": right["image_path"],
                "splits_b": right["splits"],
                "distance": distance,
            }
            if left_splits != right_splits or len(left_splits) > 1 or len(right_splits) > 1:
                near_duplicates.append(record)
            else:
                within_split_near.append(record)

    ok = not (
        scene_group_leaks
        or pair_group_leaks
        or image_path_leaks
        or exact_duplicates
        or near_duplicates
        or unreadable
    )
    return {
        "ok": ok,
        "phash_threshold": phash_threshold,
        "scene_group_leaks": scene_group_leaks,
        "pair_group_leaks": pair_group_leaks,
        "image_path_leaks": image_path_leaks,
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "within_split_near_duplicates": within_split_near,
        "unreadable_images": unreadable,
    }


def format_leakage_report(report: dict) -> str:
    """把泄漏报告打成便于阅读的中文摘要。"""
    lines = [
        f"泄漏检查: {'通过' if report['ok'] else '未通过'}（phash 阈值 {report['phash_threshold']}）",
        f"- 跨 split 的 scene_group: {len(report['scene_group_leaks'])}",
        f"- 跨 split 的 pair_group: {len(report['pair_group_leaks'])}",
        f"- 跨 split 的 image_path: {len(report['image_path_leaks'])}",
        f"- 精确重复（不同路径、相同 sha256）: {len(report['exact_duplicates'])}",
        f"- 跨 split 感知近重复: {len(report['near_duplicates'])}",
        f"- 同 split 内感知近重复（成对图通常落在这里）: {len(report['within_split_near_duplicates'])}",
        f"- 无法读取的图片: {len(report['unreadable_images'])}",
    ]
    for key, label in (
        ("scene_group_leaks", "scene_group"),
        ("pair_group_leaks", "pair_group"),
        ("image_path_leaks", "image_path"),
    ):
        for item in report[key][:10]:
            ident = item.get("scene_group_id") or item.get("pair_group_id") or item.get("image_path")
            lines.append(f"  {label} {ident} -> {', '.join(item['splits'])}")
    for item in report["near_duplicates"][:10]:
        lines.append(
            f"  近重复 距离 {item['distance']}: {item['image_a']} {item['splits_a']} |"
            f" {item['image_b']} {item['splits_b']}"
        )
    for item in report["unreadable_images"][:10]:
        lines.append(f"  无法读取 {item['image_path']}: {item['error']}")
    return "\n".join(lines)
