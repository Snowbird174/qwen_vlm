"""按场景组划分 train/val/test，并写出带哈希的 manifest。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from random import Random

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.data.io import sha256_file, write_jsonl

SPLIT_NAMES = ("train", "val", "test")


def compute_split_counts(n: int, ratios: tuple[float, float, float]) -> list[int]:
    """把 n 个划分单位按比例分成三份，总和严格等于 n。

    使用最大余额法。n >= 3 且某份比例大于 0 时，尽量不让那一份变成 0，
    这样很小的冒烟集也能同时有 train/val/test。
    """
    if n < 0:
        raise ValueError(f"n 不能为负: {n}")
    if len(ratios) != 3:
        raise ValueError("ratios 必须是 train/val/test 三个数")
    if any(r < 0 for r in ratios):
        raise ValueError(f"比例不能为负: {ratios}")
    if n == 0:
        return [0, 0, 0]
    raw = [n * r for r in ratios]
    counts = [math.floor(x) for x in raw]
    leftover = n - sum(counts)
    order = sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))
    for k in range(leftover):
        counts[order[k]] += 1
    if n >= 3:
        for _ in range(3):
            zeros = [i for i, ratio in enumerate(ratios) if ratio > 0 and counts[i] == 0]
            if not zeros:
                break
            donor = max(range(3), key=lambda j: (counts[j], -j))
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[zeros[0]] += 1
    if sum(counts) != n:
        raise RuntimeError(f"划分数量之和 {sum(counts)} != {n}")
    return counts


def assign_splits(
    samples: list[dict],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
    holdout_sessions: list[str] | tuple[str, ...] = (),
) -> tuple[dict[str, list[dict]], list[str]]:
    """按场景组划分。返回 ``(split -> 样本列表, 警告)``。

    划分单位是 ``scene_group_id``。若一个 ``pair_group_id`` 跨了多个场景组，
    这些场景组并成一个单位，保证成对样本落在同一个 split。
    ``holdout_sessions`` 里的 session 整体进入 ``test_ood``，不参与 70/15/15。
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"train/val/test 比例之和必须为 1，实际 {sum(ratios)}")

    scene_session: dict[str, str] = {}
    for sample in samples:
        scene = sample["scene_group_id"]
        session = sample["session_id"]
        previous = scene_session.setdefault(scene, session)
        if previous != session:
            raise ValueError(f"scene_group_id={scene} 对应多个 session_id: {previous} 与 {session}")

    parent = {scene: scene for scene in scene_session}

    def find(scene: str) -> str:
        while parent[scene] != scene:
            parent[scene] = parent[parent[scene]]
            scene = parent[scene]
        return scene

    def union(left: str, right: str) -> None:
        ra, rb = find(left), find(right)
        if ra == rb:
            return
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra

    pair_scenes: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        pair_id = sample.get("pair_group_id")
        if pair_id:
            pair_scenes[pair_id].add(sample["scene_group_id"])
    for scenes in pair_scenes.values():
        ordered = sorted(scenes)
        for other in ordered[1:]:
            union(ordered[0], other)

    component_of = {scene: find(scene) for scene in scene_session}
    component_sessions: dict[str, set[str]] = defaultdict(set)
    for scene, component in component_of.items():
        component_sessions[component].add(scene_session[scene])

    holdout = set(holdout_sessions)
    warnings: list[str] = []
    held: dict[str, bool] = {}
    for component, sessions in component_sessions.items():
        hit = sessions & holdout
        held[component] = bool(hit)
        if hit and sessions - holdout:
            warnings.append(
                "pair_group 把留出 session 和普通 session 连在了一起，整组进入 test_ood: "
                + "、".join(sorted(sessions))
            )

    free = sorted(component for component, is_held in held.items() if not is_held)
    counts = compute_split_counts(len(free), ratios)
    shuffled = list(free)
    Random(seed).shuffle(shuffled)
    assignment: dict[str, str] = {}
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts):
        for component in shuffled[cursor : cursor + count]:
            assignment[component] = name
        cursor += count

    scene_split = {
        scene: ("test_ood" if held[component] else assignment[component])
        for scene, component in component_of.items()
    }
    buckets: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    if holdout:
        buckets["test_ood"] = []
    for sample in samples:
        buckets[scene_split[sample["scene_group_id"]]].append(sample)
    for rows in buckets.values():
        rows.sort(key=lambda s: s["sample_id"])
    return buckets, warnings


def write_split_files(
    buckets: dict[str, list[dict]],
    out_dir: Path | str,
    *,
    source_jsonl: Path | str,
    seed: int,
    params: dict,
    warnings: list[str] | None = None,
) -> dict:
    """写出 ``{train,val,test}.jsonl`` 和 ``manifest.json``。有 test_ood 时一并写出。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [name for name in (*SPLIT_NAMES, "test_ood") if name in buckets]
    files: dict[str, dict] = {}
    for name in names:
        target = out / f"{name}.jsonl"
        write_jsonl(target, buckets[name])
        scenes = sorted({row["scene_group_id"] for row in buckets[name]})
        files[name] = {
            "path": target.name,
            "sha256": sha256_file(target),
            "num_samples": len(buckets[name]),
            "num_scene_groups": len(scenes),
            "scene_group_ids": scenes,
        }
    source = Path(source_jsonl)
    manifest = {
        "manifest_version": 1,
        "name": out.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "source_jsonl": _display_path(source),
        "source_sha256": sha256_file(source),
        "params": params,
        "warnings": list(warnings or []),
        "files": files,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
