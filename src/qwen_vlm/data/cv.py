"""固定三折交叉验证：按 scene_group_id 生成训练、真实训练和真实测试集。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.data.io import sha256_file, write_jsonl
from qwen_vlm.data.validate import validate_samples

REAL_SOURCE = "real_lab"
SYNTHETIC_SOURCE = "synthetic_edit"

# 协议固定写在 docs/NEXT_STEPS.md 任务 C。不要随机重排或把跨组 pair 拆开。
FIXED_CV3_TEST_GROUPS: dict[int, tuple[str, ...]] = {
    1: ("s001_l009",),
    2: ("s001_l008",),
    3: (
        "s001_l001",
        "s001_l002",
        "s001_l003",
        "s001_l004",
        "s001_l005",
        "s001_l006",
        "s001_l007",
    ),
}
FIXED_CV3_SCENE_GROUPS = tuple(
    sorted(group for groups in FIXED_CV3_TEST_GROUPS.values() for group in groups)
)


def build_fixed_cv3_folds(
    real_samples: list[dict],
    synthetic_samples: list[dict],
) -> dict[int, dict]:
    """返回 fold -> ``train``/``train_real``/``test`` 样本列表。

    test 只包含指定组的真实样本；train 包含其余组的真实与 synthetic_edit 样本；
    train_real 只包含其余组的真实样本。先做完整 schema/图片内一致性/成对约束校验，
    再拒绝任何可能使固定折产生泄漏的输入。
    """
    real = list(real_samples)
    synthetic = list(synthetic_samples)
    if not real:
        raise ValueError("真实样本为空")
    if not synthetic:
        raise ValueError("synthetic_edit 样本为空；任务 C 需要真实 + 合成训练集")

    combined = real + synthetic
    errors = validate_samples(combined, check_images=False)
    if errors:
        shown = "\n  ".join(errors[:20])
        more = "" if len(errors) <= 20 else f"\n  ... 另有 {len(errors) - 20} 个错误"
        raise ValueError(f"CV 输入未通过样本校验：\n  {shown}{more}")

    _assert_sources(real, synthetic)
    real_scenes = _scene_sessions(real, "真实样本")
    synthetic_scenes = _scene_sessions(synthetic, "合成样本")
    _assert_fixed_scene_groups(set(real_scenes))
    _assert_synthetic_scenes(synthetic_scenes, real_scenes)
    _assert_pair_groups(real, synthetic)

    all_real_ids = {row["sample_id"] for row in real}
    all_real_images = {row["image_path"] for row in real}
    all_scenes = set(real_scenes)
    folds: dict[int, dict] = {}
    for fold, test_groups in FIXED_CV3_TEST_GROUPS.items():
        test_set = set(test_groups)
        train_set = all_scenes - test_set
        test_rows = [row for row in real if row["scene_group_id"] in test_set]
        train_real = [row for row in real if row["scene_group_id"] in train_set]
        train_synthetic = [
            row for row in synthetic if row["scene_group_id"] in train_set
        ]
        train = train_real + train_synthetic
        for name, rows in (
            ("train", train),
            ("train_real", train_real),
            ("test", test_rows),
        ):
            rows.sort(key=lambda row: row["sample_id"])
            if not rows:
                raise ValueError(f"fold{fold}/{name}.jsonl 为空")
        folds[fold] = {
            "fold": fold,
            "test_scene_groups": sorted(test_set),
            "train_scene_groups": sorted(train_set),
            "train": train,
            "train_real": train_real,
            "test": test_rows,
        }

    _assert_fold_contract(folds, all_real_ids, all_real_images)
    return folds


def write_fixed_cv3_folds(
    folds: dict[int, dict],
    out_dir: Path | str,
    *,
    real_jsonl: Path | str,
    augment_jsonl: Path | str,
    data_dir: Path | str,
) -> dict[int, dict]:
    """写出每折三个 JSONL 与可由 train.py 直接使用的 manifest。"""
    out = Path(out_dir)
    real_path = Path(real_jsonl)
    augment_path = Path(augment_jsonl)
    real_sha256 = sha256_file(real_path)
    augment_sha256 = sha256_file(augment_path)

    for fold, payload in sorted(folds.items()):
        for name in ("train", "train_real", "test"):
            errors = validate_samples(payload[name], data_dir=data_dir, check_images=True)
            if errors:
                shown = "\n  ".join(errors[:20])
                more = "" if len(errors) <= 20 else f"\n  ... 另有 {len(errors) - 20} 个错误"
                raise ValueError(f"fold{fold}/{name}.jsonl 校验失败：\n  {shown}{more}")

    coverage = _coverage_summary(folds)
    source_display = _display_path(real_path)
    augment_display = _display_path(augment_path)
    manifests: dict[int, dict] = {}

    for fold, payload in sorted(folds.items()):
        fold_dir = out / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict] = {}
        for name in ("train", "train_real", "test"):
            target = fold_dir / f"{name}.jsonl"
            write_jsonl(target, payload[name])
            files[name] = _file_info(target.name, target, payload[name])

        manifest = {
            "manifest_version": 1,
            "name": fold_dir.name,
            "protocol": "fixed_scene_group_cv3",
            "fold": fold,
            "seed": 0,
            "source_jsonl": source_display,
            "source_sha256": real_sha256,
            "augment_jsonl": augment_display,
            "augment_sha256": augment_sha256,
            "train_scene_groups": payload["train_scene_groups"],
            "test_scene_groups": payload["test_scene_groups"],
            "params": {
                "protocol": "fixed_scene_group_cv3",
                "fold": fold,
                "randomized": False,
                "split_unit": "scene_group_id；固定映射见 docs/NEXT_STEPS.md 任务 C",
                "train_scene_groups": payload["train_scene_groups"],
                "test_scene_groups": payload["test_scene_groups"],
                "test_sources": [REAL_SOURCE],
                "train_sources": [REAL_SOURCE, SYNTHETIC_SOURCE],
            },
            "warnings": [],
            "coverage": coverage,
            "checks": {
                "test_real_only": True,
                "no_shared_scene_groups": True,
                "no_shared_sample_ids": True,
                "no_shared_image_paths": True,
                "no_shared_pair_group_ids": True,
                "real_samples_covered_once": True,
                "real_images_covered_once": True,
                "pair_groups_within_scene_group": True,
            },
            "files": files,
        }
        manifest_path = fold_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifests[fold] = manifest
    return manifests


def _assert_sources(real: list[dict], synthetic: list[dict]) -> None:
    bad_real = [row["sample_id"] for row in real if row["source"] != REAL_SOURCE]
    if bad_real:
        raise ValueError(
            f"--real 中有 {len(bad_real)} 条 source != {REAL_SOURCE}："
            + "、".join(bad_real[:5])
        )
    bad_synthetic = [
        row["sample_id"] for row in synthetic if row["source"] != SYNTHETIC_SOURCE
    ]
    if bad_synthetic:
        raise ValueError(
            f"--augment 中有 {len(bad_synthetic)} 条 source != {SYNTHETIC_SOURCE}："
            + "、".join(bad_synthetic[:5])
        )


def _scene_sessions(samples: list[dict], label: str) -> dict[str, str]:
    sessions: dict[str, set[str]] = defaultdict(set)
    for row in samples:
        sessions[row["scene_group_id"]].add(row["session_id"])
    conflicts = {
        scene: sorted(values)
        for scene, values in sessions.items()
        if len(values) != 1
    }
    if conflicts:
        detail = "；".join(
            f"{scene}={values}" for scene, values in sorted(conflicts.items())
        )
        raise ValueError(f"{label}中同一 scene_group_id 对应多个 session_id：{detail}")
    return {scene: next(iter(values)) for scene, values in sessions.items()}


def _assert_fixed_scene_groups(real_scenes: set[str]) -> None:
    expected = set(FIXED_CV3_SCENE_GROUPS)
    missing = sorted(expected - real_scenes)
    unknown = sorted(real_scenes - expected)
    if missing or unknown:
        parts = ["固定三折的真实 scene_group_id 集合与任务 C 不一致"]
        if missing:
            parts.append(f"缺少：{', '.join(missing)}")
        if unknown:
            parts.append(f"未知：{', '.join(unknown)}")
        raise ValueError("；".join(parts))


def _assert_synthetic_scenes(
    synthetic_scenes: dict[str, str], real_scenes: dict[str, str]
) -> None:
    unknown = sorted(set(synthetic_scenes) - set(real_scenes))
    if unknown:
        raise ValueError(
            "synthetic_edit 继承了真实数据中不存在的 scene_group_id："
            + "、".join(unknown)
        )
    session_conflicts = [
        f"{scene}({synthetic_scenes[scene]} != {real_scenes[scene]})"
        for scene in sorted(synthetic_scenes)
        if synthetic_scenes[scene] != real_scenes[scene]
    ]
    if session_conflicts:
        raise ValueError(
            "synthetic_edit 的 session_id 没有继承原真实场景："
            + "、".join(session_conflicts)
        )


def _assert_pair_groups(real: list[dict], synthetic: list[dict]) -> None:
    pair_scenes: dict[str, set[str]] = defaultdict(set)
    pair_sources: dict[str, set[str]] = defaultdict(set)
    for label, rows in (("real", real), ("synthetic", synthetic)):
        for row in rows:
            pair_id = row.get("pair_group_id")
            if pair_id:
                pair_scenes[pair_id].add(row["scene_group_id"])
                pair_sources[pair_id].add(label)

    cross_scene = {
        pair_id: sorted(scenes)
        for pair_id, scenes in pair_scenes.items()
        if len(scenes) > 1
    }
    if cross_scene:
        detail = "；".join(
            f"{pair_id}={scenes}" for pair_id, scenes in sorted(cross_scene.items())
        )
        raise ValueError(
            "pair_group_id 跨 scene_group，固定三折无法同时满足声明的测试组与"
            f"完整成对约束：{detail}"
        )
    cross_source = [
        pair_id
        for pair_id, sources in sorted(pair_sources.items())
        if len(sources) > 1
    ]
    if cross_source:
        raise ValueError(
            "pair_group_id 同时出现在真实与 synthetic_edit 输入中："
            + "、".join(cross_source)
        )


def _assert_fold_contract(
    folds: dict[int, dict],
    all_real_ids: set[str],
    all_real_images: set[str],
) -> None:
    seen_test_ids: Counter[str] = Counter()
    seen_test_images: Counter[str] = Counter()
    for fold, payload in sorted(folds.items()):
        train = payload["train"]
        train_real = payload["train_real"]
        test = payload["test"]
        test_scenes = {row["scene_group_id"] for row in test}
        train_scenes = {row["scene_group_id"] for row in train}
        if test_scenes & train_scenes:
            raise ValueError(f"fold{fold}: train/test scene_group_id 重叠")
        if not test_scenes.issubset(set(payload["test_scene_groups"])):
            raise ValueError(f"fold{fold}: test 含声明的测试组之外的样本")
        if not train_scenes.issubset(set(payload["train_scene_groups"])):
            raise ValueError(f"fold{fold}: train 含声明的训练组之外的样本")
        if any(row["source"] != REAL_SOURCE for row in test):
            raise ValueError(f"fold{fold}: test 只能有 {REAL_SOURCE} 样本")
        if any(row["source"] != REAL_SOURCE for row in train_real):
            raise ValueError(f"fold{fold}: train_real 只能有 {REAL_SOURCE} 样本")

        train_ids = {row["sample_id"] for row in train}
        test_ids = {row["sample_id"] for row in test}
        train_images = {row["image_path"] for row in train}
        test_images = {row["image_path"] for row in test}
        train_pairs = {
            row["pair_group_id"] for row in train if row.get("pair_group_id")
        }
        test_pairs = {
            row["pair_group_id"] for row in test if row.get("pair_group_id")
        }
        if train_ids & test_ids:
            raise ValueError(f"fold{fold}: train/test sample_id 重叠")
        if train_images & test_images:
            raise ValueError(f"fold{fold}: train/test image_path 重叠")
        if train_pairs & test_pairs:
            raise ValueError(f"fold{fold}: train/test pair_group_id 重叠")

        seen_test_ids.update(test_ids)
        seen_test_images.update(test_images)

    _assert_exactly_once("真实 sample_id", seen_test_ids, all_real_ids)
    _assert_exactly_once("真实 image_path", seen_test_images, all_real_images)


def _assert_exactly_once(
    label: str, counts: Counter[str], expected: set[str]
) -> None:
    missing = sorted(expected - set(counts))
    duplicated = sorted(key for key, count in counts.items() if count != 1)
    extra = sorted(set(counts) - expected)
    if missing or duplicated or extra:
        parts = [f"{label} 在固定三折测试集中没有恰好覆盖一次"]
        if missing:
            parts.append(f"缺少 {len(missing)}：{', '.join(missing[:5])}")
        if duplicated:
            parts.append(f"重复 {len(duplicated)}：{', '.join(duplicated[:5])}")
        if extra:
            parts.append(f"多出 {len(extra)}：{', '.join(extra[:5])}")
        raise ValueError("；".join(parts))


def _file_info(path_name: str, path: Path, rows: list[dict]) -> dict:
    scenes = sorted({row["scene_group_id"] for row in rows})
    sources = Counter(row["source"] for row in rows)
    return {
        "path": path_name,
        "sha256": sha256_file(path),
        "num_samples": len(rows),
        "num_images": len({row["image_path"] for row in rows}),
        "num_scene_groups": len(scenes),
        "scene_group_ids": scenes,
        "num_real_samples": sources[REAL_SOURCE],
        "num_synthetic_edit_samples": sources[SYNTHETIC_SOURCE],
        "source_counts": dict(sorted(sources.items())),
    }


def _coverage_summary(folds: dict[int, dict]) -> dict:
    test_ids = [row["sample_id"] for payload in folds.values() for row in payload["test"]]
    test_images = [
        image
        for payload in folds.values()
        for image in {row["image_path"] for row in payload["test"]}
    ]
    by_fold = {}
    for fold, payload in sorted(folds.items()):
        by_fold[f"fold{fold}"] = {
            "num_test_samples": len(payload["test"]),
            "num_test_images": len({row["image_path"] for row in payload["test"]}),
            "test_scene_groups": payload["test_scene_groups"],
        }
    return {
        "real_samples": len(set(test_ids)),
        "real_images": len(set(test_images)),
        "test_sample_appearances": len(test_ids),
        "test_image_appearances": len(test_images),
        "real_samples_covered_once": len(test_ids) == len(set(test_ids)),
        "real_images_covered_once": len(test_images) == len(set(test_images)),
        "by_fold": by_fold,
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
