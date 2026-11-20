"""固定三折划分的覆盖、泄漏和 manifest 契约。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.data.cv import (
    FIXED_CV3_TEST_GROUPS,
    build_fixed_cv3_folds,
    write_fixed_cv3_folds,
)
from qwen_vlm.data.io import load_jsonl, sha256_file, write_jsonl
from qwen_vlm.data.validate import validate_samples

SCENES = tuple(
    scene for fold in sorted(FIXED_CV3_TEST_GROUPS) for scene in FIXED_CV3_TEST_GROUPS[fold]
)


def _states(**overrides: str) -> dict[str, str]:
    states = {category: "present" for category in CATEGORY_IDS}
    states.update(overrides)
    return states


def _sample(
    scene: str,
    sample_id: str,
    image_path: str,
    *,
    source: str,
    pair_group_id: str | None = None,
    item_states: dict[str, str] | None = None,
    session_id: str = "s001",
    task: str = "checklist",
    target_item: str | None = None,
) -> dict:
    states = item_states or _states()
    row = {
        "sample_id": sample_id,
        "image_path": image_path,
        "session_id": session_id,
        "scene_group_id": scene,
        "pair_group_id": pair_group_id,
        "task": task,
        "required_items": [target_item] if target_item else ["screwdriver", "pliers"],
        "question_zh": "请检查画面中的器材是否齐全。",
        "item_states": states,
        "condition_tags": [],
        "hard_negative_tags": [],
        "unknown_reason": {
            category: "合成遮挡"
            for category, state in states.items()
            if state == "unknown"
        },
        "source": source,
        "annotation_verified": source == "real_lab",
    }
    if target_item:
        row["target_item"] = target_item
    return row


def _write_image(data_dir: Path, rel: str, color: tuple[int, int, int]) -> None:
    path = data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color).save(path)


def _fixture(tmp_path: Path) -> tuple[list[dict], list[dict], Path, Path, Path]:
    data_dir = tmp_path / "data"
    real_rows: list[dict] = []
    synthetic_rows: list[dict] = []
    for index, scene in enumerate(SCENES):
        image = f"raw/{scene}.png"
        _write_image(data_dir, image, (20 + index, 40, 60))
        real_rows.append(
            _sample(
                scene,
                f"{scene}_q01",
                image,
                source="real_lab",
            )
        )

        synthetic_image = f"aug/{scene}_edit.png"
        _write_image(data_dir, synthetic_image, (80, 20 + index, 40))
        synthetic_rows.append(
            _sample(
                scene,
                f"{scene}_e01_q01",
                synthetic_image,
                source="synthetic_edit",
                item_states=_states(multimeter="absent"),
            )
        )

    # 同一张真实图的两条问法都进入同一折，但图片覆盖仍只算一次。
    l009 = next(row for row in real_rows if row["scene_group_id"] == "s001_l009")
    second = dict(l009)
    second["sample_id"] = "s001_l009_q02"
    second["question_zh"] = "开始实验前，所需器材都在吗？"
    real_rows.append(second)

    # 成对真实样本在同一组内，fold2 测试时必须两张图一起进入 test。
    pair_a = "raw/pair_a.png"
    pair_b = "raw/pair_b.png"
    _write_image(data_dir, pair_a, (180, 20, 20))
    _write_image(data_dir, pair_b, (20, 20, 180))
    real_rows.extend(
        [
            _sample(
                "s001_l008",
                "s001_l008_pair_a_q01",
                pair_a,
                source="real_lab",
                pair_group_id="s001_l008_pair",
            ),
            _sample(
                "s001_l008",
                "s001_l008_pair_b_q01",
                pair_b,
                source="real_lab",
                pair_group_id="s001_l008_pair",
                item_states=_states(hammer="absent"),
            ),
        ]
    )

    # 合成成对样本在测试组时要整体排除，不应进入 train。
    synth_a = "aug/s001_l009_pair_a.png"
    synth_b = "aug/s001_l009_pair_b.png"
    _write_image(data_dir, synth_a, (200, 80, 20))
    _write_image(data_dir, synth_b, (20, 80, 200))
    synthetic_rows.extend(
        [
            _sample(
                "s001_l009",
                "s001_l009_e_pair_a_q01",
                synth_a,
                source="synthetic_edit",
                pair_group_id="synth_l009_pair",
            ),
            _sample(
                "s001_l009",
                "s001_l009_e_pair_b_q01",
                synth_b,
                source="synthetic_edit",
                pair_group_id="synth_l009_pair",
                item_states=_states(pliers="absent"),
            ),
        ]
    )

    real_path = tmp_path / "real_v1.jsonl"
    augment_path = tmp_path / "samples.jsonl"
    write_jsonl(real_path, real_rows)
    write_jsonl(augment_path, synthetic_rows)
    return real_rows, synthetic_rows, real_path, augment_path, data_dir


def test_fixed_cv3_partition_coverage_and_manifest(tmp_path: Path):
    real, synthetic, real_path, augment_path, data_dir = _fixture(tmp_path)
    folds = build_fixed_cv3_folds(real, synthetic)
    assert folds[1]["test_scene_groups"] == ["s001_l009"]
    assert folds[2]["test_scene_groups"] == ["s001_l008"]
    assert folds[3]["test_scene_groups"] == [
        f"s001_l00{index}" for index in range(1, 8)
    ]

    out = tmp_path / "splits" / "cv3"
    manifests = write_fixed_cv3_folds(
        folds,
        out,
        real_jsonl=real_path,
        augment_jsonl=augment_path,
        data_dir=data_dir,
    )

    all_test_ids = []
    all_test_images = []
    all_real_ids = {row["sample_id"] for row in real}
    all_real_images = {row["image_path"] for row in real}
    for fold in (1, 2, 3):
        fold_dir = out / f"fold{fold}"
        train = load_jsonl(fold_dir / "train.jsonl")
        train_real = load_jsonl(fold_dir / "train_real.jsonl")
        test = load_jsonl(fold_dir / "test.jsonl")
        manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["manifest_version"] == 1
        assert manifest["protocol"] == "fixed_scene_group_cv3"
        assert manifest["fold"] == fold
        assert manifest["test_scene_groups"] == folds[fold]["test_scene_groups"]
        assert set(manifest["files"]) == {"train", "train_real", "test"}
        for name, rows in (("train", train), ("train_real", train_real), ("test", test)):
            info = manifest["files"][name]
            assert info["path"] == f"{name}.jsonl"
            assert info["sha256"] == sha256_file(fold_dir / f"{name}.jsonl")
            assert info["num_samples"] == len(rows)
            assert validate_samples(rows, data_dir=data_dir) == []
        assert all(row["source"] == "real_lab" for row in train_real)
        assert all(row["source"] == "real_lab" for row in test)
        assert all(row["source"] == "real_lab" for row in train if row["sample_id"] in all_real_ids)
        assert not ({row["scene_group_id"] for row in train} & {row["scene_group_id"] for row in test})

        if fold == 1:
            assert not any(row["scene_group_id"] == "s001_l009" for row in train)
        if fold == 2:
            pair_ids = {
                row["sample_id"]
                for row in test
                if row["pair_group_id"] == "s001_l008_pair"
            }
            assert pair_ids == {
                "s001_l008_pair_a_q01",
                "s001_l008_pair_b_q01",
            }

        all_test_ids.extend(row["sample_id"] for row in test)
        all_test_images.extend({row["image_path"] for row in test})

    assert len(all_test_ids) == len(set(all_test_ids))
    assert set(all_test_ids) == all_real_ids
    assert len(all_test_images) == len(set(all_test_images))
    assert set(all_test_images) == all_real_images
    assert manifests[1]["coverage"]["real_samples_covered_once"] is True
    assert manifests[1]["coverage"]["real_images_covered_once"] is True


def test_cross_scene_pair_is_rejected_instead_of_silently_repartitioned(tmp_path: Path):
    real, synthetic, *_ = _fixture(tmp_path)
    left = next(row for row in real if row["scene_group_id"] == "s001_l001")
    right = next(
        row
        for row in real
        if row["scene_group_id"] == "s001_l008"
        and row["pair_group_id"] is None
    )
    left["pair_group_id"] = "cross_scene_pair"
    right["pair_group_id"] = "cross_scene_pair"
    right["item_states"] = _states(hammer="absent")

    with pytest.raises(ValueError, match="pair_group_id 跨 scene_group"):
        build_fixed_cv3_folds(real, synthetic)


def test_synthetic_must_inherit_known_scene_and_session(tmp_path: Path):
    real, synthetic, *_ = _fixture(tmp_path)
    unknown = dict(synthetic[0])
    unknown["sample_id"] = "unknown_scene"
    unknown["scene_group_id"] = "s001_l999"
    unknown["image_path"] = "aug/unknown_scene.png"
    with pytest.raises(ValueError, match="不存在的 scene_group_id"):
        build_fixed_cv3_folds(real, [*synthetic, unknown])

    wrong_session = []
    for row in synthetic:
        changed = dict(row)
        if changed["scene_group_id"] == "s001_l009":
            changed["session_id"] = "other"
            changed["image_path"] = f"wrong/{changed['image_path']}"
        wrong_session.append(changed)
    with pytest.raises(ValueError, match="session_id 没有继承"):
        build_fixed_cv3_folds(real, wrong_session)


def test_writer_rejects_missing_training_image(tmp_path: Path):
    real, synthetic, real_path, augment_path, data_dir = _fixture(tmp_path)
    folds = build_fixed_cv3_folds(real, synthetic)
    (data_dir / folds[1]["train"][0]["image_path"]).unlink()

    with pytest.raises(ValueError, match="图片不存在"):
        write_fixed_cv3_folds(
            folds,
            tmp_path / "splits" / "cv3",
            real_jsonl=real_path,
            augment_jsonl=augment_path,
            data_dir=data_dir,
        )


def test_manifest_is_byte_stable_for_identical_inputs(tmp_path: Path):
    real, synthetic, real_path, augment_path, data_dir = _fixture(tmp_path)
    folds = build_fixed_cv3_folds(real, synthetic)
    out = tmp_path / "splits" / "cv3"
    kwargs = {
        "real_jsonl": real_path,
        "augment_jsonl": augment_path,
        "data_dir": data_dir,
    }
    write_fixed_cv3_folds(folds, out, **kwargs)
    first = {
        fold: (out / f"fold{fold}" / "manifest.json").read_bytes()
        for fold in (1, 2, 3)
    }
    write_fixed_cv3_folds(folds, out, **kwargs)
    second = {
        fold: (out / f"fold{fold}" / "manifest.json").read_bytes()
        for fold in (1, 2, 3)
    }
    assert first == second
