"""合成冒烟集必须能通过完整校验，并且划分不能把同一布置拆开。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from qwen_vlm import DATA_DIR, PROJECT_ROOT
from qwen_vlm.data.io import load_jsonl, sha256_file
from qwen_vlm.data.leakage import check_leakage
from qwen_vlm.data.validate import validate_samples

ROOT = PROJECT_ROOT / "data" / "smoke" / "synthetic"
SPLIT_DIR = ROOT / "splits" / "v0"


def _splits() -> dict[str, list[dict]]:
    return {name: load_jsonl(SPLIT_DIR / f"{name}.jsonl") for name in ("train", "val", "test")}


def test_synthetic_samples_and_splits_validate():
    samples = load_jsonl(ROOT / "samples.jsonl")
    assert validate_samples(samples, data_dir=DATA_DIR) == []
    splits = _splits()
    for name, rows in splits.items():
        assert rows, f"{name} 不应为空"
        assert validate_samples(rows, data_dir=DATA_DIR) == []
    report = check_leakage(splits, data_dir=DATA_DIR, phash_threshold=8)
    assert report["ok"], report
    assert report["scene_group_leaks"] == []
    assert report["pair_group_leaks"] == []


def test_synthetic_covers_required_diversity_and_is_labeled_smoke():
    samples = load_jsonl(ROOT / "samples.jsonl")
    splits = _splits()
    images = {row["image_path"] for row in samples}
    sessions = {row["session_id"] for row in samples}
    scenes = {row["scene_group_id"] for row in samples}
    pairs = {row["pair_group_id"] for row in samples if row["pair_group_id"]}
    assert len(images) >= 24
    assert len(sessions) >= 3
    assert len(scenes) >= 6
    assert len(pairs) >= 4
    assert any(state == "unknown" for row in samples for state in row["item_states"].values())
    assert any(row["hard_negative_tags"] for row in samples)
    assert any(row["task"] == "presence" for row in samples)
    assert all(row["source"] == "synthetic_smoke" for row in samples)
    assert all(row["annotation_verified"] is False for row in samples)
    assert all(not row["question_zh"].startswith("【测试问法】") for row in samples)

    seen: dict[str, str] = {}
    split_ids = []
    for name, rows in splits.items():
        for row in rows:
            previous = seen.setdefault(row["scene_group_id"], name)
            assert previous == name
            split_ids.append(row["sample_id"])
    assert sorted(split_ids) == sorted(row["sample_id"] for row in samples)
    train = splits["train"]
    assert any(row["task"] == "presence" for row in train)
    assert any(row["pair_group_id"] for row in train)
    assert any(state == "unknown" for row in train for state in row["item_states"].values())
    assert any(row["hard_negative_tags"] for row in train)

    example = next(iter(images))
    with Image.open(DATA_DIR / example) as im:
        assert im.size == (1280, 960)
    assert str(example).startswith("smoke/synthetic/images/")


def test_split_manifest_hashes_match_files():
    manifest = json.loads((SPLIT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 0
    assert manifest["source_jsonl"] == "data/smoke/synthetic/samples.jsonl"
    assert manifest["source_sha256"] == sha256_file(ROOT / "samples.jsonl")
    for name, info in manifest["files"].items():
        path = SPLIT_DIR / info["path"]
        rows = load_jsonl(path)
        assert info["sha256"] == sha256_file(path)
        assert info["num_samples"] == len(rows)
        assert info["num_scene_groups"] == len({row["scene_group_id"] for row in rows})
        assert Path(info["path"]).name == f"{name}.jsonl"
