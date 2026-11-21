"""Focused tests for deterministic evaluation-only robustness variants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.data.io import load_jsonl, sha256_file
from qwen_vlm.data.robust import (
    CONDITION_NAMES,
    CONDITIONS,
    JPEG_QUALITY,
    RobustError,
    generate_robust_dataset,
    sample_id_for_condition,
)
from qwen_vlm.data.validate import validate_samples


def _states(**changes: str) -> dict[str, str]:
    states = {category: "present" for category in CATEGORY_IDS}
    states.update(changes)
    return {category: states[category] for category in CATEGORY_IDS}


def _make_rows(data_dir: Path, *, reverse: bool = False) -> list[dict]:
    specs = (
        ("a", "s001_l001_v001", "s001_l001", "s001_l001_p01", _states()),
        (
            "b",
            "s001_l001_v002",
            "s001_l001",
            "s001_l001_p01",
            _states(screwdriver="absent"),
        ),
        ("c", "s001_l002_v003", "s001_l002", None, _states()),
    )
    rows: list[dict] = []
    for name, prefix, scene, pair_id, states in specs:
        image_path = f"raw/session_001/{name}.png"
        image_file = data_dir / image_path
        image_file.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (96, 64), (70, 90, 110))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 35, 32), fill=(210, 80, 45))
        draw.rectangle((44, 20, 77, 48), fill=(35, 175, 95))
        draw.line((5, 55, 90, 35), fill=(240, 225, 95), width=3)
        image.save(image_file)
        for seq in (1, 2):
            rows.append(
                {
                    "sample_id": f"{prefix}_q{seq:02d}",
                    "image_path": image_path,
                    "session_id": "s001",
                    "scene_group_id": scene,
                    "pair_group_id": pair_id,
                    "task": "checklist",
                    "required_items": ["screwdriver"],
                    "question_zh": f"检查第 {seq} 题",
                    "item_states": states,
                    "condition_tags": ["baseline"],
                    "hard_negative_tags": ["cable_without_multimeter"],
                    "unknown_reason": {},
                    "source": "real_lab",
                    "annotation_verified": True,
                }
            )
    return list(reversed(rows)) if reverse else rows


def _generate(tmp_path: Path, *, reverse: bool = False):
    data_dir = tmp_path / "data"
    source = data_dir / "annotations" / "real_v1.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in _make_rows(data_dir, reverse=reverse)
        ),
        encoding="utf-8",
    )
    report = generate_robust_dataset(
        real_jsonl=source,
        out_dir=data_dir / "robust" / "v1",
        data_dir=data_dir,
        seed=17,
        contact_sheet_cell_size=(48, 32),
    )
    return data_dir, source, report


def test_conditions_manifest_jpeg_and_full_sheet(tmp_path: Path):
    data_dir, source, report = _generate(tmp_path)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert CONDITION_NAMES == (
        "clean",
        "dark_l1",
        "dark_l2",
        "bright_l1",
        "bright_l2",
        "blur_l1",
        "blur_l2",
        "noise_l1",
        "noise_l2",
    )
    assert manifest["source_jsonl_sha256"] == sha256_file(source)
    assert manifest["training_eligible"] is False
    assert manifest["jpeg_quality"] == JPEG_QUALITY == 95
    assert list(manifest["conditions"]) == sorted(CONDITION_NAMES)

    for condition in CONDITIONS:
        info = manifest["conditions"][condition.name]
        assert info["parameters"] == condition.parameters
        assert info["id_suffix"] == condition.id_suffix
        samples_path = data_dir / info["samples_path"]
        samples = load_jsonl(samples_path)
        assert len(samples) == info["sample_count"] == 6
        assert sha256_file(samples_path) == info["samples_sha256"]
        assert validate_samples(samples, data_dir=data_dir) == []
        assert len(info["images"]) == info["image_count"] == 3
        for record in info["images"]:
            image_file = data_dir / record["image_path"]
            assert sha256_file(image_file) == record["sha256"]
            with Image.open(image_file) as image:
                assert image.format == "JPEG"
                assert image.size == (96, 64)

    with Image.open(report.contact_sheet_path) as sheet:
        assert sheet.size == (10 * 48, 3 * (32 + 30))


def test_metadata_ids_and_null_pair_are_preserved(tmp_path: Path):
    data_dir, source, report = _generate(tmp_path)
    originals = {row["sample_id"]: row for row in load_jsonl(source)}
    generated = load_jsonl(data_dir / "robust" / "v1" / "blur_l1" / "samples.jsonl")
    assert sample_id_for_condition("s001_l008_v010_q01", "r05") == "s001_l008_v010r05_q01"
    assert report.condition_counts["blur_l1"] == 6

    for row in generated:
        original = originals[row["sample_id"].replace("r05", "", 1)]
        assert row["pair_group_id"] == (
            None if original["pair_group_id"] is None else original["pair_group_id"] + "r05"
        )
        assert row["condition_tags"] == ["baseline", "robust_blur_l1"]
        for field, value in original.items():
            if field not in {"sample_id", "image_path", "pair_group_id", "condition_tags"}:
                assert row[field] == value


def test_results_are_order_independent_and_output_stays_evaluation_only(
    tmp_path: Path,
):
    first_data, _, first = _generate(tmp_path / "first")
    second_data, _, second = _generate(tmp_path / "second", reverse=True)
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))

    for name in CONDITION_NAMES:
        first_images = sorted(first_manifest["conditions"][name]["images"], key=lambda row: row["image_path"])
        second_images = sorted(second_manifest["conditions"][name]["images"], key=lambda row: row["image_path"])
        assert first_images == second_images
        for record in first_images:
            assert sha256_file(first_data / record["image_path"]) == record["sha256"]
            assert sha256_file(second_data / record["image_path"]) == record["sha256"]

    clean = {row["image_path"]: row["sha256"] for row in first_manifest["conditions"]["clean"]["images"]}
    noisy = {row["image_path"]: row["sha256"] for row in first_manifest["conditions"]["noise_l2"]["images"]}
    assert set(clean.values()).isdisjoint(noisy.values())

    data_dir = tmp_path / "guard" / "data"
    source = data_dir / "annotations" / "real_v1.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in _make_rows(data_dir)),
        encoding="utf-8",
    )
    with pytest.raises(RobustError, match="data/robust"):
        generate_robust_dataset(
            real_jsonl=source,
            out_dir=data_dir / "splits" / "robust",
            data_dir=data_dir,
        )
