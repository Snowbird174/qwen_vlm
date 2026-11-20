"""Tests for deterministic box-guided bait/true augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.data.augment import (
    AnnotationImage,
    AugmentError,
    BoxAnnotation,
    _prepare_managed_images,
    _removal_mask,
    attach_annotations,
    build_samples,
    load_annotation_images,
    plan_pairs,
    prepare_source_images,
    render_pair_images,
    validate_pair_integrity,
)
from qwen_vlm.data.convert import load_question_templates
from qwen_vlm.data.io import sha256_file
from qwen_vlm.data.validate import validate_samples
from qwen_vlm import PROJECT_ROOT

TEMPLATES = PROJECT_ROOT / "configs" / "data" / "question_templates.yaml"


def _states(**overrides: str) -> dict[str, str]:
    states = {item: "present" for item in CATEGORY_IDS}
    states.update(overrides)
    return {item: states[item] for item in CATEGORY_IDS}


def _source_rows(
    image_path: str = "real/source.png",
    *,
    hard_tags: list[str] | None = None,
) -> list[dict]:
    rows = []
    for seq, required in enumerate(
        (["screwdriver", "multimeter"], ["cable_bundle", "hammer"]),
        start=1,
    ):
        rows.append(
            {
                "sample_id": f"s001_l001_v001_q{seq:02d}",
                "image_path": image_path,
                "session_id": "s001",
                "scene_group_id": "s001_l001",
                "pair_group_id": None,
                "task": "checklist",
                "required_items": required,
                "question_zh": "所需器材都齐吗？",
                "item_states": _states(),
                "condition_tags": ["baseline"],
                "hard_negative_tags": list(hard_tags or []),
                "unknown_reason": {},
                "source": "real_lab",
                "annotation_verified": True,
            }
        )
    return rows


def _draw_source_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 180), (232, 234, 231))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 312, 172), radius=18, outline=(140, 145, 140), width=3)
    for color, box in (
        ((40, 90, 190), (25, 25, 55, 80)),
        ((190, 70, 35), (70, 25, 105, 80)),
        ((35, 55, 100), (125, 45, 180, 80)),
        ((145, 50, 55), (215, 30, 260, 80)),
        ((55, 80, 145), (40, 120, 100, 160)),
    ):
        draw.rectangle(box, fill=color)
    image.save(path)


def _annotations() -> AnnotationImage:
    return AnnotationImage(
        image="source.png",
        width=320,
        height=180,
        boxes=(
            BoxAnnotation("screwdriver", (25, 25, 55, 80)),
            BoxAnnotation("pliers", (70, 25, 105, 80)),
            BoxAnnotation("cable_bundle", (125, 45, 180, 80)),
            BoxAnnotation("multimeter", (215, 30, 260, 80)),
            BoxAnnotation("hammer", (40, 120, 100, 160)),
        ),
    )


def _prepared_source(tmp_path: Path, *, hard_tags: list[str] | None = None):
    data_dir = tmp_path / "data"
    _draw_source_image(data_dir / "real" / "source.png")
    sources = prepare_source_images(_source_rows(hard_tags=hard_tags))
    attached, report = attach_annotations(sources, [_annotations()], data_dir=data_dir)
    return data_dir, attached, report


def test_pairs_change_one_state_and_inherit_scene_group(tmp_path: Path):
    data_dir, sources, report = _prepared_source(tmp_path)
    assert report["present_boxes"] == 5
    plans = plan_pairs(
        sources,
        data_dir=data_dir,
        out_dir=data_dir / "aug" / "v1",
        output_size=(320, 180),
        pairs_per_image=2,
        seed=0,
    )
    assert len(plans) == 2
    assert {plan.edit_kind for plan in plans} == {"remove", "occlude"}

    for plan in plans:
        assert plan.bait_states == sources[0].item_states
        changed = [
            item
            for item in CATEGORY_IDS
            if plan.bait_states[item] != plan.true_states[item]
        ]
        assert changed == [plan.target_item]
        assert plan.source.scene_group_id == "s001_l001"
        assert plan.pair_group_id.startswith(plan.source.sample_prefix)
        assert plan.bait_image_path.startswith("aug/v1/images/s001_l001/")
        assert plan.true_image_path.startswith("aug/v1/images/s001_l001/")
        assert plan.bait_image_path != plan.true_image_path

    removal = next(plan for plan in plans if plan.edit_kind == "remove")
    assert removal.target_item == "multimeter"
    assert "cable_without_multimeter" in removal.hard_negative_tags
    occlusion = next(plan for plan in plans if plan.edit_kind == "occlude")
    assert occlusion.target_item == "screwdriver"
    assert occlusion.true_states["screwdriver"] == "unknown"
    assert occlusion.true_reasons["screwdriver"] == "合成遮挡"


def test_generated_samples_validate_and_stay_in_one_scene(tmp_path: Path):
    data_dir, sources, _ = _prepared_source(tmp_path)
    out_dir = data_dir / "aug" / "v1"
    plans = plan_pairs(
        sources,
        data_dir=data_dir,
        out_dir=out_dir,
        output_size=(320, 180),
        pairs_per_image=2,
        seed=7,
    )
    templates = load_question_templates(TEMPLATES)
    samples = build_samples(plans, templates=templates, seed=7)
    for plan in plans:
        render_pair_images(
            plan,
            data_dir=data_dir,
            output_size=(320, 180),
            seed=7,
        )

    assert len(samples) == 8
    assert validate_samples(samples, data_dir=data_dir) == []
    assert validate_pair_integrity(samples, source_images=sources) == []
    assert all(row["source"] == "synthetic_edit" for row in samples)
    assert all(row["annotation_verified"] is False for row in samples)
    assert all(row["scene_group_id"] == "s001_l001" for row in samples)
    assert all(row["session_id"] == "s001" for row in samples)
    assert all("_q01" in row["sample_id"] or "_q02" in row["sample_id"] for row in samples)
    assert not any(row["image_path"] == "real/source.png" for row in samples)
    assert not any(
        str(tag).startswith(("synthetic_", "removed_", "occluded_"))
        for row in samples
        for tag in row["hard_negative_tags"]
    )
    removal_true = [
        row
        for row in samples
        if "removed_multimeter" in row["condition_tags"]
    ]
    assert removal_true
    assert all(row["hard_negative_tags"] == ["cable_without_multimeter"] for row in removal_true)

    by_image: dict[str, list[dict]] = {}
    for row in samples:
        by_image.setdefault(row["image_path"], []).append(row)
    assert len(by_image) == 4
    assert all(len(rows) == 2 for rows in by_image.values())
    assert all(all(row["task"] == "checklist" for row in rows) for rows in by_image.values())

    for pair_id in {row["pair_group_id"] for row in samples}:
        image_paths = {
            row["image_path"] for row in samples if row["pair_group_id"] == pair_id
        }
        assert len(image_paths) == 2
        left_path, right_path = sorted(image_paths)
        left_rows = by_image[left_path]
        right_rows = by_image[right_path]
        assert [
            (row["required_items"], row["question_zh"]) for row in left_rows
        ] == [
            (row["required_items"], row["question_zh"]) for row in right_rows
        ]
        left = left_rows[0]["item_states"]
        right = right_rows[0]["item_states"]
        assert sum(left[item] != right[item] for item in CATEGORY_IDS) == 1

    bait_images = {
        row["image_path"]
        for row in samples
        if "bait_edit" in row["condition_tags"]
    }
    assert len(bait_images) == 2
    for image_path in bait_images:
        assert by_image[image_path][0]["item_states"] == sources[0].item_states


def test_rendering_is_byte_deterministic(tmp_path: Path):
    data_dir, sources, _ = _prepared_source(tmp_path)
    plan = plan_pairs(
        sources,
        data_dir=data_dir,
        out_dir=data_dir / "aug" / "v1",
        output_size=(320, 180),
        pairs_per_image=1,
        seed=11,
    )[0]
    bait_path, true_path = render_pair_images(
        plan,
        data_dir=data_dir,
        output_size=(320, 180),
        seed=11,
    )
    first = (sha256_file(bait_path), sha256_file(true_path))
    render_pair_images(
        plan,
        data_dir=data_dir,
        output_size=(320, 180),
        seed=11,
    )
    assert (sha256_file(bait_path), sha256_file(true_path)) == first
    assert np.asarray(Image.open(bait_path)).shape == (180, 320, 3)
    assert np.asarray(Image.open(true_path)).shape == (180, 320, 3)


def test_input_checks_reject_bad_labels_bounds_and_missing_present_box(tmp_path: Path):
    data_dir = tmp_path / "data"
    _draw_source_image(data_dir / "real" / "source.png")
    sources = prepare_source_images(_source_rows())

    with pytest.raises(AugmentError, match="present 但缺少框"):
        missing_box = _annotations()
        attach_annotations(
            sources,
            [
                AnnotationImage(
                    image=missing_box.image,
                    width=missing_box.width,
                    height=missing_box.height,
                    boxes=tuple(
                        box for box in missing_box.boxes if box.item != "hammer"
                    ),
                )
            ],
            data_dir=data_dir,
        )

    with pytest.raises(AugmentError, match="框越出全图"):
        out_of_bounds = _annotations()
        attach_annotations(
            sources,
            [
                AnnotationImage(
                    image=out_of_bounds.image,
                    width=out_of_bounds.width,
                    height=out_of_bounds.height,
                    boxes=tuple(
                        BoxAnnotation(box.item, (0, 0, 400, 40))
                        if box.item == "multimeter"
                        else box
                        for box in out_of_bounds.boxes
                    ),
                )
            ],
            data_dir=data_dir,
        )

    payload = {
        "images": [
            {
                "image": "source.png",
                "width": 320,
                "height": 180,
                "annotations": [{"class": "wrench", "box": [1, 2, 30, 40]}],
            }
        ]
    }
    annotation_path = tmp_path / "bad_annotations.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AugmentError, match="未知框标签"):
        load_annotation_images(annotation_path)


def test_source_requires_exactly_two_checklists(tmp_path: Path):
    rows = _source_rows()[:1]
    with pytest.raises(AugmentError, match="期望 2 条 checklist"):
        prepare_source_images(rows)


def test_hard_tags_only_preserve_applicable_source_and_relation(tmp_path: Path):
    data_dir, sources, _ = _prepared_source(
        tmp_path,
        hard_tags=["pliers_vs_metal_tool"],
    )
    plans = plan_pairs(
        sources,
        data_dir=data_dir,
        out_dir=data_dir / "aug" / "v1",
        output_size=(320, 180),
        pairs_per_image=2,
        seed=3,
    )
    samples = build_samples(
        plans,
        templates=load_question_templates(TEMPLATES),
        seed=3,
    )
    for plan in plans:
        bait = [
            row for row in samples if row["sample_id"].startswith(plan.bait_image_id)
        ]
        true = [
            row for row in samples if row["sample_id"].startswith(plan.true_image_id)
        ]
        assert bait and true
        assert all(row["hard_negative_tags"] == ["pliers_vs_metal_tool"] for row in bait)
        expected = ["pliers_vs_metal_tool"]
        if plan.edit_kind == "remove" and plan.target_item == "multimeter":
            expected.append("cable_without_multimeter")
        assert all(row["hard_negative_tags"] == expected for row in true)
        assert all(
            not any(
                str(tag).startswith(("synthetic_", "removed_", "occluded_"))
                for tag in row["hard_negative_tags"]
            )
            for row in (*bait, *true)
        )


def test_existing_images_directory_requires_ownership_marker(tmp_path: Path):
    data_dir = tmp_path / "data"
    out_dir = data_dir / "aug" / "v1"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True)
    user_file = images_dir / "keep-me.txt"
    user_file.write_text("user data", encoding="utf-8")

    with pytest.raises(AugmentError, match="缺少 .augment-v1.json"):
        _prepare_managed_images(
            out_dir,
            ["aug/v1/images/new.jpg"],
            data_dir=data_dir,
        )
    assert user_file.read_text(encoding="utf-8") == "user data"


def test_rebuild_keeps_existing_images_until_atomically_replaced(tmp_path: Path):
    data_dir = tmp_path / "data"
    out_dir = data_dir / "aug" / "v1"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True)
    stale = images_dir / "stale.jpg"
    stale.write_bytes(b"old")
    user_file = images_dir / "keep-me.txt"
    user_file.write_text("user data", encoding="utf-8")
    marker = images_dir / ".augment-v1.json"
    marker.write_text(
        json.dumps(
            {
                "version": "aug-v1",
                "generated_image_paths": ["aug/v1/images/stale.jpg"],
            }
        ),
        encoding="utf-8",
    )

    _prepare_managed_images(
        out_dir,
        ["aug/v1/images/stale.jpg", "aug/v1/images/new.jpg"],
        data_dir=data_dir,
    )
    assert stale.read_bytes() == b"old"
    assert user_file.read_text(encoding="utf-8") == "user data"
    updated = json.loads(marker.read_text(encoding="utf-8"))
    assert updated["generated_image_paths"] == [
        "aug/v1/images/new.jpg", "aug/v1/images/stale.jpg"
    ]
    with pytest.raises(AugmentError, match="拒绝覆盖未登记的图片"):
        _prepare_managed_images(
            out_dir,
            ["aug/v1/images/keep-me.txt"],
            data_dir=data_dir,
        )


def test_multimeter_red_footprint_crosses_protected_box_without_erasing_neighbor():
    image = np.full((100, 100, 3), (242, 242, 238), dtype=np.uint8)
    image[30:70, 30:55] = (210, 20, 20)
    image[25:75, 65:85] = (25, 25, 25)
    mask, _ = _removal_mask(
        (20, 20, 90, 80),
        [(0, 0, 85, 100)],
        frame_size=(100, 100),
        image_rgb=image,
        item="multimeter",
    )
    assert mask[40, 40] == 255
    assert mask[40, 50] == 255
    assert mask[25, 35] == 0
    assert mask[65, 75] == 0
