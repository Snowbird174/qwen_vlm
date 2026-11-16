"""数据 agent 的合成划分还没到时，用来做训练冒烟的最小样本。

六张图都标成 synthetic_smoke。图片和 JSONL 写在 work/train_fixture/，
image_path 相对该目录，不写到 data/。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.common.schema import validate_sample

FIXTURE_ROOT = PROJECT_ROOT / "work" / "train_fixture"


def _states(**overrides: str) -> dict[str, str]:
    states = {category: "absent" for category in CATEGORY_IDS}
    states.update(overrides)
    return states


def _sample(
    sample_id: str,
    image_name: str,
    scene: str,
    states: dict[str, str],
    question: str,
    required: list[str],
    *,
    pair: str | None = None,
    task: str = "checklist",
    target_item: str | None = None,
    hard: list[str] | None = None,
    session: str = "fixture",
) -> dict:
    sample = {
        "sample_id": sample_id,
        "image_path": f"images/{image_name}",
        "session_id": session,
        "scene_group_id": scene,
        "pair_group_id": pair,
        "task": task,
        "required_items": required,
        "question_zh": question,
        "item_states": states,
        "condition_tags": ["fixture"],
        "hard_negative_tags": list(hard or []),
        "unknown_reason": {category: "合成图里该区域被挡住" for category, state in states.items() if state == "unknown"},
        "source": "synthetic_smoke",
        "annotation_verified": True,
    }
    if target_item is not None:
        sample["target_item"] = target_item
    errors = validate_sample(sample)
    if errors:
        raise RuntimeError(f"fixture 样本 {sample_id} 不合法: {errors}")
    return sample


def _draw(path: Path, color: tuple[int, int, int], boxes: list[tuple[int, int, int, int, tuple[int, int, int]]]) -> None:
    image = Image.new("RGB", (640, 480), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([16, 16, 624, 464], outline=(40, 40, 40), width=4)
    for box in boxes:
        draw.rectangle(box[:4], fill=box[4])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_fixture(root: Path = FIXTURE_ROOT) -> Path:
    images = root / "images"
    splits = root / "splits" / "v0"
    splits.mkdir(parents=True, exist_ok=True)
    drawings = {
        "a.png": ((210, 180, 140), [(40, 200, 280, 250, (180, 40, 40)), (360, 180, 520, 300, (80, 80, 80))]),
        "b.png": ((210, 180, 140), [(40, 200, 280, 250, (180, 40, 40))]),
        "c.png": ((180, 190, 170), [(40, 60, 160, 180, (200, 30, 30)), (200, 60, 320, 200, (30, 30, 30)), (360, 60, 560, 220, (240, 200, 40)), (80, 260, 300, 420, (40, 40, 160)), (360, 280, 560, 400, (120, 80, 40))]),
        "d.png": ((120, 120, 130), [(40, 80, 200, 200, (190, 30, 30)), (240, 80, 400, 220, (20, 20, 20)), (80, 280, 360, 430, (30, 70, 160)), (430, 120, 600, 360, (90, 90, 90))]),
        "e.png": ((200, 170, 130), [(180, 160, 460, 300, (30, 90, 40))]),
        "f.png": ((200, 170, 130), []),
    }
    for name, (color, boxes) in drawings.items():
        _draw(images / name, color, boxes)

    present = _states(screwdriver="present", pliers="present", cable_bundle="present", multimeter="present", hammer="present")
    train = [
        _sample("fix_a", "a.png", "scene_a", _states(screwdriver="present", hammer="present"), "图里有哪些器材？", ["screwdriver", "hammer"], pair="pair_hammer"),
        _sample("fix_b", "b.png", "scene_b", _states(screwdriver="present", hammer="absent"), "锤子还在实验台上吗？", ["screwdriver", "hammer"], pair="pair_hammer"),
        _sample("fix_c", "c.png", "scene_c", present, "所需器材是否都在？", ["screwdriver", "pliers", "multimeter"], hard=["similar_tool"]),
        _sample("fix_d", "d.png", "scene_d", _states(screwdriver="present", pliers="present", cable_bundle="present", multimeter="unknown", hammer="absent"), "万用表能不能看清？", ["multimeter", "pliers"]),
        _sample("fix_e", "e.png", "scene_e", _states(pliers="present"), "钳子在不在？", ["pliers"], pair="pair_pliers"),
        _sample("fix_f", "f.png", "scene_f", _states(), "这张台上有钳子吗？", ["pliers"], pair="pair_pliers"),
        _sample("fix_a_presence", "a.png", "scene_a", _states(screwdriver="present", hammer="present"), "画面里有没有螺丝刀？", ["screwdriver", "hammer"], task="presence", target_item="screwdriver", pair="pair_hammer"),
    ]
    val = [
        _sample("fix_val_c", "c.png", "scene_c", present, "请核对这次实验要的器材。", ["hammer", "cable_bundle"], hard=["similar_tool"]),
        _sample("fix_val_d", "d.png", "scene_d", _states(screwdriver="present", pliers="present", cable_bundle="present", multimeter="unknown", hammer="absent"), "有没有看不清的器材？", ["screwdriver", "multimeter"]),
    ]
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (splits / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "name": "train_fixture",
        "source": "synthetic_smoke",
        "note": "训练 agent 在等待合成划分超时后自建的最小冒烟集，不是正式数据。",
        "train_jsonl": "splits/v0/train.jsonl",
        "val_jsonl": "splits/v0/val.jsonl",
        "n_images": len(drawings),
        "n_train": len(train),
        "n_val": len(val),
    }
    (splits / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return splits
