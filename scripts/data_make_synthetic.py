"""生成合成冒烟数据（简笔工作台，不是真实照片）。

固定种子，重复运行会得到相同的图片和 samples.jsonl。
manifest 里的 created_at 会随运行时间变化。

流程：画图 -> annotations.csv -> data_csv_to_jsonl.py -> data_split.py。
产物：
  data/smoke/synthetic/samples.jsonl
  data/smoke/synthetic/splits/v0/{train,val,test}.jsonl
  data/smoke/synthetic/splits/v0/manifest.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw, ImageFilter

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.data.convert import ANNOTATION_COLUMNS

WIDTH = 1280
HEIGHT = 960
QUESTION_SEED = 0
SPLIT_SEED = 0
ROOT = PROJECT_ROOT / "data" / "smoke" / "synthetic"
NOTE = "合成冒烟，不是真实照片。"

# 不同布置用不同垫板颜色。色块只代表批次/背景，不编码器材是否存在。
LOOK = {
    "s01_l001": ((214, 186, 140), (176, 52, 36), (92, 64, 40)),
    "s01_l002": ((186, 206, 214), (36, 78, 150), (48, 64, 78)),
    "s01_l003": ((196, 214, 176), (30, 120, 64), (48, 72, 44)),
    "s02_l001": ((214, 186, 186), (140, 40, 120), (90, 60, 70)),
    "s02_l002": ((176, 176, 204), (190, 110, 20), (60, 60, 90)),
    "s02_l003": ((220, 214, 176), (16, 110, 130), (80, 78, 48)),
    "s03_l001": ((128, 136, 156), (90, 48, 28), (40, 44, 58)),
    "s03_l002": ((156, 132, 116), (48, 48, 130), (58, 44, 36)),
    "s03_l003": ((120, 146, 128), (130, 140, 36), (36, 52, 40)),
}

# 每个布置的器材位置不同。感知哈希几乎不看颜色，只拉开位置才能避免跨布置被当成近重复。
# s03_l002 的万用表固定在右侧，供「出画」变体遮住。
SLOTS = {
    "s01_l001": {
        "screwdriver": (260, 300), "pliers": (560, 280), "cable_bundle": (880, 360),
        "multimeter": (1040, 660), "hammer": (340, 690),
    },
    "s01_l002": {
        "screwdriver": (980, 300), "pliers": (250, 430), "cable_bundle": (640, 660),
        "multimeter": (900, 720), "hammer": (420, 230),
    },
    "s01_l003": {
        "screwdriver": (300, 660), "pliers": (720, 250), "cable_bundle": (480, 430),
        "multimeter": (1020, 300), "hammer": (180, 280),
    },
    "s02_l001": {
        "screwdriver": (820, 690), "pliers": (280, 250), "cable_bundle": (1060, 260),
        "multimeter": (540, 300), "hammer": (360, 700),
    },
    "s02_l002": {
        "screwdriver": (220, 240), "pliers": (1040, 700), "cable_bundle": (320, 690),
        "multimeter": (760, 240), "hammer": (560, 500),
    },
    "s02_l003": {
        "screwdriver": (620, 250), "pliers": (220, 520), "cable_bundle": (960, 540),
        "multimeter": (400, 730), "hammer": (1080, 250),
    },
    "s03_l001": {
        "screwdriver": (360, 250), "pliers": (900, 690), "cable_bundle": (230, 650),
        "multimeter": (680, 380), "hammer": (1060, 280),
    },
    "s03_l002": {
        "screwdriver": (240, 280), "pliers": (500, 640), "cable_bundle": (700, 280),
        "multimeter": (1060, 430), "hammer": (250, 720),
    },
    "s03_l003": {
        "screwdriver": (1000, 690), "pliers": (300, 300), "cable_bundle": (820, 250),
        "multimeter": (520, 690), "hammer": (180, 540),
    },
}


def _states(**overrides: str) -> dict[str, str]:
    base = {cat: "present" for cat in CATEGORY_IDS}
    base.update(overrides)
    return base


def _spec(
    session: str,
    layout: str,
    variant: str,
    pair: str | None,
    states: dict[str, str],
    tags: list[str],
    *,
    hard: list[str] | None = None,
    reasons: dict[str, str] | None = None,
    notes: str,
    shift: tuple[int, int] = (0, 0),
    styles: dict[str, str] | None = None,
    probes: bool = False,
    wrench: bool = False,
) -> dict:
    return {
        "session_id": session,
        "layout_id": layout,
        "variant_id": variant,
        "pair": pair,
        "states": states,
        "tags": tags,
        "hard": hard or [],
        "reasons": reasons or {},
        "notes": f"{NOTE}{notes}",
        "shift": shift,
        "styles": styles or {},
        "probes": probes,
        "wrench": wrench,
    }


def build_specs() -> list[dict]:
    """24 张图、3 个 session、9 个布置、9 对成对样本。"""
    return [
        _spec("s01", "l001", "v01", "s01_l001_p01", _states(), ["lighting_warm", "background_wood", "baseline"], notes="齐全基准。"),
        _spec(
            "s01", "l001", "v02", "s01_l001_p01", _states(screwdriver="absent"),
            ["lighting_warm", "background_wood"], notes="相对同布置 v01 只拿走螺丝刀。",
        ),
        _spec(
            "s01", "l002", "v01", "s01_l002_p01", _states(cable_bundle="absent"),
            ["lighting_warm", "background_wood", "baseline"],
            hard=["pliers_vs_metal_tool"], wrench=True, notes="钳子在场，旁边有扳手状金属件。",
        ),
        _spec(
            "s01", "l002", "v02", "s01_l002_p01", _states(cable_bundle="absent", pliers="absent"),
            ["lighting_warm", "background_wood"],
            hard=["pliers_vs_metal_tool"], wrench=True, notes="相对 v01 只拿走钳子，扳手留在台上。",
        ),
        _spec(
            "s01", "l002", "v03", None, _states(cable_bundle="absent", hammer="unknown"),
            ["lighting_warm", "background_wood", "occlusion"],
            reasons={"hammer": "灰色块遮住锤头和锤柄，无法确认"},
            styles={"hammer": "occlusion"}, notes="锤子被灰色块挡住。",
        ),
        _spec(
            "s01", "l003", "v01", "s01_l003_p01", _states(cable_bundle="absent", hammer="absent"),
            ["lighting_warm", "background_wood", "baseline"],
            hard=["probe_leads_vs_cable"], probes=True, notes="没有成束线缆，万用表表笔线展开。",
        ),
        _spec(
            "s01", "l003", "v02", "s01_l003_p01",
            _states(cable_bundle="absent", hammer="absent", multimeter="absent"),
            ["lighting_warm", "background_wood"], notes="相对 v01 只拿走万用表，表笔一并拿走。",
        ),
        _spec("s02", "l001", "v01", "s02_l001_p01", _states(), ["lighting_cool", "background_gray", "baseline"], notes="齐全基准。"),
        _spec(
            "s02", "l001", "v02", "s02_l001_p01", _states(cable_bundle="absent"),
            ["lighting_cool", "background_gray"], notes="相对 v01 只拿走线缆。",
        ),
        _spec(
            "s02", "l001", "v03", None, _states(screwdriver="unknown"),
            ["lighting_cool", "background_gray", "occlusion"],
            reasons={"screwdriver": "灰色块遮住手柄和杆，无法确认"},
            styles={"screwdriver": "occlusion"}, notes="螺丝刀被灰色块挡住。",
        ),
        _spec(
            "s02", "l002", "v01", "s02_l002_p01", _states(pliers="absent", multimeter="absent"),
            ["lighting_cool", "background_gray", "baseline"],
            hard=["pliers_vs_metal_tool"], wrench=True, notes="没有钳子，台上有扳手状金属件。",
        ),
        _spec(
            "s02", "l002", "v02", "s02_l002_p01", _states(pliers="absent", multimeter="absent", hammer="absent"),
            ["lighting_cool", "background_gray"],
            hard=["pliers_vs_metal_tool"], wrench=True, notes="相对 v01 只拿走锤子，扳手留在台上。",
        ),
        _spec(
            "s02", "l003", "v01", "s02_l003_p01", _states(pliers="absent"),
            ["lighting_cool", "background_gray", "baseline"], notes="缺钳子，没有干扰扳手。",
        ),
        _spec(
            "s02", "l003", "v02", "s02_l003_p01", _states(pliers="absent", screwdriver="absent"),
            ["lighting_cool", "background_gray"], notes="相对 v01 只拿走螺丝刀。",
        ),
        _spec(
            "s02", "l003", "v03", None, _states(pliers="absent"),
            ["lighting_cool", "background_gray", "position_shift"],
            shift=(70, 46), notes="与 v01 的器材集合相同，只改变位置。",
        ),
        _spec("s03", "l001", "v01", "s03_l001_p01", _states(), ["lighting_dim", "background_dark", "baseline"], notes="齐全基准。"),
        _spec(
            "s03", "l001", "v02", "s03_l001_p01", _states(pliers="absent"),
            ["lighting_dim", "background_dark"], notes="相对 v01 只拿走钳子。",
        ),
        _spec(
            "s03", "l001", "v03", None, _states(cable_bundle="unknown"),
            ["lighting_dim", "background_dark", "defocus"],
            reasons={"cable_bundle": "线缆区域严重失焦，无法确认是不是一束线"},
            styles={"cable_bundle": "blur"}, notes="线缆位置失焦。",
        ),
        _spec(
            "s03", "l002", "v01", "s03_l002_p01", _states(cable_bundle="absent"),
            ["lighting_mixed", "background_dark", "baseline"],
            hard=["probe_leads_vs_cable"], probes=True, notes="表笔线展开，没有独立线缆。",
        ),
        _spec(
            "s03", "l002", "v02", "s03_l002_p01", _states(),
            ["lighting_mixed", "background_dark"],
            hard=["probe_leads_vs_cable"], probes=True, notes="相对 v01 放入一束线缆，表笔线仍在。",
        ),
        _spec(
            "s03", "l002", "v03", None, _states(multimeter="unknown"),
            ["lighting_mixed", "background_dark", "partial_frame"],
            reasons={"multimeter": "检查区域右侧出画，万用表位置不可见"},
            styles={"multimeter": "partial"}, notes="右侧出画，万用表无法确认。",
        ),
        _spec(
            "s03", "l003", "v01", "s03_l003_p01", _states(screwdriver="absent", cable_bundle="absent"),
            ["lighting_dim", "background_dark", "baseline"], notes="基准：缺螺丝刀和线缆。",
        ),
        _spec(
            "s03", "l003", "v02", "s03_l003_p01",
            _states(screwdriver="absent", cable_bundle="absent", hammer="absent"),
            ["lighting_dim", "background_dark"], notes="相对 v01 只拿走锤子。",
        ),
        _spec(
            "s03", "l003", "v03", None,
            _states(screwdriver="absent", cable_bundle="absent", pliers="unknown"),
            ["lighting_dim", "background_dark", "occlusion"],
            reasons={"pliers": "灰色块遮住钳口和手柄，无法确认"},
            styles={"pliers": "occlusion"}, notes="钳子被灰色块挡住。",
        ),
    ]


def check_specs(specs: list[dict]) -> None:
    if len(specs) < 24:
        raise RuntimeError(f"合成图应至少 24 张，实际 {len(specs)}")
    paths = [image_path(spec) for spec in specs]
    if len(paths) != len(set(paths)):
        raise RuntimeError("image_path 重复")
    groups: dict[str, list[dict]] = {}
    sessions = set()
    scenes = set()
    for spec in specs:
        sessions.add(spec["session_id"])
        scenes.add(f"{spec['session_id']}_{spec['layout_id']}")
        unknown = {cat for cat, state in spec["states"].items() if state == "unknown"}
        if set(spec["reasons"]) != unknown:
            raise RuntimeError(f"{image_path(spec)} 的 unknown_reason 与状态不一致")
        for cat, style in spec["styles"].items():
            if spec["states"][cat] != "unknown" or style not in {"occlusion", "partial", "blur"}:
                raise RuntimeError(f"{image_path(spec)} 的遮挡样式无效: {cat}={style}")
        if spec["pair"]:
            groups.setdefault(spec["pair"], []).append(spec)
    if len(sessions) < 3 or len(scenes) < 6 or len(groups) < 4:
        raise RuntimeError(f"session={len(sessions)} scene={len(scenes)} pair={len(groups)}，低于要求")
    for pair_id, items in groups.items():
        if len(items) != 2:
            raise RuntimeError(f"{pair_id} 有 {len(items)} 张图")
        diff = [cat for cat in CATEGORY_IDS if items[0]["states"][cat] != items[1]["states"][cat]]
        if len(diff) != 1:
            raise RuntimeError(f"{pair_id} 有 {len(diff)} 个类别不同: {diff}")
        if items[0]["shift"] != items[1]["shift"]:
            raise RuntimeError(f"{pair_id} 的两张图位置偏移不一致")


def image_path(spec: dict) -> str:
    return f"smoke/synthetic/images/{spec['session_id']}/{spec['layout_id']}/{spec['variant_id']}.png"


def _layout_rng(session: str, layout: str) -> Random:
    digest = hashlib.sha256(f"grain|{session}|{layout}".encode()).digest()
    return Random(int.from_bytes(digest[:8], "big"))


def _draw_screwdriver(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rectangle([x - 7, y - 70, x + 7, y + 55], fill=(188, 192, 198))
    draw.polygon([(x - 7, y + 55), (x + 7, y + 55), (x, y + 82)], fill=(160, 164, 170))
    draw.rectangle([x - 18, y - 118, x + 18, y - 62], fill=(214, 122, 36))


def _draw_pliers(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    metal = (72, 80, 88)
    draw.line([(x - 55, y + 45), (x + 8, y - 20), (x + 58, y - 72)], fill=metal, width=12)
    draw.line([(x - 55, y - 45), (x + 8, y - 8), (x + 62, y + 48)], fill=metal, width=12)
    draw.ellipse([x - 10, y - 22, x + 16, y + 4], fill=(36, 40, 46))


def _draw_cable(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    for rx, ry in ((62, 38), (44, 28), (26, 16)):
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], outline=(24, 36, 96), width=9)
    draw.rectangle([x - 9, y - 24, x + 9, y + 24], fill=(196, 164, 48))


def _draw_multimeter(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rectangle([x - 58, y - 74, x + 58, y + 74], fill=(32, 68, 56))
    draw.rectangle([x - 40, y - 56, x + 40, y - 16], fill=(176, 214, 176))
    draw.ellipse([x - 30, y + 2, x - 8, y + 24], fill=(70, 74, 78))
    draw.ellipse([x + 8, y + 2, x + 30, y + 24], fill=(70, 74, 78))
    draw.rectangle([x - 22, y + 40, x - 8, y + 58], fill=(190, 36, 36))
    draw.rectangle([x + 8, y + 40, x + 22, y + 58], fill=(24, 24, 24))


def _draw_hammer(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rectangle([x - 9, y - 10, x + 9, y + 96], fill=(148, 96, 52))
    draw.rectangle([x - 62, y - 48, x + 58, y - 12], fill=(86, 90, 98))
    draw.polygon([(x - 62, y - 48), (x - 36, y - 48), (x - 48, y - 12)], fill=(68, 72, 78))


def _draw_wrench(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    color = (168, 172, 178)
    draw.line([(x, y), (x + 78, y + 70)], fill=color, width=10)
    draw.ellipse([x - 18, y - 18, x + 18, y + 18], outline=color, width=8)
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color)


def _draw_probes(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    red = (190, 32, 32)
    black = (20, 20, 20)
    draw.line([(x - 8, y + 58), (x - 70, y + 130), (x - 150, y + 210), (x - 190, y + 280)], fill=red, width=3)
    draw.line([(x + 16, y + 58), (x + 90, y + 140), (x + 40, y + 220), (x + 110, y + 300)], fill=black, width=3)
    draw.ellipse([x - 198, y + 272, x - 182, y + 288], fill=red)
    draw.ellipse([x + 102, y + 292, x + 118, y + 308], fill=black)


def _pos(spec: dict, cat: str) -> tuple[int, int]:
    dx, dy = spec["shift"]
    x, y = SLOTS[f"{spec['session_id']}_{spec['layout_id']}"][cat]
    return x + dx, y + dy


def _mark_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.3 * bg[0] + 0.59 * bg[1] + 0.11 * bg[2]
    if luminance > 150:
        return (42, 44, 48)
    return (232, 228, 214)


def _draw_layout_mark(draw: ImageDraw.ImageDraw, key: str, color: tuple[int, int, int]) -> None:
    """大块垫板图案。同一布置的各变体共用，用来拉开不同布置的低频结构。"""
    marks = {
        "s01_l001": lambda: draw.ellipse([20, 20, 520, 520], fill=color),
        "s01_l002": lambda: draw.ellipse([760, 440, 1260, 940], fill=color),
        "s01_l003": lambda: draw.rectangle([0, 0, WIDTH, 280], fill=color),
        "s02_l001": lambda: draw.rectangle([0, 0, 300, HEIGHT], fill=color),
        "s02_l002": lambda: draw.rectangle([980, 0, WIDTH, HEIGHT], fill=color),
        "s02_l003": lambda: draw.ellipse([20, 460, 540, 960], fill=color),
        "s03_l001": lambda: draw.polygon([(0, 0), (760, 0), (0, 760)], fill=color),
        "s03_l002": lambda: draw.rectangle([0, 700, WIDTH, HEIGHT], fill=color),
        "s03_l003": lambda: draw.ellipse([760, 20, 1260, 520], fill=color),
    }
    marks[key]()


def render(spec: dict) -> Image.Image:
    key = f"{spec['session_id']}_{spec['layout_id']}"
    bg, stripe, frame = LOOK[key]
    image = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(image)
    _draw_layout_mark(draw, key, _mark_color(bg))
    draw.rectangle([0, 0, WIDTH - 1, HEIGHT - 1], outline=frame, width=26)
    draw.rectangle([78, 68, 1202, 892], outline=(54, 42, 32), width=4)
    draw.rectangle([78, 68, 1202, 132], fill=stripe)
    rng = _layout_rng(spec["session_id"], spec["layout_id"])
    for _ in range(48):
        x = rng.randint(100, 1160)
        y = rng.randint(150, 860)
        delta = rng.randint(-22, 22)
        color = tuple(max(0, min(255, channel + delta)) for channel in bg)
        draw.line([(x, y), (x + rng.randint(24, 90), y + rng.randint(-3, 3))], fill=color, width=2)

    drawers = {
        "screwdriver": _draw_screwdriver,
        "pliers": _draw_pliers,
        "cable_bundle": _draw_cable,
        "multimeter": _draw_multimeter,
        "hammer": _draw_hammer,
    }
    for cat in CATEGORY_IDS:
        style = spec["styles"].get(cat)
        if spec["states"][cat] == "present" or style == "blur":
            drawers[cat](draw, *_pos(spec, cat))
    if spec["probes"] and spec["states"]["multimeter"] == "present":
        _draw_probes(draw, *_pos(spec, "multimeter"))
    if spec["wrench"]:
        x, y = _pos(spec, "pliers")
        _draw_wrench(draw, x + 90, y + 80)

    for cat, style in spec["styles"].items():
        x, y = _pos(spec, cat)
        if style == "occlusion":
            draw.rectangle([x - 78, y - 96, x + 78, y + 96], fill=(148, 148, 148), outline=(96, 96, 96), width=3)
        elif style == "partial":
            draw.rectangle([930, 150, 1236, 900], fill=(28, 30, 34))
        elif style == "blur":
            box = (x - 100, y - 90, x + 100, y + 90)
            crop = image.crop(box).filter(ImageFilter.GaussianBlur(radius=18))
            image.paste(crop, box)
    return image


def write_images(specs: list[dict]) -> None:
    img_root = ROOT / "images"
    if img_root.exists():
        shutil.rmtree(img_root)
    for spec in specs:
        path = PROJECT_ROOT / "data" / image_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        render(spec).save(path, format="PNG")


def write_csv(specs: list[dict]) -> Path:
    path = ROOT / "annotations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "session_id": spec["session_id"],
                    "layout_id": spec["layout_id"],
                    "variant_id": spec["variant_id"],
                    "image_path": image_path(spec),
                    "pair_group_id": spec["pair"] or "",
                    "screwdriver": spec["states"]["screwdriver"],
                    "pliers": spec["states"]["pliers"],
                    "cable_bundle": spec["states"]["cable_bundle"],
                    "multimeter": spec["states"]["multimeter"],
                    "hammer": spec["states"]["hammer"],
                    "unknown_reason": json.dumps(spec["reasons"], ensure_ascii=False, sort_keys=True),
                    "condition_tags": "|".join(spec["tags"]),
                    "hard_negative_tags": "|".join(spec["hard"]),
                    "annotation_verified": "false",
                    "notes": spec["notes"],
                }
            )
    return path


def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args]
    print("运行:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> int:
    specs = build_specs()
    check_specs(specs)
    write_images(specs)
    csv_path = write_csv(specs)
    jsonl_path = ROOT / "samples.jsonl"
    split_dir = ROOT / "splits" / "v0"
    _run(
        "data_csv_to_jsonl.py",
        "--csv", str(csv_path),
        "--out", str(jsonl_path),
        "--source", "synthetic_smoke",
        "--question-set", "default",
        "--checklists-per-image", "2",
        "--presence-per-image", "1",
        "--seed", str(QUESTION_SEED),
    )
    _run(
        "data_split.py",
        "--input", str(jsonl_path),
        "--out", str(split_dir),
        "--seed", str(SPLIT_SEED),
        "--fail-on-leak",
    )
    _run(
        "data_validate.py",
        str(jsonl_path),
        str(split_dir / "train.jsonl"),
        str(split_dir / "val.jsonl"),
        str(split_dir / "test.jsonl"),
    )
    _run(
        "data_stats.py",
        "--split-dir", str(split_dir),
        "--title", "合成冒烟数据统计（不是真实实验结果）",
        "--out", str(ROOT / "stats.md"),
    )
    print(f"合成冒烟数据已写好：{jsonl_path}")
    print(f"训练划分：{split_dir / 'train.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
