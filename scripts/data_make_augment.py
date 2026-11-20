"""Generate deterministic box-guided bait/true pairs from the real lab photos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen_vlm import DATA_DIR, PROJECT_ROOT
from qwen_vlm.data.augment import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PAIRS_PER_IMAGE,
    generate_augmented_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用真实框生成成对训练数据（仅 CPU）")
    parser.add_argument(
        "--real-jsonl",
        type=Path,
        default=DATA_DIR / "annotations" / "real_v1.jsonl",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DATA_DIR / "annotations" / "annotations.json",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "question_templates.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "aug" / "v1",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs-per-image", type=int, default=DEFAULT_PAIRS_PER_IMAGE)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--contact-sheet-pairs", type=int, default=10)
    args = parser.parse_args(argv)

    if not 1 <= args.pairs_per_image <= 2:
        parser.error("--pairs-per-image 只能是 1 或 2")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality 必须在 1..100")

    try:
        report = generate_augmented_dataset(
            real_jsonl=args.real_jsonl,
            annotations_json=args.annotations,
            templates_path=args.templates,
            out_dir=args.out,
            data_dir=args.data_dir,
            output_size=DEFAULT_OUTPUT_SIZE,
            pairs_per_image=args.pairs_per_image,
            seed=args.seed,
            jpeg_quality=args.jpeg_quality,
            contact_sheet_pairs=args.contact_sheet_pairs,
        )
    except ValueError as exc:
        print(f"生成失败: {exc}", file=sys.stderr)
        return 1

    print(
        f"生成完成: {report.source_images} 张原图 -> "
        f"{report.pairs} 对 / {report.synthetic_images} 张编辑图 / "
        f"{report.samples} 条 checklist"
    )
    print(
        f"  pair 类型: {report.pair_types}，"
        f"cable_without_multimeter={report.hard_negative_pairs}"
    )
    print(f"  框检查: {report.annotation_checks}")
    print(f"  样本: {args.out / 'samples.jsonl'}")
    print(f"  抽查: {args.out / 'contact_sheet.jpg'}")
    print(f"  清单: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
