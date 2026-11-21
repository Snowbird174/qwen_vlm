"""Generate deterministic inference-only robustness variants from real photos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen_vlm import DATA_DIR
from qwen_vlm.data.robust import (
    CONDITION_NAMES,
    JPEG_QUALITY,
    generate_robust_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成鲁棒性实验图片（仅 CPU、仅推理评测，不参与训练）"
    )
    parser.add_argument(
        "--real-jsonl",
        type=Path,
        default=DATA_DIR / "annotations" / "real_v1.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "robust" / "v1",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--contact-cell-width",
        type=int,
        default=240,
        help="全量复核 contact sheet 的单格宽度",
    )
    parser.add_argument(
        "--contact-cell-height",
        type=int,
        default=135,
        help="全量复核 contact sheet 的单格高度",
    )
    args = parser.parse_args(argv)

    if args.contact_cell_width <= 0 or args.contact_cell_height <= 0:
        parser.error("contact sheet 单元格尺寸必须为正")

    try:
        report = generate_robust_dataset(
            real_jsonl=args.real_jsonl,
            out_dir=args.out,
            data_dir=args.data_dir,
            seed=args.seed,
            contact_sheet_cell_size=(
                args.contact_cell_width,
                args.contact_cell_height,
            ),
        )
    except ValueError as exc:
        print(f"生成失败: {exc}", file=sys.stderr)
        return 1

    print(
        f"生成完成: {report.source_images} 张图 / "
        f"{report.source_samples} 条真实样本 -> "
        f"{report.generated_images} 张 JPEG / "
        f"{report.generated_samples} 条评测样本"
    )
    print("  条件: " + ", ".join(CONDITION_NAMES))
    print("  条件样本数: " + ", ".join(
        f"{name}={count}" for name, count in report.condition_counts.items()
    ))
    print(f"  JPEG: q={JPEG_QUALITY}，保持原分辨率")
    print("  用途: evaluation_only；禁止传入训练或交叉验证脚本")
    print(f"  全量复核: {report.contact_sheet_path}")
    print(f"  清单: {report.manifest_path}")
    for condition in CONDITION_NAMES:
        print(f"  {condition}/samples.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
