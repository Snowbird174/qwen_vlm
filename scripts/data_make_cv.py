"""生成任务 C 的固定 scene_group 三折交叉验证目录。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen_vlm.data.cv import build_fixed_cv3_folds, write_fixed_cv3_folds
from qwen_vlm.data.io import load_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成固定三折真实测试 + 合成训练 CV 划分")
    parser.add_argument(
        "--real",
        type=Path,
        default=Path("data/annotations/real_v1.jsonl"),
        help="真实样本 JSONL，默认 data/annotations/real_v1.jsonl",
    )
    parser.add_argument(
        "--augment",
        type=Path,
        default=Path("data/aug/v1/samples.jsonl"),
        help="synthetic_edit 样本 JSONL，默认 data/aug/v1/samples.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/splits/cv3"),
        help="输出根目录，默认 data/splits/cv3",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="image_path 的根目录，默认 data",
    )
    args = parser.parse_args(argv)

    for label, path in (("真实", args.real), ("合成", args.augment)):
        if not path.is_file():
            print(f"错误：{label}样本不存在 {path}", file=sys.stderr)
            return 2
    if not args.data_dir.is_dir():
        print(f"错误：data-dir 不存在 {args.data_dir}", file=sys.stderr)
        return 2

    try:
        real = load_jsonl(args.real)
        synthetic = load_jsonl(args.augment)
        print(
            f"输入：真实 {len(real)} 条 / 合成 {len(synthetic)} 条；"
            f"固定折测试组见 docs/NEXT_STEPS.md 任务 C"
        )
        folds = build_fixed_cv3_folds(real, synthetic)
        manifests = write_fixed_cv3_folds(
            folds,
            args.out,
            real_jsonl=args.real,
            augment_jsonl=args.augment,
            data_dir=args.data_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    for fold, manifest in sorted(manifests.items()):
        train = manifest["files"]["train"]
        train_real = manifest["files"]["train_real"]
        test = manifest["files"]["test"]
        print(
            f"fold{fold}: test_scene_groups={manifest['test_scene_groups']}；"
            f"train={train['num_samples']}（real={train['num_real_samples']}, "
            f"synthetic={train['num_synthetic_edit_samples']}）"
            f" train_real={train_real['num_samples']} test={test['num_samples']}"
        )
        print(f"  manifest: {args.out / f'fold{fold}' / 'manifest.json'}")
    coverage = next(iter(manifests.values()))["coverage"]
    print(
        "真实测试覆盖："
        f"{coverage['test_sample_appearances']}/{coverage['real_samples']} 条样本，"
        f"{coverage['test_image_appearances']}/{coverage['real_images']} 张图片，均恰好一次"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
