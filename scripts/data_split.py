"""按 scene_group_id 划分 train/val/test，并写出 manifest。

成对样本和同一布置的全部变体会进同一个 split。
指定 --holdout-session 时，该 session 整体写入 test_ood.jsonl。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_vlm.data.io import load_jsonl
from qwen_vlm.data.leakage import check_leakage, format_leakage_report
from qwen_vlm.data.split import assign_splits, write_split_files
from qwen_vlm.data.validate import validate_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按场景组划分数据集并写 manifest")
    parser.add_argument("--input", type=Path, required=True, help="样本 JSONL")
    parser.add_argument("--out", type=Path, required=True, help="输出目录，例如 data/splits/v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--holdout-session", action="append", default=[], help="整个 session 留作 test_ood，可重复")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--phash-threshold", type=int, default=8, help="跨 split 感知哈希的汉明距离阈值")
    parser.add_argument("--fail-on-leak", action="store_true", help="泄漏检查未通过时返回退出码 1")
    args = parser.parse_args(argv)

    samples = load_jsonl(args.input)
    errors = validate_samples(samples, data_dir=args.data_dir, check_images=True)
    print(f"划分前校验 {args.input}：{len(samples)} 条样本，{len(errors)} 个错误")
    if errors:
        for msg in errors:
            print(f"  {msg}")
        return 1

    buckets, warnings = assign_splits(
        samples,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        holdout_sessions=args.holdout_session,
    )
    for msg in warnings:
        print(f"警告: {msg}")
    if not args.holdout_session:
        stale = args.out / "test_ood.jsonl"
        if stale.exists():
            stale.unlink()
    params = {
        "train_ratio": args.train,
        "val_ratio": args.val,
        "test_ratio": args.test,
        "holdout_sessions": list(args.holdout_session),
        "split_unit": "scene_group_id；共用 pair_group_id 的场景组合并",
        "phash_threshold": args.phash_threshold,
    }
    manifest = write_split_files(
        buckets,
        args.out,
        source_jsonl=args.input,
        seed=args.seed,
        params=params,
        warnings=warnings,
    )
    for name, info in manifest["files"].items():
        print(
            f"  {name}: {info['num_samples']} 条样本，{info['num_scene_groups']} 个场景组，"
            f"sha256={info['sha256'][:12]}"
        )
    print(f"manifest: {args.out / 'manifest.json'}，来源 sha256={manifest['source_sha256'][:12]}，种子 {manifest['seed']}")

    report = check_leakage(buckets, data_dir=args.data_dir, phash_threshold=args.phash_threshold)
    print(format_leakage_report(report))
    if args.fail_on_leak and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
