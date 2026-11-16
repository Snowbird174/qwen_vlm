"""统计类别 × 状态、条件标签、困难负样本和 split 分布，输出 Markdown。"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_vlm.data.io import load_jsonl
from qwen_vlm.data.stats import compute_stats, render_stats_markdown

SPLIT_FILES = ("train", "val", "test", "test_ood")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统计样本分布并输出 Markdown")
    parser.add_argument("jsonl", nargs="*", type=Path, help="一份或多份 JSONL，按文件名当作 split")
    parser.add_argument("--split-dir", type=Path, default=None, help="含 train/val/test.jsonl 的目录")
    parser.add_argument("--title", default="数据统计")
    parser.add_argument("--out", type=Path, default=None, help="写入 Markdown；默认只打印")
    args = parser.parse_args(argv)

    if args.split_dir is not None:
        splits = {}
        for name in SPLIT_FILES:
            path = args.split_dir / f"{name}.jsonl"
            if path.is_file():
                splits[name] = load_jsonl(path)
        if not splits:
            parser.error(f"{args.split_dir} 下没有 train/val/test.jsonl")
    elif args.jsonl:
        splits = {path.stem: load_jsonl(path) for path in args.jsonl}
    else:
        parser.error("请传入 JSONL 或 --split-dir")

    text = render_stats_markdown(compute_stats(splits), title=args.title)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"写出 {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
