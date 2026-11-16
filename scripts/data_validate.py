"""校验一个或多个样本 JSONL。有错误时退出码为 1。"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_vlm.data.io import load_jsonl
from qwen_vlm.data.validate import validate_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验样本 JSONL 的 schema、图片和成对约束")
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=None, help="image_path 的根目录，默认是 data/")
    parser.add_argument("--no-images", action="store_true", help="不检查图片文件")
    args = parser.parse_args(argv)

    failed = False
    for path in args.jsonl:
        samples = load_jsonl(path)
        errors = validate_samples(samples, data_dir=args.data_dir, check_images=not args.no_images)
        n_images = len({row.get("image_path") for row in samples if isinstance(row, dict)})
        print(f"校验 {path}：{len(samples)} 条样本，{n_images} 张图片，{len(errors)} 个错误")
        for msg in errors:
            print(f"  {msg}")
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
