"""把标注 CSV 展开成样本 JSONL。

每张图至少一条 checklist。presence 样本默认不生成，需要时用 --presence-per-image。
测试专用问法只有 --question-set test_only 才会用到。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.data.convert import ConversionError, csv_to_samples, load_question_templates
from qwen_vlm.data.io import write_jsonl

DEFAULT_TEMPLATES = PROJECT_ROOT / "configs" / "data" / "question_templates.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="标注 CSV 展开为样本 JSONL")
    parser.add_argument("--csv", type=Path, required=True, help="一图一行的标注 CSV")
    parser.add_argument("--out", type=Path, required=True, help="输出 JSONL")
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES, help="问法与清单组合池")
    parser.add_argument("--question-set", default=None, help="default 或 test_only；默认读配置文件")
    parser.add_argument("--checklists-per-image", type=int, default=None, help="每张图的 checklist 条数，至少 1")
    parser.add_argument("--presence-per-image", type=int, default=None, help="每张图额外的 presence 条数，0 表示不生成")
    parser.add_argument("--seed", type=int, default=None, help="抽问法和清单的种子")
    parser.add_argument("--source", required=True, help="real_lab / synthetic_smoke / synthetic_edit / teacher_generated")
    args = parser.parse_args(argv)

    templates = load_question_templates(args.templates)
    try:
        samples = csv_to_samples(
            args.csv,
            templates,
            question_set=args.question_set,
            checklists_per_image=args.checklists_per_image,
            presence_per_image=args.presence_per_image,
            seed=args.seed,
            source=args.source,
        )
    except ConversionError as exc:
        print(exc, file=sys.stderr)
        return 1
    write_jsonl(args.out, samples)
    n_check = sum(row["task"] == "checklist" for row in samples)
    n_presence = sum(row["task"] == "presence" for row in samples)
    n_images = len({row["image_path"] for row in samples})
    print(f"写出 {args.out}：{len(samples)} 条样本（checklist {n_check}，presence {n_presence}），{n_images} 张图片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
