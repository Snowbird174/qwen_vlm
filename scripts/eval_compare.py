#!/usr/bin/env python3
"""读取多次 eval 的 metrics.json，生成方法对比表和量化×像素预算矩阵。

表格里的数字只来自这些 JSON。缺少的 B0/B1/M/A1-A4 或优化组合写“待测”。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_vlm.evaluation.report import load_metrics_files, write_comparison


def _collect(metrics: list[str], runs_root: str | None) -> list[Path]:
    paths = [Path(item) for item in metrics]
    if runs_root is not None:
        root = Path(runs_root)
        paths.extend(sorted(root.glob("*/eval/metrics.json")))
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            raise SystemExit(f"找不到 metrics.json: {path}")
        unique.append(path)
    if not unique:
        raise SystemExit("没有传入任何 metrics.json。用法：--metrics a.json b.json 或 --runs-root runs")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="对比多次评测的 metrics.json，生成 Markdown、CSV 和柱状图。")
    parser.add_argument("--metrics", nargs="*", default=[], help="一个或多个 metrics.json")
    parser.add_argument("--runs-root", default=None, help="扫描 <root>/*/eval/metrics.json")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    paths = _collect(args.metrics, args.runs_root)
    runs = load_metrics_files(paths)
    out = Path(args.out)
    write_comparison(runs, out, make_plot=not args.no_plot)
    print(f"写出 {out}  （读取 {len(runs)} 个 metrics.json）")


if __name__ == "__main__":
    main()
