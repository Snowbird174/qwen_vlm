#!/usr/bin/env python3
"""评测一份金标准和一份逐样本预测，写出 metrics.json、per_sample.jsonl 和 metrics.md。

示例：
  python scripts/eval_predictions.py \
    --gold data/dev.jsonl --pred runs/B0/predictions.jsonl --out runs/B0/eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_vlm.evaluation.bootstrap import DEFAULT_CONFIDENCE, DEFAULT_N_ITERATIONS, DEFAULT_SEED
from qwen_vlm.evaluation.report import load_eval_config, run_eval


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测逐样本预测，写出指标、逐样本结果和可读表格。")
    parser.add_argument("--gold", required=True, help="金标准 JSONL，字段见 contracts/sample.schema.json")
    parser.add_argument("--pred", required=True, help="逐样本预测 JSONL，每行含 sample_id 和 PredictResult")
    parser.add_argument("--out", required=True, help="输出目录，例如 runs/<experiment_id>/eval")
    parser.add_argument("--experiment-id", default=None, help="实验名。缺省时用输出目录的上一级或目录名")
    parser.add_argument("--quantization", default=None, choices=["F16", "Q8_0", "Q4_K_M"], help="量化设置，用于优化矩阵")
    parser.add_argument("--pixel-budget", type=int, default=None, choices=[384, 512, 768], help="输入像素预算的边长")
    parser.add_argument("--bootstrap-iters", type=int, default=None, help=f"聚类 bootstrap 次数，默认 {DEFAULT_N_ITERATIONS}")
    parser.add_argument("--seed", type=int, default=None, help=f"bootstrap 随机种子，默认 {DEFAULT_SEED}")
    parser.add_argument("--confidence", type=float, default=None, help=f"区间置信水平，默认 {DEFAULT_CONFIDENCE}")
    parser.add_argument("--warmup", type=int, default=None, help="延迟统计时按预测文件顺序丢掉的前 k 条，默认 0")
    parser.add_argument("--tegrastats", default=None, help="可选的 tegrastats 日志。不提供则资源记为未测")
    parser.add_argument("--tegrastats-start-index", type=int, default=0)
    parser.add_argument("--tegrastats-end-index", type=int, default=None)
    parser.add_argument("--config", default=None, help="可选 YAML。命令行显式传入的参数覆盖配置文件")
    parser.add_argument("--no-plot", action="store_true", help="不写延迟分布图")
    return parser


def _pick(cli_value, config_value, fallback):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return fallback


def main() -> None:
    args = _build_parser().parse_args()
    config = load_eval_config(args.config)
    bootstrap_cfg = config.get("bootstrap") or {}
    latency_cfg = config.get("latency") or {}
    payload = run_eval(
        args.gold,
        args.pred,
        args.out,
        experiment_id=args.experiment_id,
        quantization=args.quantization if args.quantization is not None else config.get("quantization"),
        pixel_budget=args.pixel_budget if args.pixel_budget is not None else config.get("pixel_budget"),
        n_iterations=_pick(args.bootstrap_iters, bootstrap_cfg.get("n_iterations"), DEFAULT_N_ITERATIONS),
        seed=_pick(args.seed, bootstrap_cfg.get("seed"), DEFAULT_SEED),
        confidence=_pick(args.confidence, bootstrap_cfg.get("confidence"), DEFAULT_CONFIDENCE),
        warmup=_pick(args.warmup, latency_cfg.get("warmup"), 0),
        tegrastats_path=args.tegrastats,
        tegrastats_start_index=args.tegrastats_start_index,
        tegrastats_end_index=args.tegrastats_end_index,
        make_plot=not args.no_plot,
    )
    false_complete = payload["checklist"]["false_complete_rate"]
    print(
        f"写出 {args.out}  experiment={payload['experiment_id']}  "
        f"误报齐全率={false_complete['numerator']}/{false_complete['denominator']}"
    )


if __name__ == "__main__":
    main()
