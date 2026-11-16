"""评测：指标、对齐、聚类 bootstrap、延迟与 tegrastats。

指标定义以 ``contracts/metrics.md`` 为准。比例一律返回
``{value, numerator, denominator}``，分母为 0 时 ``value`` 为 ``None``。
"""

from qwen_vlm.evaluation.align import AlignError, align, group_by_tags, load_jsonl
from qwen_vlm.evaluation.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_ITERATIONS,
    DEFAULT_SEED,
    MIN_SCENE_GROUPS,
    cluster_bootstrap,
)
from qwen_vlm.evaluation.latency import latency_from_predictions, throughput_from_predictions
from qwen_vlm.evaluation.metrics import (
    checklist_metrics,
    evaluate,
    explain_sample,
    pair_consistency,
    presence_metrics,
)
from qwen_vlm.evaluation.tegrastats import parse_tegrastats_line, parse_tegrastats_log, summarize_tegrastats

__all__ = [
    "AlignError",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_N_ITERATIONS",
    "DEFAULT_SEED",
    "MIN_SCENE_GROUPS",
    "align",
    "checklist_metrics",
    "cluster_bootstrap",
    "evaluate",
    "explain_sample",
    "group_by_tags",
    "latency_from_predictions",
    "load_jsonl",
    "pair_consistency",
    "parse_tegrastats_line",
    "parse_tegrastats_log",
    "presence_metrics",
    "summarize_tegrastats",
    "throughput_from_predictions",
]
