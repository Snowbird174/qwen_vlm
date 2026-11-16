"""数据读取、校验、划分与泄漏检查。

标注 CSV 经 ``convert.csv_to_samples`` 展开成样本 JSONL。划分单位是
``scene_group_id``；共用 ``pair_group_id`` 的场景组会并到同一次划分里。
"""

from qwen_vlm.data.convert import csv_to_samples, load_question_templates
from qwen_vlm.data.io import load_jsonl, resolve_image_path, sha256_file, write_jsonl
from qwen_vlm.data.leakage import check_leakage
from qwen_vlm.data.split import assign_splits, compute_split_counts, write_split_files
from qwen_vlm.data.stats import compute_stats, render_stats_markdown
from qwen_vlm.data.validate import validate_samples

__all__ = [
    "assign_splits",
    "check_leakage",
    "compute_split_counts",
    "compute_stats",
    "csv_to_samples",
    "load_jsonl",
    "load_question_templates",
    "render_stats_markdown",
    "resolve_image_path",
    "sha256_file",
    "validate_samples",
    "write_jsonl",
    "write_split_files",
]
