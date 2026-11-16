"""从预测 JSONL 的 timings 统计延迟和两种吞吐。

延迟按预测文件的顺序丢掉前 ``warmup`` 条，再报告 p50、p95、均值和 n。
没有对应计时字段的记录不进入该字段的分母，并单独计数。这和准确率不同：
准确率必须把解析失败留在分母里，而没有测到的时间无法加进分母。

吞吐分成两个单位，不能合成一个 FPS：

* 完整点检请求数/秒 = checklist 条数 / 这些条的 ``timings.total_s`` 之和
  （含图像加载和重试）。
* 生成 token/s = ``n_generated_tokens`` 之和 / 最后一次生成的 ``generate_s`` 之和。
  ``PredictResult.n_generated_tokens`` 只来自最后一次尝试，所以分母用
  ``attempt{k}_generate_s`` 里 k 最大的那一项，不用 ``total_s``。

摄像头显示 FPS 不在预测文件里，本模块不估计。
"""

from __future__ import annotations

import re

import numpy as np

_ATTEMPT_KEY = re.compile(r"^attempt(\d+)_(.+)$")


def distribution_stats(values: list[float]) -> dict:
    """p50、p95、均值和 n。空序列的三个统计量为 None。百分位用 NumPy 线性插值。"""
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
    }


def last_attempt_timing(timings: dict | None, suffix: str) -> float | None:
    """取最后一次尝试的阶段耗时。也接受没有 attempt 前缀的同名键。"""
    if not isinstance(timings, dict):
        return None
    best_index = -1
    best_value: float | None = None
    for key, value in timings.items():
        match = _ATTEMPT_KEY.fullmatch(str(key))
        if match is None or match.group(2) != suffix or not isinstance(value, (int, float)):
            continue
        index = int(match.group(1))
        if index >= best_index:
            best_index = index
            best_value = float(value)
    if best_value is not None:
        return best_value
    plain = timings.get(suffix)
    if isinstance(plain, (int, float)):
        return float(plain)
    return None


def _drop_warmup(records: list[dict], warmup: int) -> list[dict]:
    if warmup < 0:
        raise ValueError(f"warmup 不能为负，收到 {warmup}")
    return list(records)[warmup:]


def latency_from_predictions(records: list[dict], *, warmup: int = 0) -> dict:
    """统计 ``total_s``、首 token 延迟和生成耗时。``warmup`` 按文件顺序丢掉前 k 条。"""
    kept = _drop_warmup(records, warmup)
    total_values: list[float] = []
    first_values: list[float] = []
    generate_values: list[float] = []
    missing_total = 0
    for record in kept:
        timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
        total = timings.get("total_s")
        if isinstance(total, (int, float)):
            total_values.append(float(total))
        else:
            missing_total += 1
        first = last_attempt_timing(timings, "first_token_s")
        if first is not None:
            first_values.append(first)
        generate = last_attempt_timing(timings, "generate_s")
        if generate is not None:
            generate_values.append(generate)
    return {
        "warmup_excluded": warmup,
        "n_records_after_warmup": len(kept),
        "n_missing_total_s": missing_total,
        "total_s": distribution_stats(total_values),
        "first_token_s": distribution_stats(first_values),
        "generate_s": distribution_stats(generate_values),
        "unit": "s",
    }


def _rate(numerator: float, denominator_s: float) -> dict:
    if denominator_s <= 0 or numerator < 0:
        value = None
    else:
        value = numerator / denominator_s
    return {"value": value, "numerator": numerator, "denominator_s": denominator_s}


def throughput_from_predictions(records: list[dict], *, warmup: int = 0) -> dict:
    """分开返回点检请求数/秒和生成 token/s。两者都只统计 checklist。"""
    kept = [r for r in _drop_warmup(records, warmup) if r.get("task") == "checklist"]
    n_requests = 0
    total_s = 0.0
    n_missing_total = 0
    token_sum = 0
    generate_s = 0.0
    n_token_rows = 0
    n_skipped_tokens = 0
    for record in kept:
        timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
        total = timings.get("total_s")
        if isinstance(total, (int, float)) and float(total) >= 0:
            n_requests += 1
            total_s += float(total)
        else:
            n_missing_total += 1
        tokens = record.get("n_generated_tokens")
        gen = last_attempt_timing(timings, "generate_s")
        if isinstance(tokens, int) and tokens >= 0 and gen is not None and gen >= 0:
            token_sum += tokens
            generate_s += gen
            n_token_rows += 1
        else:
            n_skipped_tokens += 1
    return {
        "warmup_excluded": warmup,
        "checklist_requests_per_s": {
            **_rate(n_requests, total_s),
            "unit": "request/s",
            "n_missing_total_s": n_missing_total,
        },
        "generated_tokens_per_s": {
            **_rate(token_sum, generate_s),
            "numerator_tokens": token_sum,
            "unit": "token/s",
            "n_records": n_token_rows,
            "n_skipped": n_skipped_tokens,
        },
        "camera_fps": {"value": None, "status": "未测"},
        "note": "点检请求数/秒和生成 token/s 单位不同，不能合并成 FPS。token/s 的分母是最后一次生成耗时，不是 total_s。",
    }
