"""按 scene_group_id 做聚类 bootstrap。

每次有放回地抽取场景组，被抽到的组里的样本全部进入该次复制；同一组被抽到
多次时，组内样本按次数重复计入。默认 1000 次、种子 0、95% 百分位区间
（2.5 与 97.5，NumPy 线性插值）。场景组少于 10 个时仍然计算区间，同时把
``insufficient_samples`` 设为 True，供报表标注“样本不足”。

指标函数返回 float，或在该次复制的分母为 0 时返回 None。None 不进入百分位。
指标函数不得修改样本。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

DEFAULT_N_ITERATIONS = 1000
DEFAULT_SEED = 0
DEFAULT_CONFIDENCE = 0.95
MIN_SCENE_GROUPS = 10

MetricFn = Callable[[list], float | None]


def _group_records(records: Sequence[dict], group_key: str) -> tuple[list[str], list[list]]:
    """按样本首次出现的顺序分组。组下标 0 是第一条样本所在的组。"""
    order: list[str] = []
    buckets: dict[str, list] = {}
    for record in records:
        if group_key in record and record[group_key] is not None:
            group_id = record[group_key]
        elif isinstance(record.get("gold"), dict) and record["gold"].get(group_key) is not None:
            group_id = record["gold"][group_key]
        else:
            raise ValueError(f"样本 {record.get('sample_id')!r} 缺少分组字段 {group_key}")
        key = str(group_id)
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(record)
    return order, [buckets[key] for key in order]


def _draws(
    n_groups: int,
    n_iterations: int,
    seed: int,
    group_draws: Sequence[Sequence[int]] | None,
) -> list[list[int]]:
    if group_draws is not None:
        return [list(draw) for draw in group_draws]
    if n_groups == 0:
        return []
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_groups, size=n_groups).tolist() for _ in range(n_iterations)]


def _resample(groups: list[list], draw: Sequence[int]) -> list:
    sample: list = []
    n_groups = len(groups)
    for index in draw:
        if index < 0 or index >= n_groups:
            raise ValueError(f"group draw 下标越界: {index}（共 {n_groups} 组）")
        sample.extend(groups[index])
    return sample


def _interval(values: list[float], confidence: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    array = np.asarray(values, dtype=float)
    low = float(np.percentile(array, 100.0 * alpha, method="linear"))
    high = float(np.percentile(array, 100.0 * (1.0 - alpha), method="linear"))
    return low, high


def _pack(
    *,
    point: float | None,
    replicates: list[float],
    n_undefined: int,
    n_draws: int,
    n_groups: int,
    group_ids: list[str],
    confidence: float,
    min_groups: int,
    seed: int | None,
    group_key: str,
    return_replicates: bool,
) -> dict:
    ci_low, ci_high = _interval(replicates, confidence)
    result = {
        "point_estimate": None if point is None else float(point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "n_iterations": n_draws,
        "n_groups": n_groups,
        "group_ids": group_ids,
        "n_valid_replicates": len(replicates),
        "n_undefined_replicates": n_undefined,
        "insufficient_samples": n_groups < min_groups,
        "min_groups": min_groups,
        "seed": seed,
        "method": "cluster_percentile",
        "group_key": group_key,
        "percentile_method": "numpy.linear",
    }
    if result["insufficient_samples"]:
        result["note"] = "样本不足"
    if return_replicates:
        result["replicates"] = replicates
    return result


def cluster_bootstrap(
    records: Sequence[dict],
    metric_fn: MetricFn,
    *,
    group_key: str = "scene_group_id",
    n_iterations: int = DEFAULT_N_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    min_groups: int = MIN_SCENE_GROUPS,
    group_draws: Sequence[Sequence[int]] | None = None,
    return_replicates: bool = False,
) -> dict:
    """对 ``metric_fn(records) -> float | None`` 求聚类 bootstrap 区间。

    ``group_draws`` 给出每次复制抽到的组下标（有放回）时不再使用随机数。
    组下标按样本首次出现的顺序，从 0 开始。
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence 必须在 (0, 1) 内，收到 {confidence}")
    group_ids, groups = _group_records(records, group_key) if records else ([], [])
    draws = _draws(len(groups), n_iterations, seed, group_draws)
    replicates: list[float] = []
    n_undefined = 0
    for draw in draws:
        value = metric_fn(_resample(groups, draw))
        if value is None:
            n_undefined += 1
        else:
            replicates.append(float(value))
    point = metric_fn(list(records))
    return _pack(
        point=None if point is None else float(point),
        replicates=replicates,
        n_undefined=n_undefined,
        n_draws=len(draws),
        n_groups=len(groups),
        group_ids=group_ids,
        confidence=confidence,
        min_groups=min_groups,
        seed=None if group_draws is not None else seed,
        group_key=group_key,
        return_replicates=return_replicates,
    )


def cluster_bootstrap_many(
    records: Sequence[dict],
    metric_map_fn: Callable[[list], dict[str, float | None]],
    *,
    group_key: str = "scene_group_id",
    n_iterations: int = DEFAULT_N_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    min_groups: int = MIN_SCENE_GROUPS,
    group_draws: Sequence[Sequence[int]] | None = None,
) -> dict[str, dict]:
    """一次重抽样同时计算多个指标。``metric_map_fn`` 返回 ``{名字: float | None}``。"""
    if not 0 < confidence < 1:
        raise ValueError(f"confidence 必须在 (0, 1) 内，收到 {confidence}")
    group_ids, groups = _group_records(records, group_key) if records else ([], [])
    draws = _draws(len(groups), n_iterations, seed, group_draws)
    point = metric_map_fn(list(records))
    names = list(point)
    series = {name: [] for name in names}
    undefined = {name: 0 for name in names}
    for draw in draws:
        values = metric_map_fn(_resample(groups, draw))
        for name in names:
            value = values.get(name)
            if value is None:
                undefined[name] += 1
            else:
                series[name].append(float(value))
    results = {}
    for name in names:
        estimate = point.get(name)
        results[name] = _pack(
            point=None if estimate is None else float(estimate),
            replicates=series[name],
            n_undefined=undefined[name],
            n_draws=len(draws),
            n_groups=len(groups),
            group_ids=group_ids,
            confidence=confidence,
            min_groups=min_groups,
            seed=None if group_draws is not None else seed,
            group_key=group_key,
            return_replicates=False,
        )
    return results
