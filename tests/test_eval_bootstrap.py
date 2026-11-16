"""聚类 bootstrap 抽的是场景组。组内全对、组内全错时，复制的准确率只能是 0、0.5 或 1。"""

from __future__ import annotations

import numpy as np
import pytest

from qwen_vlm.evaluation.bootstrap import DEFAULT_N_ITERATIONS, DEFAULT_SEED, cluster_bootstrap


def _records():
    rows = []
    for index in range(20):
        rows.append({"sample_id": f"g{index}", "scene_group_id": "good", "correct": 1})
    for index in range(20):
        rows.append({"sample_id": f"b{index}", "scene_group_id": "bad", "correct": 0})
    return rows


def _accuracy(records):
    if not records:
        return None
    return sum(row["correct"] for row in records) / len(records)


def test_default_iteration_count_is_1000():
    assert DEFAULT_N_ITERATIONS == 1000
    assert DEFAULT_SEED == 0


def test_explicit_draws_take_whole_groups_and_keep_duplicates():
    result = cluster_bootstrap(
        _records(),
        _accuracy,
        group_draws=[[0, 0], [1, 1], [0, 1]],
        return_replicates=True,
    )
    assert result["group_ids"] == ["good", "bad"]
    assert result["replicates"] == [1.0, 0.0, 0.5]
    assert result["point_estimate"] == 0.5
    assert result["insufficient_samples"] is True
    assert result["note"] == "样本不足"
    lengths = cluster_bootstrap(
        _records(),
        lambda records: float(len(records)),
        group_draws=[[0, 0], [1, 1], [0, 1]],
        return_replicates=True,
    )
    assert lengths["replicates"] == [40.0, 40.0, 40.0]


def test_random_replicates_stay_on_the_group_grid_and_honor_the_seed():
    first = cluster_bootstrap(_records(), _accuracy, n_iterations=30, seed=0, return_replicates=True)
    second = cluster_bootstrap(_records(), _accuracy, n_iterations=30, seed=0, return_replicates=True)
    third = cluster_bootstrap(_records(), _accuracy, n_iterations=30, seed=1, return_replicates=True)
    assert first["replicates"] == second["replicates"]
    assert first["replicates"] != third["replicates"]
    assert set(first["replicates"]) <= {0.0, 0.5, 1.0}
    array = np.asarray(first["replicates"], dtype=float)
    low, high = np.percentile(array, [2.5, 97.5], method="linear")
    assert first["ci_low"] == pytest.approx(float(low))
    assert first["ci_high"] == pytest.approx(float(high))
    assert first["n_groups"] == 2
    wide = cluster_bootstrap(_records(), _accuracy, n_iterations=200, seed=0)
    assert wide["ci_high"] - wide["ci_low"] > 0.5


def test_ten_groups_are_enough_and_undefined_replicates_are_dropped():
    rows = [{"sample_id": str(i), "scene_group_id": f"s{i}", "correct": 1} for i in range(10)]
    result = cluster_bootstrap(rows, _accuracy, n_iterations=5, seed=0)
    assert result["insufficient_samples"] is False
    assert "note" not in result

    def only_good(records):
        groups = {row["scene_group_id"] for row in records}
        if groups == {"bad"}:
            return None
        return 1.0

    dropped = cluster_bootstrap(
        _records(),
        only_good,
        group_draws=[[1, 1], [0, 0]],
        return_replicates=True,
    )
    assert dropped["replicates"] == [1.0]
    assert dropped["n_undefined_replicates"] == 1
    assert dropped["ci_low"] == 1.0 and dropped["ci_high"] == 1.0


def test_missing_group_key_raises():
    with pytest.raises(ValueError, match="scene_group_id"):
        cluster_bootstrap([{"sample_id": "x", "correct": 1}], _accuracy, n_iterations=1)
