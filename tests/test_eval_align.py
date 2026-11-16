"""对齐：缺预测按失败留在分母里，多出来的预测报错。"""

from __future__ import annotations

import pytest

from qwen_vlm.evaluation.align import AlignError, align
from qwen_vlm.evaluation.metrics import checklist_metrics

CATS = ["screwdriver", "pliers", "cable_bundle", "multimeter", "hammer"]


def _gold(sample_id, scene="G"):
    return {
        "sample_id": sample_id,
        "task": "checklist",
        "scene_group_id": scene,
        "pair_group_id": None,
        "required_items": ["screwdriver"],
        "item_states": {cat: "present" for cat in CATS},
        "condition_tags": [],
    }


def _pred(sample_id):
    return {
        "sample_id": sample_id,
        "task": "checklist",
        "parse_status": "ok",
        "retry_count": 0,
        "parsed": {cat: "present" for cat in CATS},
        "timings": {},
    }


def test_missing_prediction_stays_in_the_denominator_as_a_failure():
    records = align([_gold("a"), _gold("b")], [_pred("a")])
    assert records[1]["missing_prediction"] is True
    assert records[1]["pred"]["parse_status"] == "missing_prediction"
    metrics = checklist_metrics(records)
    assert metrics["state_accuracy"]["overall"]["denominator"] == 10
    assert metrics["state_accuracy"]["overall"]["numerator"] == 5
    assert metrics["failure_rate"]["numerator"] == 1 and metrics["failure_rate"]["denominator"] == 2
    assert metrics["decision_accuracy"]["denominator"] == 2
    assert metrics["decision_accuracy"]["numerator"] == 1


def test_extra_prediction_is_an_error():
    with pytest.raises(AlignError, match="没有的 sample_id"):
        align([_gold("a")], [_pred("a"), _pred("extra")])


def test_duplicate_sample_id_is_an_error():
    with pytest.raises(AlignError, match="重复"):
        align([_gold("a"), _gold("a")], [_pred("a")])
