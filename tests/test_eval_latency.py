"""延迟丢掉前 k 次预热；两种吞吐的分母不能混用。"""

from __future__ import annotations

import pytest

from qwen_vlm.evaluation.latency import latency_from_predictions, throughput_from_predictions


def _row(task, total_s, tokens, generate_s, *, second_generate=None):
    timings = {"total_s": total_s, "attempt0_generate_s": generate_s, "attempt0_first_token_s": 0.1}
    if second_generate is not None:
        timings["attempt1_generate_s"] = second_generate
        timings["attempt1_first_token_s"] = 0.2
    return {"task": task, "timings": timings, "n_generated_tokens": tokens}


def test_warmup_drops_the_first_k_records_in_file_order():
    rows = [_row("checklist", total, 10, 0.5) for total in (10, 20, 30, 40, 50)]
    stats = latency_from_predictions(rows, warmup=2)
    assert stats["warmup_excluded"] == 2
    assert stats["total_s"]["n"] == 3
    assert stats["total_s"]["mean"] == pytest.approx(40)
    assert stats["total_s"]["p50"] == pytest.approx(40)
    assert stats["total_s"]["p95"] == pytest.approx(49)
    assert stats["generate_s"]["n"] == 3


def test_missing_timing_is_counted_separately_and_not_treated_as_zero():
    rows = [
        _row("checklist", 10, 4, 0.5),
        {"task": "checklist", "timings": {}, "n_generated_tokens": None},
        _row("checklist", 30, 6, 0.5),
    ]
    stats = latency_from_predictions(rows, warmup=0)
    assert stats["total_s"]["n"] == 2
    assert stats["total_s"]["mean"] == pytest.approx(20)
    assert stats["n_missing_total_s"] == 1
    empty = latency_from_predictions(rows, warmup=3)
    assert empty["total_s"]["n"] == 0
    assert empty["total_s"]["mean"] is None
    assert empty["total_s"]["p50"] is None


def test_token_throughput_uses_the_last_attempt_not_total_time():
    rows = [
        _row("checklist", 1.2, 30, 0.4, second_generate=0.5),
        _row("presence", 100.0, 1000, 10.0),
        _row("checklist", 3.0, 10, 0.5),
    ]
    # 文件顺序的第一条是预热，丢掉后只剩一条 checklist：10 token / 0.5 秒。
    warmed = throughput_from_predictions(rows, warmup=1)
    assert warmed["checklist_requests_per_s"]["value"] == pytest.approx(1 / 3)
    assert warmed["checklist_requests_per_s"]["numerator"] == 1
    assert warmed["generated_tokens_per_s"]["value"] == pytest.approx(10 / 0.5)
    assert warmed["camera_fps"]["status"] == "未测"

    full = throughput_from_predictions(rows, warmup=0)
    assert full["checklist_requests_per_s"]["value"] == pytest.approx(2 / 4.2)
    assert full["checklist_requests_per_s"]["denominator_s"] == pytest.approx(4.2)
    assert full["checklist_requests_per_s"]["numerator"] == 2
    # 30 token 只除最后一次的 0.5 秒，不加第一次的 0.4，也不除 total_s=1.2。presence 不混进来。
    assert full["generated_tokens_per_s"]["numerator_tokens"] == 40
    assert full["generated_tokens_per_s"]["denominator_s"] == pytest.approx(1.0)
    assert full["generated_tokens_per_s"]["value"] == pytest.approx(40)
