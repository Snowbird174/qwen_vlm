"""tegrastats 三种格式都是格式示例，数字不是实测。缺字段必须是 None。"""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen_vlm.evaluation.tegrastats import parse_tegrastats_line, parse_tegrastats_log, summarize_tegrastats

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "eval"


def _read(name: str) -> list[dict]:
    return parse_tegrastats_log((FIXTURE / name).read_text(encoding="utf-8"))


def test_jetpack6_orin_line_and_window():
    rows = _read("tegrastats_jp6.log")
    assert len(rows) == 2
    first = rows[0]
    assert first["timestamp"] == "2026-11-16 13:44:15"
    assert first["ram_used_mb"] == 3021 and first["ram_total_mb"] == 7620
    assert first["ram_lfb"] == "4x4MB"
    assert first["swap_used_mb"] == 0 and first["swap_cached_mb"] == 0
    assert first["cpu_util_pct"] == [12, 8, 5, 3, 10, 7]
    assert first["cpu_freq_mhz"] == [1510, 1510, 1510, 1510, 729, 729]
    assert first["gr3d_freq_pct"] == 45 and first["gr3d_freq_mhz"] is None
    assert first["temperatures_c"]["cpu"] == pytest.approx(48.5)
    assert first["power_mw"]["VDD_IN"] == {"instant_mw": 5120, "average_mw": 5000}
    summary = summarize_tegrastats(rows)
    assert summary["ram_used_mb"]["mean"] == pytest.approx(3060.5)
    assert summary["ram_used_mb"]["peak"] == 3100
    assert summary["gr3d_freq_pct"]["mean"] == pytest.approx(62.5)
    assert summary["gr3d_freq_pct"]["peak"] == 80
    assert summary["power_mw"]["VDD_IN"]["instant_mw"]["mean"] == pytest.approx(5560)
    assert summary["power_mw"]["VDD_IN"]["instant_mw"]["peak"] == 6000
    assert summary["power_mw"]["VDD_IN"]["average_mw"]["mean"] == pytest.approx(5250)
    assert summary["cpu_util_pct"]["mean"] == pytest.approx(134 / 12)
    assert summary["cpu_util_pct"]["peak"] == 20
    second_only = summarize_tegrastats(rows, start_time="2026-11-16 13:44:16", end_time="2026-11-16 13:44:16")
    assert second_only["n"] == 1 and second_only["ram_used_mb"]["peak"] == 3100
    first_only = summarize_tegrastats(rows, start_index=0, end_index=1)
    assert first_only["n"] == 1 and first_only["ram_used_mb"]["mean"] == 3021


def test_jetpack5_off_core_and_bracketed_gpu_frequency():
    row = _read("tegrastats_jp5.log")[0]
    assert row["timestamp"] is None
    assert row["cpu_util_pct"] == [12, None, 8, 6]
    assert row["cpu_freq_mhz"] == [729, None, 729, 729]
    assert row["swap_cached_mb"] is None
    assert row["emc_freq_pct"] == 0 and row["emc_freq_mhz"] == 2133
    assert row["gr3d_freq_pct"] == 0 and row["gr3d_freq_mhz"] == 624
    assert row["temperatures_c"]["cpu"] == pytest.approx(47.5)
    assert row["power_mw"]["VDD_SOC"]["instant_mw"] == 1448
    assert "cv0" not in row["temperatures_c"]


def test_jetpack4_plain_pom_channels_and_us_timestamp():
    row = _read("tegrastats_jp4.log")[0]
    assert row["timestamp"] == "2026-11-16 13:44:15"
    assert row["ram_total_mb"] == 3964
    assert row["cpu_util_pct"] == [15, 8, 12, 10]
    assert row["gr3d_freq_pct"] == 0 and row["gr3d_freq_mhz"] is None
    assert row["temperatures_c"]["pmic"] == 50
    assert row["temperatures_c"]["thermal"] == pytest.approx(38.5)
    assert row["power_mw"]["POM_5V_IN"] == {"instant_mw": 1864, "average_mw": 1864}
    assert row["power_mw"]["POM_5V_GPU"]["instant_mw"] == 0
    assert "VDD_IN" not in row["power_mw"]


def test_missing_fields_are_none_and_comments_are_skipped():
    assert parse_tegrastats_line("# comment") is None
    assert parse_tegrastats_line("not a log") is None
    row = parse_tegrastats_line("RAM 100/200MB SWAP 1/10MB CPU [1%@100]")
    assert row is not None
    assert row["gr3d_freq_pct"] is None
    assert row["emc_freq_pct"] is None
    assert row["swap_cached_mb"] is None
    assert row["temperatures_c"] == {}
    assert row["power_mw"] == {}
    assert row["ram_lfb"] is None
    summary = summarize_tegrastats([row])
    assert summary["gr3d_freq_pct"]["mean"] is None
    assert summary["power_mw"] == {}
