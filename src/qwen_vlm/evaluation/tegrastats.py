"""解析 Jetson ``tegrastats`` 日志行。

不同 JetPack 的行格式并不相同，本解析器按常见片段容错提取，缺字段时返回
None，不估算、不补 0：

* JetPack 4（Nano）功率通道多为 ``POM_5V_IN 1864/1864``，数值不带 ``mW``，
  CPU 通常 4 核，温度名如 ``PLL``、``PMIC``、``AO``、``thermal``。
* JetPack 5/6（Xavier、Orin）功率通道多为 ``VDD_IN 5120mW/5000mW``。
  ``GR3D_FREQ`` 可能是 ``45%``、``45%@918`` 或 ``0%@[624]``。
  CPU 里可以出现 ``off``。温度传感器的名字和大小写随 BSP 变化。
* 行首可能有外部工具加的时间戳（``YYYY-MM-DD HH:MM:SS`` 或
  ``MM-DD-YYYY HH:MM:SS``），也可能没有。

``tests/fixtures/eval/tegrastats_*.log`` 里的数字是格式示例，不是实测数据。
"""

from __future__ import annotations

import re
from datetime import datetime

_TIMESTAMP = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})\s+"
)
_RAM = re.compile(r"RAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB(?:\s*\(lfb\s+([^)]+)\))?", re.IGNORECASE)
_SWAP = re.compile(
    r"SWAP\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB(?:\s*\(cached\s+(\d+(?:\.\d+)?)MB\))?",
    re.IGNORECASE,
)
_CPU = re.compile(r"CPU\s+\[([^\]]+)\]", re.IGNORECASE)
_CPU_SLOT = re.compile(r"^(?:off|(\d+(?:\.\d+)?)%?(?:@(\d+))?)$", re.IGNORECASE)
_EMC = re.compile(r"EMC_FREQ\s+(\d+(?:\.\d+)?)%(?:@(\d+))?", re.IGNORECASE)
_GR3D = re.compile(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%(?:@\[?(\d+)\]?)?", re.IGNORECASE)
_TEMP = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)@(-?\d+(?:\.\d+)?)C\b")
_POWER_MW = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+(\d+(?:\.\d+)?)mW/(\d+(?:\.\d+)?)mW")
_POWER_PLAIN = re.compile(r"\b((?:POM|VDD)_[A-Z0-9_]+)\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)")


def _optional_float(match: re.Match | None, group: int) -> float | None:
    if match is None:
        return None
    text = match.group(group)
    if text is None:
        return None
    return float(text)


def _parse_timestamp(line: str) -> tuple[datetime | None, str]:
    match = _TIMESTAMP.match(line)
    if match is None:
        return None, line
    text = match.group("ts").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m-%d-%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt), line[match.end():]
        except ValueError:
            continue
    return None, line


def parse_tegrastats_line(line: str) -> dict | None:
    """解析一行。空行、注释和无法识别的行返回 None。缺字段为 None。"""
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    timestamp, body = _parse_timestamp(text)
    ram = _RAM.search(body)
    if ram is None:
        return None
    swap = _SWAP.search(body)
    cpu_utils: list[float | None] | None = None
    cpu_freqs: list[int | None] | None = None
    cpu = _CPU.search(body)
    if cpu is not None:
        cpu_utils = []
        cpu_freqs = []
        for slot in cpu.group(1).split(","):
            piece = slot.strip()
            slot_match = _CPU_SLOT.fullmatch(piece)
            if slot_match is None or piece.lower() == "off":
                cpu_utils.append(None)
                cpu_freqs.append(None)
                continue
            util = slot_match.group(1)
            freq = slot_match.group(2)
            cpu_utils.append(None if util is None else float(util))
            cpu_freqs.append(None if freq is None else int(freq))
    emc = _EMC.search(body)
    gr3d = _GR3D.search(body)
    temperatures = {name.lower(): float(value) for name, value in _TEMP.findall(body)}
    power_matches = _POWER_MW.findall(body)
    if not power_matches:
        power_matches = _POWER_PLAIN.findall(body)
    power = {
        name: {"instant_mw": float(instant), "average_mw": float(average)}
        for name, instant, average in power_matches
    }
    return {
        "timestamp": None if timestamp is None else timestamp.isoformat(sep=" "),
        "ram_used_mb": float(ram.group(1)),
        "ram_total_mb": float(ram.group(2)),
        "ram_lfb": ram.group(3),
        "swap_used_mb": _optional_float(swap, 1),
        "swap_total_mb": _optional_float(swap, 2),
        "swap_cached_mb": _optional_float(swap, 3),
        "cpu_util_pct": cpu_utils,
        "cpu_freq_mhz": cpu_freqs,
        "emc_freq_pct": _optional_float(emc, 1),
        "emc_freq_mhz": None if emc is None or emc.group(2) is None else int(emc.group(2)),
        "gr3d_freq_pct": _optional_float(gr3d, 1),
        "gr3d_freq_mhz": None if gr3d is None or gr3d.group(2) is None else int(gr3d.group(2)),
        "temperatures_c": temperatures,
        "power_mw": power,
    }


def parse_tegrastats_log(text: str) -> list[dict]:
    """解析整段日志，跳过无法识别的行。"""
    rows = []
    for line in text.splitlines():
        parsed = parse_tegrastats_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _mean_peak(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "peak": None}
    return {"n": len(values), "mean": sum(values) / len(values), "peak": max(values)}


def _parse_bound(value: datetime | str | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = value.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m-%d-%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间窗边界: {value!r}")


def select_window(
    rows: list[dict],
    *,
    start_index: int = 0,
    end_index: int | None = None,
    start_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
) -> list[dict]:
    """按样本下标（左闭右开）或时间戳（两端都包含）截取窗口。"""
    window = rows[start_index:end_index]
    start = _parse_bound(start_time)
    end = _parse_bound(end_time)
    if start is None and end is None:
        return window
    selected = []
    for row in window:
        stamp = row.get("timestamp")
        if stamp is None:
            continue
        current = datetime.fromisoformat(stamp)
        if start is not None and current < start:
            continue
        if end is not None and current > end:
            continue
        selected.append(row)
    return selected


def summarize_tegrastats(rows: list[dict], **window) -> dict:
    """对一个时间窗求均值和峰值。某个字段整窗都缺失时，均值和峰值为 None。"""
    chosen = select_window(rows, **window) if window else list(rows)
    ram_used, ram_total, swap_used = [], [], []
    gr3d, emc = [], []
    cpu_utils: list[float] = []
    temperatures: dict[str, list[float]] = {}
    power: dict[str, dict[str, list[float]]] = {}
    for row in chosen:
        if row.get("ram_used_mb") is not None:
            ram_used.append(row["ram_used_mb"])
        if row.get("ram_total_mb") is not None:
            ram_total.append(row["ram_total_mb"])
        if row.get("swap_used_mb") is not None:
            swap_used.append(row["swap_used_mb"])
        if row.get("gr3d_freq_pct") is not None:
            gr3d.append(row["gr3d_freq_pct"])
        if row.get("emc_freq_pct") is not None:
            emc.append(row["emc_freq_pct"])
        for util in row.get("cpu_util_pct") or []:
            if util is not None:
                cpu_utils.append(util)
        for name, value in (row.get("temperatures_c") or {}).items():
            temperatures.setdefault(name, []).append(value)
        for name, channel in (row.get("power_mw") or {}).items():
            bucket = power.setdefault(name, {"instant_mw": [], "average_mw": []})
            if channel.get("instant_mw") is not None:
                bucket["instant_mw"].append(channel["instant_mw"])
            if channel.get("average_mw") is not None:
                bucket["average_mw"].append(channel["average_mw"])
    return {
        "n": len(chosen),
        "status": "ok" if chosen else "empty",
        "ram_used_mb": _mean_peak(ram_used),
        "ram_total_mb": _mean_peak(ram_total),
        "swap_used_mb": _mean_peak(swap_used),
        "gr3d_freq_pct": _mean_peak(gr3d),
        "emc_freq_pct": _mean_peak(emc),
        "cpu_util_pct": _mean_peak(cpu_utils),
        "temperatures_c": {name: _mean_peak(values) for name, values in sorted(temperatures.items())},
        "power_mw": {
            name: {
                "instant_mw": _mean_peak(channels["instant_mw"]),
                "average_mw": _mean_peak(channels["average_mw"]),
            }
            for name, channels in sorted(power.items())
        },
    }
