"""用 sample_id 把金标准 JSONL 和逐样本预测 JSONL 对齐。

金标准里有、预测里没有的样本按失败计入，不从分母中删除。
预测里多出来的 sample_id 直接报错，避免悄悄丢掉无法对上的结果。
"""

from __future__ import annotations

import json
from pathlib import Path


class AlignError(ValueError):
    """金标准和预测无法对齐。"""


def load_jsonl(path: str | Path) -> list[dict]:
    """读取 JSONL。空行跳过，非法 JSON 带行号报错。"""
    rows: list[dict] = []
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AlignError(f"{file_path}:{line_no} 不是合法 JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise AlignError(f"{file_path}:{line_no} 必须是 JSON 对象")
            rows.append(obj)
    return rows


def _index_by_id(rows: list[dict], what: str) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for position, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise AlignError(f"{what} 第 {position} 行缺少 sample_id")
        if sample_id in index:
            raise AlignError(f"{what} 中 sample_id 重复: {sample_id}")
        index[sample_id] = row
    return index


def missing_prediction(gold: dict) -> dict:
    """金标准有、预测文件没有的样本。parse_status 不是 ok，评测时按失败计。"""
    return {
        "sample_id": gold["sample_id"],
        "task": gold.get("task"),
        "raw_text": "",
        "raw_attempts": [],
        "parsed": None,
        "parse_status": "missing_prediction",
        "retry_count": 0,
        "n_generated_tokens": None,
        "n_image_tokens": None,
        "image_size_after_resize": None,
        "timings": {},
        "backend": None,
        "prompt_version": None,
    }


def align(gold_rows: list[dict], pred_rows: list[dict]) -> list[dict]:
    """按金标准的顺序对齐。返回的记录可直接交给指标函数。

    每条记录包含 ``sample_id``、``scene_group_id``、``gold``、``pred``、
    ``missing_prediction``。预测多出来的 id 抛 ``AlignError``。
    """
    gold_index = _index_by_id(gold_rows, "金标准")
    pred_index = _index_by_id(pred_rows, "预测")
    extra = [sample_id for sample_id in pred_index if sample_id not in gold_index]
    if extra:
        raise AlignError(f"预测中存在金标准没有的 sample_id: {extra}")
    aligned: list[dict] = []
    for gold in gold_rows:
        sample_id = gold["sample_id"]
        scene_group_id = gold.get("scene_group_id")
        if sample_id in pred_index:
            pred = pred_index[sample_id]
            missing = False
        else:
            pred = missing_prediction(gold)
            missing = True
        aligned.append(
            {
                "sample_id": sample_id,
                "scene_group_id": scene_group_id,
                "gold": gold,
                "pred": pred,
                "missing_prediction": missing,
            }
        )
    return aligned


def group_by_tags(records: list[dict], field: str = "condition_tags") -> dict[str, list[dict]]:
    """按金标准里的标签分组。标签原样使用，一个样本可以进入多个组。"""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        for tag in record["gold"].get(field) or []:
            grouped.setdefault(str(tag), []).append(record)
    return dict(sorted(grouped.items()))
