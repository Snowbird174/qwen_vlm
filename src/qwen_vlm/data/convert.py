"""把「一图一行」的标注 CSV 展开成样本 JSONL。

问法和清单组合从 ``configs/data/question_templates.yaml`` 抽取。
``test_only`` 是测试专用问法，只有显式指定才会使用。
"""

from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from random import Random

import yaml

from qwen_vlm import CONTRACTS_DIR
from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATES, normalize_required_items
from qwen_vlm.common.schema import validate_sample

ANNOTATION_COLUMNS = [
    "session_id",
    "layout_id",
    "variant_id",
    "image_path",
    "pair_group_id",
    "screwdriver",
    "pliers",
    "cable_bundle",
    "multimeter",
    "hammer",
    "unknown_reason",
    "condition_tags",
    "hard_negative_tags",
    "annotation_verified",
    "notes",
]

_TRUE = {"true", "1", "yes", "y", "是"}
_FALSE = {"false", "0", "no", "n", "否"}
_EMPTY_PAIR = {"", "null", "none", "nan"}


class ConversionError(ValueError):
    """CSV 无法展开。``errors`` 是逐行说明。"""

    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@lru_cache(maxsize=1)
def _allowed_sources() -> tuple[str, ...]:
    with open(CONTRACTS_DIR / "sample.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    return tuple(schema["properties"]["source"]["enum"])


def load_question_templates(path: Path | str) -> dict:
    """读取问法集合与清单组合池，并检查组合长度。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "question_sets" not in data:
        raise ValueError(f"问法配置缺少 question_sets: {path}")
    defaults = data.get("defaults") or {}
    data["defaults"] = {
        "question_set": defaults.get("question_set", "default"),
        "checklists_per_image": int(defaults.get("checklists_per_image", 2)),
        "presence_per_image": int(defaults.get("presence_per_image", 0)),
        "seed": int(defaults.get("seed", 0)),
    }
    normalized = {}
    for name, group in data["question_sets"].items():
        normalized[name] = _normalize_question_set(name, group)
    data["question_sets"] = normalized
    return data


def _normalize_question_set(name: str, group: dict) -> dict:
    if not isinstance(group, dict):
        raise ValueError(f"问法集合 {name} 不是映射")
    questions = list(group.get("checklist_questions") or [])
    presence = list(group.get("presence_questions") or [])
    pools_raw = list(group.get("checklist_pools") or [])
    if not questions or not presence or not pools_raw:
        raise ValueError(f"问法集合 {name} 需要 checklist_questions、presence_questions 和 checklist_pools")
    pools = []
    for pool in pools_raw:
        items = normalize_required_items([str(x) for x in pool])
        if len(items) != len(list(pool)) or not 2 <= len(items) <= 4:
            raise ValueError(f"问法集合 {name} 的清单组合必须是 2-4 个不重复类别: {pool}")
        pools.append(items)
    return {
        "checklist_questions": [str(q) for q in questions],
        "presence_questions": [str(q) for q in presence],
        "checklist_pools": pools,
    }


def csv_to_samples(
    csv_path: Path | str,
    templates: dict,
    *,
    question_set: str | None = None,
    checklists_per_image: int | None = None,
    presence_per_image: int | None = None,
    seed: int | None = None,
    source: str,
) -> list[dict]:
    """每张图展开成若干 checklist 样本，并按配置追加 presence 样本。

    presence 样本的 ``required_items`` 只有 ``target_item``。
    该器材的真值状态是 ``item_states[target_item]``，schema 里没有第二个状态字段。
    CSV 的 notes 不写入 JSONL。
    """
    if source not in _allowed_sources():
        raise ConversionError([f"source 必须是 {_allowed_sources()} 之一，实际 {source!r}"])
    defaults = templates["defaults"]
    set_name = question_set or defaults["question_set"]
    if set_name not in templates["question_sets"]:
        known = "、".join(sorted(templates["question_sets"]))
        raise ConversionError([f"未知问法集合 {set_name!r}，可选: {known}"])
    group = templates["question_sets"][set_name]
    n_check = defaults["checklists_per_image"] if checklists_per_image is None else checklists_per_image
    n_presence = defaults["presence_per_image"] if presence_per_image is None else presence_per_image
    base_seed = defaults["seed"] if seed is None else seed
    if n_check < 1:
        raise ConversionError(["checklists_per_image 至少为 1"])
    if n_presence < 0:
        raise ConversionError(["presence_per_image 不能为负"])

    rows = _read_csv(Path(csv_path))
    errors: list[str] = []
    samples: list[dict] = []
    seen_images: set[tuple[str, str, str]] = set()
    for line_no, row in rows:
        built, row_errors = _expand_row(
            line_no,
            row,
            group=group,
            n_check=n_check,
            n_presence=n_presence,
            seed=base_seed,
            source=source,
        )
        key = (row.get("session_id", "").strip(), row.get("layout_id", "").strip(), row.get("variant_id", "").strip())
        if all(key) and key in seen_images:
            row_errors.append(f"第 {line_no} 行: session/layout/variant 重复 {key}")
        elif all(key):
            seen_images.add(key)
        errors.extend(row_errors)
        samples.extend(built)
    if errors:
        raise ConversionError(errors)
    return samples


def _read_csv(path: Path) -> list[tuple[int, dict]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ConversionError([f"{path}: CSV 没有表头"])
        missing = [col for col in ANNOTATION_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ConversionError([f"{path}: CSV 缺少列: {', '.join(missing)}"])
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            rows.append((line_no, row))
    return rows


def _expand_row(line_no, row, *, group, n_check, n_presence, seed, source) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    session = (row.get("session_id") or "").strip()
    layout = (row.get("layout_id") or "").strip()
    variant = (row.get("variant_id") or "").strip()
    image_path = (row.get("image_path") or "").strip()
    for label, value in (("session_id", session), ("layout_id", layout), ("variant_id", variant)):
        if not value:
            errors.append(f"第 {line_no} 行: {label} 为空")
        elif "_" in value:
            errors.append(f"第 {line_no} 行: {label} 不能包含下划线（sample_id 用下划线拼接）: {value}")
    if not image_path:
        errors.append(f"第 {line_no} 行: image_path 为空")
    elif image_path.startswith("/") or ".." in Path(image_path).parts or "\\" in image_path:
        errors.append(f"第 {line_no} 行: image_path 必须是相对 data/ 的正斜杠路径: {image_path}")

    states: dict[str, str] = {}
    for cat in CATEGORY_IDS:
        value = (row.get(cat) or "").strip()
        if value not in STATES:
            errors.append(f"第 {line_no} 行: {cat} 必须是 present/absent/unknown，实际 {value!r}")
        else:
            states[cat] = value
    try:
        reasons = _parse_unknown_reason(row.get("unknown_reason") or "")
    except ValueError as exc:
        errors.append(f"第 {line_no} 行: unknown_reason 无法解析: {exc}")
        reasons = {}
    unknown = {cat for cat, state in states.items() if state == "unknown"}
    if set(reasons) != unknown:
        errors.append(
            f"第 {line_no} 行: unknown_reason 的键 {sorted(reasons)} 必须正好等于 unknown 类别 {sorted(unknown)}"
        )
    try:
        verified = _parse_bool(row.get("annotation_verified") or "")
    except ValueError as exc:
        errors.append(f"第 {line_no} 行: {exc}")
        verified = False
    if errors:
        return [], errors

    pair_id = _parse_pair(row.get("pair_group_id") or "")
    tags = _parse_tags(row.get("condition_tags") or "")
    hard = _parse_tags(row.get("hard_negative_tags") or "")
    ordered_states = {cat: states[cat] for cat in CATEGORY_IDS}
    ordered_reasons = {cat: reasons[cat] for cat in CATEGORY_IDS if cat in reasons}
    rng = _image_rng(seed, session, layout, variant)
    pools = group["checklist_pools"]
    if n_check <= len(pools):
        chosen = rng.sample(pools, n_check)
    else:
        chosen = [rng.choice(pools) for _ in range(n_check)]
    questions = [rng.choice(group["checklist_questions"]) for _ in chosen]
    targets: list[str] = []
    presence_questions: list[str] = []
    if n_presence:
        cats = list(CATEGORY_IDS)
        if n_presence <= len(cats):
            targets = rng.sample(cats, n_presence)
        else:
            targets = [rng.choice(cats) for _ in range(n_presence)]
        for target in targets:
            template = rng.choice(group["presence_questions"])
            try:
                presence_questions.append(template.format(name=NAME_ZH[target], id=target))
            except KeyError as exc:
                return [], [f"第 {line_no} 行: presence 问法缺少占位符 {exc}"]

    samples: list[dict] = []
    seq = 1
    for required, question in zip(chosen, questions):
        samples.append(
            _base_sample(
                session, layout, variant, seq, image_path, pair_id, ordered_states,
                ordered_reasons, tags, hard, source, verified,
                task="checklist", required_items=list(required), question_zh=question,
            )
        )
        seq += 1
    for target, question in zip(targets, presence_questions):
        samples.append(
            _base_sample(
                session, layout, variant, seq, image_path, pair_id, ordered_states,
                ordered_reasons, tags, hard, source, verified,
                task="presence", required_items=[target], question_zh=question, target_item=target,
            )
        )
        seq += 1

    schema_errors = []
    for sample in samples:
        schema_errors.extend(
            f"第 {line_no} 行 sample_id={sample['sample_id']}: {msg}" for msg in validate_sample(sample)
        )
    return ([], schema_errors) if schema_errors else (samples, [])


def _base_sample(
    session, layout, variant, seq, image_path, pair_id, states, reasons, tags, hard,
    source, verified, *, task, required_items, question_zh, target_item=None,
) -> dict:
    sample = {
        "sample_id": f"{session}_{layout}_{variant}_q{seq:02d}",
        "image_path": image_path,
        "session_id": session,
        "scene_group_id": f"{session}_{layout}",
        "pair_group_id": pair_id,
        "task": task,
        "required_items": required_items,
        "question_zh": question_zh,
        "item_states": states,
        "condition_tags": tags,
        "hard_negative_tags": hard,
        "unknown_reason": reasons,
        "source": source,
        "annotation_verified": verified,
    }
    if target_item is not None:
        sample["target_item"] = target_item
    return sample


def _image_rng(seed: int, session: str, layout: str, variant: str) -> Random:
    digest = hashlib.sha256(f"{seed}|{session}|{layout}|{variant}".encode()).digest()
    return Random(int.from_bytes(digest[:8], "big"))


def _parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"annotation_verified 必须是 true/false，实际 {text!r}")


def _parse_pair(text: str) -> str | None:
    value = text.strip()
    if value.lower() in _EMPTY_PAIR:
        return None
    return value


def _parse_tags(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        return []
    separator = "|" if "|" in raw else ";"
    seen: list[str] = []
    for part in raw.split(separator):
        tag = part.strip()
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def _parse_unknown_reason(text: str) -> dict[str, str]:
    """接受 JSON 对象，或 ``类别=原因;类别=原因``。"""
    raw = text.strip()
    if raw.lower() in _EMPTY_PAIR or raw == "{}":
        return {}
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"不是合法 JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError("JSON 必须是对象")
        parsed = obj
    else:
        parsed = {}
        for part in raw.split(";"):
            piece = part.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise ValueError(f"缺少等号: {piece}")
            key, value = piece.split("=", 1)
            parsed[key.strip()] = value.strip()
    result = {}
    for key, value in parsed.items():
        if key not in CATEGORY_IDS:
            raise ValueError(f"未知类别 {key!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} 的原因必须是非空字符串")
        result[key] = value.strip()
    return result
