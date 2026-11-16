"""Deterministic checklist post-processing: item states + required items -> decision.

Rules (frozen):
  * any required item ``absent``                 -> ``incomplete``
  * else any required item ``unknown``           -> ``unknown``
  * else (all required items ``present``)        -> ``complete``
  * items not in ``required_items`` never affect the decision
  * an empty required list is trivially ``complete``
"""

from __future__ import annotations

from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATES, normalize_required_items


def _join_zh(items: list[str]) -> str:
    return "、".join(NAME_ZH[i] for i in items)


def derive(item_states: dict[str, str], required_items: list[str]) -> dict:
    """Return ``{decision, missing_required, uncertain_required, answer_zh}``.

    Raises ``UnsupportedCategoryError`` for unknown required categories and
    ``ValueError`` if ``item_states`` is incomplete or contains illegal states.
    """
    required = normalize_required_items(required_items)
    missing_keys = [c for c in CATEGORY_IDS if c not in item_states]
    if missing_keys:
        raise ValueError(f"item_states 缺少类别: {missing_keys}")
    bad = {k: v for k, v in item_states.items() if k not in CATEGORY_IDS or v not in STATES}
    if bad:
        raise ValueError(f"item_states 含非法类别或状态: {bad}")

    missing = [c for c in required if item_states[c] == "absent"]
    uncertain = [c for c in required if item_states[c] == "unknown"]

    if missing:
        decision = "incomplete"
    elif uncertain:
        decision = "unknown"
    else:
        decision = "complete"

    if not required:
        answer = "清单为空，无需检查器材。"
    elif decision == "complete":
        answer = f"所需器材齐全：{_join_zh(required)}。"
    else:
        parts = []
        if missing:
            parts.append(f"缺少{_join_zh(missing)}")
        if uncertain:
            parts.append(f"{_join_zh(uncertain)}是否齐备暂时无法确认")
        answer = "；".join(parts) + "。"

    return {
        "decision": decision,
        "missing_required": missing,
        "uncertain_required": uncertain,
        "answer_zh": answer,
    }
