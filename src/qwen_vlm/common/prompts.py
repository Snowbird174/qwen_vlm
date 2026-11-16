"""Frozen task prompts shared by every backend (Transformers, llama.cpp, training data).

Changing any text here changes the experimental condition; bump PROMPT_VERSION and
record it in run manifests.
"""

from __future__ import annotations

import json

from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, normalize_required_items

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "你是实验室器材点检助手。只根据图片中检查区域内可见的内容判断，不要根据常识或清单猜测。"
    "对每类器材输出三种状态之一：present（画面中有足够证据确认存在）、"
    "absent（检查区域完整可见且确认没有该器材）、unknown（遮挡、模糊、画面不完整或无法辨认）。"
    "万用表的表笔线属于万用表，不算线缆。只输出 JSON，不要输出其他文字。"
)

_CATEGORY_LINE = "，".join(f"{c}={NAME_ZH[c]}" for c in CATEGORY_IDS)
_CHECKLIST_FORMAT = (
    '{"item_states": {' + ", ".join(f'"{c}": "present|absent|unknown"' for c in CATEGORY_IDS) + "}}"
)


def checklist_user_prompt(question_zh: str, required_items: list[str]) -> str:
    """User turn text for the main task (five-category, three-state JSON)."""
    required = normalize_required_items(required_items)
    req_text = "、".join(NAME_ZH[c] for c in required) if required else "（空）"
    return (
        f"器材类别：{_CATEGORY_LINE}。\n"
        f"本次实验所需器材：{req_text}。\n"
        f"问题：{question_zh}\n"
        f"请判断全部五类器材的状态，严格按以下格式输出：{_CHECKLIST_FORMAT}"
    )


def presence_user_prompt(target_item: str, question_zh: str) -> str:
    """User turn text for the auxiliary single-item presence task."""
    return (
        f"问题：{question_zh}\n"
        f"只判断器材 {target_item}（{NAME_ZH[target_item]}），严格按以下格式输出："
        f'{{"item": "{target_item}", "state": "present|absent|unknown"}}'
    )


def checklist_target_text(item_states: dict[str, str]) -> str:
    """Canonical assistant answer for the main task (training target)."""
    return json.dumps({"item_states": {c: item_states[c] for c in CATEGORY_IDS}}, ensure_ascii=False)


def presence_target_text(target_item: str, state: str) -> str:
    """Canonical assistant answer for the auxiliary presence task (training target)."""
    return json.dumps({"item": target_item, "state": state}, ensure_ascii=False)


def build_messages(user_text: str, image) -> list[dict]:
    """Chat messages in the Qwen3-VL processor format; ``image`` is a PIL image or path."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
    ]
