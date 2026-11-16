"""Frozen category / state vocabulary, loaded from contracts/categories.yaml."""

from __future__ import annotations

from functools import lru_cache

import yaml

from qwen_vlm import CONTRACTS_DIR


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(CONTRACTS_DIR / "categories.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CATEGORY_IDS: tuple[str, ...] = tuple(c["id"] for c in _load()["categories"])
STATES: tuple[str, ...] = tuple(s["id"] for s in _load()["states"])
DECISIONS: tuple[str, ...] = tuple(_load()["decisions"])
NAME_ZH: dict[str, str] = {c["id"]: c["name_zh"] for c in _load()["categories"]}
STATE_NAME_ZH: dict[str, str] = {s["id"]: s["name_zh"] for s in _load()["states"]}
_ALIAS_TO_ID: dict[str, str] = {
    alias: c["id"] for c in _load()["categories"] for alias in [c["id"], c["name_zh"], *c["aliases_zh"]]
}


class UnsupportedCategoryError(ValueError):
    """Raised when a required item is not one of the five supported categories."""


def normalize_category(name: str) -> str:
    """Map a category id or Chinese name/alias to its canonical id."""
    key = name.strip()
    if key not in _ALIAS_TO_ID:
        raise UnsupportedCategoryError(f"不支持的器材类别: {name!r}")
    return _ALIAS_TO_ID[key]


def normalize_required_items(items: list[str]) -> list[str]:
    """Normalize, de-duplicate and sort required items into canonical category order."""
    ids = {normalize_category(x) for x in items}
    return [c for c in CATEGORY_IDS if c in ids]
