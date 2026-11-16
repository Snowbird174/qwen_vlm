"""Validation and parsing of model outputs and data samples against the frozen JSON Schemas.

Parsing only strips surrounding whitespace / Markdown code fences and takes the first
JSON object in the text. It never repairs keys or values: anything that does not
validate is reported with a failure status and must be counted as such by evaluation.
"""

from __future__ import annotations

import json
from functools import lru_cache

import jsonschema

from qwen_vlm import CONTRACTS_DIR

PARSE_OK = "ok"
PARSE_NO_JSON = "no_json"
PARSE_INVALID_JSON = "invalid_json"
PARSE_SCHEMA_ERROR = "schema_error"
PARSE_STATUSES = (PARSE_OK, PARSE_NO_JSON, PARSE_INVALID_JSON, PARSE_SCHEMA_ERROR)


@lru_cache(maxsize=None)
def _validator(name: str) -> jsonschema.Draft202012Validator:
    with open(CONTRACTS_DIR / name, encoding="utf-8") as f:
        return jsonschema.Draft202012Validator(json.load(f))


def item_states_validator() -> jsonschema.Draft202012Validator:
    return _validator("item_states.schema.json")


def presence_validator() -> jsonschema.Draft202012Validator:
    return _validator("presence.schema.json")


def sample_validator() -> jsonschema.Draft202012Validator:
    return _validator("sample.schema.json")


def extract_first_json_object(text: str) -> tuple[object | None, str]:
    """Return ``(obj, status)`` for the first JSON object embedded in ``text``."""
    start = text.find("{")
    if start < 0:
        return None, PARSE_NO_JSON
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None, PARSE_INVALID_JSON
    return obj, PARSE_OK


def parse_model_output(text: str, task: str = "checklist") -> tuple[dict | None, str]:
    """Parse raw model text for ``task`` in {"checklist", "presence"}.

    Returns ``(parsed, status)``; ``parsed`` is ``None`` unless status is ``ok``.
    For ``checklist`` the parsed value is the ``item_states`` dict; for ``presence``
    it is ``{"item": ..., "state": ...}``.
    """
    obj, status = extract_first_json_object(text)
    if status != PARSE_OK:
        return None, status
    validator = item_states_validator() if task == "checklist" else presence_validator()
    if not validator.is_valid(obj):
        return None, PARSE_SCHEMA_ERROR
    return (obj["item_states"] if task == "checklist" else obj), PARSE_OK


def validate_sample(sample: dict) -> list[str]:
    """Return a list of human-readable schema errors for one JSONL sample (empty if valid)."""
    errors = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in sample_validator().iter_errors(sample)]
    if not errors:
        unknown_cats = {k for k, v in sample["item_states"].items() if v == "unknown"}
        if set(sample["unknown_reason"]) != unknown_cats:
            errors.append(
                f"unknown_reason keys {sorted(sample['unknown_reason'])} != unknown categories {sorted(unknown_cats)}"
            )
    return errors
