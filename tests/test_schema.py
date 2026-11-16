import json

from qwen_vlm.common.schema import (
    PARSE_INVALID_JSON,
    PARSE_NO_JSON,
    PARSE_OK,
    PARSE_SCHEMA_ERROR,
    parse_model_output,
    validate_sample,
)

GOOD = {"screwdriver": "present", "pliers": "absent", "cable_bundle": "present", "multimeter": "unknown", "hammer": "present"}


def test_plain_json():
    parsed, status = parse_model_output(json.dumps({"item_states": GOOD}))
    assert status == PARSE_OK and parsed == GOOD


def test_code_fenced_json():
    text = "```json\n" + json.dumps({"item_states": GOOD}, ensure_ascii=False) + "\n```"
    parsed, status = parse_model_output(text)
    assert status == PARSE_OK and parsed == GOOD


def test_no_json():
    assert parse_model_output("所有器材都在。") == (None, PARSE_NO_JSON)


def test_truncated_json():
    assert parse_model_output('{"item_states": {"screwdriver": "pres')[1] == PARSE_INVALID_JSON


def test_missing_category_is_schema_error():
    bad = dict(GOOD)
    del bad["hammer"]
    assert parse_model_output(json.dumps({"item_states": bad}))[1] == PARSE_SCHEMA_ERROR


def test_illegal_state_is_not_repaired():
    bad = {**GOOD, "pliers": "missing"}
    assert parse_model_output(json.dumps({"item_states": bad}))[1] == PARSE_SCHEMA_ERROR


def test_extra_category_is_schema_error():
    bad = {**GOOD, "wrench": "present"}
    assert parse_model_output(json.dumps({"item_states": bad}))[1] == PARSE_SCHEMA_ERROR


def test_presence_task():
    parsed, status = parse_model_output('{"item": "pliers", "state": "absent"}', task="presence")
    assert status == PARSE_OK and parsed == {"item": "pliers", "state": "absent"}
    assert parse_model_output('{"item": "pliers", "state": "gone"}', task="presence")[1] == PARSE_SCHEMA_ERROR


def _sample(**kw):
    s = {
        "sample_id": "s01_l001_v01_q01",
        "image_path": "raw/s01/l001/v01.jpg",
        "session_id": "s01",
        "scene_group_id": "s01_l001",
        "pair_group_id": None,
        "task": "checklist",
        "required_items": ["screwdriver", "pliers"],
        "question_zh": "还缺少什么器材？",
        "item_states": dict(GOOD),
        "condition_tags": ["normal_light"],
        "hard_negative_tags": [],
        "unknown_reason": {"multimeter": "被手遮挡"},
        "source": "real_lab",
        "annotation_verified": True,
    }
    s.update(kw)
    return s


def test_valid_sample():
    assert validate_sample(_sample()) == []


def test_unknown_reason_must_match_unknown_states():
    assert validate_sample(_sample(unknown_reason={})) != []


def test_presence_sample_needs_target_item():
    assert validate_sample(_sample(task="presence")) != []
    assert validate_sample(_sample(task="presence", target_item="pliers")) == []


def test_bad_required_item():
    assert validate_sample(_sample(required_items=["wrench"])) != []
