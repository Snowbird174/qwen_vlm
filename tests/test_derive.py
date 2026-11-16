import pytest

from qwen_vlm.common import UnsupportedCategoryError, derive, normalize_required_items

ALL_PRESENT = {c: "present" for c in ["screwdriver", "pliers", "cable_bundle", "multimeter", "hammer"]}


def states(**overrides):
    return {**ALL_PRESENT, **overrides}


def test_plan_example():
    out = derive(states(pliers="absent", multimeter="unknown"), ["screwdriver", "pliers", "multimeter"])
    assert out == {
        "decision": "incomplete",
        "missing_required": ["pliers"],
        "uncertain_required": ["multimeter"],
        "answer_zh": "缺少钳子；万用表是否齐备暂时无法确认。",
    }


def test_absent_dominates_unknown():
    assert derive(states(hammer="absent", pliers="unknown"), ["hammer", "pliers"])["decision"] == "incomplete"


def test_unknown_without_absent():
    out = derive(states(cable_bundle="unknown"), ["cable_bundle", "hammer"])
    assert out["decision"] == "unknown"
    assert out["missing_required"] == []
    assert out["uncertain_required"] == ["cable_bundle"]


def test_complete():
    out = derive(ALL_PRESENT, ["hammer", "screwdriver"])
    assert out["decision"] == "complete"
    assert out["answer_zh"] == "所需器材齐全：螺丝刀、锤子。"


def test_items_outside_list_are_ignored():
    out = derive(states(pliers="absent", multimeter="unknown"), ["screwdriver"])
    assert out["decision"] == "complete"


def test_empty_list_is_complete():
    out = derive(states(pliers="absent"), [])
    assert out["decision"] == "complete"
    assert out["missing_required"] == [] and out["uncertain_required"] == []


def test_duplicates_and_chinese_names_are_normalized_in_canonical_order():
    assert normalize_required_items(["锤子", "hammer", "螺丝刀", "钳子"]) == ["screwdriver", "pliers", "hammer"]
    out = derive(states(hammer="absent", screwdriver="absent"), ["hammer", "锤子", "screwdriver"])
    assert out["missing_required"] == ["screwdriver", "hammer"]


def test_unsupported_category_raises():
    with pytest.raises(UnsupportedCategoryError):
        derive(ALL_PRESENT, ["screwdriver", "soldering_iron"])


def test_incomplete_or_illegal_states_raise():
    partial = dict(ALL_PRESENT)
    del partial["hammer"]
    with pytest.raises(ValueError):
        derive(partial, ["screwdriver"])
    with pytest.raises(ValueError):
        derive(states(hammer="maybe"), ["screwdriver"])
