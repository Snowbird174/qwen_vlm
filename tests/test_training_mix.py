"""mix_cfg 各开关：unknown、成对 presence、困难样本权重。"""

import copy

from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.common.prompts import presence_target_text, presence_user_prompt
from qwen_vlm.common.schema import validate_sample
from qwen_vlm.training.mix import AUX_PAIR_ID_PREFIX, build_training_mix


def _states(**overrides):
    states = {category: "present" for category in CATEGORY_IDS}
    states.update(overrides)
    return states


def _sample(index, states, **kw):
    sample = {
        "sample_id": f"s{index}",
        "image_path": kw.pop("image", f"img/{index}.jpg"),
        "session_id": "sess",
        "scene_group_id": kw.pop("scene", f"scene{index}"),
        "pair_group_id": kw.pop("pair", None),
        "task": kw.pop("task", "checklist"),
        "required_items": ["screwdriver"],
        "question_zh": kw.pop("question", f"问题{index}"),
        "item_states": states,
        "condition_tags": [],
        "hard_negative_tags": list(kw.pop("hard", [])),
        "unknown_reason": {category: "遮挡" for category, state in states.items() if state == "unknown"},
        "source": "synthetic_smoke",
        "annotation_verified": True,
    }
    if "target_item" in kw:
        sample["target_item"] = kw.pop("target_item")
    if kw:
        raise TypeError(kw)
    return sample


def _cfg(**overrides):
    cfg = {
        "include_unknown": True,
        "aux_presence": "none",
        "aux_ratio": 1.0,
        "hard_negative_weight": 1.0,
        "balance_states": False,
        "epoch_size": None,
    }
    cfg.update(overrides)
    return cfg


def test_include_unknown_false_drops_unknown_samples():
    samples = [
        _sample(1, _states()),
        _sample(2, _states(multimeter="unknown")),
        _sample(3, _states(hammer="absent")),
    ]
    original = copy.deepcopy(samples)
    mixed, stats = build_training_mix(samples, _cfg(include_unknown=False), seed=0)
    assert samples == original
    assert {row["sample_id"] for row in mixed} == {"s1", "s3"}
    assert stats["unknown_samples"] == 0
    assert all(not any(state == "unknown" for state in row["item_states"].values()) for row in mixed)


def test_pairs_share_question_and_differ_in_answer():
    left = _states(hammer="present")
    right = _states(hammer="absent")
    samples = [
        _sample(1, left, pair="p1", image="img/a.jpg", question="左图"),
        _sample(2, right, pair="p1", image="img/b.jpg", question="右图"),
        _sample(3, _states(pliers="absent"), question="无关"),
    ]
    mixed, stats = build_training_mix(samples, _cfg(aux_presence="pairs", aux_ratio=5), seed=1)
    generated = [row for row in mixed if str(row["sample_id"]).startswith(AUX_PAIR_ID_PREFIX)]
    assert len(generated) == 2
    assert stats["aux_presence_from_pairs"] == 2
    assert stats["n_presence_generated"] == 2
    assert {row["target_item"] for row in generated} == {"hammer"}
    assert generated[0]["question_zh"] == generated[1]["question_zh"]
    assert generated[0]["item_states"]["hammer"] != generated[1]["item_states"]["hammer"]
    prompts = [presence_user_prompt(row["target_item"], row["question_zh"]) for row in generated]
    assert prompts[0] == prompts[1]
    answers = [presence_target_text(row["target_item"], row["item_states"][row["target_item"]]) for row in generated]
    assert len(set(answers)) == 2
    assert all(validate_sample(row) == [] for row in generated)
    assert {row["sample_id"] for row in mixed if row["task"] == "checklist"} == {"s1", "s2", "s3"}


def test_aux_none_keeps_existing_presence_but_does_not_invent_pairs():
    samples = [
        _sample(1, _states(hammer="present"), pair="p1", image="img/a.jpg"),
        _sample(2, _states(hammer="absent"), pair="p1", image="img/b.jpg"),
        _sample(9, _states(), task="presence", target_item="screwdriver", question="已有的存在性问法"),
    ]
    mixed, stats = build_training_mix(samples, _cfg(aux_presence="none"), seed=0)
    assert stats["n_presence_generated"] == 0
    assert stats["aux_presence_from_pairs"] == 0
    assert any(row["sample_id"] == "s9" and row["task"] == "presence" for row in mixed)
    assert stats["by_task"]["presence"] == 1
    assert stats["by_task"]["checklist"] == 2


def test_unknown_pair_is_not_used_when_include_unknown_is_false():
    samples = [
        _sample(1, _states(hammer="present"), pair="p1", image="img/a.jpg"),
        _sample(2, _states(hammer="unknown"), pair="p1", image="img/b.jpg"),
    ]
    mixed, stats = build_training_mix(samples, _cfg(include_unknown=False, aux_presence="pairs", aux_ratio=5), seed=0)
    assert [row["sample_id"] for row in mixed] == ["s1"] or {row["sample_id"] for row in mixed} == {"s1"}
    assert stats["n_presence_generated"] == 0
    assert stats["unknown_samples"] == 0


def test_aux_ratio_caps_presence():
    samples = [_sample(i, _states()) for i in range(4)]
    for j in range(6):
        samples.append(_sample(100 + j, _states(), task="presence", target_item="pliers", question=f"存在性{j}"))
    mixed, stats = build_training_mix(samples, _cfg(aux_ratio=0.5), seed=2)
    assert stats["n_checklist_pool"] == 4
    assert stats["aux_ratio_cap"] == 2
    assert stats["n_presence_pool"] == 2
    assert stats["by_task"]["presence"] == 2
    assert stats["by_task"]["checklist"] == 4
    assert len(mixed) == 6


def test_hard_negative_weight_changes_exposure():
    samples = [_sample(i, _states(hammer="absent")) for i in range(10)]
    samples += [_sample(100 + i, _states(), hard=["lookalike"]) for i in range(10)]
    plain, plain_stats = build_training_mix(samples, _cfg(), seed=0)
    assert plain_stats["hard_negative_samples"] == 10
    assert len(plain) == 20
    weighted, stats = build_training_mix(samples, _cfg(hard_negative_weight=9.0, epoch_size=4000), seed=0)
    ratio = stats["hard_negative_samples"] / stats["total_samples"]
    assert ratio > 0.8
    assert len(weighted) == 4000


def test_balance_states_upsamples_rare_state():
    samples = [_sample(i, _states()) for i in range(9)]
    samples.append(_sample(99, _states(**{category: "absent" for category in CATEGORY_IDS})))
    mixed, stats = build_training_mix(samples, _cfg(balance_states=True, epoch_size=2000), seed=3)
    rare = sum(row["sample_id"] == "s99" for row in mixed)
    assert rare > 700
    assert stats["by_category_state"]["screwdriver"]["absent"] == rare


def test_epoch_size_and_seed():
    samples = [_sample(i, _states()) for i in range(10)]
    short, _ = build_training_mix(samples, _cfg(epoch_size=5), seed=4)
    assert len(short) == 5
    assert len({row["sample_id"] for row in short}) == 5
    long, _ = build_training_mix(samples, _cfg(epoch_size=25), seed=4)
    assert len(long) == 25
    counts = {}
    for row in long:
        counts[row["sample_id"]] = counts.get(row["sample_id"], 0) + 1
    assert min(counts.values()) >= 2
    again, _ = build_training_mix(samples, _cfg(epoch_size=25), seed=4)
    assert [row["sample_id"] for row in long] == [row["sample_id"] for row in again]


def test_two_moved_categories_make_four_presence_samples():
    left = _states(hammer="present", pliers="present")
    right = _states(hammer="absent", pliers="absent")
    samples = [
        _sample(1, left, pair="p", image="img/a.jpg"),
        _sample(2, right, pair="p", image="img/b.jpg"),
    ]
    _, stats = build_training_mix(samples, _cfg(aux_presence="pairs", aux_ratio=10), seed=0)
    assert stats["n_presence_generated"] == 4
