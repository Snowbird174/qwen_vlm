"""对照手工构造的 10 条样本核对全部分子和分母。

数据在 tests/fixtures/eval/case_data.py，不是实验结果。下面的计数按
contracts/metrics.md 手算。类别顺序 S/P/C/M/H = screwdriver/pliers/cable_bundle/multimeter/hammer。

八条 checklist 的（真值, 预测）：

c1 所需 S,P,M。真值 P,A,P,U,P，预测相同。决策都是 incomplete（缺 P，M 不确定）。完全匹配。
c2 所需 S,P,M。真值 P,P,P,U,P，预测相同。决策都是 unknown。完全匹配。与 c1 只差钳子，钳子两次都预测对。
c3 所需 S,H。真值 A,P,P,P,P，预测全 P。真值 incomplete，预测 complete。这是唯一的误报齐全。
    预测文件里带了 decision=incomplete 的 derived，必须忽略。
c4 所需 S,H。真值全 P，预测 H=A。真值 complete，预测 incomplete。与 c3 只差螺丝刀：c3 预测错，c4 预测对，对子不一致。
c5 所需五类。真值 U,A,P,P,U。解析失败，五类预测 invalid，决策 invalid。真值 incomplete，留在误报齐全率分母，分子不加。
c6 所需清单为空。真值全 P，预测全 U。状态全错，但派生决策都是 complete，两个集合都空，完全匹配为真，不贡献 TP/FP/FN。
    预测文件里带了 decision=incomplete 的 derived，必须忽略。
c7 所需 P,C。真值 P,P,A,U,P，预测 P,U,A,P,P。缺失集合都是 {C}，不确定集合真值空、预测 {P}，不完全匹配。
    retry_count=1 且最终 ok：不算 JSON 首次成功，算重试，不算失败。
c8 所需 M。真值 P,P,P,U,P，预测全 P。真值 unknown，预测 complete。

状态格 8×5=40。预测对的格子：
S: c1 c2 c4 c7 c8 = 5/8（c3 A→P，c5 U→invalid，c6 P→U）
P: c1 c2 c3 c4 c8 = 5/8（c5 A→invalid，c6 P→U，c7 P→U）
C: 除 c5、c6 外都对 = 6/8
M: c1 c2 c3 c4 = 4/8（c5 P→invalid，c6 P→U，c7 U→P，c8 U→P）
H: c1 c2 c3 c7 c8 = 5/8（c4 P→A，c5 U→invalid，c6 P→U）
合计 25/40。若把 c5 从分母去掉会得到 25/35，那就是错的。

混淆矩阵（行 present/absent/unknown，列 present/absent/unknown/invalid）：
present: 21, 1, 6, 2
absent:   1, 2, 0, 1
unknown:  2, 0, 2, 2
行和 30+4+6=40。

F1：invalid 只进 FN。
present TP21 FP3 FN9，F1=42/54
absent  TP2  FP1 FN2，F1=4/7
unknown TP2  FP6 FN4，F1=4/14
宏平均 (42/54 + 4/7 + 4/14) / 3 = (7/9 + 6/7) / 3 = 103/189

缺失集合（只看所需器材里的 absent；c5 失败预测不产生 FP）：
c1 TP {P}；c2 两边空，不贡献；c3 FN {S}；c4 FP {H}；c5 FN {P}；
c6 两边空，不贡献；c7 TP {C}；c8 两边空，不贡献。
微平均 TP2 FP1 FN2。精确率 2/3，召回率 2/4，F1 4/7。
逐类：S 0/0/1，P 1/0/1，C 1/0/0，M 0/0/0，H 0/1/0。

集合完全匹配：c1、c2、c6 为真，3/8。c5 失败，即使不算，也不能把失败当成空集合匹配。
决策：c1、c2、c6、c7 正确，4/8。
误报齐全：真值 incomplete 是 c1 c3 c5 c7，只有 c3 预测 complete，1/4。
unknown 样本误判齐全：真值 unknown 是 c2 c8，c8 预测 complete，1/2。
unknown 召回：真值 unknown 共 6 格，预测对 2 格（c1、c2 的万用表），2/6。
明确回答覆盖：真值 present/absent 共 34 格，预测也明确的 25 格，25/34。
明确回答子集准确：这 25 格里对 23 格，23/25。
JSON 首次成功 6/8（c5 失败，c7 重试后才成功）。重试 2/8。失败 1/8。
presence：p1 对，p2 解析失败，准确率 1/2。
成对：pair1 一致，pair2 不一致，1/2。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from qwen_vlm.common.schema import validate_sample
from qwen_vlm.evaluation.align import align, load_jsonl
from qwen_vlm.evaluation.metrics import checklist_metrics, evaluate, explain_sample, pair_consistency, presence_metrics

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "eval"


def _case():
    spec = importlib.util.spec_from_file_location("case_data", FIXTURE / "case_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASE = _case()


def _records():
    return align(CASE.GOLD, CASE.PRED)


def _checklist():
    return checklist_metrics(_records())


def test_fixture_files_match_the_handwritten_case():
    assert load_jsonl(FIXTURE / "gold.jsonl") == CASE.GOLD
    assert load_jsonl(FIXTURE / "pred.jsonl") == CASE.PRED
    for sample in CASE.GOLD:
        assert validate_sample(sample) == []


def test_hand_counted_state_accuracy_keeps_parse_failure_in_the_denominator():
    metrics = _checklist()
    overall = metrics["state_accuracy"]["overall"]
    assert overall == {"value": pytest.approx(25 / 40), "numerator": 25, "denominator": 40, "n_samples": 8}
    per_category = metrics["state_accuracy"]["per_category"]
    assert per_category["screwdriver"]["numerator"] == 5 and per_category["screwdriver"]["denominator"] == 8
    assert per_category["pliers"]["numerator"] == 5
    assert per_category["cable_bundle"]["numerator"] == 6
    assert per_category["multimeter"]["numerator"] == 4
    assert per_category["hammer"]["numerator"] == 5
    counts = metrics["confusion_matrix"]["counts"]
    assert counts["present"] == {"present": 21, "absent": 1, "unknown": 6, "invalid": 2}
    assert counts["absent"] == {"present": 1, "absent": 2, "unknown": 0, "invalid": 1}
    assert counts["unknown"] == {"present": 2, "absent": 0, "unknown": 2, "invalid": 2}
    assert metrics["confusion_matrix"]["matrix"] == [
        [21, 1, 6, 2],
        [1, 2, 0, 1],
        [2, 0, 2, 2],
    ]


def test_hand_counted_f1_counts_invalid_only_as_false_negative():
    per_class = _checklist()["macro_f1"]["per_class"]
    assert (per_class["present"]["tp"], per_class["present"]["fp"], per_class["present"]["fn"]) == (21, 3, 9)
    assert per_class["present"]["f1"]["numerator"] == 42
    assert per_class["present"]["f1"]["denominator"] == 54
    assert (per_class["absent"]["tp"], per_class["absent"]["fp"], per_class["absent"]["fn"]) == (2, 1, 2)
    assert per_class["absent"]["f1"]["numerator"] == 4 and per_class["absent"]["f1"]["denominator"] == 7
    assert (per_class["unknown"]["tp"], per_class["unknown"]["fp"], per_class["unknown"]["fn"]) == (2, 6, 4)
    assert per_class["unknown"]["f1"]["numerator"] == 4 and per_class["unknown"]["f1"]["denominator"] == 14
    macro = _checklist()["macro_f1"]
    assert macro["value"] == pytest.approx(103 / 189)
    assert macro["n_classes_defined"] == 3
    assert "invalid" not in per_class


def test_hand_counted_missing_sets_decisions_and_conditional_rates():
    metrics = _checklist()
    missing = metrics["missing_set"]
    assert (missing["tp"], missing["fp"], missing["fn"]) == (2, 1, 2)
    assert missing["precision"]["numerator"] == 2 and missing["precision"]["denominator"] == 3
    assert missing["recall"]["numerator"] == 2 and missing["recall"]["denominator"] == 4
    assert missing["f1"]["numerator"] == 4 and missing["f1"]["denominator"] == 7
    per_category = missing["per_category"]
    assert (per_category["screwdriver"]["tp"], per_category["screwdriver"]["fp"], per_category["screwdriver"]["fn"]) == (0, 0, 1)
    assert per_category["screwdriver"]["precision"]["value"] is None
    assert per_category["screwdriver"]["recall"]["numerator"] == 0 and per_category["screwdriver"]["recall"]["denominator"] == 1
    assert (per_category["pliers"]["tp"], per_category["pliers"]["fp"], per_category["pliers"]["fn"]) == (1, 0, 1)
    assert per_category["pliers"]["f1"]["numerator"] == 2 and per_category["pliers"]["f1"]["denominator"] == 3
    assert (per_category["cable_bundle"]["tp"], per_category["cable_bundle"]["fp"], per_category["cable_bundle"]["fn"]) == (1, 0, 0)
    assert per_category["cable_bundle"]["f1"]["value"] == 1
    assert per_category["multimeter"]["precision"]["value"] is None
    assert per_category["multimeter"]["recall"]["value"] is None
    assert per_category["multimeter"]["f1"]["value"] is None
    assert (per_category["hammer"]["tp"], per_category["hammer"]["fp"], per_category["hammer"]["fn"]) == (0, 1, 0)
    assert per_category["hammer"]["recall"]["value"] is None
    assert per_category["hammer"]["f1"]["value"] == 0

    assert metrics["set_exact_match"]["numerator"] == 3 and metrics["set_exact_match"]["denominator"] == 8
    assert metrics["decision_accuracy"]["numerator"] == 4 and metrics["decision_accuracy"]["denominator"] == 8
    decision = metrics["decision_confusion_matrix"]["counts"]
    assert decision["complete"] == {"complete": 1, "incomplete": 1, "unknown": 0, "invalid": 0}
    assert decision["incomplete"] == {"complete": 1, "incomplete": 2, "unknown": 0, "invalid": 1}
    assert decision["unknown"] == {"complete": 1, "incomplete": 0, "unknown": 1, "invalid": 0}
    assert metrics["false_complete_rate"]["numerator"] == 1 and metrics["false_complete_rate"]["denominator"] == 4
    assert metrics["unknown_sample_false_complete_rate"]["numerator"] == 1
    assert metrics["unknown_sample_false_complete_rate"]["denominator"] == 2
    assert metrics["unknown_recall"]["numerator"] == 2 and metrics["unknown_recall"]["denominator"] == 6
    assert metrics["explicit_coverage"]["numerator"] == 25 and metrics["explicit_coverage"]["denominator"] == 34
    assert metrics["explicit_subset_accuracy"]["numerator"] == 23 and metrics["explicit_subset_accuracy"]["denominator"] == 25
    assert metrics["json_valid_rate"]["numerator"] == 6 and metrics["json_valid_rate"]["denominator"] == 8
    assert metrics["retry_rate"]["numerator"] == 2 and metrics["retry_rate"]["denominator"] == 8
    assert metrics["failure_rate"]["numerator"] == 1 and metrics["failure_rate"]["denominator"] == 8


def test_per_sample_flags_ignore_the_misleading_derived_field():
    rows = {row["sample_id"]: explain_sample(row) for row in _records()}
    assert rows["c3"]["false_complete"] is True
    assert rows["c3"]["pred_derived"]["decision"] == "complete"
    assert rows["c5"]["false_complete"] is False
    assert rows["c5"]["pred_states"] == {cat: "invalid" for cat in rows["c5"]["gold_states"]}
    assert rows["c5"]["pred_derived"]["decision"] == "invalid"
    assert rows["c5"]["set_exact_match"] is False
    assert rows["c6"]["decision_correct"] is True
    assert rows["c6"]["set_exact_match"] is True
    assert rows["c6"]["n_state_correct"] == 0
    assert rows["c6"]["pred_derived"]["decision"] == "complete"
    assert rows["c7"]["decision_correct"] is True
    assert rows["c7"]["set_exact_match"] is False
    assert rows["c7"]["pred_derived"]["missing_required"] == ["cable_bundle"]
    assert rows["c7"]["pred_derived"]["uncertain_required"] == ["pliers"]
    assert rows["c8"]["unknown_judged_complete"] is True
    assert rows["p1"]["correct"] is True
    assert rows["p2"]["correct"] is False
    assert rows["p2"]["pred_state"] == "invalid"


def test_presence_and_pair_consistency():
    presence = presence_metrics(_records())
    assert presence["n_samples"] == 2
    assert presence["accuracy"]["numerator"] == 1 and presence["accuracy"]["denominator"] == 2
    assert presence["failure_rate"]["numerator"] == 1 and presence["json_valid_rate"]["numerator"] == 1
    assert presence["confusion_matrix"]["counts"]["present"]["present"] == 1
    assert presence["confusion_matrix"]["counts"]["absent"]["invalid"] == 1
    pairs = pair_consistency(_records())
    assert pairs["rate"]["numerator"] == 1 and pairs["rate"]["denominator"] == 2
    by_id = {item["pair_group_id"]: item for item in pairs["pairs"]}
    assert by_id["pair1"]["consistent"] is True and by_id["pair1"]["moved_categories"] == ["pliers"]
    assert by_id["pair2"]["consistent"] is False and by_id["pair2"]["moved_categories"] == ["screwdriver"]


def test_condition_groups_keep_a_sample_in_every_tag_it_has():
    evaluated = evaluate(_records())
    assert evaluated["n_checklist"] == 8 and evaluated["n_presence"] == 2
    normal = evaluated["by_condition"]["normal_light"]["checklist"]
    assert normal["n_samples"] == 2
    assert normal["state_accuracy"]["overall"]["numerator"] == 8
    assert normal["state_accuracy"]["overall"]["denominator"] == 10
    assert normal["false_complete_rate"]["numerator"] == 0 and normal["false_complete_rate"]["denominator"] == 2
    dim = evaluated["by_condition"]["dim_light"]
    assert dim["n_records"] == 2
    assert dim["checklist"]["false_complete_rate"]["numerator"] == 1
    assert dim["checklist"]["false_complete_rate"]["denominator"] == 1
    reflection = evaluated["by_condition"]["reflection"]
    assert reflection["n_records"] == 2
    assert evaluated["by_condition"]["moved"]["checklist"]["state_accuracy"]["overall"]["numerator"] == 5
    slight = evaluated["by_condition"]["slight_occlusion"]["checklist"]
    assert slight["state_accuracy"]["overall"]["denominator"] == 5
    assert slight["false_complete_rate"]["denominator"] == 1
    assert slight["false_complete_rate"]["numerator"] == 0
    heavy = evaluated["by_condition"]["heavy_occlusion"]
    assert heavy["checklist"]["n_samples"] == 1
    assert heavy["checklist"]["set_exact_match"]["numerator"] == 1
    assert heavy["presence"]["n_samples"] == 1
    assert heavy["insufficient_samples"] is True


def test_parse_failure_with_empty_gold_missing_set_is_not_an_exact_match():
    """失败预测不能借‘两边都像空集合’记成完全匹配，同时仍留在分母里。"""
    sample = {
        "sample_id": "only",
        "task": "checklist",
        "scene_group_id": "G",
        "pair_group_id": None,
        "required_items": ["screwdriver"],
        "item_states": {cat: "present" for cat in ("screwdriver", "pliers", "cable_bundle", "multimeter", "hammer")},
        "condition_tags": [],
    }
    record = {
        "sample_id": "only",
        "scene_group_id": "G",
        "gold": sample,
        "pred": {"sample_id": "only", "task": "checklist", "parsed": None, "parse_status": "no_json", "retry_count": 0},
        "missing_prediction": False,
    }
    metrics = checklist_metrics([record])
    assert metrics["state_accuracy"]["overall"]["denominator"] == 5
    assert metrics["state_accuracy"]["overall"]["numerator"] == 0
    assert metrics["set_exact_match"] == {"value": 0.0, "numerator": 0, "denominator": 1, "n_samples": 1}
    assert metrics["missing_set"]["tp"] == 0 and metrics["missing_set"]["fp"] == 0 and metrics["missing_set"]["fn"] == 0
    assert metrics["missing_set"]["f1"]["value"] is None
    assert metrics["decision_accuracy"]["numerator"] == 0 and metrics["decision_accuracy"]["denominator"] == 1
    assert metrics["false_complete_rate"]["denominator"] == 0 and metrics["false_complete_rate"]["value"] is None


def test_pair_consistency_requires_both_images_to_match_gold_not_merely_differ():
    """两次预测不同但都猜错，不能算一致。"""

    def record(sample_id, gold_state, pred_state):
        item_states = {cat: "present" for cat in ("screwdriver", "pliers", "cable_bundle", "multimeter", "hammer")}
        item_states["screwdriver"] = gold_state
        return {
            "sample_id": sample_id,
            "scene_group_id": "G",
            "gold": {
                "sample_id": sample_id,
                "task": "checklist",
                "scene_group_id": "G",
                "pair_group_id": "moved-screwdriver",
                "required_items": ["screwdriver"],
                "item_states": item_states,
                "condition_tags": [],
            },
            "pred": {
                "sample_id": sample_id,
                "task": "checklist",
                "parse_status": "ok",
                "retry_count": 0,
                "parsed": {**item_states, "screwdriver": pred_state},
            },
            "missing_prediction": False,
        }

    rows = [record("a", "present", "absent"), record("b", "absent", "present")]
    result = pair_consistency(rows)
    assert result["rate"]["denominator"] == 1
    assert result["rate"]["numerator"] == 0


def test_presence_item_mismatch_is_invalid_for_the_target():
    item_states = {cat: "unknown" for cat in ("screwdriver", "pliers", "cable_bundle", "multimeter", "hammer")}
    record = {
        "sample_id": "p",
        "scene_group_id": "G",
        "gold": {
            "sample_id": "p",
            "task": "presence",
            "target_item": "screwdriver",
            "scene_group_id": "G",
            "required_items": ["screwdriver"],
            "item_states": item_states,
            "condition_tags": [],
        },
        "pred": {
            "sample_id": "p",
            "task": "presence",
            "parse_status": "ok",
            "retry_count": 0,
            "parsed": {"item": "hammer", "state": "unknown"},
        },
        "missing_prediction": False,
    }
    metrics = presence_metrics([record])
    assert metrics["accuracy"]["numerator"] == 0 and metrics["accuracy"]["denominator"] == 1
    assert metrics["item_mismatch"] == 1
    assert metrics["confusion_matrix"]["counts"]["unknown"]["invalid"] == 1
    assert metrics["json_valid_rate"]["numerator"] == 1
    assert metrics["failure_rate"]["numerator"] == 0
