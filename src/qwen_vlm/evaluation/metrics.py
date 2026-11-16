"""清单点检与存在性任务的指标。

计算规则以 ``contracts/metrics.md`` 为唯一依据，并在模块常量 ``METHODOLOGY_NOTES``
里记录对原文歧义的处理。所有函数都是纯函数：不读文件、不改入参。

解析失败（``parse_status != "ok"``）的样本留在准确率、决策、集合完全匹配和
误报齐全率的分母里。类别状态记为 ``invalid``，完整性决策也记为 ``invalid``，
此时不调用 ``derive``。``invalid`` 不是 ``absent``，也不等于空缺失集合。
"""

from __future__ import annotations

from qwen_vlm.common.categories import CATEGORY_IDS, DECISIONS, STATES, normalize_required_items
from qwen_vlm.common.derive import derive

INVALID = "invalid"
PRED_STATE_LABELS: tuple[str, ...] = tuple(STATES) + (INVALID,)
PRED_DECISION_LABELS: tuple[str, ...] = tuple(DECISIONS) + (INVALID,)

METHODOLOGY_NOTES: tuple[str, ...] = (
    "每个比例都返回 value、numerator、denominator；分母为 0 时 value 为 None。n_samples 是参与该指标的样本数，类别级指标的分母可能是样本数乘以类别数。",
    "parse_status 不是 ok 的样本不从分母中删除。五个类别的预测都记为 invalid，完整性决策记为 invalid，并且不对预测调用 derive。",
    "解析失败时不把缺失集合当成空集合去和真值比‘完全相同’：集合完全匹配一律为否。微平均 P/R/F1 里，失败预测不产生任何‘预测为缺失’的元素，真值缺失项记为 FN、不记 FP；真值缺失集合也为空时，该样本不贡献 TP/FP/FN。",
    "JSON 有效率按原文定义为首次即解析成功：parse_status==ok 且 retry_count==0。它不是最终有效率。失败率是最终 parse_status!=ok 的比例。重试率是 retry_count>0 的比例。",
    "宏 F1 对 present/absent/unknown 三类的 F1 做算术平均。invalid 只进入真实类别的 FN，不进入任何一类的 FP，也不单独成类。某一类的 2TP+FP+FN 为 0 时该类 F1 的 value 为 None，宏平均只对有定义的类别计算；三类都无定义时宏平均 value 为 None。",
    "明确回答覆盖率的分母是真值为 present 或 absent 的全部（样本, 类别），包括预测成 unknown 或 invalid 的格子。明确回答子集准确率的分母只保留预测也是 present 或 absent 的格子，这是条件子集，不是把失败样本从覆盖率里删掉。",
    "presence 任务只评 target_item。解析成功但 item 不等于 target_item 时，不采用其 state，混淆矩阵记入 invalid，并另计 item_mismatch。JSON 有效率仍只看 parse_status 与 retry_count。",
    "成对一致性只使用 checklist 样本。pair_group_id 下不是恰好两条 checklist，或没有任何类别的真值不同，则跳过该组并计入 skipped。被移动的类别是真值不同的类别；若多于一类，必须每一类在两张图上都预测正确才算该图片对一致。分母是这样的图片对数量，不是‘两次回答不同’的数量。",
    "完整性决策、missing_required、uncertain_required 一律用 common.derive 根据金标准 required_items 重算，忽略预测文件里自带的 derived 字段。",
    "按 condition_tags 分组时，标签字符串原样使用，一个样本可以同时进入多个标签。分组后再计算指标，因此成对一致性只统计过滤后仍然成对的图片。",
)


def ratio(numerator: int, denominator: int, *, n_samples: int | None = None) -> dict:
    """返回比例。分母为 0 时 value 为 None，不把缺失定义记成 0。"""
    numer = int(numerator)
    denom = int(denominator)
    out = {
        "value": None if denom == 0 else numer / denom,
        "numerator": numer,
        "denominator": denom,
    }
    if n_samples is not None:
        out["n_samples"] = int(n_samples)
    return out


def _retry_count(pred: dict) -> int:
    value = pred.get("retry_count") or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_ok(pred: dict) -> bool:
    return pred.get("parse_status") == "ok" and isinstance(pred.get("parsed"), dict)


def checklist_pred_states(record: dict) -> dict[str, str]:
    """清单样本的五类预测。失败、任务不符或字段不合法时，五类都是 invalid。"""
    pred = record["pred"]
    if not _parse_ok(pred) or pred.get("task") not in (None, "checklist"):
        return {c: INVALID for c in CATEGORY_IDS}
    parsed = pred["parsed"]
    if any(parsed.get(c) not in STATES for c in CATEGORY_IDS):
        return {c: INVALID for c in CATEGORY_IDS}
    return {c: parsed[c] for c in CATEGORY_IDS}


def _required(record: dict) -> list[str]:
    return normalize_required_items(list(record["gold"].get("required_items") or []))


def gold_derived(record: dict) -> dict:
    """金标准决策。金标准不合法时抛出 ValueError，避免静默算错。"""
    try:
        return derive(record["gold"]["item_states"], _required(record))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"样本 {record.get('sample_id')} 的金标准无法派生决策: {exc}") from exc


def pred_derived(record: dict) -> dict:
    """预测决策。解析失败时不调用 derive，决策为 invalid，两个集合为 None。"""
    states = checklist_pred_states(record)
    if any(state == INVALID for state in states.values()):
        return {
            "decision": INVALID,
            "missing_required": None,
            "uncertain_required": None,
            "answer_zh": None,
        }
    return derive(states, _required(record))


def _checklist_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r["gold"]["task"] == "checklist"]


def _presence_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r["gold"]["task"] == "presence"]


def _empty_confusion(gold_labels: tuple[str, ...] | list[str], pred_labels: tuple[str, ...] | list[str]) -> dict:
    counts = {g: {p: 0 for p in pred_labels} for g in gold_labels}
    return {
        "gold_labels": list(gold_labels),
        "pred_labels": list(pred_labels),
        "counts": counts,
        "matrix": [[0 for _ in pred_labels] for _ in gold_labels],
    }


def _confusion_from_pairs(
    pairs: list[tuple[str, str]],
    gold_labels: tuple[str, ...] | list[str],
    pred_labels: tuple[str, ...] | list[str],
) -> dict:
    counts = {g: {p: 0 for p in pred_labels} for g in gold_labels}
    for gold, pred in pairs:
        if gold not in counts or pred not in counts[gold]:
            raise ValueError(f"混淆矩阵出现未定义的标签: gold={gold!r} pred={pred!r}")
        counts[gold][pred] += 1
    matrix = [[counts[g][p] for p in pred_labels] for g in gold_labels]
    return {
        "gold_labels": list(gold_labels),
        "pred_labels": list(pred_labels),
        "counts": counts,
        "matrix": matrix,
    }


def _class_scores(confusion: dict) -> dict[str, dict]:
    """由三行四列的状态混淆矩阵计算三类的 P/R/F1。invalid 只出现在 FN。"""
    counts = confusion["counts"]
    per_class = {}
    for state in STATES:
        tp = counts[state][state]
        fp = sum(counts[other][state] for other in STATES if other != state)
        fn = sum(count for label, count in counts[state].items() if label != state)
        per_class[state] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
        }
    return per_class


def _macro_f1(per_class: dict[str, dict]) -> dict:
    defined = [per_class[state]["f1"]["value"] for state in STATES if per_class[state]["f1"]["value"] is not None]
    n_defined = len(defined)
    return {
        "value": None if n_defined == 0 else sum(defined) / n_defined,
        "numerator": n_defined,
        "denominator": 0 if n_defined == 0 else len(STATES),
        "n_classes": len(STATES),
        "n_classes_defined": n_defined,
        "aggregation": "arithmetic_mean_of_defined_class_f1",
        "per_class": per_class,
    }


def _json_rates(rows: list[dict]) -> dict:
    n = len(rows)
    first_ok = 0
    retried = 0
    failed = 0
    for record in rows:
        pred = record["pred"]
        status = pred.get("parse_status")
        retries = _retry_count(pred)
        if status == "ok" and retries == 0:
            first_ok += 1
        if retries > 0:
            retried += 1
        if status != "ok":
            failed += 1
    return {
        "json_valid_rate": ratio(first_ok, n, n_samples=n),
        "retry_rate": ratio(retried, n, n_samples=n),
        "failure_rate": ratio(failed, n, n_samples=n),
    }


def checklist_metrics(records: list[dict]) -> dict:
    """计算主任务（checklist）的全部指标。非 checklist 样本被忽略。"""
    rows = _checklist_records(records)
    n = len(rows)
    state_pairs: list[tuple[str, str]] = []
    per_cat_correct = {c: 0 for c in CATEGORY_IDS}
    per_cat_total = {c: 0 for c in CATEGORY_IDS}
    decision_pairs: list[tuple[str, str]] = []
    decision_correct = 0
    exact_match = 0
    false_complete_num = 0
    false_complete_den = 0
    unknown_complete_num = 0
    unknown_complete_den = 0
    micro_tp = micro_fp = micro_fn = 0
    cat_tp = {c: 0 for c in CATEGORY_IDS}
    cat_fp = {c: 0 for c in CATEGORY_IDS}
    cat_fn = {c: 0 for c in CATEGORY_IDS}

    for record in rows:
        gold_states = record["gold"]["item_states"]
        pred_states = checklist_pred_states(record)
        for category in CATEGORY_IDS:
            gold_state = gold_states[category]
            pred_state = pred_states[category]
            state_pairs.append((gold_state, pred_state))
            per_cat_total[category] += 1
            if pred_state == gold_state:
                per_cat_correct[category] += 1

        gold_view = gold_derived(record)
        pred_view = pred_derived(record)
        decision_pairs.append((gold_view["decision"], pred_view["decision"]))
        if pred_view["decision"] == gold_view["decision"]:
            decision_correct += 1
        if gold_view["decision"] == "incomplete":
            false_complete_den += 1
            if pred_view["decision"] == "complete":
                false_complete_num += 1
        if gold_view["decision"] == "unknown":
            unknown_complete_den += 1
            if pred_view["decision"] == "complete":
                unknown_complete_num += 1

        gold_missing = set(gold_view["missing_required"])
        gold_uncertain = set(gold_view["uncertain_required"])
        if pred_view["decision"] == INVALID:
            exact = False
            pred_missing: set[str] = set()
        else:
            pred_missing = set(pred_view["missing_required"])
            pred_uncertain = set(pred_view["uncertain_required"])
            exact = pred_missing == gold_missing and pred_uncertain == gold_uncertain
        if exact:
            exact_match += 1
        # 两边缺失集合都为空时不贡献 TP/FP/FN。失败预测的缺失集合按“没有声称缺失”处理。
        if gold_missing or pred_missing:
            micro_tp += len(gold_missing & pred_missing)
            micro_fp += len(pred_missing - gold_missing)
            micro_fn += len(gold_missing - pred_missing)
            for category in CATEGORY_IDS:
                in_gold = category in gold_missing
                in_pred = category in pred_missing
                if in_gold and in_pred:
                    cat_tp[category] += 1
                elif in_pred:
                    cat_fp[category] += 1
                elif in_gold:
                    cat_fn[category] += 1

    confusion = (
        _confusion_from_pairs(state_pairs, STATES, PRED_STATE_LABELS)
        if rows
        else _empty_confusion(STATES, PRED_STATE_LABELS)
    )
    decision_cm = (
        _confusion_from_pairs(decision_pairs, DECISIONS, PRED_DECISION_LABELS)
        if rows
        else _empty_confusion(DECISIONS, PRED_DECISION_LABELS)
    )
    per_class = _class_scores(confusion)
    correct_cells = sum(per_cat_correct.values())
    explicit_den = 0
    explicit_num = 0
    subset_den = 0
    subset_num = 0
    unknown_den = 0
    unknown_num = 0
    for gold_state in STATES:
        for pred_state, count in confusion["counts"][gold_state].items():
            if gold_state == "unknown":
                unknown_den += count
                if pred_state == "unknown":
                    unknown_num += count
            if gold_state in ("present", "absent"):
                explicit_den += count
                if pred_state in ("present", "absent"):
                    explicit_num += count
                    subset_den += count
                    if pred_state == gold_state:
                        subset_num += count

    per_category_missing = {}
    for category in CATEGORY_IDS:
        tp, fp, fn = cat_tp[category], cat_fp[category], cat_fn[category]
        per_category_missing[category] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
        }

    out = {
        "n_samples": n,
        "state_accuracy": {
            "overall": ratio(correct_cells, n * len(CATEGORY_IDS), n_samples=n),
            "per_category": {
                category: ratio(per_cat_correct[category], per_cat_total[category], n_samples=n)
                for category in CATEGORY_IDS
            },
        },
        "macro_f1": _macro_f1(per_class),
        "confusion_matrix": confusion,
        "missing_set": {
            "precision": ratio(micro_tp, micro_tp + micro_fp, n_samples=n),
            "recall": ratio(micro_tp, micro_tp + micro_fn, n_samples=n),
            "f1": ratio(2 * micro_tp, 2 * micro_tp + micro_fp + micro_fn, n_samples=n),
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "per_category": per_category_missing,
        },
        "set_exact_match": ratio(exact_match, n, n_samples=n),
        "decision_accuracy": ratio(decision_correct, n, n_samples=n),
        "decision_confusion_matrix": decision_cm,
        "false_complete_rate": ratio(false_complete_num, false_complete_den, n_samples=n),
        "unknown_sample_false_complete_rate": ratio(unknown_complete_num, unknown_complete_den, n_samples=n),
        "unknown_recall": ratio(unknown_num, unknown_den, n_samples=n),
        "explicit_coverage": ratio(explicit_num, explicit_den, n_samples=n),
        "explicit_subset_accuracy": ratio(subset_num, subset_den, n_samples=n),
    }
    out.update(_json_rates(rows))
    return out


def _presence_label(record: dict) -> tuple[str, bool]:
    """返回 (用于混淆矩阵的预测状态, 是否 item 不匹配)。

    item 不匹配时预测状态为 invalid，但 JSON 仍然可能是有效的。
    """
    pred = record["pred"]
    target = record["gold"].get("target_item")
    if not _parse_ok(pred) or pred.get("task") not in (None, "presence"):
        return INVALID, False
    parsed = pred["parsed"]
    item = parsed.get("item")
    state = parsed.get("state")
    if item != target or state not in STATES:
        return INVALID, item != target and pred.get("parse_status") == "ok"
    return state, False


def presence_metrics(records: list[dict]) -> dict:
    """辅助任务：只看 target_item 的三状态准确率和混淆矩阵。"""
    rows = _presence_records(records)
    n = len(rows)
    pairs: list[tuple[str, str]] = []
    correct = 0
    item_mismatch = 0
    per_cat_pairs: dict[str, list[tuple[str, str]]] = {c: [] for c in CATEGORY_IDS}
    per_cat_correct = {c: 0 for c in CATEGORY_IDS}
    per_cat_total = {c: 0 for c in CATEGORY_IDS}
    for record in rows:
        target = record["gold"]["target_item"]
        gold_state = record["gold"]["item_states"][target]
        pred_state, mismatch = _presence_label(record)
        if mismatch:
            item_mismatch += 1
        pairs.append((gold_state, pred_state))
        per_cat_pairs[target].append((gold_state, pred_state))
        per_cat_total[target] += 1
        if pred_state == gold_state:
            correct += 1
            per_cat_correct[target] += 1
    confusion = (
        _confusion_from_pairs(pairs, STATES, PRED_STATE_LABELS)
        if rows
        else _empty_confusion(STATES, PRED_STATE_LABELS)
    )
    per_category = {}
    for category in CATEGORY_IDS:
        cat_rows = per_cat_pairs[category]
        per_category[category] = {
            "accuracy": ratio(per_cat_correct[category], per_cat_total[category]),
            "confusion_matrix": (
                _confusion_from_pairs(cat_rows, STATES, PRED_STATE_LABELS)
                if cat_rows
                else _empty_confusion(STATES, PRED_STATE_LABELS)
            ),
        }
    out = {
        "n_samples": n,
        "accuracy": ratio(correct, n, n_samples=n),
        "item_mismatch": item_mismatch,
        "confusion_matrix": confusion,
        "per_category": per_category,
    }
    out.update(_json_rates(rows))
    return out


def pair_consistency(records: list[dict]) -> dict:
    """成对一致性。分母是‘恰好两张 checklist、且至少有一个类别真值不同’的图片对。"""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for record in _checklist_records(records):
        pair_id = record["gold"].get("pair_group_id")
        if not pair_id:
            continue
        if pair_id not in groups:
            order.append(pair_id)
            groups[pair_id] = []
        groups[pair_id].append(record)

    skipped: list[dict] = []
    details: list[dict] = []
    consistent = 0
    for pair_id in order:
        items = groups[pair_id]
        if len(items) != 2:
            skipped.append({"pair_group_id": pair_id, "reason": "not_exactly_two_checklist_samples", "n": len(items)})
            continue
        left, right = items
        moved = [
            category
            for category in CATEGORY_IDS
            if left["gold"]["item_states"][category] != right["gold"]["item_states"][category]
        ]
        if not moved:
            skipped.append({"pair_group_id": pair_id, "reason": "no_moved_category", "n": 2})
            continue
        left_pred = checklist_pred_states(left)
        right_pred = checklist_pred_states(right)
        per_category = {}
        ok = True
        for category in moved:
            left_ok = left_pred[category] == left["gold"]["item_states"][category]
            right_ok = right_pred[category] == right["gold"]["item_states"][category]
            per_category[category] = {"left_correct": left_ok, "right_correct": right_ok}
            if not (left_ok and right_ok):
                ok = False
        if ok:
            consistent += 1
        details.append(
            {
                "pair_group_id": pair_id,
                "sample_ids": [left["sample_id"], right["sample_id"]],
                "moved_categories": moved,
                "consistent": ok,
                "per_category": per_category,
            }
        )
    return {
        "rate": ratio(consistent, len(details), n_samples=len(details)),
        "n_pairs": len(details),
        "n_skipped_groups": len(skipped),
        "skipped_groups": skipped,
        "pairs": details,
    }


def explain_sample(record: dict) -> dict:
    """一条样本的金标准、预测、派生决策和对错，写入 per_sample.jsonl。"""
    gold = record["gold"]
    pred = record["pred"]
    base = {
        "sample_id": record["sample_id"],
        "task": gold["task"],
        "scene_group_id": gold.get("scene_group_id"),
        "pair_group_id": gold.get("pair_group_id"),
        "condition_tags": list(gold.get("condition_tags") or []),
        "required_items": _required(record),
        "parse_status": pred.get("parse_status"),
        "retry_count": _retry_count(pred),
        "missing_prediction": bool(record.get("missing_prediction")),
        "gold_states": dict(gold["item_states"]),
    }
    if gold["task"] == "checklist":
        gold_view = gold_derived(record)
        pred_view = pred_derived(record)
        pred_states = checklist_pred_states(record)
        state_correct = {c: pred_states[c] == gold["item_states"][c] for c in CATEGORY_IDS}
        return {
            **base,
            "pred_states": pred_states,
            "gold_derived": gold_view,
            "pred_derived": pred_view,
            "state_correct": state_correct,
            "n_state_correct": sum(state_correct.values()),
            "decision_correct": pred_view["decision"] == gold_view["decision"],
            "set_exact_match": _set_exact(gold_view, pred_view),
            "false_complete": gold_view["decision"] == "incomplete" and pred_view["decision"] == "complete",
            "unknown_judged_complete": gold_view["decision"] == "unknown" and pred_view["decision"] == "complete",
        }
    if gold["task"] == "presence":
        target = gold["target_item"]
        pred_state, mismatch = _presence_label(record)
        gold_state = gold["item_states"][target]
        parsed = pred.get("parsed") if isinstance(pred.get("parsed"), dict) else None
        return {
            **base,
            "target_item": target,
            "gold_state": gold_state,
            "pred_item": None if parsed is None else parsed.get("item"),
            "pred_state": pred_state,
            "item_mismatch": mismatch,
            "correct": pred_state == gold_state,
        }
    raise ValueError(f"样本 {record.get('sample_id')} 的 task 无法评测: {gold.get('task')!r}")


def _set_exact(gold_view: dict, pred_view: dict) -> bool:
    if pred_view["decision"] == INVALID:
        return False
    return set(pred_view["missing_required"]) == set(gold_view["missing_required"]) and set(
        pred_view["uncertain_required"]
    ) == set(gold_view["uncertain_required"])


def evaluate(records: list[dict], *, condition_field: str = "condition_tags") -> dict:
    """主指标、存在性、成对一致性，以及按条件标签分组后的同一套指标。"""
    grouped: dict[str, list[dict]] = {}
    n_untagged = 0
    for record in records:
        tags = list(record["gold"].get(condition_field) or [])
        if not tags:
            n_untagged += 1
        for tag in tags:
            grouped.setdefault(str(tag), []).append(record)

    by_condition = {}
    for tag in sorted(grouped):
        subset = grouped[tag]
        scene_ids = {r["gold"].get("scene_group_id") for r in subset}
        by_condition[tag] = {
            "n_records": len(subset),
            "n_scene_groups": len(scene_ids),
            "insufficient_samples": len(scene_ids) < 10,
            "checklist": checklist_metrics(subset),
            "presence": presence_metrics(subset),
            "pair_consistency": pair_consistency(subset),
        }
    return {
        "n_records": len(records),
        "n_checklist": len(_checklist_records(records)),
        "n_presence": len(_presence_records(records)),
        "n_untagged": n_untagged,
        "checklist": checklist_metrics(records),
        "presence": presence_metrics(records),
        "pair_consistency": pair_consistency(records),
        "by_condition": by_condition,
        "methodology_notes": list(METHODOLOGY_NOTES),
    }


def scalar_metrics(records: list[dict]) -> dict[str, float | None]:
    """给聚类 bootstrap 用的标量指标。分母为 0 时对应值为 None。"""
    checklist = checklist_metrics(records)
    presence = presence_metrics(records)
    pairs = pair_consistency(records)
    return {
        "checklist.state_accuracy": checklist["state_accuracy"]["overall"]["value"],
        "checklist.macro_f1": checklist["macro_f1"]["value"],
        "checklist.decision_accuracy": checklist["decision_accuracy"]["value"],
        "checklist.false_complete_rate": checklist["false_complete_rate"]["value"],
        "checklist.unknown_sample_false_complete_rate": checklist["unknown_sample_false_complete_rate"]["value"],
        "checklist.unknown_recall": checklist["unknown_recall"]["value"],
        "checklist.explicit_coverage": checklist["explicit_coverage"]["value"],
        "checklist.explicit_subset_accuracy": checklist["explicit_subset_accuracy"]["value"],
        "checklist.set_exact_match": checklist["set_exact_match"]["value"],
        "checklist.missing_set_f1": checklist["missing_set"]["f1"]["value"],
        "checklist.json_valid_rate": checklist["json_valid_rate"]["value"],
        "checklist.retry_rate": checklist["retry_rate"]["value"],
        "checklist.failure_rate": checklist["failure_rate"]["value"],
        "presence.accuracy": presence["accuracy"]["value"],
        "pair_consistency": pairs["rate"]["value"],
    }
