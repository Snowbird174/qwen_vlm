"""手工构造的 8 条 checklist + 2 条 presence。不是模型输出，也不是实验结果。

状态顺序：screwdriver, pliers, cable_bundle, multimeter, hammer。
手算过程写在 tests/test_metrics.py 顶部。预测里的 derived 字段是故意放错的，
评测必须忽略它并重新调用 derive。
"""

from __future__ import annotations

import json
from pathlib import Path

CATS = ["screwdriver", "pliers", "cable_bundle", "multimeter", "hammer"]


def states(screwdriver, pliers, cable_bundle, multimeter, hammer) -> dict:
    return {
        "screwdriver": screwdriver,
        "pliers": pliers,
        "cable_bundle": cable_bundle,
        "multimeter": multimeter,
        "hammer": hammer,
    }


def gold(sample_id, task, scene, pair, required, item_states, tags, target=None) -> dict:
    row = {
        "sample_id": sample_id,
        "image_path": f"fixtures/eval/{sample_id}.png",
        "session_id": "fixture-session",
        "scene_group_id": scene,
        "pair_group_id": pair,
        "task": task,
        "required_items": required,
        "question_zh": "请检查画面中所需器材是否齐全。",
        "item_states": item_states,
        "condition_tags": tags,
        "hard_negative_tags": [],
        "unknown_reason": {key: "遮挡，无法确认" for key, value in item_states.items() if value == "unknown"},
        "source": "synthetic_smoke",
        "annotation_verified": True,
    }
    if target is not None:
        row["target_item"] = target
    return row


def prediction(
    sample_id,
    task,
    parsed,
    *,
    status="ok",
    retry=0,
    tokens=16,
    total_s=1.0,
    generate_s=0.4,
    first_s=0.05,
    derived=None,
) -> dict:
    timings = {
        "image_load_s": 0.01,
        "attempt0_preprocess_s": 0.02,
        "attempt0_first_token_s": first_s,
        "attempt0_generate_s": generate_s if retry == 0 else 0.2,
        "total_s": total_s,
    }
    attempts = ["raw-0"]
    if retry:
        timings["attempt1_first_token_s"] = first_s
        timings["attempt1_generate_s"] = generate_s
        attempts.append("raw-1")
    row = {
        "sample_id": sample_id,
        "task": task,
        "raw_text": attempts[-1],
        "raw_attempts": attempts,
        "parsed": parsed,
        "parse_status": status,
        "retry_count": retry,
        "n_generated_tokens": tokens,
        "n_image_tokens": 100,
        "image_size_after_resize": [512, 512],
        "timings": timings,
        "backend": "fixture",
        "prompt_version": "v1",
    }
    if derived is not None:
        row["derived"] = derived
    return row


ALL = CATS[:]

# c1 全对。所需钳子缺失、万用表 unknown → incomplete。
# c2 只把钳子从 absent 改成 present，决策变为 unknown。与 c1 组成 pair1。
# c3 螺丝刀真值 absent，预测 present → 误报齐全。与 c4 组成 pair2，移动的是螺丝刀。
# c4 真值齐全，预测把锤子说成 absent。
# c5 解析失败，五类都是 invalid，决策 invalid。真值 incomplete，所以留在误报齐全率分母里，分子不加。
# c6 所需清单为空。预测状态全错，但派生决策仍是 complete，缺失集合两边都空，完全匹配为真。
# c7 重试后成功。缺失集合同为 {cable_bundle}，不确定集合不同，所以集合不完全匹配。
# c8 真值决策 unknown，预测 complete。
GOLD = [
    gold("c1", "checklist", "G1", "pair1", ["screwdriver", "pliers", "multimeter"], states("present", "absent", "present", "unknown", "present"), ["normal_light"]),
    gold("c2", "checklist", "G1", "pair1", ["screwdriver", "pliers", "multimeter"], states("present", "present", "present", "unknown", "present"), ["moved"]),
    gold("c3", "checklist", "G2", "pair2", ["screwdriver", "hammer"], states("absent", "present", "present", "present", "present"), ["dim_light"]),
    gold("c4", "checklist", "G2", "pair2", ["screwdriver", "hammer"], states("present", "present", "present", "present", "present"), ["dim_light", "reflection"]),
    gold("c5", "checklist", "G3", None, ALL, states("unknown", "absent", "present", "present", "unknown"), ["slight_occlusion"]),
    gold("c6", "checklist", "G3", None, [], states("present", "present", "present", "present", "present"), ["heavy_occlusion"]),
    gold("c7", "checklist", "G4", None, ["pliers", "cable_bundle"], states("present", "present", "absent", "unknown", "present"), ["normal_light"]),
    gold("c8", "checklist", "G4", None, ["multimeter"], states("present", "present", "present", "unknown", "present"), ["reflection"]),
    gold("p1", "presence", "G5", None, ["hammer"], states("present", "present", "present", "present", "present"), ["normal_light"], target="hammer"),
    gold("p2", "presence", "G5", None, ["pliers"], states("present", "absent", "present", "present", "present"), ["heavy_occlusion"], target="pliers"),
]

PRED = [
    prediction("c1", "checklist", states("present", "absent", "present", "unknown", "present"), total_s=1.0, generate_s=0.4, tokens=10),
    prediction("c2", "checklist", states("present", "present", "present", "unknown", "present"), total_s=2.0, generate_s=0.5, tokens=12),
    prediction(
        "c3",
        "checklist",
        states("present", "present", "present", "present", "present"),
        total_s=3.0,
        generate_s=0.6,
        tokens=14,
        derived={"decision": "incomplete", "missing_required": ["screwdriver"], "uncertain_required": [], "answer_zh": "这是故意写错的派生字段"},
    ),
    prediction("c4", "checklist", states("present", "present", "present", "present", "absent"), total_s=4.0, generate_s=0.7, tokens=16),
    prediction("c5", "checklist", None, status="schema_error", retry=1, total_s=5.0, generate_s=0.8, tokens=8),
    prediction(
        "c6",
        "checklist",
        states("unknown", "unknown", "unknown", "unknown", "unknown"),
        total_s=6.0,
        generate_s=0.9,
        tokens=18,
        derived={"decision": "incomplete", "missing_required": ["hammer"], "uncertain_required": [], "answer_zh": "空清单不应采用这个字段"},
    ),
    prediction(
        "c7",
        "checklist",
        states("present", "unknown", "absent", "present", "present"),
        retry=1,
        total_s=7.0,
        generate_s=1.1,
        tokens=20,
    ),
    prediction("c8", "checklist", states("present", "present", "present", "present", "present"), total_s=8.0, generate_s=1.2, tokens=22),
    prediction("p1", "presence", {"item": "hammer", "state": "present"}, total_s=0.5, generate_s=0.2, tokens=6),
    prediction("p2", "presence", None, status="no_json", total_s=0.5, generate_s=0.2, tokens=2),
]


def dump(directory: Path | None = None) -> None:
    folder = directory or Path(__file__).resolve().parent
    for name, rows in (("gold.jsonl", GOLD), ("pred.jsonl", PRED)):
        path = folder / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    dump()
