"""把训练集 JSONL 配成一个 epoch 的监督样本。

语义以 contracts/training_mix.md 为准。成对场景只用来多写几条「单个器材
在不在」的监督样本（同一句问法、两张图的答案不同），不是新的训练目标形式。

随机数顺序（同一个 seed 可复现）：
1. 按 pair_group_id、类别顺序，为每个被移动的类别抽一次问法；
2. presence 样本超过 aux_ratio 上限时，无放回下采样；
3. 按权重生成本 epoch：权重全相同则打乱（不够就整轮重复），否则按权重有放回抽样。
"""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict

import numpy as np

from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATES

AUX_PAIR_ID_PREFIX = "pairaux__"

# 成对场景共用的问法。{name} 会换成该类器材的中文名。两张图抽到同一句。
PRESENCE_QUESTION_TEMPLATES: tuple[str, ...] = (
    "画面中有没有{name}？",
    "检查区域内能否确认{name}存在？",
    "{name}在不在这张实验台图片里？",
    "请判断{name}是存在、缺失，还是无法确认。",
)

_REQUIRED_MIX_KEYS = (
    "include_unknown",
    "aux_presence",
    "aux_ratio",
    "hard_negative_weight",
    "balance_states",
    "epoch_size",
)


def _require_mix_cfg(mix_cfg: dict) -> None:
    missing = [k for k in _REQUIRED_MIX_KEYS if k not in mix_cfg]
    if missing:
        raise KeyError(f"mix_cfg 缺少字段: {missing}")
    if not isinstance(mix_cfg["include_unknown"], bool):
        raise TypeError("include_unknown 必须是布尔值")
    if mix_cfg["aux_presence"] not in ("none", "pairs"):
        raise ValueError(f"aux_presence 只能是 none 或 pairs，收到 {mix_cfg['aux_presence']!r}")
    if not isinstance(mix_cfg["balance_states"], bool):
        raise TypeError("balance_states 必须是布尔值")
    if float(mix_cfg["aux_ratio"]) < 0:
        raise ValueError("aux_ratio 不能为负")
    if float(mix_cfg["hard_negative_weight"]) < 0:
        raise ValueError("hard_negative_weight 不能为负")
    epoch_size = mix_cfg["epoch_size"]
    if epoch_size is not None and int(epoch_size) < 0:
        raise ValueError("epoch_size 不能为负")


def _has_unknown(sample: dict) -> bool:
    return any(state == "unknown" for state in sample["item_states"].values())


def _is_hard_negative(sample: dict) -> bool:
    return bool(sample.get("hard_negative_tags"))


def _representative(samples_for_image: list[dict]) -> dict:
    """同一张图优先用 checklist 行的真值。同一张图的真值必须一致。"""
    ordered = sorted(samples_for_image, key=lambda s: (0 if s["task"] == "checklist" else 1, s["sample_id"]))
    rep = ordered[0]
    for other in ordered[1:]:
        if other["item_states"] != rep["item_states"]:
            raise ValueError(
                f"同一张图的 item_states 不一致: {rep['sample_id']} vs {other['sample_id']} ({rep['image_path']})"
            )
    return rep


def _presence_from_pair(rep: dict, category: str, question_zh: str, pair_group_id: str) -> dict:
    digest = hashlib.sha1(f"{pair_group_id}|{rep['image_path']}|{category}".encode()).hexdigest()[:12]
    sample = {
        "sample_id": f"{AUX_PAIR_ID_PREFIX}{category}__{digest}",
        "image_path": rep["image_path"],
        "session_id": rep["session_id"],
        "scene_group_id": rep["scene_group_id"],
        "pair_group_id": pair_group_id,
        "task": "presence",
        "target_item": category,
        "required_items": list(rep["required_items"]),
        "question_zh": question_zh,
        "item_states": dict(rep["item_states"]),
        "condition_tags": list(rep.get("condition_tags") or []),
        "hard_negative_tags": list(rep.get("hard_negative_tags") or []),
        "unknown_reason": dict(rep.get("unknown_reason") or {}),
        "source": rep["source"],
        "annotation_verified": bool(rep["annotation_verified"]),
    }
    return sample


def _generate_pair_presence(samples: list[dict], rng: np.random.Generator) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        pair_id = sample.get("pair_group_id")
        if pair_id:
            groups[str(pair_id)].append(sample)

    generated: list[dict] = []
    for pair_id in sorted(groups):
        by_image: dict[str, list[dict]] = defaultdict(list)
        for sample in groups[pair_id]:
            by_image[sample["image_path"]].append(sample)
        if len(by_image) < 2:
            continue
        reps = {path: _representative(group) for path, group in by_image.items()}
        paths = sorted(reps)
        for category in CATEGORY_IDS:
            states = [reps[path]["item_states"][category] for path in paths]
            if len(set(states)) < 2:
                continue
            template = PRESENCE_QUESTION_TEMPLATES[int(rng.integers(0, len(PRESENCE_QUESTION_TEMPLATES)))]
            question = template.format(name=NAME_ZH[category])
            for path in paths:
                generated.append(_presence_from_pair(reps[path], category, question, pair_id))
    return generated


def _supervised_keys(sample: dict) -> list[tuple[str, str]]:
    if sample["task"] == "checklist":
        return [(category, sample["item_states"][category]) for category in CATEGORY_IDS]
    if sample["task"] == "presence":
        item = sample["target_item"]
        return [(item, sample["item_states"][item])]
    raise ValueError(f"未知 task: {sample.get('task')!r}")


def _balance_factor(sample: dict, inverse: dict[tuple[str, str], float]) -> float:
    keys = _supervised_keys(sample)
    return float(sum(inverse[key] for key in keys) / len(keys))


def _sample_weights(pool: list[dict], mix_cfg: dict) -> list[float]:
    hard_weight = float(mix_cfg["hard_negative_weight"])
    factors = [1.0] * len(pool)
    if mix_cfg["balance_states"] and pool:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for sample in pool:
            for key in _supervised_keys(sample):
                counts[key] += 1
        inverse = {key: 1.0 / count for key, count in counts.items()}
        factors = [_balance_factor(sample, inverse) for sample in pool]
    weights = []
    for sample, factor in zip(pool, factors):
        weight = factor * (hard_weight if _is_hard_negative(sample) else 1.0)
        if weight < 0:
            raise ValueError("样本权重不能为负")
        weights.append(float(weight))
    return weights


def _is_uniform(weights: np.ndarray) -> bool:
    if weights.size == 0:
        return True
    return bool(np.allclose(weights, weights[0], rtol=1e-6, atol=1e-8))


def _materialize(
    pool: list[dict], weights: list[float], epoch_size: int | None, rng: np.random.Generator
) -> list[dict]:
    n = len(pool)
    if n == 0:
        return []
    target = n if epoch_size is None else int(epoch_size)
    if target < 0:
        raise ValueError("epoch_size 不能为负")
    if target == 0:
        return []
    array = np.asarray(weights, dtype=np.float64)
    if _is_uniform(array):
        order = rng.permutation(n)
        if target <= n:
            indices = order[:target]
        else:
            repeats = int(np.ceil(target / n))
            indices = np.tile(order, repeats)[:target]
    else:
        total = float(array.sum())
        if total <= 0:
            raise ValueError("权重和必须为正，无法抽样")
        indices = rng.choice(n, size=target, replace=True, p=array / total)
    return [copy.deepcopy(pool[int(i)]) for i in indices]


def _exposure_stats(epoch_samples: list[dict], pool_info: dict) -> dict:
    by_task = {"checklist": 0, "presence": 0}
    by_category_state = {category: {state: 0 for state in STATES} for category in CATEGORY_IDS}
    hard_negative_samples = 0
    unknown_samples = 0
    aux_presence_from_pairs = 0
    images: set[str] = set()
    for sample in epoch_samples:
        task = sample["task"]
        if task not in by_task:
            raise ValueError(f"未知 task: {task!r}")
        by_task[task] += 1
        images.add(sample["image_path"])
        if _is_hard_negative(sample):
            hard_negative_samples += 1
        if _has_unknown(sample):
            unknown_samples += 1
        if str(sample["sample_id"]).startswith(AUX_PAIR_ID_PREFIX):
            aux_presence_from_pairs += 1
        for category, state in _supervised_keys(sample):
            by_category_state[category][state] += 1
    stats = {
        "total_samples": len(epoch_samples),
        "unique_images": len(images),
        "by_task": by_task,
        "by_category_state": by_category_state,
        "hard_negative_samples": hard_negative_samples,
        "unknown_samples": unknown_samples,
        "aux_presence_from_pairs": aux_presence_from_pairs,
    }
    stats.update(pool_info)
    return stats


def build_training_mix(samples: list[dict], mix_cfg: dict, seed: int) -> tuple[list[dict], dict]:
    """返回（本 epoch 的样本, 暴露统计）。

    暴露统计按本 epoch 实际抽到的样本计数（重复抽到会重复计数）。
    ``by_category_state`` 只计监督标签：checklist 计五类，presence 只计 target_item。
    ``pool_*`` 字段是重采样之前的数量，方便对照 aux_ratio 是否截断了 presence。
    """
    _require_mix_cfg(mix_cfg)
    rng = np.random.default_rng(int(seed))

    kept = list(samples)
    if not mix_cfg["include_unknown"]:
        kept = [sample for sample in kept if not _has_unknown(sample)]

    for sample in kept:
        if sample["task"] not in ("checklist", "presence"):
            raise ValueError(f"未知 task: {sample['task']!r} ({sample.get('sample_id')})")

    checklist = [sample for sample in kept if sample["task"] == "checklist"]
    existing_presence = [sample for sample in kept if sample["task"] == "presence"]
    generated: list[dict] = []
    if mix_cfg["aux_presence"] == "pairs":
        generated = _generate_pair_presence(kept, rng)

    presence = existing_presence + generated
    n_before_cap = len(presence)
    max_presence = int(np.floor(float(mix_cfg["aux_ratio"]) * len(checklist) + 1e-9))
    if len(presence) > max_presence:
        if max_presence <= 0:
            presence = []
        else:
            picked = np.sort(rng.choice(len(presence), size=max_presence, replace=False))
            presence = [presence[int(i)] for i in picked]

    pool = checklist + presence
    seen: set[str] = set()
    for sample in pool:
        if sample["sample_id"] in seen:
            raise ValueError(f"混合后 sample_id 重复: {sample['sample_id']}")
        seen.add(sample["sample_id"])

    weights = _sample_weights(pool, mix_cfg)
    epoch = _materialize(pool, weights, mix_cfg["epoch_size"], rng)
    pool_info = {
        "pool_size": len(pool),
        "n_checklist_pool": len(checklist),
        "n_presence_pool": len(presence),
        "n_presence_before_cap": n_before_cap,
        "n_presence_generated": len(generated),
        "n_presence_existing": len(existing_presence),
        "aux_ratio_cap": max_presence,
    }
    return epoch, _exposure_stats(epoch, pool_info)
