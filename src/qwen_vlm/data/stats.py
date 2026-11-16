"""按类别、状态、标签和 split 统计样本分布，输出 Markdown。"""

from __future__ import annotations

from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATES
from qwen_vlm.common.derive import derive


def compute_stats(splits: dict[str, list[dict]]) -> dict:
    """``splits`` 的键是 split 名。只有一份 JSONL 时用 ``{"all": samples}``。"""
    flat: list[tuple[str, dict]] = []
    for name, samples in splits.items():
        for sample in samples:
            flat.append((name, sample))

    images: dict[str, dict] = {}
    for _, sample in flat:
        images.setdefault(sample["image_path"], sample)

    def count_tags(field: str) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = {}
        image_seen: dict[str, set[str]] = {}
        for _, sample in flat:
            for tag in sample.get(field) or []:
                bucket = stats.setdefault(tag, {"samples": 0, "images": 0})
                bucket["samples"] += 1
                seen = image_seen.setdefault(tag, set())
                if sample["image_path"] not in seen:
                    seen.add(sample["image_path"])
                    bucket["images"] += 1
        return dict(sorted(stats.items()))

    category_images = {cat: {state: 0 for state in STATES} for cat in CATEGORY_IDS}
    for sample in images.values():
        for cat in CATEGORY_IDS:
            category_images[cat][sample["item_states"][cat]] += 1
    category_samples = {cat: {state: 0 for state in STATES} for cat in CATEGORY_IDS}
    for _, sample in flat:
        for cat in CATEGORY_IDS:
            category_samples[cat][sample["item_states"][cat]] += 1

    by_task: dict[str, int] = {}
    by_source: dict[str, int] = {}
    verified = {"true": 0, "false": 0}
    decisions = {"complete": 0, "incomplete": 0, "unknown": 0}
    for _, sample in flat:
        by_task[sample["task"]] = by_task.get(sample["task"], 0) + 1
        by_source[sample["source"]] = by_source.get(sample["source"], 0) + 1
        verified["true" if sample["annotation_verified"] else "false"] += 1
        if sample["task"] == "checklist":
            decision = derive(sample["item_states"], sample["required_items"])["decision"]
            decisions[decision] += 1

    by_split = {}
    for name, samples in splits.items():
        by_split[name] = {
            "samples": len(samples),
            "images": len({row["image_path"] for row in samples}),
            "scene_groups": len({row["scene_group_id"] for row in samples}),
            "pair_groups": len({row["pair_group_id"] for row in samples if row.get("pair_group_id")}),
        }

    pair_ids = {sample["pair_group_id"] for sample in images.values() if sample.get("pair_group_id")}
    return {
        "num_samples": len(flat),
        "num_images": len(images),
        "num_scene_groups": len({sample["scene_group_id"] for sample in images.values()}),
        "num_pair_groups": len(pair_ids),
        "by_task": dict(sorted(by_task.items())),
        "by_source": dict(sorted(by_source.items())),
        "verified": verified,
        "category_state_images": category_images,
        "category_state_samples": category_samples,
        "condition_tags": count_tags("condition_tags"),
        "hard_negative_tags": count_tags("hard_negative_tags"),
        "by_split": by_split,
        "checklist_decisions": decisions,
    }


def render_stats_markdown(stats: dict, *, title: str) -> str:
    """渲染统计表。数字全部来自传入样本，不填真实实验结论。"""
    lines = [
        f"# {title}",
        "",
        f"- 样本数：{stats['num_samples']}",
        f"- 图片数：{stats['num_images']}",
        f"- 场景组数：{stats['num_scene_groups']}",
        f"- 成对组数：{stats['num_pair_groups']}",
        f"- 已复核 / 未复核：{stats['verified']['true']} / {stats['verified']['false']}",
        "",
        "## 按任务",
        "",
    ]
    lines.extend(_count_table(stats["by_task"]))
    lines.extend(["", "## 按来源", ""])
    lines.extend(_count_table(stats["by_source"]))
    lines.extend(["", "## 按 split", "", "| split | 样本 | 图片 | 场景组 | 成对组 |", "| --- | --- | --- | --- | --- |"])
    for name, row in stats["by_split"].items():
        lines.append(
            f"| {name} | {row['samples']} | {row['images']} | {row['scene_groups']} | {row['pair_groups']} |"
        )
    lines.extend(["", "## 类别 × 状态（按图片，每张图计一次）", ""])
    lines.extend(_state_table(stats["category_state_images"]))
    lines.extend(["", "## 类别 × 状态（按样本，同一张图的多条问法会重复计数）", ""])
    lines.extend(_state_table(stats["category_state_samples"]))
    lines.extend(["", "## 条件标签", ""])
    lines.extend(_tag_table(stats["condition_tags"]))
    lines.extend(["", "## 困难负样本标签", ""])
    lines.extend(_tag_table(stats["hard_negative_tags"]))
    lines.extend(
        [
            "",
            "## 清单结论（checklist 样本，由冻结的 derive 规则推出，不是模型输出）",
            "",
        ]
    )
    lines.extend(_count_table(stats["checklist_decisions"]))
    lines.append("")
    return "\n".join(lines)


def _count_table(counts: dict) -> list[str]:
    if not counts:
        return ["（无）", ""]
    lines = ["| 项 | 数量 |", "| --- | --- |"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    return lines


def _state_table(table: dict[str, dict[str, int]]) -> list[str]:
    header = "| 类别 | " + " | ".join(STATES) + " |"
    sep = "| --- | " + " | ".join("---" for _ in STATES) + " |"
    lines = [header, sep]
    for cat in CATEGORY_IDS:
        cells = " | ".join(str(table[cat][state]) for state in STATES)
        lines.append(f"| {cat}（{NAME_ZH[cat]}） | {cells} |")
    return lines


def _tag_table(tags: dict[str, dict[str, int]]) -> list[str]:
    if not tags:
        return ["（无）"]
    lines = ["| 标签 | 样本 | 图片 |", "| --- | --- | --- |"]
    for tag, row in tags.items():
        lines.append(f"| {tag} | {row['samples']} | {row['images']} |")
    return lines
