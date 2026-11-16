"""把对齐后的样本写成 metrics.json、per_sample.jsonl 和可读的 Markdown。

表格里的数字只来自指标函数的返回值。没有测到的格子写“待测”或“未测”，
不填估计值。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH
from qwen_vlm.evaluation.align import align, load_jsonl
from qwen_vlm.evaluation.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_ITERATIONS,
    DEFAULT_SEED,
    MIN_SCENE_GROUPS,
    cluster_bootstrap_many,
)
from qwen_vlm.evaluation.latency import latency_from_predictions, throughput_from_predictions
from qwen_vlm.evaluation.metrics import METHODOLOGY_NOTES, evaluate, explain_sample, scalar_metrics
from qwen_vlm.evaluation.tegrastats import parse_tegrastats_log, summarize_tegrastats

METHOD_ORDER = ("B0", "B1", "M", "A1", "A2", "A3")
QUANT_ORDER = ("F16", "Q8_0", "Q4_K_M")
PIXEL_ORDER = (384, 512, 768)

_BOOTSTRAP_KEYS = {
    ("checklist", "state_accuracy", "overall"): "checklist.state_accuracy",
    ("checklist", "macro_f1"): "checklist.macro_f1",
    ("checklist", "decision_accuracy"): "checklist.decision_accuracy",
    ("checklist", "false_complete_rate"): "checklist.false_complete_rate",
    ("checklist", "unknown_sample_false_complete_rate"): "checklist.unknown_sample_false_complete_rate",
    ("checklist", "unknown_recall"): "checklist.unknown_recall",
    ("checklist", "explicit_coverage"): "checklist.explicit_coverage",
    ("checklist", "explicit_subset_accuracy"): "checklist.explicit_subset_accuracy",
    ("checklist", "missing_set", "precision"): "checklist.missing_set_precision",
    ("checklist", "missing_set", "recall"): "checklist.missing_set_recall",
    ("checklist", "missing_set", "f1"): "checklist.missing_set_f1",
    ("checklist", "set_exact_match"): "checklist.set_exact_match",
    ("checklist", "json_valid_rate"): "checklist.json_valid_rate",
    ("checklist", "retry_rate"): "checklist.retry_rate",
    ("checklist", "failure_rate"): "checklist.failure_rate",
    ("presence", "accuracy"): "presence.accuracy",
    ("pair_consistency", "rate"): "pair_consistency",
}

METHOD_ROWS = (
    ("三状态准确率", ("checklist", "state_accuracy", "overall")),
    ("宏平均 F1", ("checklist", "macro_f1")),
    ("决策准确率", ("checklist", "decision_accuracy")),
    ("误报齐全率", ("checklist", "false_complete_rate")),
    ("unknown 样本误判齐全率", ("checklist", "unknown_sample_false_complete_rate")),
    ("unknown 召回", ("checklist", "unknown_recall")),
    ("明确回答覆盖率", ("checklist", "explicit_coverage")),
    ("明确回答子集准确率", ("checklist", "explicit_subset_accuracy")),
    ("缺失集合精确率", ("checklist", "missing_set", "precision")),
    ("缺失集合召回率", ("checklist", "missing_set", "recall")),
    ("缺失集合 F1", ("checklist", "missing_set", "f1")),
    ("集合完全匹配率", ("checklist", "set_exact_match")),
    ("JSON 有效率", ("checklist", "json_valid_rate")),
    ("重试率", ("checklist", "retry_rate")),
    ("失败率", ("checklist", "failure_rate")),
    ("presence 准确率", ("presence", "accuracy")),
    ("成对一致性", ("pair_consistency", "rate")),
)

PENDING = "待测"
UNMEASURED = "未测"


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(value) for value in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_ratio(item: dict | None) -> str:
    if not isinstance(item, dict) or "value" not in item or "denominator" not in item:
        return PENDING
    if item["denominator"] == 0 or item["value"] is None:
        return "无定义（分母为 0）"
    return f"{item['value']:.4f} ({item['numerator']}/{item['denominator']})"


def format_macro_f1(item: dict | None) -> str:
    if not isinstance(item, dict) or item.get("value") is None:
        return "无定义（分母为 0）" if isinstance(item, dict) else PENDING
    defined = item.get("n_classes_defined")
    total = item.get("n_classes")
    return f"{item['value']:.4f}（{defined}/{total} 类）"


def format_ci(interval: dict | None) -> str:
    if not isinstance(interval, dict) or interval.get("ci_low") is None or interval.get("ci_high") is None:
        return ""
    text = f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
    if interval.get("insufficient_samples"):
        text += "（样本不足）"
    return text


def _dig(payload: dict, path: tuple[str, ...]):
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _fmt_stat(stats: dict | None) -> str:
    if not isinstance(stats, dict) or not stats.get("n"):
        return UNMEASURED
    return f"n={stats['n']} mean={stats['mean']:.4f} p50={stats['p50']:.4f} p95={stats['p95']:.4f}"


def _fmt_optional_number(value, digits: int = 4) -> str:
    if value is None:
        return UNMEASURED
    return f"{value:.{digits}f}"


def render_metrics_md(metrics: dict) -> str:
    """人能读的指标表。数字全部来自 metrics 字典。"""
    lines = [
        f"# 评测结果 {metrics.get('experiment_id') or ''}".rstrip(),
        "",
        "本文件由评测程序生成。没有测到的项目写“待测”或“未测”，不手填数字。",
        "",
        "## 设置",
        "",
        f"- 样本数：{metrics.get('n_records')}（checklist {metrics.get('n_checklist')}，presence {metrics.get('n_presence')}）",
        f"- 预测缺失：{metrics.get('n_missing_predictions')}",
        f"- 量化：{metrics.get('quantization') or '未标注'}",
        f"- 输入像素预算边长：{metrics.get('pixel_budget') or '未标注'}",
        "",
        "## 主任务",
        "",
        "| 指标 | 结果 | 95% 区间 |",
        "| --- | --- | --- |",
    ]
    bootstrap = (metrics.get("bootstrap") or {}).get("metrics") or {}
    checklist = metrics.get("checklist") or {}
    for label, path in METHOD_ROWS:
        if path[0] != "checklist":
            continue
        value = _dig({"checklist": checklist}, path)
        shown = format_macro_f1(value) if path == ("checklist", "macro_f1") else format_ratio(value)
        lines.append(f"| {label} | {shown} | {format_ci(bootstrap.get(_BOOTSTRAP_KEYS.get(path)))} |")

    lines += ["", "### 逐类三状态准确率", "", "| 类别 | 准确率 |", "| --- | --- |"]
    per_category = (checklist.get("state_accuracy") or {}).get("per_category") or {}
    for category in CATEGORY_IDS:
        lines.append(f"| {NAME_ZH.get(category, category)} `{category}` | {format_ratio(per_category.get(category))} |")

    lines += ["", "### 状态混淆矩阵", "", _matrix_md(checklist.get("confusion_matrix"))]
    lines += ["", "### 决策混淆矩阵", "", _matrix_md(checklist.get("decision_confusion_matrix"))]
    lines += ["", "### 缺失集合（逐类）", "", "| 类别 | 精确率 | 召回率 | F1 | TP/FP/FN |", "| --- | --- | --- | --- | --- |"]
    missing = (checklist.get("missing_set") or {}).get("per_category") or {}
    for category in CATEGORY_IDS:
        item = missing.get(category) or {}
        counts = f"{item.get('tp', '待测')}/{item.get('fp', '待测')}/{item.get('fn', '待测')}"
        lines.append(
            f"| {NAME_ZH.get(category, category)} | {format_ratio(item.get('precision'))} | "
            f"{format_ratio(item.get('recall'))} | {format_ratio(item.get('f1'))} | {counts} |"
        )

    presence = metrics.get("presence") or {}
    lines += [
        "",
        "## 辅助任务 presence",
        "",
        f"- 准确率：{format_ratio(presence.get('accuracy'))}",
        f"- item 与 target 不一致：{presence.get('item_mismatch', '待测')}",
        f"- JSON 有效率：{format_ratio(presence.get('json_valid_rate'))}",
        f"- 重试率：{format_ratio(presence.get('retry_rate'))}",
        f"- 失败率：{format_ratio(presence.get('failure_rate'))}",
        "",
        _matrix_md(presence.get("confusion_matrix")),
        "",
        "## 成对一致性",
        "",
        f"- {format_ratio((metrics.get('pair_consistency') or {}).get('rate'))}",
        f"- 跳过的组：{(metrics.get('pair_consistency') or {}).get('n_skipped_groups', 0)}",
        "",
        "## 按条件分组",
        "",
        "一个样本可以带多个 condition_tags，会同时出现在多个组里。场景组少于 10 个时标注样本不足。",
        "",
        "| 条件 | 样本数 | 场景组 | 三状态准确率 | 误报齐全率 | 决策准确率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for tag, block in (metrics.get("by_condition") or {}).items():
        block_checklist = block.get("checklist") or {}
        flag = "（样本不足）" if block.get("insufficient_samples") else ""
        lines.append(
            f"| `{tag}` | {block.get('n_records')} | {block.get('n_scene_groups')}{flag} | "
            f"{format_ratio((block_checklist.get('state_accuracy') or {}).get('overall'))} | "
            f"{format_ratio(block_checklist.get('false_complete_rate'))} | "
            f"{format_ratio(block_checklist.get('decision_accuracy'))} |"
        )
    if not metrics.get("by_condition"):
        lines.append("| 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |")

    latency = metrics.get("latency") or {}
    throughput = metrics.get("throughput") or {}
    lines += [
        "",
        "## 速度",
        "",
        "预热条数按预测文件顺序从前面丢掉，不进入下面的 n。请求吞吐和 token 吞吐分开写。",
        "",
        f"- 丢掉的预热条数：{latency.get('warmup_excluded', 0)}",
        f"- 端到端 total_s：{_fmt_stat(latency.get('total_s'))}",
        f"- 首 token 延迟（秒）：{_fmt_stat(latency.get('first_token_s'))}",
        f"- 生成耗时 generate_s：{_fmt_stat(latency.get('generate_s'))}",
        f"- 完整点检请求数/秒：{_fmt_optional_number((throughput.get('checklist_requests_per_s') or {}).get('value'))}",
        f"- 生成 token/s：{_fmt_optional_number((throughput.get('generated_tokens_per_s') or {}).get('value'))}",
        f"- 摄像头显示 FPS：{(throughput.get('camera_fps') or {}).get('status', UNMEASURED)}",
        "",
        "## 资源",
        "",
    ]
    resources = metrics.get("resources") or {"status": UNMEASURED}
    if resources.get("status") == "未测":
        lines.append("未提供 tegrastats 日志，内存、GPU 利用率和功耗均为未测。")
    else:
        ram = resources.get("ram_used_mb") or {}
        gr3d = resources.get("gr3d_freq_pct") or {}
        lines.append(f"- 样本行数：{resources.get('n')}")
        lines.append(f"- RAM 已用 MB：均值 {_fmt_optional_number(ram.get('mean'), 1)}，峰值 {_fmt_optional_number(ram.get('peak'), 1)}")
        lines.append(f"- GR3D_FREQ %：均值 {_fmt_optional_number(gr3d.get('mean'), 1)}，峰值 {_fmt_optional_number(gr3d.get('peak'), 1)}")
        lines.append("- 功率通道见 metrics.json 的 resources.power_mw。日志里没有的通道保持为空，不估算。")
    bootstrap_meta = metrics.get("bootstrap") or {}
    lines += [
        "",
        "## 区间",
        "",
        f"- 方法：按 scene_group_id 聚类 bootstrap，{bootstrap_meta.get('n_iterations', '待测')} 次，种子 {bootstrap_meta.get('seed', '待测')}",
        f"- 场景组数：{bootstrap_meta.get('n_groups', '待测')}",
        f"- 样本不足：{'是' if bootstrap_meta.get('insufficient_samples') else '否'}",
        "",
        "## 计算方法说明",
        "",
    ]
    for note in metrics.get("methodology_notes") or METHODOLOGY_NOTES:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _matrix_md(confusion: dict | None) -> str:
    if not isinstance(confusion, dict) or not confusion.get("gold_labels"):
        return "待测"
    pred_labels = confusion["pred_labels"]
    header = "| 真值 \\ 预测 | " + " | ".join(pred_labels) + " |"
    sep = "| --- | " + " | ".join("---" for _ in pred_labels) + " |"
    rows = [header, sep]
    for gold, counts in zip(confusion["gold_labels"], confusion["matrix"]):
        rows.append("| " + gold + " | " + " | ".join(str(value) for value in counts) + " |")
    return "\n".join(rows)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_latency_hist(values: list[float], path: Path) -> None:
    if not values:
        return
    plt = _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    bins = min(20, max(5, len(values)))
    axis.hist(values, bins=bins, color="#3b6ea5", edgecolor="white")
    axis.set_xlabel("total_s")
    axis.set_ylabel("count")
    axis.set_title("latency distribution")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_false_complete(rows: list[tuple[str, float]], path: Path) -> None:
    """柱状图只画 metrics.json 里已经有数值的实验。"""
    if not rows:
        return
    plt = _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(6.8, 4.0))
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    axis.bar(labels, values, color="#b4534b")
    axis.set_ylim(0, 1)
    axis.set_ylabel("false complete rate")
    axis.set_title("false complete rate")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_latency_bars(rows: list[tuple[str, float]], path: Path) -> None:
    if not rows:
        return
    plt = _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    axis.bar(labels, values, color="#3b6ea5")
    axis.set_ylabel("p50 total_s")
    axis.set_title("latency p50")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _latency_values(pred_rows: list[dict], warmup: int) -> list[float]:
    values = []
    for record in pred_rows[warmup:]:
        timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
        total = timings.get("total_s")
        if isinstance(total, (int, float)):
            values.append(float(total))
    return values


def build_metrics_payload(
    gold_rows: list[dict],
    pred_rows: list[dict],
    *,
    experiment_id: str | None = None,
    quantization: str | None = None,
    pixel_budget: int | None = None,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    warmup: int = 0,
    tegrastats_text: str | None = None,
    tegrastats_start_index: int = 0,
    tegrastats_end_index: int | None = None,
) -> tuple[dict, list[dict]]:
    """计算全部产物内容。返回 (metrics 字典, per_sample 行)。"""
    aligned = align(gold_rows, pred_rows)
    evaluated = evaluate(aligned)
    bootstrap = cluster_bootstrap_many(
        aligned,
        scalar_metrics,
        n_iterations=n_iterations,
        seed=seed,
        confidence=confidence,
        min_groups=MIN_SCENE_GROUPS,
    )
    n_groups = 0 if not bootstrap else next(iter(bootstrap.values()))["n_groups"]
    insufficient = True if not bootstrap else next(iter(bootstrap.values()))["insufficient_samples"]
    resources: dict
    if tegrastats_text is None:
        resources = {"status": "未测", "note": "未提供 tegrastats 日志"}
    else:
        parsed = parse_tegrastats_log(tegrastats_text)
        resources = summarize_tegrastats(parsed, start_index=tegrastats_start_index, end_index=tegrastats_end_index)
        resources["n_lines_parsed"] = len(parsed)
    payload = {
        "experiment_id": experiment_id,
        "quantization": quantization,
        "pixel_budget": pixel_budget,
        "n_records": evaluated["n_records"],
        "n_checklist": evaluated["n_checklist"],
        "n_presence": evaluated["n_presence"],
        "n_untagged": evaluated["n_untagged"],
        "n_missing_predictions": sum(1 for row in aligned if row["missing_prediction"]),
        "checklist": evaluated["checklist"],
        "presence": evaluated["presence"],
        "pair_consistency": evaluated["pair_consistency"],
        "by_condition": evaluated["by_condition"],
        "bootstrap": {
            "n_iterations": n_iterations,
            "seed": seed,
            "confidence": confidence,
            "n_groups": n_groups,
            "insufficient_samples": insufficient,
            "min_groups": MIN_SCENE_GROUPS,
            "metrics": bootstrap,
        },
        "latency": latency_from_predictions(pred_rows, warmup=warmup),
        "throughput": throughput_from_predictions(pred_rows, warmup=warmup),
        "resources": resources,
        "methodology_notes": list(METHODOLOGY_NOTES),
    }
    per_sample = [explain_sample(row) for row in aligned]
    return payload, per_sample


def write_eval_dir(payload: dict, per_sample: list[dict], out_dir: Path, pred_rows: list[dict] | None = None, *, make_plot: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", payload)
    with (out_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")
    (out_dir / "metrics.md").write_text(render_metrics_md(payload), encoding="utf-8")
    if make_plot and pred_rows is not None:
        warmup = int((payload.get("latency") or {}).get("warmup_excluded") or 0)
        plot_latency_hist(_latency_values(pred_rows, warmup), out_dir / "latency_hist.png")


def load_eval_config(path: str | Path | None) -> dict:
    if path is None:
        return {}
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"评测配置必须是映射: {path}")
    return loaded


def run_eval(
    gold_path: str | Path,
    pred_path: str | Path,
    out_dir: str | Path,
    *,
    experiment_id: str | None = None,
    quantization: str | None = None,
    pixel_budget: int | None = None,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    warmup: int = 0,
    tegrastats_path: str | Path | None = None,
    tegrastats_start_index: int = 0,
    tegrastats_end_index: int | None = None,
    make_plot: bool = True,
) -> dict:
    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(pred_path)
    tegrastats_text = None
    if tegrastats_path is not None:
        tegrastats_text = Path(tegrastats_path).read_text(encoding="utf-8")
    out = Path(out_dir)
    if experiment_id is None:
        experiment_id = out.parent.name if out.name == "eval" else out.name
    payload, per_sample = build_metrics_payload(
        gold_rows,
        pred_rows,
        experiment_id=experiment_id,
        quantization=quantization,
        pixel_budget=pixel_budget,
        n_iterations=n_iterations,
        seed=seed,
        confidence=confidence,
        warmup=warmup,
        tegrastats_text=tegrastats_text,
        tegrastats_start_index=tegrastats_start_index,
        tegrastats_end_index=tegrastats_end_index,
    )
    write_eval_dir(payload, per_sample, out, pred_rows, make_plot=make_plot)
    return payload


def _cell_ratio(metrics: dict | None, path: tuple[str, ...], *, macro: bool = False) -> str:
    if metrics is None:
        return PENDING
    value = _dig(metrics, path)
    if macro:
        return format_macro_f1(value)
    return format_ratio(value)


def _lookup_method(runs: list[dict], experiment_id: str) -> dict | None:
    matched = [run for run in runs if run.get("experiment_id") == experiment_id]
    if len(matched) > 1:
        raise ValueError(f"experiment_id 重复，无法比较: {experiment_id}")
    return matched[0] if matched else None


def render_comparison_md(runs: list[dict]) -> str:
    header = "| 指标 | " + " | ".join(METHOD_ORDER) + " |"
    sep = "| --- | " + " | ".join("---" for _ in METHOD_ORDER) + " |"
    lines = [
        "# 方法对比",
        "",
        "单元格只来自各次运行的 metrics.json。没有对应实验时写“待测”。",
        "",
        header,
        sep,
    ]
    for label, path in METHOD_ROWS:
        cells = []
        for experiment_id in METHOD_ORDER:
            run = _lookup_method(runs, experiment_id)
            cells.append(_cell_ratio(run, path, macro=path == ("checklist", "macro_f1")))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 端侧优化矩阵",
        "",
        "行是量化，列是输入像素预算的边长。单元格依次为三状态准确率、误报齐全率、total_s 的 p50、RAM 已用均值。",
        "",
        "| 量化 \\ 像素 | " + " | ".join(f"{side}²" for side in PIXEL_ORDER) + " |",
        "| --- | " + " | ".join("---" for _ in PIXEL_ORDER) + " |",
    ]
    index = {(run.get("quantization"), run.get("pixel_budget")): run for run in runs if run.get("quantization") and run.get("pixel_budget")}
    seen = {}
    for run in runs:
        key = (run.get("quantization"), run.get("pixel_budget"))
        if key[0] is None or key[1] is None:
            continue
        if key in seen:
            raise ValueError(f"量化与像素预算的组合重复: {key}")
        seen[key] = run
    for quant in QUANT_ORDER:
        cells = []
        for pixel in PIXEL_ORDER:
            run = seen.get((quant, pixel))
            cells.append(PENDING if run is None else _optimization_cell(run))
        lines.append("| " + quant + " | " + " | ".join(cells) + " |")
    others = [run for run in runs if run.get("experiment_id") not in METHOD_ORDER]
    if others:
        lines += ["", "## 其他实验", ""]
        for run in others:
            lines.append(
                f"- `{run.get('experiment_id')}`：三状态准确率 "
                f"{format_ratio(_dig(run, ('checklist', 'state_accuracy', 'overall')))}，误报齐全率 "
                f"{format_ratio(_dig(run, ('checklist', 'false_complete_rate')))}"
            )
    lines.append("")
    return "\n".join(lines)


def _optimization_cell(run: dict) -> str:
    accuracy = format_ratio(_dig(run, ("checklist", "state_accuracy", "overall")))
    false_complete = format_ratio(_dig(run, ("checklist", "false_complete_rate")))
    p50 = _dig(run, ("latency", "total_s", "p50"))
    ram = _dig(run, ("resources", "ram_used_mb", "mean"))
    latency = UNMEASURED if p50 is None else f"{p50:.4f}s"
    memory = UNMEASURED if ram is None else f"{ram:.1f}MB"
    return f"{accuracy} / {false_complete} / {latency} / {memory}"


def _write_method_csv(runs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *METHOD_ORDER])
        for label, metric_path in METHOD_ROWS:
            row = [label]
            for experiment_id in METHOD_ORDER:
                run = _lookup_method(runs, experiment_id)
                row.append(_cell_ratio(run, metric_path, macro=metric_path == ("checklist", "macro_f1")))
            writer.writerow(row)


def _write_optimization_csv(runs: list[dict], path: Path) -> None:
    seen = {}
    for run in runs:
        key = (run.get("quantization"), run.get("pixel_budget"))
        if key[0] is None or key[1] is None:
            continue
        if key in seen:
            raise ValueError(f"量化与像素预算的组合重复: {key}")
        seen[key] = run
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "quantization",
                "pixel_budget",
                "state_accuracy",
                "false_complete_rate",
                "latency_p50_s",
                "ram_used_mb_mean",
            ]
        )
        for quant in QUANT_ORDER:
            for pixel in PIXEL_ORDER:
                run = seen.get((quant, pixel))
                if run is None:
                    writer.writerow([quant, pixel, PENDING, PENDING, PENDING, PENDING])
                    continue
                p50 = _dig(run, ("latency", "total_s", "p50"))
                ram = _dig(run, ("resources", "ram_used_mb", "mean"))
                writer.writerow(
                    [
                        quant,
                        pixel,
                        format_ratio(_dig(run, ("checklist", "state_accuracy", "overall"))),
                        format_ratio(_dig(run, ("checklist", "false_complete_rate"))),
                        PENDING if p50 is None else p50,
                        PENDING if ram is None else ram,
                    ]
                )


def _false_complete_points(runs: list[dict]) -> list[tuple[str, float]]:
    points = []
    for experiment_id in METHOD_ORDER:
        run = _lookup_method(runs, experiment_id)
        value = None if run is None else _dig(run, ("checklist", "false_complete_rate", "value"))
        if isinstance(value, (int, float)):
            points.append((experiment_id, float(value)))
    return points


def _latency_points(runs: list[dict]) -> list[tuple[str, float]]:
    points = []
    for run in runs:
        p50 = _dig(run, ("latency", "total_s", "p50"))
        if not isinstance(p50, (int, float)):
            continue
        label = str(run.get("experiment_id") or "run")
        if run.get("quantization") and run.get("pixel_budget"):
            label = f"{run['quantization']}/{run['pixel_budget']}"
        points.append((label, float(p50)))
    return points


def write_comparison(runs: list[dict], out_dir: Path, *, make_plot: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.md").write_text(render_comparison_md(runs), encoding="utf-8")
    _write_method_csv(runs, out_dir / "comparison_methods.csv")
    _write_optimization_csv(runs, out_dir / "comparison_optimization.csv")
    if make_plot:
        plot_false_complete(_false_complete_points(runs), out_dir / "false_complete_rate.png")
        plot_latency_bars(_latency_points(runs), out_dir / "latency_p50.png")


def load_metrics_files(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "checklist" not in payload:
            raise ValueError(f"{path} 不是 eval 写出的 metrics.json")
        runs.append(payload)
    return runs
