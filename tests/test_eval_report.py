"""用命令行把夹具跑通，并检查对比表里的数字来自 metrics.json。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "envs" / "qwen_vlm" / "bin" / "python"
FIXTURE = ROOT / "tests" / "fixtures" / "eval"
OUT = FIXTURE / "e2e_out"


def _run(args: list[str]) -> None:
    command = " ".join(["cd", str(ROOT), "&&", "source", "env/activate.sh", "&&", str(PYTHON), *args])
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")


def test_eval_predictions_cli_writes_the_hand_counted_rates():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    try:
        eval_dir = OUT / "fixture_not_a_result" / "eval"
        _run(
            [
                "scripts/eval_predictions.py",
                "--gold",
                str(FIXTURE / "gold.jsonl"),
                "--pred",
                str(FIXTURE / "pred.jsonl"),
                "--out",
                str(eval_dir),
                "--experiment-id",
                "fixture_not_a_result",
                "--bootstrap-iters",
                "20",
                "--seed",
                "0",
                "--tegrastats",
                str(FIXTURE / "tegrastats_jp6.log"),
                "--config",
                str(ROOT / "configs" / "infer" / "eval_default.yaml"),
            ]
        )
        metrics_path = eval_dir / "metrics.json"
        per_sample_path = eval_dir / "per_sample.jsonl"
        markdown_path = eval_dir / "metrics.md"
        plot_path = eval_dir / "latency_hist.png"
        assert metrics_path.is_file() and per_sample_path.is_file() and markdown_path.is_file()
        assert plot_path.read_bytes().startswith(b"\x89PNG")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["checklist"]["false_complete_rate"]["numerator"] == 1
        assert metrics["checklist"]["false_complete_rate"]["denominator"] == 4
        assert metrics["checklist"]["state_accuracy"]["overall"]["numerator"] == 25
        assert metrics["checklist"]["state_accuracy"]["overall"]["denominator"] == 40
        assert metrics["bootstrap"]["insufficient_samples"] is True
        assert metrics["bootstrap"]["metrics"]["checklist.false_complete_rate"]["point_estimate"] == 0.25
        assert metrics["resources"]["ram_used_mb"]["mean"] == 3060.5
        assert sum(1 for line in per_sample_path.read_text(encoding="utf-8").splitlines() if line) == 10
        text = markdown_path.read_text(encoding="utf-8")
        assert "误报齐全率" in text
        assert "0.2500 (1/4)" in text
        assert "0.6250 (25/40)" in text
        assert "样本不足" in text

        method_dir = OUT / "B0" / "eval"
        opt_dir = OUT / "opt_q4_512" / "eval"
        _run(
            [
                "scripts/eval_predictions.py",
                "--gold",
                str(FIXTURE / "gold.jsonl"),
                "--pred",
                str(FIXTURE / "pred.jsonl"),
                "--out",
                str(method_dir),
                "--experiment-id",
                "B0",
                "--bootstrap-iters",
                "5",
                "--no-plot",
            ]
        )
        _run(
            [
                "scripts/eval_predictions.py",
                "--gold",
                str(FIXTURE / "gold.jsonl"),
                "--pred",
                str(FIXTURE / "pred.jsonl"),
                "--out",
                str(opt_dir),
                "--experiment-id",
                "opt_q4_512",
                "--quantization",
                "Q4_K_M",
                "--pixel-budget",
                "512",
                "--bootstrap-iters",
                "5",
                "--no-plot",
            ]
        )
        compare_dir = OUT / "comparison"
        _run(
            [
                "scripts/eval_compare.py",
                "--metrics",
                str(method_dir / "metrics.json"),
                str(opt_dir / "metrics.json"),
                "--out",
                str(compare_dir),
            ]
        )
        comparison = (compare_dir / "comparison.md").read_text(encoding="utf-8")
        assert "0.2500 (1/4)" in comparison
        assert "| 误报齐全率 | 0.2500 (1/4) | 待测 | 待测 | 待测 | 待测 | 待测 |" in comparison
        assert (compare_dir / "false_complete_rate.png").read_bytes().startswith(b"\x89PNG")
        csv_text = (compare_dir / "comparison_optimization.csv").read_text(encoding="utf-8")
        assert "Q4_K_M,512,0.6250 (25/40),0.2500 (1/4)" in csv_text
        assert "F16,384,待测,待测,待测,待测" in csv_text
        assert "待测" in (compare_dir / "comparison_methods.csv").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(OUT, ignore_errors=True)
