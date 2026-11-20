#!/usr/bin/env python3
"""依次训练和预测三折，按真实金标准顺序汇总折外预测。"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.data.cv import FIXED_CV3_TEST_GROUPS, REAL_SOURCE, SYNTHETIC_SOURCE
from qwen_vlm.data.io import load_jsonl, sha256_file, write_jsonl
from qwen_vlm.training.config import load_train_config

EXPERIMENTS = ("B1", "M", "A1", "A2", "A3", "A4")
REAL_ONLY = {"B1", "A4"}


def _index(path: Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    by_id = {row["sample_id"]: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError(f"{path} 为空或含重复 sample_id")
    return by_id


def _check_folds(
    splits: Path, train_name: str, gold: dict[str, dict], gold_hash: str
) -> list[tuple[Path, Path, Path]]:
    seen = set()
    folds = []
    for number, expected_groups in FIXED_CV3_TEST_GROUPS.items():
        directory = splits / f"fold{number}"
        train_path, test_path = directory / train_name, directory / "test.jsonl"
        manifest = directory / "manifest.json"
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record["source_sha256"] != gold_hash or any(
            record["files"][path.stem]["sha256"] != sha256_file(path)
            for path in (train_path, test_path)
        ):
            raise ValueError(f"{directory} 的 JSONL 与 manifest 不一致")
        train, test = _index(train_path), _index(test_path)
        train_groups = {row["scene_group_id"] for row in train.values()}
        test_groups = {row["scene_group_id"] for row in test.values()}
        if test_groups != set(expected_groups) or train_groups & test_groups or train.keys() & test.keys():
            raise ValueError(f"{directory} 的训练或测试分组与三折协议冲突")
        if train_name == "train_real.jsonl" and any(row["source"] == SYNTHETIC_SOURCE for row in train.values()):
            raise ValueError(f"{train_path} 中包含 synthetic_edit")
        if seen.intersection(test) or any(
            row["source"] != REAL_SOURCE or row != gold.get(sample_id)
            for sample_id, row in test.items()
        ):
            raise ValueError(f"{test_path} 的测试样本重复或与真实金标准不一致")
        seen.update(test)
        folds.append((train_path, test_path, manifest))
    if seen != gold.keys():
        raise ValueError("三折测试集未覆盖完整金标准")
    return folds


def _run(script: str, *args: str | Path) -> None:
    subprocess.run(
        [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / script), *map(str, args)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _predict_fold(
    run_dir: Path, config: Path, max_steps: int, train: Path, test: Path, split_manifest: Path
) -> dict[str, dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = run_dir / "adapter"
    manifest_path = run_dir / "run_manifest.json"
    output = run_dir / "predictions.jsonl"
    test_ids = _index(test).keys()
    expected_config = load_train_config(config)
    expected_config["optim"]["max_steps"] = max_steps
    expected_data = {
        "train_jsonl_sha256": sha256_file(train),
        "split_manifest_sha256": sha256_file(split_manifest),
        "val_jsonl": None,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["config"] != expected_config or any(
            previous["data"].get(key) != value for key, value in expected_data.items()
        ):
            raise ValueError(f"{run_dir} 对应的训练配置或划分已变更，请指定新的 --runs-root")
        trained = (
            previous["steps_completed"] == max_steps
            and previous["error"] is None
            and (adapter / "adapter_config.json").is_file()
            and (adapter / "adapter_model.safetensors").is_file()
        )
    else:
        trained = False
    if not trained:
        output.unlink(missing_ok=True)
        _run(
            "train.py", "--config", config, "--train-jsonl", train, "--no-validation",
            "--manifest", split_manifest, "--output-dir", run_dir, "--max-steps", str(max_steps),
        )
    if not output.exists():
        temporary = run_dir / "predictions.partial.jsonl"
        _run(
            "predict_transformers.py", "--jsonl", test, "--experiment-id", run_dir.name,
            "--adapter", adapter, "--output", temporary, "--max-pixels",
            str(expected_config["pixels"]["max_pixels"]),
            "--min-pixels", str(expected_config["pixels"]["min_pixels"]),
        )
        predictions = _index(temporary)
        if predictions.keys() != test_ids:
            raise ValueError(f"{temporary} 的 sample_id 与该折测试集不一致")
        temporary.replace(output)
    predictions = _index(output)
    if predictions.keys() != test_ids:
        raise ValueError(f"{output} 的 sample_id 与该折测试集不一致")
    checkpoints = run_dir / "checkpoints"
    if checkpoints.is_dir():
        shutil.rmtree(checkpoints)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--max-steps", type=int, required=True, help="所有实验和折共用的优化步数")
    parser.add_argument("--splits-root", type=Path, default=PROJECT_ROOT / "data/splits/cv3")
    parser.add_argument("--gold", type=Path, default=PROJECT_ROOT / "data/annotations/real_v1.jsonl")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps 必须大于 0")

    gold_path = args.gold.resolve()
    gold = _index(gold_path)
    gold_hash = sha256_file(gold_path)
    splits = args.splits_root.resolve()
    runs = args.runs_root.resolve()
    fold_runs = runs / "cv3_folds"
    logs = runs / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "cv3.lock.log").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"已有三折任务在 {runs} 下运行") from None
        for experiment in dict.fromkeys(args.experiments):
            config = PROJECT_ROOT / "configs/train" / f"{experiment}.yaml"
            train_name = "train_real.jsonl" if experiment in REAL_ONLY else "train.jsonl"
            folds = _check_folds(splits, train_name, gold, gold_hash)
            predictions = {}
            for number, (train, test, manifest) in enumerate(folds, 1):
                print(f"{experiment} fold{number}", flush=True)
                run_dir = fold_runs / f"{experiment}_fold{number}"
                predictions.update(_predict_fold(run_dir, config, args.max_steps, train, test, manifest))
            experiment_id = f"{experiment}_cv3"
            out = runs / experiment_id
            output = out / "predictions.jsonl"
            temporary = out / "predictions.partial.jsonl"
            write_jsonl(temporary, [predictions[sample_id] for sample_id in gold])
            temporary.replace(output)
            _run(
                "eval_predictions.py", "--gold", gold_path, "--pred", output, "--out", out / "eval",
                "--experiment-id", experiment_id, "--pixel-budget", "512",
            )


if __name__ == "__main__":
    main()
