"""用 TransformersRunner 对 JSONL 逐条推理。

可选加载 LoRA adapter（peft 包一层后作为 model 传给 TransformersRunner）。
每行是 {"sample_id": ..., **PredictResult.to_dict()}。presence 样本会带上 task 和 target_item。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import PeftModel
from transformers import AutoProcessor

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.inference.transformers_runner import TransformersRunner
from qwen_vlm.training.dataset import read_jsonl, resolve_image_path, resolve_pixel_budget
from qwen_vlm.training.lora import load_base_model
from qwen_vlm.training.resources import disk_guard, wait_for_gpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformers 批量推理，写出 predictions.jsonl")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model-dir", default="artifacts/base/Qwen3-VL-2B-Instruct")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = PROJECT_ROOT
    disk_guard(root)
    wait_for_gpu()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = root / jsonl_path
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    output = Path(args.output) if args.output else root / "runs" / args.experiment_id / "predictions.jsonl"
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    max_pixels, min_pixels = resolve_pixel_budget(processor, args.max_pixels, args.min_pixels)
    model = load_base_model(model_dir)
    if args.adapter:
        adapter = Path(args.adapter)
        if not adapter.is_absolute():
            adapter = root / adapter
        model = PeftModel.from_pretrained(model, adapter).to("cuda")
    model.eval()
    runner = TransformersRunner(
        model=model,
        processor=processor,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        max_new_tokens=args.max_new_tokens,
        device="cuda",
    )

    samples = read_jsonl(jsonl_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    with output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            task = sample.get("task", "checklist")
            result = runner.predict(
                resolve_image_path(sample, data_root),
                sample["question_zh"],
                sample["required_items"],
                task=task,
                target_item=sample.get("target_item"),
            )
            record = {"sample_id": sample["sample_id"], **result.to_dict()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(samples)}] {sample['sample_id']} task={task} parse={result.parse_status}")
    print(f"写出 {len(samples)} 行到 {output}")
    del runner, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
