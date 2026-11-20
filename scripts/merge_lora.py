"""把 LoRA adapter 合并进基座，存成带 processor 的 Hugging Face 目录。

用同一条固定输入比较合并前（基座加 adapter）和合并后的贪心文本，并记录 logits 最大绝对误差。
冒烟时加上 --delete-merged，检查写完后删掉完整模型，只留 merge_check.json。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoProcessor

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.prompts import checklist_user_prompt, presence_user_prompt
from qwen_vlm.inference.base import load_image
from qwen_vlm.inference.transformers_runner import TransformersRunner
from qwen_vlm.training.dataset import read_jsonl, resolve_image_path, resolve_pixel_budget
from qwen_vlm.training.lora import load_base_model
from qwen_vlm.training.resources import disk_guard, free_bytes, wait_for_gpu

SMALL_FILES = (
    "chat_template.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "video_preprocessor_config.json",
    "generation_config.json",
    "processor_config.json",
)
FORWARD_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")


def _user_text(sample: dict) -> str:
    if sample.get("task", "checklist") == "presence":
        return presence_user_prompt(sample["target_item"], sample["question_zh"])
    return checklist_user_prompt(sample["question_zh"], sample["required_items"])


def _eval(model, processor, sample, image, max_pixels: int, min_pixels: int, max_new_tokens: int):
    model.eval()
    runner = TransformersRunner(
        model=model,
        processor=processor,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        max_new_tokens=max_new_tokens,
        device="cuda",
    )
    task = sample.get("task", "checklist")
    result = runner.predict(
        image,
        sample["question_zh"],
        sample["required_items"],
        task=task,
        target_item=sample.get("target_item"),
    )
    inputs = runner.prepare_inputs(load_image(image), _user_text(sample))
    payload = {key: inputs[key] for key in FORWARD_KEYS if key in inputs}
    with torch.inference_mode():
        logits = model(**payload).logits.float()
    return result.raw_text, logits


def _copy_missing(base_dir: Path, merged_dir: Path) -> list[str]:
    copied = []
    for name in SMALL_FILES:
        source = base_dir / name
        target = merged_dir / name
        if source.is_file() and not target.is_file():
            shutil.copy2(source, target)
            copied.append(name)
    return copied


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"logits 形状不一致: {tuple(left.shape)} vs {tuple(right.shape)}")
    return float((left - right).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA 并检查与合并前的贪心输出是否一致")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default="artifacts/base/Qwen3-VL-2B-Instruct")
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--merge-check", required=True)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--delete-merged", action="store_true")
    args = parser.parse_args()

    root = PROJECT_ROOT
    disk_guard(root)
    merged_dir = Path(args.merged_dir)
    if not merged_dir.is_absolute():
        merged_dir = root / merged_dir
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    needed = 5 * 1024**3 + int(4.8 * 1024**3)
    if free_bytes(merged_dir.parent) < needed:
        raise SystemExit(f"{merged_dir.parent} 的可用空间不足以写入合并模型并保留 5 GiB。")
    wait_for_gpu()

    base_dir = Path(args.base_model)
    if not base_dir.is_absolute():
        base_dir = root / base_dir
    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_absolute():
        adapter_dir = root / adapter_dir
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    check_path = Path(args.merge_check)
    if not check_path.is_absolute():
        check_path = root / check_path

    sample = read_jsonl(args.sample_jsonl if Path(args.sample_jsonl).is_absolute() else root / args.sample_jsonl)[0]
    image = resolve_image_path(sample, data_root)
    processor = AutoProcessor.from_pretrained(base_dir, local_files_only=True)
    max_pixels, min_pixels = resolve_pixel_budget(processor, args.max_pixels, args.min_pixels)

    from peft import PeftModel

    base = load_base_model(base_dir)
    peft_model = PeftModel.from_pretrained(base, adapter_dir).to("cuda").eval()
    text_before, logits_before = _eval(peft_model, processor, sample, image, max_pixels, min_pixels, args.max_new_tokens)
    logits_before = logits_before.cpu()

    merged = peft_model.merge_and_unload()
    merged.eval()
    text_merged, logits_merged = _eval(merged, processor, sample, image, max_pixels, min_pixels, args.max_new_tokens)
    in_memory_error = _max_abs(logits_before, logits_merged.cpu())
    del logits_merged

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)
    copied = _copy_missing(base_dir, merged_dir)
    saved_files = sorted(path.name for path in merged_dir.iterdir())
    del merged, peft_model, base
    torch.cuda.empty_cache()

    reloaded = load_base_model(merged_dir)
    text_reloaded, logits_reloaded = _eval(reloaded, processor, sample, image, max_pixels, min_pixels, args.max_new_tokens)
    reload_error = _max_abs(logits_before, logits_reloaded.cpu())
    del logits_reloaded, logits_before, reloaded
    torch.cuda.empty_cache()

    text_match = text_before == text_merged == text_reloaded
    payload = {
        "adapter_dir": str(adapter_dir),
        "base_model_dir": str(base_dir),
        "merged_dir": str(merged_dir),
        "sample_id": sample["sample_id"],
        "task": sample.get("task", "checklist"),
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
        "text_before": text_before,
        "text_merged_in_memory": text_merged,
        "text_reloaded": text_reloaded,
        "greedy_text_match": text_match,
        "logits_max_abs_error_in_memory": in_memory_error,
        "logits_max_abs_error_reloaded": reload_error,
        "copied_small_files": copied,
        "saved_files": saved_files,
        "merged_deleted": False,
    }
    if args.delete_merged:
        shutil.rmtree(merged_dir)
        payload["merged_deleted"] = True
        payload["merged_dir_exists_after_delete"] = merged_dir.exists()
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("greedy_text_match", "logits_max_abs_error_in_memory", "logits_max_abs_error_reloaded", "merged_deleted")}, ensure_ascii=False))
    if not text_match:
        raise SystemExit("合并前后的贪心文本不一致")


if __name__ == "__main__":
    main()
