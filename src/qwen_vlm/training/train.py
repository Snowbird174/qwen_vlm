"""LoRA 监督微调循环。

一个 optimizer step 由 micro-batch 加梯度累积组成。视觉分支冻结，损失只来自
assistant 回答上的 token。本文件不改模型结构。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
from pathlib import Path

import accelerate
import numpy as np
import peft
import torch
import transformers
from peft import PeftModel
from transformers import (
    AutoProcessor,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.common.prompts import PROMPT_VERSION
from qwen_vlm.common.schema import parse_model_output, validate_sample
from qwen_vlm.inference.transformers_runner import TransformersRunner
from qwen_vlm.training.collator import MultimodalCollator
from qwen_vlm.training.config import load_train_config
from qwen_vlm.training.dataset import (
    TrainDataset,
    read_jsonl,
    resolve_image_path,
    resolve_pixel_budget,
    target_text_for,
)
from qwen_vlm.training.lora import (
    LANGUAGE_LORA_TARGET_PATTERN,
    _assert_targets,
    attach_lora,
    dump_named_modules,
    keep_visual_frozen_eval,
    load_base_model,
    matched_module_names,
    prepare_for_training,
    write_trainable_params,
)
from qwen_vlm.training.mix import build_training_mix
from qwen_vlm.training.resources import cap_process_memory, disk_guard, wait_for_gpu

MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "pixel_values",
    "image_grid_thw",
    "labels",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(root: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"commit": None, "dirty": None, "note": f"无法读取 git: {exc}"}
    dirty = bool(status.strip())
    return {"commit": commit, "dirty": dirty, "note": "dirty" if dirty else "clean"}


def _base_revision(model_dir: Path) -> str:
    manifest = PROJECT_ROOT / "artifacts/base/manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("revision"):
            return payload["revision"]
    return "unknown"


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_token_id(processor) -> int:
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if pad_id is None or int(pad_id) < 0:
        raise RuntimeError("tokenizer 没有可用的 pad token")
    return int(pad_id)


def _move_batch(batch: dict, device: str) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _build_scheduler(optimizer, name: str, warmup_steps: int, max_steps: int):
    name = name.lower()
    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    if name in ("constant", "constant_with_warmup"):
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    raise ValueError(f"不认识的学习率调度: {name}")


def _environment() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "gpu": gpu,
    }


def _validate(samples: list[dict], path: Path) -> None:
    bad = []
    for sample in samples:
        errors = validate_sample(sample)
        if errors:
            bad.append({"sample_id": sample.get("sample_id"), "errors": errors})
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"{path} 有不符合 sample.schema.json 的样本: {json.dumps(bad, ensure_ascii=False)}")


class EpochCursor:
    """按 epoch 调用 build_training_mix，再逐条往外送。"""

    def __init__(self, raw, mix_cfg, seed, processor, data_root, max_pixels, min_pixels, cache, start_epoch=0, start_index=0):
        self.raw = raw
        self.mix_cfg = mix_cfg
        self.seed = seed
        self.processor = processor
        self.data_root = data_root
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.cache = cache
        self.epoch = start_epoch
        self.index = start_index
        self.dataset = None
        self.samples = None
        self.exposure_stats = None

    def _rebuild(self) -> None:
        mixed, stats = build_training_mix(self.raw, self.mix_cfg, self.seed + self.epoch)
        if not mixed:
            raise RuntimeError("mix 得到空 epoch。检查 include_unknown、aux_ratio 和训练 JSONL。")
        if self.exposure_stats is None and self.epoch == 0:
            self.exposure_stats = stats
        self.samples = mixed
        self.dataset = TrainDataset(
            mixed,
            self.processor,
            self.data_root,
            self.max_pixels,
            self.min_pixels,
            cache=self.cache,
        )
        if self.index >= len(self.dataset):
            self.epoch += 1
            self.index = 0
            self.dataset = None
            self._rebuild()

    def ensure(self) -> None:
        if self.dataset is not None and self.index < len(self.dataset):
            return
        if self.dataset is not None and self.index >= len(self.dataset):
            self.epoch += 1
            self.index = 0
            self.dataset = None
        self._rebuild()

    def take(self, count: int) -> tuple[list[dict], list[dict]]:
        features, samples = [], []
        for _ in range(count):
            self.ensure()
            features.append(self.dataset[self.index])
            samples.append(self.samples[self.index])
            self.index += 1
        return features, samples

    def position(self) -> tuple[int, int]:
        if self.dataset is not None and self.index >= len(self.dataset):
            return self.epoch + 1, 0
        return self.epoch, self.index


def _overlap(text: str, sample: dict) -> dict:
    parsed, status = parse_model_output(text, sample["task"])
    if sample["task"] == "checklist":
        matched = 0 if parsed is None else sum(parsed[item] == sample["item_states"][item] for item in CATEGORY_IDS)
        total = len(CATEGORY_IDS)
    else:
        item = sample["target_item"]
        matched = int(parsed is not None and parsed.get("item") == item and parsed.get("state") == sample["item_states"][item])
        total = 1
    return {"parse_status": status, "matched_fields": matched, "n_fields": total}


def _generate(model, processor, sample, data_root, max_pixels, min_pixels, max_new_tokens: int):
    was_training = model.training
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    previous_cache = base.config.use_cache
    base.config.use_cache = True
    if getattr(base.config, "text_config", None) is not None:
        base.config.text_config.use_cache = True
    model.eval()
    runner = TransformersRunner(
        model=model,
        processor=processor,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        max_new_tokens=max_new_tokens,
        device="cuda",
    )
    result = runner.predict(
        resolve_image_path(sample, data_root),
        sample["question_zh"],
        sample["required_items"],
        task=sample["task"],
        target_item=sample.get("target_item"),
    )
    base.config.use_cache = previous_cache
    if getattr(base.config, "text_config", None) is not None:
        base.config.text_config.use_cache = previous_cache
    if hasattr(base, "model") and hasattr(base.model, "rope_deltas"):
        base.model.rope_deltas = None
    if was_training:
        model.train()
        keep_visual_frozen_eval(model)
    return result


def _vision_diff(model, batch: dict) -> float:
    was_training = model.training
    model.eval()
    payload = {key: value for key, value in batch.items() if key != "labels"}
    with torch.inference_mode():
        real = model(**payload).logits
        blank = dict(payload)
        blank["pixel_values"] = torch.zeros_like(payload["pixel_values"])
        zeroed = model(**blank).logits
        diff = float((real - zeroed).abs().max().item())
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        if hasattr(base, "model") and hasattr(base.model, "rope_deltas"):
            base.model.rope_deltas = None
    if was_training:
        model.train()
        keep_visual_frozen_eval(model)
    if not diff > 0:
        raise RuntimeError("图像像素清零后 logits 没有变化，视觉输入没有进入前向计算")
    return diff


def _save_checkpoint(model, optimizer, scheduler, path: Path, state: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path / "rng.pt",
    )
    (path / "trainer_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _prune_checkpoints(root: Path, limit: int) -> None:
    if limit is None or limit <= 0:
        return
    folders = sorted(path for path in root.glob("step_*") if path.is_dir())
    while len(folders) > limit:
        shutil.rmtree(folders.pop(0))


def _logged_loss(log_path: Path, index: int, fallback):
    """续训时本次进程只有后半段 loss，清单里的首尾值以完整 jsonl 为准。"""
    if log_path.is_file():
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            return rows[index]["loss"]
    return fallback


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL 实验室点检 LoRA 微调")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint 目录，例如 runs/train_smoke/checkpoints/step_000015")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config)
    root = PROJECT_ROOT
    if args.max_steps is not None:
        cfg["optim"]["max_steps"] = args.max_steps
    if args.max_samples is not None:
        cfg.setdefault("data", {})["max_samples"] = args.max_samples

    experiment_id = cfg["experiment_id"]
    run_dir = _resolve(root, args.output_dir) if args.output_dir else root / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg["data"]
    train_path = _resolve(root, args.train_jsonl or data_cfg["train_jsonl"])
    val_path = _resolve(root, args.val_jsonl or data_cfg.get("val_jsonl"))
    data_root = _resolve(root, args.data_root or data_cfg.get("root") or "data")
    manifest_path = _resolve(root, args.manifest or data_cfg.get("manifest"))
    if manifest_path is None and train_path is not None:
        sibling = train_path.parent / "manifest.json"
        if sibling.is_file():
            manifest_path = sibling

    disk_guard(root)
    if not torch.cuda.is_available():
        raise SystemExit("训练需要 CUDA。")
    free_mib, _, total_mib = wait_for_gpu()
    cap_process_memory(total_mib, free_mib)
    torch.cuda.reset_peak_memory_stats()

    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    optim_cfg = cfg["optim"]
    max_steps = int(optim_cfg["max_steps"])
    micro_batch = int(optim_cfg["micro_batch_size"])
    grad_accum = int(optim_cfg["grad_accum_steps"])
    if micro_batch < 1 or grad_accum < 1:
        raise ValueError("micro_batch_size 和 grad_accum_steps 都必须 >= 1")
    warmup_steps = optim_cfg.get("warmup_steps")
    warmup_steps = int(warmup_steps) if warmup_steps is not None else int(float(optim_cfg.get("warmup_ratio") or 0) * max_steps)

    if not train_path.is_file():
        raise SystemExit(f"找不到训练集 {train_path}")
    samples = read_jsonl(train_path)
    limit = data_cfg.get("max_samples")
    if limit is not None:
        samples = samples[: int(limit)]
    _validate(samples, train_path)
    print(f"训练样本 {len(samples)} 条，来自 {train_path}")

    model_dir = _resolve(root, cfg.get("model_dir") or "artifacts/base/Qwen3-VL-2B-Instruct")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    max_pixels, min_pixels = resolve_pixel_budget(processor, cfg["pixels"]["max_pixels"], cfg["pixels"].get("min_pixels"))
    print(f"像素预算 max={max_pixels} min={min_pixels}")

    resume_path = Path(args.resume) if args.resume else None
    start_step = 0
    start_epoch = 0
    start_index = 0
    saved_exposure = None
    if resume_path is not None:
        state_path = resume_path / "trainer_state.json"
        if not state_path.is_file():
            raise SystemExit(f"断点缺少 trainer_state.json: {resume_path}")
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        start_step = int(saved["optimizer_step"])
        start_epoch = int(saved["epoch"])
        start_index = int(saved["index"])
        saved_exposure = saved.get("exposure_stats")
        base = load_base_model(model_dir)
        names = dump_named_modules(base, run_dir / "named_modules.txt")
        matched = matched_module_names(base)
        _assert_targets(names, matched)
        model = PeftModel.from_pretrained(base, resume_path, is_trainable=True).to("cuda")
        prepare_for_training(model, gradient_checkpointing=bool(optim_cfg.get("gradient_checkpointing", True)))
        trainable_summary = write_trainable_params(model, run_dir / "trainable_params.txt")
        print(f"从 step {start_step} 继续，epoch={start_epoch} index={start_index}")
    else:
        base = load_base_model(model_dir)
        model, matched, trainable_summary = attach_lora(base, cfg["lora"], run_dir)
        prepare_for_training(model, gradient_checkpointing=bool(optim_cfg.get("gradient_checkpointing", True)))

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(optim_cfg["lr"]), weight_decay=float(optim_cfg.get("weight_decay", 0.0)))
    scheduler = _build_scheduler(optimizer, optim_cfg.get("lr_scheduler", "cosine"), warmup_steps, max_steps)
    if resume_path is not None:
        optimizer.load_state_dict(torch.load(resume_path / "optimizer.pt", map_location="cuda", weights_only=False))
        scheduler.load_state_dict(torch.load(resume_path / "scheduler.pt", map_location="cpu", weights_only=False))
        rng_state = torch.load(resume_path / "rng.pt", map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    cache = {} if cfg.get("cache_encodings", False) else None
    cursor = EpochCursor(
        samples,
        cfg["mix"],
        seed,
        processor,
        data_root,
        max_pixels,
        min_pixels,
        cache,
        start_epoch=start_epoch,
        start_index=start_index,
    )
    if saved_exposure is not None:
        cursor.exposure_stats = saved_exposure
    cursor.ensure()
    print("本 epoch 暴露统计:", json.dumps(cursor.exposure_stats or {}, ensure_ascii=False))

    collator = MultimodalCollator(_pad_token_id(processor))
    preview = cursor.dataset[0]
    supervised = preview["input_ids"][preview["labels"] != -100]
    preview_text = processor.tokenizer.decode(supervised.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print(f"首条监督解码: {preview_text}")

    smoke_records = []
    vision_diff = None
    check_samples = []
    if cfg.get("smoke_check") and start_step == 0:
        seen = set()
        for sample in cursor.samples:
            if sample["task"] != "checklist" or sample["sample_id"] in seen:
                continue
            seen.add(sample["sample_id"])
            check_samples.append(sample)
            if len(check_samples) >= 2:
                break
        print("冒烟：训练前生成（LoRA B 仍为 0，对应未训练的基座）")
        for sample in check_samples:
            before = _generate(model, processor, sample, data_root, max_pixels, min_pixels, int(cfg.get("max_new_tokens", 128)))
            smoke_records.append(
                {
                    "sample_id": sample["sample_id"],
                    "task": sample["task"],
                    "target": target_text_for(sample),
                    "before": before.raw_text,
                    "before_overlap": _overlap(before.raw_text, sample),
                }
            )
            print(f"  before {sample['sample_id']}: {before.raw_text}")
        probe = _move_batch(collator([cursor.dataset[0]]), "cuda")
        vision_diff = _vision_diff(model, probe)
        print(f"视觉输入检查：真实像素与全零像素的 logits 最大差 {vision_diff:.6g}")
        del probe
        torch.cuda.empty_cache()

    log_path = run_dir / "train_log.jsonl"
    if resume_path is None and log_path.exists():
        log_path.unlink()
    log_handle = log_path.open("a", encoding="utf-8")
    losses: list[float] = []
    step = start_step
    error = None
    try:
        model.train()
        keep_visual_frozen_eval(model)
        while step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for _ in range(grad_accum):
                features, _ = cursor.take(micro_batch)
                batch = _move_batch(collator(features), "cuda")
                outputs = model(**{key: batch[key] for key in MODEL_INPUT_KEYS})
                loss = outputs.loss
                (loss / grad_accum).backward()
                micro_losses.append(float(loss.detach().float().cpu()))
                del outputs, loss, batch
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(optim_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            step += 1
            mean_loss = sum(micro_losses) / len(micro_losses)
            losses.append(mean_loss)
            lr = float(scheduler.get_last_lr()[0])
            epoch_now, index_now = cursor.position()
            row = {"step": step, "loss": mean_loss, "lr": lr, "epoch": epoch_now, "grad_norm": float(grad_norm)}
            log_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_handle.flush()
            print(f"step {step}/{max_steps} loss={mean_loss:.4f} lr={lr:.3e} grad_norm={float(grad_norm):.3f}")
            save_every = int(cfg.get("save_steps") or optim_cfg.get("save_steps") or 0)
            if save_every and (step % save_every == 0 or step == max_steps):
                epoch_now, index_now = cursor.position()
                ckpt = run_dir / "checkpoints" / f"step_{step:06d}"
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    ckpt,
                    {
                        "optimizer_step": step,
                        "epoch": epoch_now,
                        "index": index_now,
                        "exposure_stats": cursor.exposure_stats,
                        "seed": seed,
                    },
                )
                _prune_checkpoints(run_dir / "checkpoints", int(cfg.get("save_total_limit") or optim_cfg.get("save_total_limit") or 0))
                print(f"已保存断点 {ckpt}")
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        log_handle.close()
        torch.cuda.synchronize()
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        manifest = {
            "experiment_id": experiment_id,
            "git": _git_info(root),
            "base_model_dir": str(model_dir),
            "base_model_revision": _base_revision(model_dir),
            "prompt_version": PROMPT_VERSION,
            "data": {
                "train_jsonl": str(train_path),
                "train_jsonl_sha256": _sha256(train_path),
                "val_jsonl": None if val_path is None else str(val_path),
                "val_jsonl_sha256": _sha256(val_path) if val_path is not None and val_path.is_file() else None,
                "data_root": str(data_root),
                "split_manifest": None if manifest_path is None else str(manifest_path),
                "split_manifest_sha256": _sha256(manifest_path) if manifest_path is not None and manifest_path.is_file() else None,
                "n_train_samples_used": len(samples),
            },
            "config": cfg,
            "seed": seed,
            "max_steps": max_steps,
            "steps_completed": step,
            "exposure_stats": cursor.exposure_stats,
            "environment": _environment(),
            "lora_target_pattern": LANGUAGE_LORA_TARGET_PATTERN,
            "lora_matched_modules": len(matched),
            "lora_matched_examples": matched[:4] + matched[-2:],
            "trainable": trainable_summary,
            "artifacts": {
                "run_dir": str(run_dir),
                "train_log": str(log_path),
                "trainable_params": str(run_dir / "trainable_params.txt"),
                "named_modules": str(run_dir / "named_modules.txt"),
            },
            "loss_first": _logged_loss(log_path, 0, None if not losses else losses[0]),
            "loss_last": _logged_loss(log_path, -1, None if not losses else losses[-1]),
            "vision_logits_max_abs_diff": vision_diff,
            "peak_gpu_memory_allocated_bytes": peak_alloc,
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "error": error,
        }
        _write_json(run_dir / "run_manifest.json", manifest)

    if step == 0:
        return

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    if not args.no_export and cfg.get("export_adapter_dir"):
        export_dir = _resolve(root, cfg["export_adapter_dir"])
        if export_dir.exists():
            shutil.rmtree(export_dir)
        model.save_pretrained(export_dir)
        print(f"adapter 已导出到 {export_dir}")
    print(f"adapter 已保存到 {adapter_dir}")

    if smoke_records:
        print("冒烟：训练后生成")
        for record, sample in zip(smoke_records, check_samples):
            after = _generate(model, processor, sample, data_root, max_pixels, min_pixels, int(cfg.get("max_new_tokens", 128)))
            record["after"] = after.raw_text
            record["after_overlap"] = _overlap(after.raw_text, sample)
            record["text_changed"] = after.raw_text != record["before"]
            print(f"  after {sample['sample_id']}: {after.raw_text}")
        smoke_payload = {
            "note": "只说明这几条训练样本被拟合过，不是准确率。",
            "vision_logits_max_abs_diff": vision_diff,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "samples": smoke_records,
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        _write_json(run_dir / "smoke_check.json", smoke_payload)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"]["adapter"] = str(adapter_dir)
        manifest["artifacts"]["smoke_check"] = str(run_dir / "smoke_check.json")
        manifest["peak_gpu_memory_allocated_bytes"] = smoke_payload["peak_gpu_memory_allocated_bytes"]
        manifest["peak_gpu_memory_reserved_bytes"] = smoke_payload["peak_gpu_memory_reserved_bytes"]
        manifest["smoke_check"] = {
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "vision_logits_max_abs_diff": vision_diff,
            "samples": [
                {
                    "sample_id": record["sample_id"],
                    "text_changed": record["text_changed"],
                    "before_overlap": record["before_overlap"],
                    "after_overlap": record["after_overlap"],
                }
                for record in smoke_records
            ],
        }
        _write_json(run_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
