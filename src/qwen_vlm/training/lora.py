"""把 LoRA 只挂到语言模型的注意力和 MLP 投影上。

Qwen3-VL-2B 的真实模块名（由模型结构以及训练启动时打印的 named_modules 确认）：

- 视觉分支在 ``model.visual`` 下，注意力是 ``attn.qkv`` / ``attn.proj``，
  MLP 和 merger 是 ``linear_fc1`` / ``linear_fc2``，没有语言侧那种 ``q_proj``。
- 语言分支在 ``model.language_model.layers.{N}`` 下，投影是
  ``self_attn.q_proj/k_proj/v_proj/o_proj`` 和 ``mlp.gate_proj/up_proj/down_proj``。

target_modules 用整段路径的正则，避免以后视觉分支出现同名投影时被一起匹配上。
视觉编码器和 merger 全部冻结。
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import Qwen3VLForConditionalGeneration

LANGUAGE_LORA_TARGET_PATTERN = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
)

_LORA_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def module_matches_lora_target(name: str) -> bool:
    """与 peft 一样，用 fullmatch 判断整条模块路径。"""
    return re.fullmatch(LANGUAGE_LORA_TARGET_PATTERN, name) is not None


def load_base_model(model_dir: str | Path, device: str = "cuda"):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if device:
        model.to(device)
    return model


def matched_module_names(model) -> list[str]:
    return [name for name, _ in model.named_modules() if module_matches_lora_target(name)]


def dump_named_modules(model, path: Path) -> list[str]:
    names = [name for name, _ in model.named_modules()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def _assert_targets(names: list[str], matched: list[str]) -> None:
    visual_hits = [name for name in matched if ".visual." in f".{name}." or name.startswith("visual.")]
    if visual_hits:
        raise AssertionError(f"LoRA 正则匹配到了视觉模块: {visual_hits[:8]}")
    leaked = [
        name
        for name in names
        if ".visual." in f".{name}." and any(name.endswith(key) or f".{key}" in name for key in _LORA_KEYS)
    ]
    leaked = [name for name in leaked if module_matches_lora_target(name)]
    if leaked:
        raise AssertionError(f"视觉分支里的同名投影被匹配到了: {leaked}")
    if not matched:
        raise AssertionError("LoRA 正则没有匹配到任何语言模型投影，请核对 named_modules")
    bad_suffix = [name for name in matched if name.split(".")[-1] not in _LORA_KEYS]
    if bad_suffix:
        raise AssertionError(f"匹配到了非 q/k/v/o/gate/up/down 投影: {bad_suffix[:8]}")


def write_trainable_params(model, path: Path) -> dict:
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable_count, total = model.get_nb_trainable_parameters()
    else:
        total = sum(param.numel() for param in model.parameters())
        trainable_count = sum(param.numel() for _, param in trainable)
    names = [name for name, _ in trainable]
    visual = [name for name in names if "visual" in name]
    if visual:
        raise AssertionError(f"可训练参数里出现了 visual: {visual[:8]}")
    outside = [name for name in names if "language_model" not in name or "lora_" not in name]
    if outside:
        raise AssertionError(f"可训练参数应当只来自语言模型 LoRA: {outside[:8]}")
    ratio = trainable_count / total if total else 0.0
    dtypes = sorted({str(param.dtype) for _, param in trainable})
    lines = [
        f"trainable_params: {trainable_count}",
        f"total_params: {total}",
        f"trainable_ratio: {ratio:.8f}",
        f"trainable_tensors: {len(names)}",
        f"dtypes: {', '.join(dtypes)}",
        "names:",
        *names,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "trainable_params": trainable_count,
        "total_params": total,
        "trainable_ratio": ratio,
        "trainable_tensors": len(names),
        "dtypes": dtypes,
    }


def prepare_for_training(model, gradient_checkpointing: bool) -> None:
    """冻结视觉分支，并只给语言模型开梯度检查点。"""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.config.use_cache = False
    if getattr(base.config, "text_config", None) is not None:
        base.config.text_config.use_cache = False
    if hasattr(base, "model") and hasattr(base.model, "rope_deltas"):
        base.model.rope_deltas = None
    visual = base.model.visual
    for param in visual.parameters():
        param.requires_grad_(False)
    if gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        visual.gradient_checkpointing = False
        for module in visual.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
    visual.eval()
    model.train()


def keep_visual_frozen_eval(model) -> None:
    """model.train() 会把子模块一并设回 train，每个 step 再把视觉分支拨回 eval。"""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual = base.model.visual
    visual.eval()
    for param in visual.parameters():
        param.requires_grad_(False)


def attach_lora(model, lora_cfg: dict, run_dir: Path):
    """打印模块名、挂上 LoRA，并写出可训练参数名单。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    names = dump_named_modules(model, run_dir / "named_modules.txt")
    matched = matched_module_names(model)
    _assert_targets(names, matched)
    (run_dir / "lora_matched_modules.txt").write_text("\n".join(matched) + "\n", encoding="utf-8")

    visual_example = [name for name in names if name.startswith("model.visual.blocks.0.")][:12]
    language_example = [name for name in names if name.startswith("model.language_model.layers.0.")][:16]
    print(f"named_modules 共 {len(names)} 个，完整名单: {run_dir / 'named_modules.txt'}")
    print("视觉分支样例:")
    for name in visual_example:
        print(f"  {name}")
    print("语言分支样例:")
    for name in language_example:
        print(f"  {name}")
    print(f"LoRA 正则匹配到 {len(matched)} 个模块，样例:")
    for name in matched[:4] + matched[-2:]:
        print(f"  {name}")

    config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=LANGUAGE_LORA_TARGET_PATTERN,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, config)
    prepare_for_training(peft_model, gradient_checkpointing=False)
    summary = write_trainable_params(peft_model, run_dir / "trainable_params.txt")
    peft_model.print_trainable_parameters()
    print(
        f"可训练参数 {summary['trainable_params']} / {summary['total_params']}"
        f"（{summary['trainable_ratio'] * 100:.4f}%）"
    )
    return peft_model, matched, summary


def load_peft_model(model_dir: str | Path, adapter_dir: str | Path, device: str = "cuda", is_trainable: bool = False):
    base = load_base_model(model_dir, device=device)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)
    if device:
        model.to(device)
    model.eval()
    return model
