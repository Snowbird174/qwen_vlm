"""把一条样本编码成 Qwen3-VL 的训练张量。

用户轮次走冻结的 build_messages / checklist_user_prompt / presence_user_prompt，
答案走 checklist_target_text / presence_target_text，和推理端是同一套文本。

labels 只保留 assistant 回答和结尾的 ``<|im_end|>``。定位方法是：同一张图
分别编码「只有 system+user」和「加上回答」的序列，确认前者是后者的 token 前缀，
再把前缀以及 ``<|im_end|>`` 之后的换行打成 -100。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from qwen_vlm.common.prompts import (
    build_messages,
    checklist_target_text,
    checklist_user_prompt,
    presence_target_text,
    presence_user_prompt,
)
from qwen_vlm.inference.base import load_image

IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"


def resolve_pixel_budget(processor, max_pixels: int | None, min_pixels: int | None) -> tuple[int, int]:
    """训练像素预算。默认最长边 512*512，最短边跟处理器默认值。"""
    if max_pixels is None:
        max_pixels = 512 * 512
    if min_pixels is None:
        min_pixels = int(processor.image_processor.size["shortest_edge"])
    max_pixels = int(max_pixels)
    min_pixels = int(min_pixels)
    if min_pixels > max_pixels:
        raise ValueError(f"min_pixels ({min_pixels}) 大于 max_pixels ({max_pixels})")
    return max_pixels, min_pixels


def resolve_image_path(sample: dict, data_root: Path) -> Path:
    path = Path(sample["image_path"])
    if path.is_absolute():
        return path
    return Path(data_root) / path


def user_text_for(sample: dict) -> str:
    if sample["task"] == "checklist":
        return checklist_user_prompt(sample["question_zh"], sample["required_items"])
    if sample["task"] == "presence":
        if not sample.get("target_item"):
            raise ValueError(f"presence 样本缺少 target_item: {sample.get('sample_id')}")
        return presence_user_prompt(sample["target_item"], sample["question_zh"])
    raise ValueError(f"未知 task: {sample.get('task')!r}")


def target_text_for(sample: dict) -> str:
    if sample["task"] == "checklist":
        return checklist_target_text(sample["item_states"])
    if sample["task"] == "presence":
        item = sample["target_item"]
        return presence_target_text(item, sample["item_states"][item])
    raise ValueError(f"未知 task: {sample.get('task')!r}")


def _squeeze_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 1 and tensor.shape[0] == 1 and tensor.ndim == 2:
        return tensor[0]
    return tensor


def _processor_kwargs(max_pixels: int, min_pixels: int) -> dict:
    return {
        "return_tensors": "pt",
        "images_kwargs": {"size": {"longest_edge": max_pixels, "shortest_edge": min_pixels}},
    }


def encode_sample(sample: dict, processor, image: Image.Image, *, max_pixels: int, min_pixels: int) -> dict:
    """编码一条样本。回答被截断或图像 token 对不上时直接报错。"""
    tokenizer = processor.tokenizer
    user_text = user_text_for(sample)
    target = target_text_for(sample)
    prompt_messages = build_messages(user_text, image)
    full_messages = prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
    kwargs = _processor_kwargs(max_pixels, min_pixels)
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs=kwargs,
    )
    full = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        processor_kwargs=kwargs,
    )

    prompt_ids = _squeeze_batch(prompt["input_ids"]).to(dtype=torch.long).contiguous()
    input_ids = _squeeze_batch(full["input_ids"]).to(dtype=torch.long).contiguous()
    attention_mask = _squeeze_batch(full["attention_mask"]).to(dtype=torch.long).contiguous()
    mm_token_type_ids = _squeeze_batch(full["mm_token_type_ids"]).to(dtype=torch.long).contiguous()
    if prompt_ids.ndim != 1 or input_ids.ndim != 1:
        raise RuntimeError(f"input_ids 形状异常: prompt {tuple(prompt_ids.shape)} full {tuple(input_ids.shape)}")
    if input_ids.shape != attention_mask.shape or input_ids.shape != mm_token_type_ids.shape:
        raise RuntimeError("input_ids / attention_mask / mm_token_type_ids 长度不一致")

    prompt_len = int(prompt_ids.shape[0])
    if prompt_len >= int(input_ids.shape[0]) or not torch.equal(input_ids[:prompt_len], prompt_ids):
        raise RuntimeError(
            f"无法把 assistant 回答对齐到 token 前缀（样本 {sample.get('sample_id')}）。"
            "带回答的序列应当以生成提示为前缀。"
        )

    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    if im_end_id is None or im_end_id < 0 or image_pad_id is None or image_pad_id < 0:
        raise RuntimeError("词表里找不到 <|im_end|> 或 <|image_pad|>")

    labels = input_ids.clone()
    labels[:prompt_len] = -100
    relative = (input_ids[prompt_len:] == im_end_id).nonzero(as_tuple=False)
    if relative.numel() == 0:
        raise RuntimeError(f"回答被截掉了，监督段里没有 <|im_end|>（样本 {sample.get('sample_id')}）")
    answer_end = prompt_len + int(relative[0]) + 1
    labels[answer_end:] = -100

    supervised = input_ids[labels != -100]
    decoded = tokenizer.decode(supervised.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    expected = target + IM_END
    if decoded != expected:
        raise RuntimeError(
            "监督 token 解码结果不是目标 JSON 加 <|im_end|>。\n"
            f"样本: {sample.get('sample_id')}\n解码: {decoded!r}\n期望: {expected!r}"
        )

    n_pad_full = int((input_ids == image_pad_id).sum())
    n_pad_prompt = int((prompt_ids == image_pad_id).sum())
    grid = full["image_grid_thw"]
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    if tuple(grid.shape) != (1, 3):
        raise RuntimeError(f"一条样本应只有一张图，image_grid_thw 形状为 {tuple(grid.shape)}")
    merge = int(processor.image_processor.merge_size)
    n_grid = int(grid[0].prod().item()) // (merge**2)
    if n_pad_full != n_grid or n_pad_prompt != n_grid:
        raise RuntimeError(
            f"图像 token 数与 image_grid_thw 不一致（样本 {sample.get('sample_id')}）: "
            f"full={n_pad_full} prompt={n_pad_prompt} grid={n_grid}"
        )
    if int((supervised == image_pad_id).sum()) != 0:
        raise RuntimeError("图像占位符被放进了监督标签")

    pixels = full["pixel_values"]
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2:
        raise RuntimeError(f"意外的 pixel_values 形状: {tuple(pixels.shape)}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "labels": labels,
        "pixel_values": pixels.contiguous(),
        "image_grid_thw": grid.to(dtype=torch.long).contiguous(),
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON") from exc
    return rows


def cache_key(sample: dict) -> tuple:
    return (
        sample["sample_id"],
        sample["image_path"],
        sample["task"],
        sample.get("target_item"),
        sample["question_zh"],
        json.dumps(sample["item_states"], ensure_ascii=False, sort_keys=True),
    )


class TrainDataset(torch.utils.data.Dataset):
    """一个 epoch 的样本。cache 可在 epoch 之间共用，避免过拟合时重复做图像预处理。"""

    def __init__(
        self,
        samples: list[dict],
        processor,
        data_root: Path,
        max_pixels: int,
        min_pixels: int,
        cache: dict | None = None,
        cache_limit: int = 128,
    ):
        self.samples = samples
        self.processor = processor
        self.data_root = Path(data_root)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.cache = cache
        self.cache_limit = cache_limit

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        key = cache_key(sample)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        path = resolve_image_path(sample, self.data_root)
        if not path.is_file():
            raise FileNotFoundError(f"找不到图片 {path}（样本 {sample.get('sample_id')}）")
        encoded = encode_sample(
            sample,
            self.processor,
            load_image(path),
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        if self.cache is not None and len(self.cache) < self.cache_limit:
            self.cache[key] = encoded
        return encoded
