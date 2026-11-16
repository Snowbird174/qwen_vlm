"""监督标签只覆盖目标 JSON 和结尾的 <|im_end|>，补齐位置是 -100。"""

from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.common.prompts import SYSTEM_PROMPT
from qwen_vlm.training.collator import MultimodalCollator
from qwen_vlm.training.dataset import encode_sample, target_text_for

MODEL_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "base" / "Qwen3-VL-2B-Instruct"
IM_END = "<|im_end|>"


pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="基座模型不在本地")


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)


def _states():
    return {category: "present" for category in CATEGORY_IDS}


def _checklist():
    states = _states()
    states["pliers"] = "absent"
    states["multimeter"] = "unknown"
    return {
        "sample_id": "lab-checklist",
        "image_path": "unused.png",
        "session_id": "s",
        "scene_group_id": "g",
        "pair_group_id": None,
        "task": "checklist",
        "required_items": ["screwdriver", "pliers"],
        "question_zh": "还缺少什么器材？",
        "item_states": states,
        "condition_tags": [],
        "hard_negative_tags": [],
        "unknown_reason": {"multimeter": "被挡住"},
        "source": "synthetic_smoke",
        "annotation_verified": True,
    }


def _presence():
    sample = _checklist()
    sample["sample_id"] = "lab-presence"
    sample["task"] = "presence"
    sample["target_item"] = "pliers"
    sample["question_zh"] = "画面中有没有钳子？"
    return sample


def _supervised_text(processor, encoded):
    ids = encoded["input_ids"][encoded["labels"] != -100]
    return processor.tokenizer.decode(ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def test_supervised_tokens_are_exactly_the_target_json(processor):
    image = Image.new("RGB", (320, 240), (180, 140, 90))
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for sample in (_checklist(), _presence()):
        encoded = encode_sample(sample, processor, image, max_pixels=384 * 384, min_pixels=384 * 384)
        target = target_text_for(sample)
        assert _supervised_text(processor, encoded) == target + IM_END
        labels = encoded["labels"]
        input_ids = encoded["input_ids"]
        supervised = labels != -100
        assert torch.equal(labels[supervised], input_ids[supervised])
        assert int((input_ids[supervised] == pad_id).sum()) == 0
        start = int(supervised.nonzero()[0])
        prefix = processor.tokenizer.decode(input_ids[:start].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert "实验室器材点检助手" in prefix
        assert SYSTEM_PROMPT[:12] in prefix
        assert sample["question_zh"] in prefix
        assert target not in prefix
        assert "<|image_pad|>" in prefix
        grid = encoded["image_grid_thw"][0]
        n_grid = int(grid.prod().item()) // int(processor.image_processor.merge_size) ** 2
        assert int((input_ids == pad_id).sum()) == n_grid


def test_collator_pads_labels_with_ignore_index():
    collator = MultimodalCollator(pad_token_id=7)
    first = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "mm_token_type_ids": torch.tensor([0, 1, 0]),
        "labels": torch.tensor([-100, 5, 6]),
        "pixel_values": torch.zeros(2, 4),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    second = {
        "input_ids": torch.tensor([4, 5]),
        "attention_mask": torch.tensor([1, 1]),
        "mm_token_type_ids": torch.tensor([0, 1]),
        "labels": torch.tensor([-100, 8]),
        "pixel_values": torch.ones(3, 4),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    batch = collator([first, second])
    assert batch["labels"].tolist() == [[-100, 5, 6], [-100, 8, -100]]
    assert batch["input_ids"][1].tolist() == [4, 5, 7]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["mm_token_type_ids"][1].tolist() == [0, 1, 0]
    assert tuple(batch["pixel_values"].shape) == (5, 4)
    assert batch["image_grid_thw"].tolist() == [[1, 2, 2], [1, 4, 4]]


def test_collator_keeps_real_supervised_span(processor):
    image = Image.new("RGB", (320, 240), (20, 80, 140))
    checklist = encode_sample(_checklist(), processor, image, max_pixels=384 * 384, min_pixels=384 * 384)
    presence = encode_sample(_presence(), processor, image, max_pixels=384 * 384, min_pixels=384 * 384)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    batch = MultimodalCollator(pad_id)([checklist, presence])
    assert checklist["input_ids"].shape[0] != presence["input_ids"].shape[0]
    for row, sample, encoded in (
        (0, _checklist(), checklist),
        (1, _presence(), presence),
    ):
        length = encoded["input_ids"].shape[0]
        assert torch.equal(batch["labels"][row, :length], encoded["labels"])
        assert torch.equal(batch["labels"][row, length:], torch.full((batch["labels"].shape[1] - length,), -100))
        ids = batch["input_ids"][row][batch["labels"][row] != -100]
        text = processor.tokenizer.decode(ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert text == target_text_for(sample) + IM_END
