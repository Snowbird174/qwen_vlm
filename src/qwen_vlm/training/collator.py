"""把变长的图文样本补成一个 batch。

文本和 mm_token_type_ids 右侧补齐。labels 的补齐位置是 -100。
pixel_values / image_grid_thw 按 Qwen3-VL 的习惯在样本维上拼接，不补齐。
"""

from __future__ import annotations

import torch


class MultimodalCollator:
    def __init__(self, pad_token_id: int):
        if pad_token_id is None or int(pad_token_id) < 0:
            raise ValueError(f"无效的 pad_token_id: {pad_token_id}")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict]) -> dict:
        if not features:
            raise ValueError("空 batch")

        def pad_stack(key: str, padding: int) -> torch.Tensor:
            rows = [feature[key] for feature in features]
            return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=padding)

        grids = []
        pixels = []
        for feature in features:
            grid = feature["image_grid_thw"]
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)
            grids.append(grid)
            pixels.append(feature["pixel_values"])

        return {
            "input_ids": pad_stack("input_ids", self.pad_token_id),
            "attention_mask": pad_stack("attention_mask", 0),
            "mm_token_type_ids": pad_stack("mm_token_type_ids", 0),
            "labels": pad_stack("labels", -100),
            "pixel_values": torch.cat(pixels, dim=0),
            "image_grid_thw": torch.cat(grids, dim=0),
        }
