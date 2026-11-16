"""读取 configs/train 下的 YAML，做 LoRA 监督微调。"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from qwen_vlm.training.train import main

if __name__ == "__main__":
    main()
