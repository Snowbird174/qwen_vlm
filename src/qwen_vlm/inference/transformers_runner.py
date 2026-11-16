"""Reference backend: Hugging Face Transformers on CUDA (server baseline and training-side eval)."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.generation.streamers import BaseStreamer

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.prompts import build_messages
from qwen_vlm.inference.base import GenerationOutput, Runner

DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts/base/Qwen3-VL-2B-Instruct"


class _TimingStreamer(BaseStreamer):
    """Records wall-clock time of the first generated token (``put`` #2; #1 is the prompt)."""

    def __init__(self, sync: bool):
        self.sync = sync
        self.calls = 0
        self.first_token_time: float | None = None

    def put(self, value):
        self.calls += 1
        if self.calls == 2 and self.first_token_time is None:
            if self.sync:
                torch.cuda.synchronize()
            self.first_token_time = time.perf_counter()

    def end(self):
        pass


class TransformersRunner(Runner):
    backend_name = "transformers"

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        max_pixels: int = 512 * 512,
        min_pixels: int | None = None,
        max_new_tokens: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        model=None,
        processor=None,
    ):
        super().__init__(max_pixels=max_pixels, min_pixels=min_pixels, max_new_tokens=max_new_tokens)
        self.device = device
        self.processor = processor or AutoProcessor.from_pretrained(model_dir)
        self.model = model or Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype, attn_implementation=attn_implementation
        ).to(device)
        self.model.eval()
        ip = self.processor.image_processor
        self.patch_size = ip.patch_size
        self.merge_size = ip.merge_size
        self.min_pixels = min_pixels if min_pixels is not None else ip.size["shortest_edge"]

    def prepare_inputs(self, image: Image.Image, user_text: str) -> dict:
        """Tokenize chat + image with the official template and the configured pixel budget."""
        messages = build_messages(user_text, image)
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "return_tensors": "pt",
                "images_kwargs": {"size": {"longest_edge": self.max_pixels, "shortest_edge": self.min_pixels}},
            },
        ).to(self.device)

    @torch.inference_mode()
    def _generate(self, image: Image.Image, user_text: str) -> GenerationOutput:
        sync = self.device.startswith("cuda")
        t0 = time.perf_counter()
        inputs = self.prepare_inputs(image, user_text)
        if sync:
            torch.cuda.synchronize()
        t_pre = time.perf_counter()

        streamer = _TimingStreamer(sync)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, streamer=streamer)
        if sync:
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        t_h, t_w = inputs["image_grid_thw"][0, 1].item(), inputs["image_grid_thw"][0, 2].item()
        n_img_tokens = int(inputs["image_grid_thw"][0].prod().item()) // (self.merge_size**2)
        first = streamer.first_token_time or t_end
        return GenerationOutput(
            text=text,
            n_generated_tokens=int(new_tokens.numel()),
            n_image_tokens=n_img_tokens,
            image_size_after_resize=(t_w * self.patch_size, t_h * self.patch_size),
            timings={
                "preprocess_s": t_pre - t0,
                "first_token_s": first - t_pre,
                "generate_s": t_end - t_pre,
            },
        )
