"""Backend-independent runner interface (frozen).

Every backend (Transformers, llama.cpp, ...) subclasses ``Runner`` and implements only
``_generate``. Prompt construction, parsing, the single retry and timing bookkeeping
live here so that all backends are compared under the same policy.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from PIL import Image, ImageOps

from qwen_vlm.common.prompts import PROMPT_VERSION, checklist_user_prompt, presence_user_prompt
from qwen_vlm.common.schema import PARSE_OK, parse_model_output

MAX_RETRIES = 1


@dataclass
class GenerationOutput:
    """What a backend returns for one generation call."""

    text: str
    n_generated_tokens: int | None = None
    n_image_tokens: int | None = None
    image_size_after_resize: tuple[int, int] | None = None  # (width, height) fed to the vision encoder
    timings: dict[str, float] = field(default_factory=dict)  # backend-specific stage timings, seconds


@dataclass
class PredictResult:
    task: str
    raw_text: str  # text of the final attempt
    raw_attempts: list[str]
    parsed: dict | None  # item_states dict (checklist) or {"item","state"} (presence)
    parse_status: str
    retry_count: int
    n_generated_tokens: int | None
    n_image_tokens: int | None
    image_size_after_resize: tuple[int, int] | None
    timings: dict[str, float]  # includes "total_s" covering all attempts
    backend: str
    prompt_version: str = PROMPT_VERSION

    @property
    def item_states(self) -> dict | None:
        return self.parsed if self.task == "checklist" else None

    def to_dict(self) -> dict:
        return asdict(self)


def load_image(image) -> Image.Image:
    """Load to RGB and apply EXIF orientation; accepts a path or a PIL image."""
    img = Image.open(image) if not isinstance(image, Image.Image) else image
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class Runner(ABC):
    backend_name: str = "abstract"

    def __init__(self, max_pixels: int, min_pixels: int | None = None, max_new_tokens: int = 128):
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def _generate(self, image: Image.Image, user_text: str) -> GenerationOutput:
        """Greedy-decode one answer for ``image`` and ``user_text`` with the shared system prompt."""

    def predict(
        self,
        image,
        question_zh: str,
        required_items: list[str],
        task: str = "checklist",
        target_item: str | None = None,
    ) -> PredictResult:
        t0 = time.perf_counter()
        img = load_image(image)
        t_load = time.perf_counter() - t0
        if task == "checklist":
            user_text = checklist_user_prompt(question_zh, required_items)
        elif task == "presence":
            if target_item is None:
                raise ValueError("presence task requires target_item")
            user_text = presence_user_prompt(target_item, question_zh)
        else:
            raise ValueError(f"unknown task {task!r}")

        attempts: list[str] = []
        timings: dict[str, float] = {"image_load_s": t_load}
        out = None
        parsed, status = None, None
        for attempt in range(MAX_RETRIES + 1):
            out = self._generate(img, user_text)
            attempts.append(out.text)
            for k, v in out.timings.items():
                timings[f"attempt{attempt}_{k}"] = v
            parsed, status = parse_model_output(out.text, task)
            if status == PARSE_OK:
                break
        timings["total_s"] = time.perf_counter() - t0

        return PredictResult(
            task=task,
            raw_text=attempts[-1],
            raw_attempts=attempts,
            parsed=parsed,
            parse_status=status,
            retry_count=len(attempts) - 1,
            n_generated_tokens=out.n_generated_tokens,
            n_image_tokens=out.n_image_tokens,
            image_size_after_resize=out.image_size_after_resize,
            timings=timings,
            backend=self.backend_name,
        )
