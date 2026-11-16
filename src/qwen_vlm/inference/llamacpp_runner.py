"""llama.cpp 多模态后端。

主路径是 llama-server 的 HTTP 客户端（OpenAI 兼容的 ``/v1/chat/completions``）。
同一份代码可以在 Jetson 上跑，导入链不包含 torch / transformers。

像素预算：本文件把 ``max_pixels`` / ``min_pixels`` 换成 llama.cpp 的
``--image-max-tokens`` / ``--image-min-tokens``。Qwen3-VL 的 patch 为 16、
spatial merge 为 2，一个视觉 token 对应 32×32 = 1024 像素。llama.cpp 在加载
mmproj 时执行 ``image_*_pixels = tokens * 1024``（见
``tools/mtmd/mtmd-model.h`` 的 ``set_limit_image_tokens``）。因此

* 384² → 144 tokens
* 512² → 256 tokens
* 768² → 576 tokens

``min_pixels`` 缺省时用 65536，与基础模型 ``preprocessor_config.json`` 的
``shortest_edge`` 以及 TransformersRunner 的缺省值一致。这些参数是服务进程
的启动参数，改档位必须重启 llama-server，请求体里带不过去。
"""

from __future__ import annotations

import base64
import io
import math
import struct
import subprocess
import time
from pathlib import Path

import requests
from PIL import Image

from qwen_vlm.common.prompts import SYSTEM_PROMPT
from qwen_vlm.inference.base import GenerationOutput, Runner

PATCH_SIZE = 16
MERGE_SIZE = 2
PIXELS_PER_TOKEN = (PATCH_SIZE * MERGE_SIZE) ** 2  # 1024

# artifacts/base/Qwen3-VL-2B-Instruct/preprocessor_config.json
DEFAULT_MIN_PIXELS = 65536

PIXEL_TIERS: dict[str, int] = {
    "384": 384 * 384,
    "512": 512 * 512,
    "768": 768 * 768,
}


def pixels_to_tokens(n_pixels: int) -> int:
    """把像素预算换成最接近的 llama.cpp 视觉 token 上限（至少 1）。"""
    return max(1, int(round(n_pixels / PIXELS_PER_TOKEN)))


def server_image_token_flags(max_pixels: int, min_pixels: int | None) -> tuple[int, int]:
    """返回 ``(image_min_tokens, image_max_tokens)``。"""
    max_tokens = pixels_to_tokens(max_pixels)
    min_tokens = pixels_to_tokens(DEFAULT_MIN_PIXELS if min_pixels is None else min_pixels)
    if min_tokens > max_tokens:
        min_tokens = max_tokens
    return min_tokens, max_tokens


def _f32(value: float) -> float:
    """把数值收成 IEEE float32，用来对照 llama.cpp dyn_size 的 float 路径。"""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = PATCH_SIZE * MERGE_SIZE,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = 512 * 512,
) -> tuple[int, int]:
    """返回 ``(高度, 宽度)``，算法与 Transformers 的 Qwen2-VL ``smart_resize`` 相同。

    llama.cpp 对 Qwen3-VL 静止图走 ``calc_size_preserved_ratio``（float32），
    视频走 ``mtmd_image_preprocessor_qwen3vl::smart_resize``（double，与本函数同式）。
    1280×960 在三档像素预算上两条路径的结果相同；是否每次都相同以实测日志为准。
    """
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


def smart_resize_llama_still(
    height: int,
    width: int,
    *,
    factor: int = PATCH_SIZE * MERGE_SIZE,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = 512 * 512,
) -> tuple[int, int]:
    """对照 ``img_tool::calc_size_preserved_ratio`` 的 float32 实现，返回 ``(高度, 宽度)``。"""

    def round_by_factor(x: float) -> int:
        q = _f32(_f32(x) / _f32(factor))
        return int(round(q)) * factor

    def floor_by_factor(x: float) -> int:
        return int(math.floor(_f32(_f32(x) / _f32(factor)))) * factor

    def ceil_by_factor(x: float) -> int:
        return int(math.ceil(_f32(x) / _f32(factor))) * factor

    w_bar = max(factor, round_by_factor(width))
    h_bar = max(factor, round_by_factor(height))
    if max_pixels > 0 and h_bar * w_bar > max_pixels:
        beta = _f32(math.sqrt(_f32(float(height) * width / max_pixels)))
        h_bar = max(factor, floor_by_factor(_f32(height / beta)))
        w_bar = max(factor, floor_by_factor(_f32(width / beta)))
    elif min_pixels > 0 and h_bar * w_bar < min_pixels:
        beta = _f32(math.sqrt(_f32(min_pixels / (float(height) * width))))
        h_bar = ceil_by_factor(_f32(height * beta))
        w_bar = ceil_by_factor(_f32(width * beta))
    return h_bar, w_bar


def vision_token_count(height: int, width: int) -> int:
    side = PATCH_SIZE * MERGE_SIZE
    return (height // side) * (width // side)


def llama_server_command(
    binary: str | Path,
    model: str | Path,
    mmproj: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_pixels: int = 512 * 512,
    min_pixels: int | None = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = 99,
    n_batch: int = 2048,
    n_ubatch: int = 2048,
    verbosity: int = 3,
    extra: list[str] | None = None,
) -> list[str]:
    """组装 llama-server 命令。``n_ubatch`` 设得不小于像素档，避免服务端把视觉 token 上限截掉。"""
    min_tokens, max_tokens = server_image_token_flags(max_pixels, min_pixels)
    cmd = [
        str(binary),
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--host",
        host,
        "--port",
        str(port),
        "--jinja",
        "--temp",
        "0",
        "-c",
        str(n_ctx),
        "-ngl",
        str(n_gpu_layers),
        "-b",
        str(n_batch),
        "-ub",
        str(n_ubatch),
        "--image-min-tokens",
        str(min_tokens),
        "--image-max-tokens",
        str(max_tokens),
        "--no-warmup",
        "--fit",
        "off",
        "-lv",
        str(verbosity),
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def mtmd_cli_command(
    binary: str | Path,
    model: str | Path,
    mmproj: str | Path,
    image_path: str | Path,
    user_text: str,
    *,
    max_pixels: int = 512 * 512,
    min_pixels: int | None = None,
    max_new_tokens: int = 128,
    n_gpu_layers: int = 99,
) -> list[str]:
    """llama-mtmd-cli 单轮命令。stdout 里夹着日志，只作为 HTTP 不可用时的备用。"""
    min_tokens, max_tokens = server_image_token_flags(max_pixels, min_pixels)
    return [
        str(binary),
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--image",
        str(image_path),
        "-sys",
        SYSTEM_PROMPT,
        "-p",
        user_text,
        "--temp",
        "0",
        "-n",
        str(max_new_tokens),
        "-ngl",
        str(n_gpu_layers),
        "--jinja",
        "--image-min-tokens",
        str(min_tokens),
        "--image-max-tokens",
        str(max_tokens),
        "--no-display-prompt",
        "-lv",
        "0",
    ]


def encode_image_data_url(image: Image.Image) -> str:
    """把已经转成 RGB、做过 EXIF 校正的 PIL 图编成 PNG data URL。不再做颜色或方向处理。"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_chat_completion(data: dict, *, encode_s: float, http_s: float) -> GenerationOutput:
    """从 llama-server 的 chat completion JSON 取出文本和计时。拿不到的字段保持 None。"""
    text = ""
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type", "text") == "text":
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(part)
            text = "".join(parts)
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    n_gen = timings.get("predicted_n", usage.get("completion_tokens"))
    prompt_n = timings.get("prompt_n", usage.get("prompt_tokens"))
    out_timings: dict[str, float] = {"encode_s": float(encode_s), "http_s": float(http_s)}
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")
    # llama.cpp 没有单独的首 token 字段。prompt_ms 是 prefill 耗时，生成从这之后开始。
    if isinstance(prompt_ms, (int, float)):
        out_timings["first_token_s"] = float(prompt_ms) / 1000.0
        out_timings["prompt_s"] = float(prompt_ms) / 1000.0
    if isinstance(predicted_ms, (int, float)):
        out_timings["generate_s"] = float(predicted_ms) / 1000.0
    if isinstance(prompt_n, (int, float)):
        out_timings["prompt_tokens"] = float(prompt_n)
    if not isinstance(n_gen, int):
        n_gen = None
    return GenerationOutput(
        text=text,
        n_generated_tokens=n_gen,
        n_image_tokens=None,
        image_size_after_resize=None,
        timings=out_timings,
    )


def wait_until_healthy(base_url: str, timeout_s: float = 180.0, process: subprocess.Popen | None = None) -> None:
    """轮询 ``/health``，直到返回 200。进程提前退出则抛错。"""
    deadline = time.monotonic() + timeout_s
    url = base_url.rstrip("/") + "/health"
    last_error = "尚未连接"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"llama-server 已退出，code={process.returncode}")
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"等待 llama-server 健康检查超时：{last_error}")


class LlamaCppRunner(Runner):
    """通过本机或远端 llama-server 做贪心解码。"""

    backend_name = "llamacpp"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        max_pixels: int = 512 * 512,
        min_pixels: int | None = None,
        max_new_tokens: int = 128,
        timeout_s: float = 300.0,
        model_name: str = "qwen3-vl",
    ):
        super().__init__(max_pixels=max_pixels, min_pixels=min_pixels, max_new_tokens=max_new_tokens)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.model_name = model_name
        self.min_image_tokens, self.max_image_tokens = server_image_token_flags(max_pixels, min_pixels)
        self.last_raw: dict | None = None

    def build_request(self, image: Image.Image, user_text: str) -> dict:
        """构造与 Transformers 后端同一套 system / user 文本的请求。图像在文本前面。"""
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_image_data_url(image)}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "stream": False,
        }

    def generate_with_raw(self, image: Image.Image, user_text: str) -> tuple[GenerationOutput, dict]:
        t0 = time.perf_counter()
        payload = self.build_request(image, user_text)
        t_enc = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        t_end = time.perf_counter()
        self.last_raw = data
        output = parse_chat_completion(data, encode_s=t_enc - t0, http_s=t_end - t_enc)
        return output, data

    def _generate(self, image: Image.Image, user_text: str) -> GenerationOutput:
        output, _raw = self.generate_with_raw(image, user_text)
        return output


class LlamaCppCliRunner(Runner):
    """每次调用拉起一次 llama-mtmd-cli。模型会反复加载，只在没有 llama-server 时使用。"""

    backend_name = "llamacpp-cli"

    def __init__(
        self,
        binary: str | Path,
        model: str | Path,
        mmproj: str | Path,
        max_pixels: int = 512 * 512,
        min_pixels: int | None = None,
        max_new_tokens: int = 128,
        n_gpu_layers: int = 99,
        timeout_s: float = 600.0,
    ):
        super().__init__(max_pixels=max_pixels, min_pixels=min_pixels, max_new_tokens=max_new_tokens)
        self.binary = Path(binary)
        self.model = Path(model)
        self.mmproj = Path(mmproj)
        self.n_gpu_layers = n_gpu_layers
        self.timeout_s = timeout_s

    def _generate(self, image: Image.Image, user_text: str) -> GenerationOutput:
        import tempfile

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="qwen_vlm_cli_") as tmp:
            image_path = Path(tmp) / "frame.png"
            image.save(image_path, format="PNG")
            cmd = mtmd_cli_command(
                self.binary,
                self.model,
                self.mmproj,
                image_path,
                user_text,
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
                max_new_tokens=self.max_new_tokens,
                n_gpu_layers=self.n_gpu_layers,
            )
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        elapsed = time.perf_counter() - t0
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-2000:]
            raise RuntimeError(f"llama-mtmd-cli 退出码 {completed.returncode}: {tail}")
        text = (completed.stdout or "").strip()
        return GenerationOutput(
            text=text,
            n_generated_tokens=None,
            n_image_tokens=None,
            image_size_after_resize=None,
            timings={"generate_s": elapsed},
        )
