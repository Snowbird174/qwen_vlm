#!/usr/bin/env python3
"""服务器冒烟。

1. 同一张 1280×960 图，三档像素下对比 llama.cpp 的视觉 token 和 Transformers。
2. 三张冒烟图 × F16/Q8_0/Q4_K_M，与 TransformersRunner(bf16) 对照原始文本和解析结果。
3. 用 Q4_K_M 拉起演示界面，通过 gradio_client 走图片模式。

这里的差异只用于定位，不是精度结论。没有板端数字。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import requests

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.derive import derive
from qwen_vlm.common.prompts import checklist_user_prompt
from qwen_vlm.inference.base import load_image
from qwen_vlm.inference.llamacpp_runner import (
    PIXEL_TIERS,
    LlamaCppRunner,
    llama_server_command,
    smart_resize,
    vision_token_count,
    wait_until_healthy,
)

QUESTION = "按照本次实验清单，还缺少什么器材？"
REQUIRED = ["screwdriver", "pliers", "cable_bundle", "multimeter", "hammer"]
SMOKE_DIR = PROJECT_ROOT / "data" / "smoke" / "b0"
RUN_DIR = PROJECT_ROOT / "runs" / "deploy_smoke"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "base" / "Qwen3-VL-2B-Instruct"
QUANT_DIR = PROJECT_ROOT / "artifacts" / "quantized" / "base"
SERVER_BIN = PROJECT_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o))


def gpu_free_mib() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(text.strip().splitlines()[0])


def wait_gpu(min_free_mib: int, timeout_s: float = 2400) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        free = gpu_free_mib()
        print(f"GPU 空闲 {free} MiB", flush=True)
        if free >= min_free_mib:
            return free
        if time.monotonic() >= deadline:
            raise TimeoutError(f"等待 GPU 空闲显存 >= {min_free_mib} MiB 超时，当前 {free} MiB")
        time.sleep(20)


class Server:
    def __init__(self, model: Path, mmproj: Path, max_pixels: int, port: int, log_path: Path, verbosity: int = 3):
        self.model = model
        self.mmproj = mmproj
        self.max_pixels = max_pixels
        self.port = port
        self.log_path = log_path
        self.verbosity = verbosity
        self.proc: subprocess.Popen | None = None
        self.log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = llama_server_command(
            SERVER_BIN,
            self.model,
            self.mmproj,
            port=self.port,
            max_pixels=self.max_pixels,
            verbosity=self.verbosity,
        )
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.log_handle.write("CMD " + " ".join(cmd) + "\n")
        self.log_handle.flush()
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_until_healthy(self.base_url, timeout_s=240, process=self.proc)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        self.proc = None

    def log_text(self) -> str:
        if not self.log_path.is_file():
            return ""
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def quant_paths() -> dict[str, Path]:
    return {
        "f16": QUANT_DIR / "base-f16.gguf",
        "q8_0": QUANT_DIR / "base-q8_0.gguf",
        "q4_k_m": QUANT_DIR / "base-q4_k_m.gguf",
    }


def mmproj_path() -> Path:
    return QUANT_DIR / "mmproj-base-f16.gguf"


def transformers_token_reference(image_path: Path) -> dict:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    image = load_image(image_path)
    user_text = checklist_user_prompt(QUESTION, REQUIRED)
    from qwen_vlm.common.prompts import build_messages

    messages = build_messages(user_text, image)
    image_processor = processor.image_processor
    min_pixels = int(image_processor.size["shortest_edge"])
    out = {}
    for tier, max_pixels in PIXEL_TIERS.items():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "return_tensors": "pt",
                "images_kwargs": {"size": {"longest_edge": max_pixels, "shortest_edge": min_pixels}},
            },
        )
        grid = inputs["image_grid_thw"][0]
        merge = int(image_processor.merge_size)
        n_tokens = int(grid.prod().item()) // (merge ** 2)
        height = int(grid[1].item()) * int(image_processor.patch_size)
        width = int(grid[2].item()) * int(image_processor.patch_size)
        pred_h, pred_w = smart_resize(image.height, image.width, min_pixels=min_pixels, max_pixels=max_pixels)
        out[tier] = {
            "max_pixels": max_pixels,
            "min_pixels": min_pixels,
            "n_image_tokens": n_tokens,
            "image_size_after_resize_wh": [width, height],
            "smart_resize_hw": [pred_h, pred_w],
            "smart_resize_tokens": vision_token_count(pred_h, pred_w),
        }
    return {"image": str(image_path.relative_to(PROJECT_ROOT)), "size_wh": [image.width, image.height], "tiers": out}


def parse_llama_image_tokens(log_text: str) -> dict:
    nxs = [int(v) for v in re.findall(r"image_tokens->nx = (\d+)", log_text)]
    nys = [int(v) for v in re.findall(r"image_tokens->ny = (\d+)", log_text)]
    max_pixels = [int(v) for v in re.findall(r"image_max_pixels:\s+(\d+)", log_text)]
    min_pixels = [int(v) for v in re.findall(r"image_min_pixels:\s+(\d+)", log_text)]
    capped = "cap image_max_tokens" in log_text
    n_tokens = nxs[-1] * nys[-1] if nxs and nys else None
    return {
        "nx": nxs[-1] if nxs else None,
        "ny": nys[-1] if nys else None,
        "n_image_tokens": n_tokens,
        "logged_image_max_pixels": max_pixels[-1] if max_pixels else None,
        "logged_image_min_pixels": min_pixels[-1] if min_pixels else None,
        "server_capped_image_tokens": capped,
    }


def resolution_check(image_path: Path, reference: dict) -> dict:
    """用 Q4_K_M 测三档。视觉 token 由预处理决定，和语言权重量化档无关。"""
    model = quant_paths()["q4_k_m"]
    mmproj = mmproj_path()
    rows = []
    for tier, max_pixels in PIXEL_TIERS.items():
        wait_gpu(12000)
        log_path = RUN_DIR / "logs" / f"resolution_{tier}.log"
        server = Server(model, mmproj, max_pixels, port=8091, log_path=log_path, verbosity=5)
        server.start()
        try:
            runner = LlamaCppRunner(base_url=server.base_url, max_pixels=max_pixels, max_new_tokens=16)
            output, raw = runner.generate_with_raw(load_image(image_path), checklist_user_prompt(QUESTION, REQUIRED))
        finally:
            server.stop()
        parsed = parse_llama_image_tokens(server.log_text())
        ref = reference["tiers"][tier]
        rows.append(
            {
                "tier": tier,
                "max_pixels": max_pixels,
                "transformers_n_image_tokens": ref["n_image_tokens"],
                "transformers_size_wh": ref["image_size_after_resize_wh"],
                "llamacpp": parsed,
                "prompt_tokens": output.timings.get("prompt_tokens"),
                "raw_text_preview": output.text[:200],
                "finish_reason": (raw.get("choices") or [{}])[0].get("finish_reason"),
                "tokens_match_transformers": parsed["n_image_tokens"] == ref["n_image_tokens"],
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    prompt_counts = [row["prompt_tokens"] for row in rows]
    deltas = None
    if all(isinstance(v, (int, float)) for v in prompt_counts):
        deltas = {
            "prompt_tokens": prompt_counts,
            "delta_512_minus_384": prompt_counts[1] - prompt_counts[0],
            "delta_768_minus_512": prompt_counts[2] - prompt_counts[1],
            "expected_delta_512_minus_384": reference["tiers"]["512"]["n_image_tokens"]
            - reference["tiers"]["384"]["n_image_tokens"],
            "expected_delta_768_minus_512": reference["tiers"]["768"]["n_image_tokens"]
            - reference["tiers"]["512"]["n_image_tokens"],
        }
    matched = [row["tokens_match_transformers"] for row in rows]
    if all(item is True for item in matched):
        conclusion = "三档 llama.cpp 日志里的 nx*ny 与 Transformers 的视觉 token 数一致。"
    elif any(row["llamacpp"]["n_image_tokens"] is None for row in rows):
        conclusion = (
            "服务日志里没有 image_tokens->nx/ny（调试日志没打出来）。"
            "下面用三次请求的 prompt token 差值对照 Transformers 的视觉 token 差值；"
            "差值只说明档位之间差了多少视觉 token，不是绝对个数。"
        )
    else:
        conclusion = "三档里至少有一档和 Transformers 的视觉 token 数不同，见各档 llamacpp 字段。"
    return {"image": reference["image"], "quant_used": "q4_k_m", "rows": rows, "prompt_token_deltas": deltas, "conclusion": conclusion}


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def run_transformers_parity(images: list[Path]) -> list[dict]:
    import torch

    from qwen_vlm.inference.transformers_runner import TransformersRunner

    wait_gpu(16000)
    runner = TransformersRunner(max_pixels=PIXEL_TIERS["512"])
    records = []
    try:
        for image in images:
            result = runner.predict(image, QUESTION, REQUIRED)
            derived = derive(result.parsed, REQUIRED) if result.parsed else None
            records.append(
                {
                    "image": str(image.relative_to(PROJECT_ROOT)),
                    "backend": "transformers",
                    "quant": "bf16",
                    **result.to_dict(),
                    "derived": derived,
                }
            )
            print(image.name, result.parse_status, result.n_image_tokens, flush=True)
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def run_llamacpp_parity(images: list[Path]) -> list[dict]:
    records = []
    mmproj = mmproj_path()
    for quant, model in quant_paths().items():
        if not model.is_file():
            records.append({"quant": quant, "error": f"缺少 {model}"})
            continue
        wait_gpu(16000 if quant == "f16" else 12000)
        log_path = RUN_DIR / "logs" / f"parity_{quant}.log"
        server = Server(model, mmproj, PIXEL_TIERS["512"], port=8091, log_path=log_path, verbosity=3)
        server.start()
        try:
            runner = LlamaCppRunner(base_url=server.base_url, max_pixels=PIXEL_TIERS["512"])
            for image in images:
                result = runner.predict(image, QUESTION, REQUIRED)
                raw = runner.last_raw or {}
                derived = derive(result.parsed, REQUIRED) if result.parsed else None
                finish = (raw.get("choices") or [{}])[0].get("finish_reason")
                records.append(
                    {
                        "image": str(image.relative_to(PROJECT_ROOT)),
                        "quant": quant,
                        "pixels": "512",
                        "finish_reason": finish,
                        "has_cjk": has_cjk(result.raw_text),
                        "item_states_repeats": result.raw_text.count("item_states"),
                        **result.to_dict(),
                        "derived": derived,
                    }
                )
                print(quant, image.name, result.parse_status, finish, flush=True)
        finally:
            server.stop()
    return records


def compare_parity(tf_rows: list[dict], llama_rows: list[dict]) -> list[dict]:
    by_image = {row["image"]: row for row in tf_rows}
    compared = []
    for row in llama_rows:
        if row.get("error"):
            compared.append(row)
            continue
        ref = by_image.get(row["image"])
        compared.append(
            {
                "image": row["image"],
                "quant": row["quant"],
                "llamacpp_parse_status": row.get("parse_status"),
                "transformers_parse_status": None if ref is None else ref.get("parse_status"),
                "raw_text_equal": None if ref is None else row.get("raw_text") == ref.get("raw_text"),
                "parsed_equal": None if ref is None else row.get("parsed") == ref.get("parsed"),
                "llamacpp_raw_text": row.get("raw_text"),
                "transformers_raw_text": None if ref is None else ref.get("raw_text"),
                "finish_reason": row.get("finish_reason"),
                "has_cjk": row.get("has_cjk"),
                "item_states_repeats": row.get("item_states_repeats"),
                "retry_count": row.get("retry_count"),
            }
        )
    return compared


def app_smoke(image: Path) -> dict:
    wait_gpu(12000)
    log_path = RUN_DIR / "logs" / "app_server.log"
    server = Server(quant_paths()["q4_k_m"], mmproj_path(), PIXEL_TIERS["512"], port=8092, log_path=log_path)
    server.start()
    app_log = RUN_DIR / "logs" / "app_gradio.log"
    app_log.parent.mkdir(parents=True, exist_ok=True)
    handle = app_log.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["GRADIO_ANALYTICS_ENABLED"] = "0"
    proc = subprocess.Popen(
        [
            str(PROJECT_ROOT / "envs" / "qwen_vlm" / "bin" / "python"),
            "-m",
            "qwen_vlm.app.app",
            "--server",
            server.base_url,
            "--pixels",
            "512",
            "--host",
            "127.0.0.1",
            "--port",
            "7860",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"gradio 退出 code={proc.returncode}，看 {app_log}")
            try:
                response = requests.get("http://127.0.0.1:7860/", timeout=2)
                if response.status_code < 500:
                    break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise TimeoutError("gradio 没有在 90 秒内监听 7860")
        from gradio_client import Client, handle_file

        client = Client("http://127.0.0.1:7860")
        result = client.predict(
            "读取保存图片",
            handle_file(str(image)),
            None,
            REQUIRED,
            QUESTION,
            api_name="/infer",
        )
        text = json.dumps(result, ensure_ascii=False, default=str)
        ok = "正忙" not in text and ("原始输出" in text or "parse" in text or "耗时" in text or "秒" in text)
        return {
            "ok": ok,
            "camera_tested": False,
            "camera_note": "服务器没有可用的采集摄像头，摄像头预览没有测。",
            "result_preview": text[:2000],
        }
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        handle.close()
        server.stop()


def write_summary(resolution: dict | None, compared: list[dict] | None, app: dict | None) -> None:
    lines = ["# 服务器部署冒烟", "", "这些数字只说明服务器上的流程是否走通，不是板端性能，也不是精度结论。", ""]
    if resolution:
        lines.append("## 像素档")
        lines.append("")
        lines.append(resolution["conclusion"])
        lines.append("")
        lines.append("| 档 | Transformers 视觉 token | llama.cpp nx*ny | prompt tokens | 服务是否截断上限 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in resolution["rows"]:
            llama_n = row["llamacpp"]["n_image_tokens"]
            lines.append(
                f"| {row['tier']} | {row['transformers_n_image_tokens']} | {llama_n} | {row['prompt_tokens']} | {row['llamacpp']['server_capped_image_tokens']} |"
            )
        lines.append("")
        if resolution.get("prompt_token_deltas"):
            lines.append("prompt token 差值：" + json.dumps(resolution["prompt_token_deltas"], ensure_ascii=False))
            lines.append("")
    if compared is not None:
        lines.append("## 与 Transformers bf16 的文本对照（像素档 512）")
        lines.append("")
        lines.append("| 图 | 量化 | 原文一致 | 解析一致 | llama.cpp 解析 | finish_reason | 含中文 | item_states 次数 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in compared:
            if row.get("error"):
                lines.append(f"|  | {row.get('quant')} |  |  |  |  |  | 缺文件 |")
                continue
            lines.append(
                "| {image} | {quant} | {raw_text_equal} | {parsed_equal} | {llamacpp_parse_status} | {finish_reason} | {has_cjk} | {item_states_repeats} |".format(
                    **row
                )
            )
        lines.append("")
        lines.append(
            "颜色和方向：runner 收到的是基类 `load_image` 之后的 RGB，再编码成 PNG 发给 llama-server。"
            "服务端用图像库按 PNG 解成 RGB，runner 里没有 OpenCV，也不会把通道换成 BGR。"
            "EXIF 方向由基类处理，有单测覆盖。中文问题走 UTF-8 JSON；模型若遵守提示词，正文是英文枚举的 JSON，表里的「含中文」因此可能是 false。"
        )
        lines.append("finish_reason 为 stop 表示生成停在 EOS 或停止符；length 表示撞上 max_new_tokens。`item_states` 次数大于 1 视为重复生成。")
        lines.append("")
    if app:
        lines.append("## 演示界面")
        lines.append("")
        lines.append(f"图片模式调用：{'通过' if app.get('ok') else '未通过'}。{app.get('camera_note', '')}")
        lines.append("")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--skip-resolution", action="store_true")
    parser.add_argument("--skip-app", action="store_true")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    images = sorted(SMOKE_DIR.glob("*.png"))
    if len(images) < 3:
        raise SystemExit(f"冒烟图片不够：{SMOKE_DIR}")
    if not SERVER_BIN.is_file():
        raise SystemExit(f"没有 {SERVER_BIN}")
    if not mmproj_path().is_file():
        raise SystemExit(f"没有 {mmproj_path()}")

    reference = transformers_token_reference(images[0])
    resolution = None
    compared = None
    app = None
    if not args.skip_resolution:
        resolution = resolution_check(images[0], reference)
        resolution["transformers_reference"] = reference
        (RUN_DIR / "resolution_check.json").write_text(dump(resolution) + "\n", encoding="utf-8")
    if not args.skip_parity:
        tf_rows = run_transformers_parity(images)
        llama_rows = run_llamacpp_parity(images)
        compared = compare_parity(tf_rows, llama_rows)
        with (RUN_DIR / "parity.jsonl").open("w", encoding="utf-8") as handle:
            for row in tf_rows + llama_rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, tuple) else str(o)) + "\n")
            for row in compared:
                handle.write(json.dumps({"kind": "compare", **row}, ensure_ascii=False) + "\n")
    if not args.skip_app:
        app = app_smoke(images[0])
        (RUN_DIR / "app_smoke.json").write_text(dump(app) + "\n", encoding="utf-8")
    write_summary(resolution, compared, app)
    print("SMOKE 写出", RUN_DIR / "summary.md", flush=True)


if __name__ == "__main__":
    main()
