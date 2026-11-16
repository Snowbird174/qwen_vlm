"""用 LlamaCppRunner（llama-server HTTP）对 JSONL 逐条推理，输出格式与 predict_transformers.py 相同。

两种用法：
  1. 连接已在运行的 llama-server（例如 Jetson 上由 deploy/jetson 脚本启动的服务）：
       --server http://127.0.0.1:8080
     此时服务端启动时的 --image-max-tokens 必须与 --max-pixels 对应（每个视觉 token 为 32x32 像素）。
  2. 由本脚本在本机启动 llama-server，推理结束后关闭：
       --model artifacts/quantized/base/base-q4_k_m.gguf --mmproj artifacts/quantized/base/mmproj-base-f16.gguf

不导入 torch/transformers，可在 Jetson 上运行。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.data.io import load_jsonl, resolve_image_path
from qwen_vlm.inference.llamacpp_runner import LlamaCppRunner, llama_server_command, wait_until_healthy

DEFAULT_BIN = PROJECT_ROOT / "vendor/llama.cpp/build/bin/llama-server"


def _abs(p: str | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    ap = argparse.ArgumentParser(description="llama.cpp 批量推理，写出 predictions.jsonl")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--server", default=None, help="已运行的 llama-server 地址；不给则用 --model/--mmproj 启动本机服务")
    ap.add_argument("--model", default=None, help="语言 GGUF")
    ap.add_argument("--mmproj", default=None, help="视觉 mmproj GGUF")
    ap.add_argument("--server-bin", default=str(DEFAULT_BIN))
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-pixels", type=int, default=512 * 512)
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    output = _abs(args.output) or PROJECT_ROOT / "runs" / args.experiment_id / "predictions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(_abs(args.jsonl))
    if args.limit is not None:
        samples = samples[: args.limit]

    proc = None
    log_file = None
    base_url = args.server
    if base_url is None:
        if not (args.model and args.mmproj):
            ap.error("不给 --server 时必须同时给 --model 和 --mmproj")
        cmd = llama_server_command(
            args.server_bin, _abs(args.model), _abs(args.mmproj),
            port=args.port, max_pixels=args.max_pixels, min_pixels=args.min_pixels,
        )
        log_file = open(output.parent / "llama_server.log", "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        base_url = f"http://127.0.0.1:{args.port}"
    try:
        wait_until_healthy(base_url, process=proc)
        runner = LlamaCppRunner(
            base_url=base_url, max_pixels=args.max_pixels, min_pixels=args.min_pixels,
            max_new_tokens=args.max_new_tokens,
        )
        data_root = _abs(args.data_root)
        with output.open("w", encoding="utf-8") as handle:
            for index, sample in enumerate(samples, 1):
                task = sample.get("task", "checklist")
                result = runner.predict(
                    resolve_image_path(sample["image_path"], data_root),
                    sample["question_zh"],
                    sample["required_items"],
                    task=task,
                    target_item=sample.get("target_item"),
                )
                handle.write(json.dumps({"sample_id": sample["sample_id"], **result.to_dict()}, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{index}/{len(samples)}] {sample['sample_id']} task={task} parse={result.parse_status}")
        print(f"写出 {len(samples)} 行到 {output}")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=30)
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
