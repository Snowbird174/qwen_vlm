"""单图点检命令行：输入一张图片和所需器材清单，输出五类状态和完整性结论。

例：
  python scripts/check_image.py photo.jpg --items 螺丝刀 钳子 万用表
  python scripts/check_image.py photo.jpg --backend llamacpp --gguf q4_k_m --max-pixels 147456
  python scripts/check_image.py photo.jpg --adapter runs/M/adapter      # 微调后的模型
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATE_NAME_ZH
from qwen_vlm.common.derive import derive

DECISION_ZH = {"complete": "齐全", "incomplete": "不齐全", "unknown": "无法确认"}
QUANT_DIR = PROJECT_ROOT / "artifacts/quantized/base"


def build_runner(args):
    if args.backend == "transformers":
        from qwen_vlm.inference.transformers_runner import TransformersRunner

        runner = TransformersRunner(max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens)
        if args.adapter:
            from peft import PeftModel

            runner.model = PeftModel.from_pretrained(runner.model, PROJECT_ROOT / args.adapter).eval()
        return runner, None

    from qwen_vlm.inference.llamacpp_runner import LlamaCppRunner, llama_server_command, wait_until_healthy

    proc = None
    url = args.server
    if url is None:
        cmd = llama_server_command(
            PROJECT_ROOT / "vendor/llama.cpp/build/bin/llama-server",
            QUANT_DIR / f"base-{args.gguf}.gguf",
            QUANT_DIR / "mmproj-base-f16.gguf",
            port=args.port,
            max_pixels=args.max_pixels,
        )
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{args.port}"
    wait_until_healthy(url, process=proc)
    return LlamaCppRunner(base_url=url, max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens), proc


def main() -> None:
    ap = argparse.ArgumentParser(description="单图器材点检")
    ap.add_argument("image")
    ap.add_argument("--items", nargs="*", default=list(CATEGORY_IDS), help="所需器材，ID 或中文名；默认五类全部")
    ap.add_argument("--question", default="按照本次实验清单，器材是否齐全？缺少什么？")
    ap.add_argument("--backend", choices=["transformers", "llamacpp"], default="transformers")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录（仅 transformers 后端）")
    ap.add_argument("--gguf", choices=["f16", "q8_0", "q4_k_m"], default="q4_k_m", help="llamacpp 后端的量化档")
    ap.add_argument("--server", default=None, help="已运行的 llama-server 地址；不给则自动启动")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--max-pixels", type=int, default=512 * 512, help="像素预算：147456=384²，262144=512²，589824=768²")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    runner, proc = build_runner(args)
    try:
        res = runner.predict(args.image, args.question, args.items)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=30)

    derived = derive(res.item_states, args.items) if res.item_states else None
    if args.json:
        print(json.dumps({"result": res.to_dict(), "derived": derived}, ensure_ascii=False, indent=2))
        return

    print(f"后端：{res.backend}  像素预算：{args.max_pixels}  实际输入尺寸：{res.image_size_after_resize}  视觉 token：{res.n_image_tokens}")
    print(f"耗时：{res.timings['total_s']:.2f} 秒  解析：{res.parse_status}  重试：{res.retry_count}")
    if res.item_states is None:
        print("模型输出无法解析，原始输出：")
        print(res.raw_text)
        sys.exit(1)
    print("\n器材状态（* 为本次清单内）：")
    for c in CATEGORY_IDS:
        mark = "*" if c in args.items or NAME_ZH[c] in args.items else " "
        print(f"  {mark} {NAME_ZH[c]:<4} {STATE_NAME_ZH[res.item_states[c]]}（{res.item_states[c]}）")
    print(f"\n结论：{DECISION_ZH[derived['decision']]}")
    print(f"回答：{derived['answer_zh']}")


if __name__ == "__main__":
    main()
