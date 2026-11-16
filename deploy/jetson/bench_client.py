#!/usr/bin/env python3
"""对已经启动的 llama-server 跑一组图片，写成评测脚本能读的 JSONL。

每行是 {"sample_id": ..., **PredictResult.to_dict()}。失败也写一行，不让整批中断。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from qwen_vlm.common.derive import derive
from qwen_vlm.inference.llamacpp_runner import PIXEL_TIERS, LlamaCppRunner


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, tuple) else str(o))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--pixels", choices=sorted(PIXEL_TIERS), required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--question", default="按照本次实验清单，还缺少什么器材？")
    parser.add_argument(
        "--required",
        default="screwdriver,pliers,cable_bundle,multimeter,hammer",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    required = [item for item in args.required.split(",") if item]
    runner = LlamaCppRunner(
        base_url=args.server,
        max_pixels=PIXEL_TIERS[args.pixels],
        max_new_tokens=args.max_new_tokens,
    )
    images = [Path(p) for p in args.image]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def one(image: Path, sample_id: str, warmup: bool) -> None:
        record: dict = {
            "sample_id": sample_id,
            "image": str(image),
            "quant": args.quant,
            "pixels": args.pixels,
            "warmup": warmup,
        }
        try:
            result = runner.predict(image, args.question, required)
            record.update(result.to_dict())
            record["derived"] = derive(result.parsed, required) if result.parsed else None
            record["error"] = None
        except Exception as exc:  # noqa: BLE001 — 单次失败要留下来，不能弄垮整批
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc(limit=8)
            print(record["error"], file=sys.stderr)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(dump(record) + "\n")

    for index in range(args.warmup):
        image = images[index % len(images)]
        one(image, f"warmup_{args.quant}_{args.pixels}_{index:03d}", True)
    for index in range(args.repeat):
        image = images[index % len(images)]
        one(image, f"{image.stem}_{args.quant}_{args.pixels}_{index:03d}", False)
    print(f"写出 {args.out}")


if __name__ == "__main__":
    main()
