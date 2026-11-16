"""B0 smoke test: base Qwen3-VL-2B through TransformersRunner on synthetic placeholder images.

Checks (none of these are accuracy measurements):
  1. single-image inference runs and the output goes through the frozen parser + derive();
  2. the number of <|image_pad|> tokens in input_ids equals the visual token count from
     image_grid_thw, and changes with max_pixels;
  3. the image actually influences the forward pass: first-step logits differ between the
     synthetic image and a uniform gray image.

Writes runs/smoke_b0/<timestamp>/predictions.jsonl and summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.derive import derive
from qwen_vlm.common.prompts import checklist_user_prompt
from qwen_vlm.inference.transformers_runner import TransformersRunner

SMOKE_DIR = PROJECT_ROOT / "data/smoke/b0"


def make_images() -> list[tuple[Path, list[str]]]:
    """Three crude drawings on a 'bench'; they are placeholders, not tool photos."""
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("bench_hammer_screwdriver.png", ["hammer", "screwdriver"]),
        ("bench_empty.png", []),
        ("bench_multimeter.png", ["multimeter"]),
    ]
    out = []
    for name, items in specs:
        img = Image.new("RGB", (1280, 960), (186, 160, 120))
        d = ImageDraw.Draw(img)
        d.rectangle([40, 40, 1240, 920], outline=(80, 60, 40), width=6)
        if "hammer" in items:
            d.rectangle([200, 400, 620, 440], fill=(120, 70, 30))
            d.rectangle([580, 330, 660, 510], fill=(90, 90, 95))
        if "screwdriver" in items:
            d.rectangle([760, 600, 900, 650], fill=(200, 30, 30))
            d.rectangle([900, 615, 1120, 635], fill=(170, 170, 175))
        if "multimeter" in items:
            d.rounded_rectangle([450, 250, 800, 750], radius=30, fill=(240, 200, 0))
            d.rectangle([500, 300, 750, 420], fill=(60, 80, 60))
            d.ellipse([560, 480, 690, 610], fill=(40, 40, 40))
        path = SMOKE_DIR / name
        img.save(path)
        out.append((path, items))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pixels", type=int, default=512 * 512)
    args = ap.parse_args()

    run_dir = PROJECT_ROOT / "runs/smoke_b0" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    runner = TransformersRunner(max_pixels=args.max_pixels)
    load_s = time.perf_counter() - t0
    tok = runner.processor.tokenizer
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    required = ["screwdriver", "pliers", "multimeter", "hammer"]
    question = "按照本次实验清单，还缺少什么器材？"

    checks: dict[str, object] = {"model_load_s": load_s}
    preds = []
    for path, drawn in make_images():
        res = runner.predict(path, question, required)
        derived = derive(res.parsed, required) if res.parsed else None
        rec = {"image": str(path.relative_to(PROJECT_ROOT)), "drawn_items": drawn, **res.to_dict(), "derived": derived}
        preds.append(rec)
        print(json.dumps({k: rec[k] for k in ["image", "parse_status", "raw_text", "n_image_tokens",
                                              "image_size_after_resize", "derived"]}, ensure_ascii=False))

    img = Image.open(make_images()[0][0]).convert("RGB")
    user_text = checklist_user_prompt(question, required)
    pad_counts = {}
    for mp in (384 * 384, 512 * 512, 768 * 768):
        runner.max_pixels = mp
        inputs = runner.prepare_inputs(img, user_text)
        n_pad = int((inputs["input_ids"] == image_pad_id).sum())
        n_grid = int(inputs["image_grid_thw"][0].prod()) // runner.merge_size**2
        pad_counts[mp] = {"image_pad_tokens": n_pad, "grid_tokens": n_grid,
                          "grid_thw": inputs["image_grid_thw"][0].tolist()}
        assert n_pad == n_grid, (mp, n_pad, n_grid)
    checks["max_pixels_to_tokens"] = pad_counts
    assert len({v["image_pad_tokens"] for v in pad_counts.values()}) == 3, "max_pixels did not change token count"

    runner.max_pixels = args.max_pixels
    with torch.inference_mode():
        la = runner.model(**runner.prepare_inputs(img, user_text)).logits[0, -1].float()
        gray = Image.new("RGB", img.size, (128, 128, 128))
        lb = runner.model(**runner.prepare_inputs(gray, user_text)).logits[0, -1].float()
    diff = (la - lb).abs().max().item()
    checks["first_step_logit_maxabs_diff_vs_gray"] = diff
    assert diff > 1e-3, "image does not influence the logits"

    checks["parse_status"] = [p["parse_status"] for p in preds]
    checks["peak_gpu_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    with open(run_dir / "predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))
    print("SMOKE_B0 PASS", run_dir)
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
