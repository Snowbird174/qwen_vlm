"""Gradio 演示：摄像头或保存的图片 → LlamaCppRunner → 五类状态和 derive 中文回答。

摄像头在服务器上没有设备，只能在 Jetson 上试。图片模式不依赖摄像头。
本模块不导入 torch 或 transformers。
"""

from __future__ import annotations

import argparse
from datetime import datetime

import gradio as gr
from PIL import Image

from qwen_vlm.app.gate import BusyError, InferenceGate
from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATE_NAME_ZH
from qwen_vlm.common.derive import derive
from qwen_vlm.inference.llamacpp_runner import PIXEL_TIERS, LlamaCppRunner, server_image_token_flags

PRESET_QUESTIONS = [
    "按照本次实验清单，还缺少什么器材？",
    "画面里五类器材分别在不在？",
    "检查区域里的器材是否齐全？",
]

BUSY_TEXT = "正忙：上一个推理还没结束，请等它完成后再试。同一时间只处理一个请求，新请求不会排队。"


def grab_opencv_frame(device: int) -> Image.Image:
    """从 /dev/videoN 读一帧。OpenCV 默认是 BGR，这里转成 RGB 再交给 runner。"""
    import cv2

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"打不开摄像头 /dev/video{device}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"从 /dev/video{device} 读帧失败")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _as_pil(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return None


def format_states(parsed: dict | None, required: list[str]) -> str:
    if not parsed:
        return "模型输出没有通过解析，下面没有状态表。"
    lines = ["| 类别 | 状态 | 是否在本次清单 |", "| --- | --- | --- |"]
    required_set = set(required)
    for cat in CATEGORY_IDS:
        state = parsed.get(cat, "")
        state_zh = STATE_NAME_ZH.get(state, state)
        in_list = "是" if cat in required_set else "否"
        lines.append(f"| {NAME_ZH[cat]} | {state_zh}（{state}） | {in_list} |")
    return "\n".join(lines)


def format_timings(result) -> str:
    parts = [f"总耗时 {result.timings.get('total_s', 0):.3f} 秒", f"解析 {result.parse_status}", f"重试 {result.retry_count} 次"]
    if result.n_generated_tokens is not None:
        parts.append(f"生成 {result.n_generated_tokens} token")
    prompt_tokens = result.timings.get("attempt0_prompt_tokens")
    if prompt_tokens is not None:
        parts.append(f"prompt {int(prompt_tokens)} token")
    first = result.timings.get("attempt0_first_token_s")
    if first is not None:
        parts.append(f"prefill {first:.3f} 秒")
    gen = result.timings.get("attempt0_generate_s")
    if gen is not None:
        parts.append(f"生成 {gen:.3f} 秒")
    return "；".join(parts) + "。"


def build_demo(runner: LlamaCppRunner, camera_device: int, pixel_tier: str) -> gr.Blocks:
    gate = InferenceGate()
    min_tokens, max_tokens = runner.min_image_tokens, runner.max_image_tokens
    flag_text = (
        f"当前界面的像素档是 {pixel_tier}（max_pixels={runner.max_pixels}）。"
        f"llama-server 必须带 `--image-min-tokens {min_tokens} --image-max-tokens {max_tokens}`，"
        "改档位要重启服务，不能只改这个网页。"
    )

    def preview_camera():
        try:
            frame = grab_opencv_frame(camera_device)
        except Exception as exc:  # noqa: BLE001 — 预览失败要显示在界面上
            return None, f"预览失败：{exc}"
        return frame, f"预览时间 {datetime.now().isoformat(timespec='seconds')}（还没用于推理）"

    def run_infer(mode: str, saved_image, webcam_image, required: list[str], question: str):
        captured_at = datetime.now().isoformat(timespec="seconds")
        if mode == "OpenCV 摄像头":
            try:
                image = grab_opencv_frame(camera_device)
            except Exception as exc:  # noqa: BLE001
                return None, captured_at, "", "", "", f"摄像头失败：{exc}"
        elif mode == "浏览器摄像头":
            image = _as_pil(webcam_image)
        else:
            image = _as_pil(saved_image)
        if image is None:
            return None, captured_at, "", "", "", "没有图片。请上传图片，或切换到摄像头模式。"
        question = (question or "").strip()
        if not question:
            return image, captured_at, "", "", "", "请填写中文问题。"
        try:
            with gate:
                result = runner.predict(image, question, list(required or []))
        except BusyError:
            return image, captured_at, "", "", "", BUSY_TEXT
        except Exception as exc:  # noqa: BLE001
            return image, captured_at, "", "", "", f"推理失败：{exc}"

        if result.parsed is None:
            answer = "模型输出无法解析，没有调用 derive。"
            missing = ""
        else:
            decision = derive(result.parsed, list(required or []))
            answer = decision["answer_zh"]
            missing_ids = decision["missing_required"]
            missing = "、".join(NAME_ZH[c] for c in missing_ids) if missing_ids else "（清单内没有缺失项）"
            if decision["uncertain_required"]:
                missing += "；无法确认：" + "、".join(NAME_ZH[c] for c in decision["uncertain_required"])
        table = format_states(result.parsed, list(required or []))
        raw = result.raw_text.strip()
        body = f"{answer}\n\n原始输出：\n{raw}"
        return image, captured_at, table, missing, body, format_timings(result)

    with gr.Blocks(title="实验室器材完整性点检") as demo:
        gr.Markdown(
            "## 实验室器材完整性点检\n"
            "固定俯视画面 + 所需器材清单。模型只判断五类器材在不在，是否齐全由程序推导。\n\n"
            f"**并发规则：** {BUSY_TEXT}\n\n"
            f"{flag_text}\n\n"
            "服务器上通常没有 `/dev/video*`，摄像头预览要到 Jetson 上才测得了。这里可以用「读取保存图片」。"
        )
        with gr.Row():
            mode = gr.Radio(
                ["读取保存图片", "浏览器摄像头", "OpenCV 摄像头"],
                value="读取保存图片",
                label="画面来源",
            )
            question = gr.Textbox(label="中文问题", value=PRESET_QUESTIONS[0], lines=2)
        preset = gr.Dropdown(PRESET_QUESTIONS, label="预设问法", value=PRESET_QUESTIONS[0])
        required = gr.CheckboxGroup(
            choices=[(f"{NAME_ZH[c]}（{c}）", c) for c in CATEGORY_IDS],
            value=list(CATEGORY_IDS),
            label="本次实验所需器材（可改）",
        )
        with gr.Row():
            saved = gr.Image(label="保存的图片", type="pil", sources=["upload", "clipboard"])
            webcam = gr.Image(label="浏览器摄像头", type="pil", sources=["webcam"])
        with gr.Row():
            preview = gr.Image(label="OpenCV 预览（未送去推理）", type="pil", interactive=False)
            used = gr.Image(label="本次用于推理的画面", type="pil", interactive=False)
        preview_note = gr.Textbox(label="预览状态", interactive=False)
        refresh = gr.Button("刷新 OpenCV 预览")
        go = gr.Button("采集并推理", variant="primary")
        capture_time = gr.Textbox(label="采集时间", interactive=False)
        state_table = gr.Markdown(label="五类状态")
        missing = gr.Textbox(label="缺失与无法确认", interactive=False)
        answer = gr.Textbox(label="derive 的中文回答和原始输出", lines=8, interactive=False)
        status = gr.Textbox(label="耗时与状态", interactive=False)

        preset.change(lambda text: text, inputs=preset, outputs=question)
        refresh.click(preview_camera, outputs=[preview, preview_note])
        go.click(
            run_infer,
            inputs=[mode, saved, webcam, required, question],
            outputs=[used, capture_time, state_table, missing, answer, status],
            api_name="infer",
        )
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="器材点检演示界面")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="llama-server 地址")
    parser.add_argument("--pixels", choices=sorted(PIXEL_TIERS), default="512", help="像素预算档位")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 摄像头设备号，对应 /dev/videoN")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    max_pixels = PIXEL_TIERS[args.pixels]
    runner = LlamaCppRunner(base_url=args.server, max_pixels=max_pixels, max_new_tokens=args.max_new_tokens)
    min_tokens, max_tokens = server_image_token_flags(max_pixels, runner.min_pixels)
    print(
        f"像素档 {args.pixels}：请确认 llama-server 使用 "
        f"--image-min-tokens {min_tokens} --image-max-tokens {max_tokens}",
        flush=True,
    )
    demo = build_demo(runner, args.camera, args.pixels)
    demo.launch(server_name=args.host, server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()
