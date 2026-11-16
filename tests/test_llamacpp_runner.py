"""不占 GPU 的 llama.cpp runner 测试：请求构造、响应解析、像素档换算。"""

import base64
import io
import json

from PIL import Image

from qwen_vlm.common.prompts import SYSTEM_PROMPT, build_messages, checklist_user_prompt
from qwen_vlm.inference.llamacpp_runner import (
    PIXEL_TIERS,
    LlamaCppRunner,
    encode_image_data_url,
    llama_server_command,
    mtmd_cli_command,
    parse_chat_completion,
    server_image_token_flags,
    smart_resize,
    smart_resize_llama_still,
    vision_token_count,
)

GOOD = {
    "screwdriver": "present",
    "pliers": "absent",
    "cable_bundle": "present",
    "multimeter": "unknown",
    "hammer": "present",
}


def test_pixel_tiers_match_transformers_budget():
    assert server_image_token_flags(PIXEL_TIERS["384"], None) == (64, 144)
    assert server_image_token_flags(PIXEL_TIERS["512"], None) == (64, 256)
    assert server_image_token_flags(PIXEL_TIERS["768"], None) == (64, 576)


def test_smart_resize_matches_known_1280x960_counts():
    expected = {384 * 384: 130, 512 * 512: 234, 768 * 768: 540}
    for max_pixels, tokens in expected.items():
        height, width = smart_resize(960, 1280, max_pixels=max_pixels)
        height_f, width_f = smart_resize_llama_still(960, 1280, max_pixels=max_pixels)
        assert (height, width) == (height_f, width_f)
        assert vision_token_count(height, width) == tokens


def test_request_uses_shared_prompts_and_rgb_png():
    image = Image.new("RGB", (8, 6), (255, 0, 0))
    user = checklist_user_prompt("还缺少什么器材？", ["锤子", "钳子"])
    runner = LlamaCppRunner(max_pixels=512 * 512, max_new_tokens=128)
    payload = runner.build_request(image, user)
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 128
    assert payload["messages"][0]["content"] == SYSTEM_PROMPT
    assert payload["messages"][1]["content"][1]["text"] == user
    messages = build_messages(user, image)
    assert messages[1]["content"][1]["text"] == user
    url = payload["messages"][1]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (255, 0, 0)


def test_parse_server_response_timings():
    data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps({"item_states": GOOD}, ensure_ascii=False)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 400, "completion_tokens": 40},
        "timings": {"prompt_n": 400, "prompt_ms": 120.0, "predicted_n": 40, "predicted_ms": 80.0},
    }
    out = parse_chat_completion(data, encode_s=0.01, http_s=0.2)
    assert out.n_generated_tokens == 40
    assert out.n_image_tokens is None
    assert out.image_size_after_resize is None
    assert out.timings["first_token_s"] == 0.12
    assert out.timings["generate_s"] == 0.08
    assert out.timings["prompt_tokens"] == 400.0
    assert "螺丝刀" not in out.text  # 规范输出是英文枚举；这里只确认 JSON 被原样保留
    assert "present" in out.text


def test_parse_list_content_and_missing_timings():
    data = {"choices": [{"message": {"content": [{"type": "text", "text": "中文说明"}, {"type": "text", "text": " {"}]}}]}
    out = parse_chat_completion(data, encode_s=0.0, http_s=0.0)
    assert out.text == "中文说明 {"
    assert out.n_generated_tokens is None
    assert "first_token_s" not in out.timings


def test_server_command_contains_token_flags():
    cmd = llama_server_command(
        "llama-server",
        "model.gguf",
        "mmproj.gguf",
        max_pixels=768 * 768,
        port=8088,
    )
    text = " ".join(cmd)
    assert "--image-min-tokens 64" in text
    assert "--image-max-tokens 576" in text
    assert "--jinja" in text
    assert "--temp 0" in text
    assert "--mmproj mmproj.gguf" in text
    cli = mtmd_cli_command("llama-mtmd-cli", "m.gguf", "p.gguf", "a.png", "问题：齐全吗？", max_pixels=384 * 384)
    assert "--image-max-tokens" in cli
    assert cli[cli.index("--image-max-tokens") + 1] == "144"
    assert "问题：齐全吗？" in cli


def test_runner_sees_exif_transposed_image(tmp_path):
    """基类先做 EXIF 校正。runner 收到的是校正后的 RGB，不再自己转一次。"""
    image = Image.new("RGB", (4, 2), (0, 0, 255))
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "turned.jpg"
    image.save(path, format="JPEG", exif=exif.tobytes())

    class Capture(LlamaCppRunner):
        def _generate(self, image, user_text):
            from qwen_vlm.inference.base import GenerationOutput

            self.seen = image
            self.user_text = user_text
            return GenerationOutput(text=json.dumps({"item_states": GOOD}))

    runner = Capture()
    result = runner.predict(path, "画面里有没有锤子？", ["hammer"])
    assert result.parse_status == "ok"
    assert runner.seen.size == (2, 4)
    assert runner.seen.mode == "RGB"
    assert "锤子" in runner.user_text
    assert encode_image_data_url(runner.seen).startswith("data:image/png;base64,")
