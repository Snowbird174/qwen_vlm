# Runner 接口（冻结）

代码：`src/qwen_vlm/inference/base.py`。只有主 agent 能修改；其他 agent 发现问题要上报，不能自行改动。

## 调用

```python
runner.predict(image, question_zh, required_items, task="checklist", target_item=None) -> PredictResult
```

- `image`：图片路径或 PIL 图片。`load_image` 统一应用 EXIF 方向并转成 RGB，各后端不得再做额外的颜色或方向处理。
- `required_items`：类别 ID 或中文名都可以，内部用 `normalize_required_items` 规范化；遇到不支持的类别抛 `UnsupportedCategoryError`。
- `task="presence"` 时必须传 `target_item`。
- 构造参数 `max_pixels`/`min_pixels` 是视觉输入的像素预算，`max_new_tokens` 是生成上限。所有后端统一贪心解码。

## 后端实现

子类只实现 `_generate(image: PIL.Image, user_text: str) -> GenerationOutput`：
- system prompt 用 `common/prompts.py` 里的 `SYSTEM_PROMPT`，user turn 是 `[image, user_text]`，使用官方 chat template。
- 返回生成文本、生成 token 数、视觉 token 数、实际送入视觉编码器的 `(宽, 高)`，以及各阶段耗时（秒）。后端拿不到的字段填 `None`，不要估算。
- GPU 计时前要同步。

已有后端：`transformers_runner.TransformersRunner`。部署 agent 将添加 `llamacpp_runner.LlamaCppRunner`。

## 统一策略（在基类中实现）

- 解析：`common/schema.parse_model_output`。只去掉首尾空白和代码块标记，取第一个 JSON 对象，然后按 schema 校验，**不修正任何值**。
- 重试：解析失败时最多重试 1 次（`MAX_RETRIES = 1`），重试耗时计入 `timings["total_s"]`。
- 失败状态包括 `no_json`、`invalid_json`、`schema_error`。评测时这些样本保留在分母中，按失败计。

## PredictResult 字段

`task, raw_text, raw_attempts, parsed, parse_status, retry_count, n_generated_tokens, n_image_tokens, image_size_after_resize, timings, backend, prompt_version`

逐样本预测文件（JSONL）中每行写 `{"sample_id": ..., **PredictResult.to_dict()}`，评测代码只读取这个格式。
