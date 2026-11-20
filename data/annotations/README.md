# 标注文件

- `s001.csv`：session_001 的一图一行状态标注（用 `scripts/data_label_tool.py` 标注）。`layout_id` 按器材框位置自动分组后确认：第 1-6 张各一组，第 7-8 张 l007，第 9-25 张 l008，第 26-50 张 l009（含遮挡照片）。
- `real_v1.jsonl`：由 `s001.csv` 展开的样本（100 条 checklist，种子 0）。
- `annotations.json`：每张图各器材的框，Grounding DINO（base）自动检测后由标注者逐张复核修正，可作为检测真值。坐标为原图像素 `[x1, y1, x2, y2]`（1920x1080），类别 `cable` 对应本项目的 `cable_bundle`。与状态标注一致：标为 present 的器材都有框；第 45、46 张部分被遮挡的器材标为 unknown 但仍有框。
