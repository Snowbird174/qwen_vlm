# 指标定义（冻结）

实现：`src/qwen_vlm/evaluation/`（由评测 agent 负责）。输入是数据 JSONL（金标准，字段见 `sample.schema.json`）和逐样本预测 JSONL（见 `runner_api.md`），通过 `sample_id` 对齐。

## 通用规则

- 解析失败的样本（`parse_status != ok`）留在所有分母中。计算类别状态时，失败样本的每个类别都记为特殊预测 `invalid`，任何真值下都算错。完整性决策也记为 `invalid`，同样算错。
- 完整性决策只能由 `common/derive.derive(pred_states, required_items)` 计算，与金标准 `derive(gold_states, required_items)` 比较。
- 所有比例都要同时报告分子、分母和样本数。

## 主任务（task = checklist）

| 指标 | 定义 |
| --- | --- |
| 三状态准确率 | 在所有 checklist 样本 × 5 个类别上，预测状态等于真值的比例；同时报逐类结果 |
| 宏平均 F1 | 以 present/absent/unknown 三类分别算 F1 再平均；`invalid` 只计入 FN，不单独成一类 |
| 混淆矩阵 | 行是真值（3 类），列是预测（3 类加 invalid） |
| 缺失集合 P/R/F1 | 只看所需器材：预测的 `missing_required` 与真值做集合比较，在全部样本上微平均（TP/FP/FN 累加）；同时报逐类结果。如果某个样本真值和预测的缺失集合都为空，该样本不贡献 TP/FP/FN，但计入"集合完全匹配率" |
| 集合完全匹配率 | 预测的 `missing_required` 和 `uncertain_required` 都与真值完全相同的样本比例 |
| 决策准确率 | complete/incomplete/unknown 三分类准确率及其混淆矩阵 |
| **误报齐全率** | 真值决策为 `incomplete` 的样本中，预测决策为 `complete` 的比例 |
| unknown 样本误判齐全率 | 真值决策为 `unknown` 的样本中，预测为 `complete` 的比例 |
| unknown 召回 | 真值为 unknown 的（样本, 类别）中，预测也为 unknown 的比例 |
| 明确回答覆盖率 | 真值为 present/absent 的（样本, 类别）中，预测为 present/absent 的比例 |
| 明确回答子集准确率 | 在上一行中预测为明确回答的那部分里，预测正确的比例 |
| JSON 有效率 / 重试率 / 失败率 | 首次即解析成功的比例 / `retry_count > 0` 的比例 / 最终 `parse_status != ok` 的比例 |

## 辅助任务（task = presence）

只看 `target_item` 的三状态准确率，以及该类别的混淆矩阵。

## 成对一致性

对每个 `pair_group_id` 中，被移动的那件器材（两张图里该类别真值不同）：只有两张图上该类别的预测都等于各自的真值才算一致。分母是这样的图片对的数量。不能只检查两次回答是否不同。

## 置信区间

按 `scene_group_id` 做聚类 bootstrap：每次有放回地抽取场景组，组内样本全部带上。默认 1000 次，报 95% 百分位区间，固定随机种子。样本太少时（场景组少于 10 个）在表中标注"样本不足"。

## 速度与资源（板端）

- 每个配置先预热至少 5 次，正式测量至少 100 次或整个测试集，batch=1；报告 p50、p95、均值和 n。
- 下列指标单位不同，必须分开报告：完整点检请求数/秒，生成 token/s，首 token 延迟（秒），摄像头显示 FPS。
- 资源数据来自 `tegrastats`：进程和整机内存（Orin 上 CPU/GPU 共享）、GPU 利用率、温度、功耗模式。拿不到实测功率时标"未测"。
