# 训练样本混合配置（冻结）

由训练 agent 实现在 `src/qwen_vlm/training/mix.py`：

```python
build_training_mix(samples: list[dict], mix_cfg: dict, seed: int) -> tuple[list[dict], dict]
```

返回 `(本 epoch 的训练样本列表, 暴露统计)`。输入 `samples` 是 split 后的训练集 JSONL（`sample.schema.json` 格式），只用 train split。

## mix_cfg 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `include_unknown` | bool | false 时丢弃所有含 unknown 真值的样本（A3 用） |
| `aux_presence` | `none` / `pairs` | `pairs`：对每个 `pair_group_id` 中被移动的器材（两张图该类别真值不同），在两张图上各生成一条 presence 样本（`task=presence`、`target_item` 为该器材、问题从固定问法集合中选），同问不同答。`none`：不生成（B1、A1 用） |
| `aux_ratio` | float | presence 样本数 ÷ checklist 样本数的上限；超出时随机下采样 |
| `hard_negative_weight` | float | 带 `hard_negative_tags` 的样本的重复采样权重；1.0 等于不做针对性采样（B1、A2 用） |
| `balance_states` | bool | 是否按（类别, 状态）做反频率重采样，让各类别缺失/存在的分布均衡 |
| `epoch_size` | int 或 null | 每个 epoch 的样本数；null 表示等于重采样后的样本数 |

数据 JSONL 中已有的 presence 样本直接保留，不受 `aux_presence` 控制；`aux_presence` 只控制是否从成对场景自动生成。

## 实验预设（`configs/train/*.yaml`）

| 实验 | include_unknown | aux_presence | hard_negative_weight | balance_states |
| --- | --- | --- | --- | --- |
| B1 | true | none | 1.0 | false |
| M | true | pairs | 选定值（如 3.0） | true |
| A1 | true | none | 同 M | true |
| A2 | true | pairs | 1.0 | true |
| A3 | false | pairs | 同 M | true |

各实验应尽量匹配总优化步数和有效 batch（固定 `max_steps`），差异记录在运行 manifest 中。

## 暴露统计（必须写入每次运行的 manifest）

按 task、（类别, 状态）、hard_negative、unknown、来自 pairs 的 presence 样本分别计数，另记总样本数和唯一图片数。
