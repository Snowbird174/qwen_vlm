# 实验登记表

所有实验使用同一基础 revision（`89644892e4d85e24eaac8bacfd4f463576704203`）、同一 prompt 版本（`common/prompts.PROMPT_VERSION`）和同一评分脚本。B0 在全部 50 张真实图上推理，其余方法把固定三折的真实测试预测合并后，用同一份 100 条 checklist 统一评分。训练 manifest 位于 `runs/cv3_folds/<ID>_fold<k>/`。

## 1. 模型改进（服务器，Transformers bf16）

| ID | 配置 | 回答的问题 | 训练配置 | 状态 |
| --- | --- | --- | --- | --- |
| B0 | 原始 Qwen3-VL-2B，统一提示，直接推理 | 原模型在真实场景中的表现 | — | `runs/B0_s001/eval/` |
| B1 | 真实图，普通领域 LoRA，只有主任务，随机采样 | 单纯领域适应的效果 | `configs/train/B1.yaml` | `runs/B1_cv3/eval/` |
| M | 真实图与合成图，成对辅助、困难负样本加权、状态均衡与 unknown 监督 | 完整方法是否有效 | `configs/train/M.yaml` | `runs/M_cv3/eval/` |
| A1 | M 去掉成对辅助任务 | 成对存在性监督的贡献 | `configs/train/A1.yaml` | `runs/A1_cv3/eval/` |
| A2 | M 去掉困难负样本加权 | 困难负样本策略的贡献 | `configs/train/A2.yaml` | `runs/A2_cv3/eval/` |
| A3 | M 去掉 unknown 样本 | unknown 监督的收益与代价 | `configs/train/A3.yaml` | `runs/A3_cv3/eval/` |
| A4 | M 的混合策略，仅用真实图 | 合成训练图的贡献 | `configs/train/A4.yaml` | `runs/A4_cv3/eval/` |

- 固定三折按摆放组划分。三折测试分别有 25、17、8 张真实图，同一图的两条问法随图分组。合成图继承来源图的摆放组，仅用于训练折。
- 所有方法和折共用 50 个优化步、有效 batch 16 和随机种子 42。步数依据训练 loss 与各折训练池大小确定，见 `docs/NEXT_STEPS.md`。测试折用于最终预测和评分。数据、配置和实际步数由 manifest 记录。
- 主要指标：误报齐全率（附分母）、决策准确率、三状态宏 F1、缺失集合 F1、unknown 召回与覆盖率、成对一致性、JSON 有效率；按 `scene_group_id` 做 bootstrap 95% 置信区间。定义见 `contracts/metrics.md`。
- 真实样本来自 1 个批次、9 个摆放组。聚类 bootstrap 标注样本不足，三折大小不均。全量 checklist 的每个真实 pair 有四条样本，冻结的成对一致性规则要求恰好两条，因此主表该项为无定义。

### 折外结果与逐样本复核

| 方法 | 决策准确率 | 误报齐全率 | unknown 召回 |
| --- | ---: | ---: | ---: |
| B0 | 80/100 | 7/31 | 2/18 |
| B1 | 87/100 | 2/31 | 0/18 |
| M | 86/100 | 2/31 | 1/18 |
| A1 | 88/100 | 2/31 | 1/18 |
| A2 | 87/100 | 2/31 | 2/18 |
| A3 | 91/100 | 2/31 | 0/18 |
| A4 | 87/100 | 2/31 | 0/18 |

- B1 和 M 各少了 5 次误报齐全，点估计从 7/31 降为 2/31。与 B0 相比，两者的决策准确率分别多 7、6 条。按摆放组成对 bootstrap，准确率差的 95% 区间分别为 [−2.1, +21.6]、[−4.0, +21.6] 个百分点，差异不显著，样本不足。
- 在相同混合策略下，M 相比真实图 A4 少判对 1 条，误报齐全持平，unknown 多召回 1 个。准确率差异不显著，样本不足，合成训练图的整体收益在这组数据中未得到验证。
- A1、A2 分别比 M 多判对 2 条和 1 条，准确率差异不显著，样本不足。A1 的 unknown 召回与 M 相同，A2 多召回 1 个。A3 去掉 unknown 监督后多判对 5 条，unknown 召回为 0/18。所有消融的误报齐全率均为 2/31。
- 第 14、22、27、32 张各两条问法。B0 将缺失的万用表判为 `present` 共 6/8 条，其中 4 条造成误报齐全。M 八条均判为 `absent`。B1 七条判为 `absent`，第 14 张第一条判为 `unknown`。B0、B1、M 分别把 23、1、4 条真值明确的螺丝刀状态判成 `unknown`。真实遮挡条件下 14 个 unknown 类别标签，B0 和 B1 均召回 0 个，M 召回 1 个。

各区间与样例 ID 见 [逐样本复核](runs/comparison/cv_case_review.txt)，完整指标见 [方法对比](runs/comparison/comparison.md)。两类真实干扰均集中在单个摆放组，unknown 结果需用独立采集验证。

## 2. 分辨率与量化

服务器 B0 以 Transformers BF16 扫描 384²、512²、768²，以 llama.cpp 在 512² 比较 F16、Q8_0、Q4_K_M，各档 mmproj 均为 F16。使用同一份真实金标准。结果见 `runs/comparison/`。

### Jetson Orin Nano 8GB

模型使用 M（或最终选定的模型）合并后转换的 GGUF。功耗模式固定，记录 `nvpmodel -q`。

| 步骤 | 变量 | 固定条件 | 状态 |
| --- | --- | --- | --- |
| O1 分辨率扫描 | max_pixels ∈ {384², 512², 768²} | 较高精度（F16，装不下时用 Q8_0） | 待板端 |
| O2 量化对比 | {F16, Q8_0, Q4_K_M}（mmproj 精度单独记录） | O1 在验证集上选出的分辨率 | 待板端 |
| O3 交叉复核 | 高/低分辨率 × 高/低精度 | — | 待板端 |

每个配置记录：任务指标（同一测试集）、首 token 延迟和总延迟的 p50/p95、生成 token/s、完整请求数/秒、峰值内存（tegrastats）、OOM 情况。

## 3. 冒烟运行（不是实验结果）

| 运行 | 内容 | 位置 |
| --- | --- | --- |
| smoke_b0 | 基础模型、3 张合成图：解析、视觉 token 数、图像影响 logits | `runs/smoke_b0/` |
| train_smoke | 合成数据过拟合、adapter 保存加载、合并一致性 | `runs/train_smoke*/`（训练 agent） |
| deploy_smoke | GGUF F16/Q8/Q4 与 Transformers 的输出对照、分辨率对照 | `runs/deploy_smoke/`（部署 agent） |
