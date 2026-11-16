# 实验登记表

所有实验使用同一基础 revision（`89644892e4d85e24eaac8bacfd4f463576704203`）、同一数据划分、同一 prompt 版本（`common/prompts.PROMPT_VERSION`）、同一评分脚本。每个实验的完整记录在 `runs/<experiment_id>/run_manifest.json`。**目前没有真实数据，下表的结果列全部为空。**

## 1. 模型改进（服务器，Transformers bf16）

| ID | 配置 | 回答的问题 | 训练配置 | 状态 |
| --- | --- | --- | --- | --- |
| B0 | 原始 Qwen3-VL-2B，统一提示，不训练 | 原模型在真实场景中的表现 | — | 待真实测试集 |
| B1 | 普通领域 LoRA，只有主任务，随机采样 | 单纯领域适应的效果 | `configs/train/B1.yaml` | 待数据 |
| M | 成对辅助任务 + 困难负样本加权 + unknown 监督 | 完整方法是否有效 | `configs/train/M.yaml` | 待数据 |
| A1 | M 去掉成对辅助任务 | 成对存在性监督的贡献 | `configs/train/A1.yaml` | 待数据 |
| A2 | M 去掉困难负样本加权 | 困难负样本策略的贡献 | `configs/train/A2.yaml` | 待数据 |
| A3 | M 去掉 unknown 样本 | unknown 监督的收益与代价 | `configs/train/A3.yaml` | 待数据 |

- 训练预算匹配：所有训练实验固定 `max_steps` 和有效 batch；差异写入 manifest。
- 随机种子：B1 和 M 计划各跑 3 个种子；消融先跑单种子。算力不足时如实写明只有单种子。
- 主要指标：误报齐全率（附分母）、决策准确率、三状态宏 F1、缺失集合 F1、unknown 召回与覆盖率、成对一致性、JSON 有效率；按 `scene_group_id` 做 bootstrap 95% 置信区间。定义见 `contracts/metrics.md`。

## 2. 端侧优化（Jetson Orin Nano 8GB，llama.cpp）

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
