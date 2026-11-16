# 实验与评测

本章说明比较什么、每个指标怎么算、区间怎么来。结果表里的数字全部留空，见 `reports/templates/results_tables.md`。在对应的 `runs/<experiment_id>/eval/metrics.json` 产生之前，一律写“待测”或“未测”，不预填。

## 1. 实验目的

任务是判断实验室画面里五类器材是否齐备。类别 ID 为 screwdriver、pliers、cable_bundle、multimeter、hammer。模型对每一类输出 present、absent 或 unknown。程序用冻结的 `common.derive` 结合所需器材清单得到 complete、incomplete 或 unknown。

最重要的指标是误报齐全率：真值决策已经是 incomplete，系统却报 complete。这种错误会让实验员以为器材已经齐了。解析失败必须留在分母里，不能把“没答上来”算成没参加评测。

## 2. 实验设置

### 2.1 数据与划分

目前还没有可用于报告结论的真实实验室数据。真实采集完成前，本章所有准确率、延迟和资源都是待测。划分单位是 `scene_group_id`：同一套摆放的全部图片必须进同一个 split，避免一张图训练、挪动后的另一张图测试。`pair_group_id` 连接只挪动一件器材的两张图，用来算成对一致性，不用来划分训练和测试。

金标准是符合 `contracts/sample.schema.json` 的 JSONL。预测是符合 `contracts/runner_api.md` 的 JSONL，每行是 `{"sample_id": ..., **PredictResult.to_dict()}`。checklist 的 `parsed` 是五类状态字典；presence 的 `parsed` 是 `{"item", "state"}`；解析失败时 `parsed` 为 null。

### 2.2 对比系统

训练混合的定义以 `contracts/training_mix.md` 为准。

| 实验 | 设置 |
| --- | --- |
| B0 | 未微调的 Qwen3-VL-2B-Instruct |
| B1 | 普通 LoRA：保留 unknown，不生成 presence 辅助样本，困难负样本权重 1.0，不做状态均衡 |
| M | 完整方法：保留 unknown，由成对场景生成 presence 辅助样本，困难负样本加权，并做状态均衡 |
| A1 | 相对 M 去掉 presence 辅助样本 |
| A2 | 相对 M 把困难负样本权重改为 1.0 |
| A3 | 相对 M 在训练时丢掉含 unknown 真值的样本 |

各微调实验应匹配总优化步数和有效 batch。实际权重、步数和学习率以该次运行的 manifest 为准，不在本章另写一套数字。解码统一贪心。提示词、解析和最多一次重试都走冻结的 Runner 基类，评测不再改模型输出里的任何字段。

### 2.3 端侧优化

部署目标是 Jetson Orin Nano 8GB。两项优化交叉：

- 量化：F16、Q8_0、Q4_K_M
- 输入像素预算：384²、512²、768²（传给视觉编码器的像素上限）

每个配置 batch=1，先预热至少 5 次，再测至少 100 次或整份测试集，以实际跑到的次数为准。服务器上的 Transformers 结果和板上的 llama.cpp 结果分开列表，不混在一个速度表里。

### 2.4 评测命令

```bash
python scripts/eval_predictions.py \
  --gold data/<split>.jsonl \
  --pred runs/<experiment_id>/predictions.jsonl \
  --out runs/<experiment_id>/eval \
  --experiment-id <experiment_id>
```

量化运行额外传入 `--quantization` 和 `--pixel-budget`。有板端日志时传入 `--tegrastats`。比较：

```bash
python scripts/eval_compare.py --runs-root runs --out runs/comparison
```

输出的 Markdown 和 CSV 只转录 metrics.json 里的分子、分母和区间。

## 3. 指标定义和计算方法

定义以 `contracts/metrics.md` 为准，实现在 `src/qwen_vlm/evaluation/`。每个比例同时给出 value、numerator、denominator。分母为 0 时 value 为空，不记成 0。

### 3.1 解析失败

`parse_status != ok`（包括 no_json、invalid_json、schema_error，以及金标准有但预测文件没有的 missing_prediction）时：

- 五个类别的预测都记为 invalid，和任何真值都算错
- 决策记为 invalid，不调用 derive
- 该样本仍留在三状态准确率、决策准确率、集合完全匹配率和误报齐全率的分母中

invalid 不是 absent，也不是 unknown。算三类 F1 时，它只进入真实类别的 FN，不进入任何一类的 FP，也不单独成一类。

集合完全匹配还要求预测的 `missing_required` 和 `uncertain_required` 都与真值相同。解析失败时这两个集合没有定义，匹配记为否。即使真值两边都为空，也不把失败预测当成空集合去判对。缺失集合的微平均里，失败预测不产生“预测为缺失”的元素：真值缺失项记 FN，不记 FP；真值缺失集合也为空时，该样本不贡献 TP/FP/FN，只影响完全匹配率的分母。

预测文件里如果自带 `derived` 字段，评测忽略它，只根据解析出的状态和金标准的 `required_items` 重新调用 `derive`。

### 3.2 主任务 checklist

记测试集中 checklist 样本数为 N，类别数为 5。

- 三状态准确率 = 预测状态等于真值的（样本, 类别）个数 / (N × 5)。同时报告每一类的分子和分母，分母是 N。
- 宏平均 F1：对 present、absent、unknown 各自计算 F1 = 2TP / (2TP + FP + FN)，再取算术平均。某一类的分母为 0 时该类 F1 无定义，宏平均只对有定义的类别计算；三类都无定义则为空。
- 混淆矩阵：行是三种真值，列是三种预测再加 invalid。
- 缺失集合：只看所需器材里状态为 absent 的集合。在全部样本上把 TP、FP、FN 累加后算微平均 P/R/F1，并按类别再报一遍。某个样本的真值缺失集合和预测缺失集合都为空时，不贡献这三个计数，但仍参加集合完全匹配率。
- 集合完全匹配率 = missing_required 与 uncertain_required 都相同的样本数 / N。
- 决策准确率 = 派生决策相同的样本数 / N。混淆矩阵的列同样包含 invalid。
- 误报齐全率 = 真值为 incomplete 且预测为 complete 的样本数 / 真值为 incomplete 的样本数。
- unknown 样本误判齐全率 = 真值为 unknown 且预测为 complete 的样本数 / 真值为 unknown 的样本数。
- unknown 召回 = 真值为 unknown 且预测也为 unknown 的（样本, 类别）个数 / 真值为 unknown 的（样本, 类别）个数。这里不限于所需器材。
- 明确回答覆盖率 = 真值为 present 或 absent、且预测也是 present 或 absent 的格子数 / 真值为 present 或 absent 的格子数。预测成 unknown 或 invalid 算未覆盖，留在分母里。
- 明确回答子集准确率 = 上一子集里预测等于真值的格子数 / 该子集的格子数。这是条件子集，分母小于覆盖率的分母。
- JSON 有效率 = 首次就解析成功的样本数 / N，也就是 `parse_status == ok` 且 `retry_count == 0`。这不是最终有效率。
- 重试率 = `retry_count > 0` 的样本数 / N。重试后成功的样本算重试，不算失败。
- 失败率 = 最终 `parse_status != ok` 的样本数 / N。

### 3.3 presence 与成对一致性

presence 只看 `target_item` 的三状态是否预测对，并给出该任务的混淆矩阵。解析成功但 `item` 不是 `target_item` 时，不采用它的 state，记入 invalid，并单独计数。

成对一致性：在每个 `pair_group_id` 里取恰好两条 checklist。被移动的器材是两张图里真值不同的类别。只有这些类别在两张图上的预测都等于各自的真值，这对图才算一致。分母是这样的图片对数量。只发现两次回答不一样，不算一致。组里不是恰好两张图，或没有任何类别真值不同，则跳过并在结果里列出。多于一个类别真值不同时，必须每一类都预测对。

### 3.4 条件分组

按 `condition_tags` 原样分组，例如 normal_light、dim_light、reflection、slight_occlusion、heavy_occlusion、moved。一个样本有多个标签就同时进入多个组。分组之后再用同一套公式计算，因此成对一致性只统计过滤后仍然成对的图片。

## 4. 统计方法

准确率的置信区间按 `scene_group_id` 做聚类 bootstrap，不用按单张图片重抽样。同一套摆放的图片会一起对或一起错，按图片抽样会把区间算得过窄。

步骤：

1. 按样本在文件中首次出现的顺序确定场景组。
2. 每次有放回地抽取与原来相同数量的组，抽中的组内样本全部带上。同一组被抽到两次就计两次。
3. 在这批样本上重算指标。该次复制的分母为 0 时，这次结果不进入百分位。
4. 默认重复 1000 次，随机种子固定为 0。报告 2.5 与 97.5 百分位，使用 NumPy 的线性插值。
5. 场景组少于 10 个时仍然给出区间，并标注“样本不足”。样本不足的区间不能写成已经稳定的结论。

点估计是全量测试集上的指标，不是 bootstrap 复制的平均值。

## 5. 速度与资源

预测文件里的 `timings.total_s` 是含图像加载和重试的端到端时间。正式测量丢掉前 k 次预热（板上 k 至少为 5）后再算 p50、p95、均值和 n。没有计时字段的记录不进入延迟分母，并单独计数；这和准确率必须保留失败样本不同，因为缺失的时间没法相加。

两个吞吐分开报：

- 完整点检请求数/秒 = checklist 条数 / 这些条的 `total_s` 之和
- 生成 token/s = `n_generated_tokens` 之和 / 最后一次生成的 `generate_s` 之和

`n_generated_tokens` 只对应最后一次生成，所以不能除以 `total_s`，也不能把两次尝试的生成时间加在一起再除。摄像头显示 FPS 另计，没有测量时写“未测”。

资源来自 `tegrastats`。解析器提取 RAM 已用/总量、SWAP、各 CPU 利用率、GR3D_FREQ、温度，以及各功率通道的瞬时值和平均值，再对时间窗求均值和峰值。JetPack 4 与 JetPack 5/6 的字段名不同，缺字段时记为空，不补 0。拿不到实测功率时写“未测”。格式示例在 `tests/fixtures/eval/tegrastats_*.log`，那些数字不是本次实验的测量。

## 6. 结果

方法对比、优化矩阵和分条件表的数字位置均为待测，见 `reports/templates/results_tables.md`。对应图（误报齐全率柱状图、延迟分布）在每次 eval 写出 PNG 之后再贴上，图中的高度必须和 metrics.json 一致。

失败案例、局限和改进待真实测试集评测完成后再写，不在本章预判哪种方法更好。
