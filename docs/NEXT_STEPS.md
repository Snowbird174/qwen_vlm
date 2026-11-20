# 下一步执行计划（交给接手的模型/同学）

更新：2026-11-20。项目概况和接手规则见 `HANDOFF.md`。本文件记录实验计划、已验证的产物和恢复命令。

## 本轮推进顺序

1. 主代理在 L40 上串行跑任务 A。DeepSeek Flash 子代理（`max` 推理级别）并行实现任务 B 的成对图生成和任务 C 的固定划分。
2. 生成合成图并核查编辑效果后，主代理复验三折的图片、摆放组、pair 和样本覆盖。用真实折外预测统一评测 B1、M、A1–A4。
3. 对比表、真实指标和恢复命令随里程碑写回 `STATUS.md`、本文件和 `reports/readiness.md`，运行记录保存在 `runs/`。
4. 全量部署模型走任务 E。板端与服务器分别记载运行证据，等 Jetson 脚本的实际日志回传后补齐板端结果。

## 0. 当前状态（已完成，不要重做）

- 环境、基础模型、冻结接口、训练/评测/GGUF/演示/Jetson 脚本全部就绪。
- 真实数据：`data/raw/session_001/` 50 张（1920×1080），**用户决定不再拍新数据**。
  - 状态标注：`data/annotations/s001.csv` → `data/annotations/real_v1.jsonl`（100 条 checklist，0 错误）。
  - 器材框（已人工复核，可当真值）：`data/annotations/annotations.json`，格式见 `data/annotations/README.md`（`cable` = `cable_bundle`，坐标 xyxy 像素）。
  - 摆放分组（按框位置确定，跨组无近重复）：第 1–6 张 l001–l006 各一组，7–8 张 l007，9–25 张 l008，26–50 张 l009。
- B0（原模型，Transformers bf16，512²）已在全部 50 张上评测：`runs/B0_s001/eval/metrics.md`。
  决策准确率 80/100，**误报齐全率 7/31**，unknown 召回 2/18。典型错误：桌上有红黑线缆但没有万用表时判"万用表在"（第 14、22、27、32 张）；螺丝刀常被判 unknown（14 张）；遮挡时不说 unknown。

## 1. 已定的方案（用户已确认）

不再采集数据，改为：**按摆放组 3 折交叉验证 + 用框合成成对训练图**。测试永远只用真实照片。

| 折 | 测试组 | 训练组 |
| --- | --- | --- |
| fold1 | l009（第 26–50 张） | 其余 |
| fold2 | l008（第 9–25 张） | 其余 |
| fold3 | l001–l007（第 1–8 张） | 其余 |

每个实验训练 3 次（每折一次），把三折测试集预测拼成覆盖全部 50 张的一份 `predictions.jsonl`，再用 `scripts/eval_predictions.py` 对 `real_v1.jsonl` 统一评测（bootstrap 按 scene_group）。

## 2. 任务清单（按顺序）

### 任务 A：B0 两项优化扫描（不需要训练，先做）

**完成。** 六个 L40 配置各有 100 条真实预测，结果在 `runs/comparison/comparison.md`。BF16 384²/512²/768² 的螺丝刀准确率分别为 63/100、67/100、91/100，误报齐全分别为 7/31、7/31、6/31。llama.cpp 在 512² 的 F16/Q8_0/Q4_K_M 决策准确率分别为 80/100、80/100、73/100，端到端延迟 p50 分别为 1.3821、1.2693、1.2110 秒。Q4 的 unknown 召回为 7/18，F16 为 2/18。结果来自 9 个摆放组，置信区间标为样本不足。

1. 分辨率：已有 512²；补跑 384²（`--max-pixels 147456`）和 768²（`589824`）：
   ```bash
   P=envs/qwen_vlm/bin/python
   $P scripts/predict_transformers.py --jsonl data/annotations/real_v1.jsonl --experiment-id B0_s001_r384 --max-pixels 147456
   $P scripts/eval_predictions.py --gold data/annotations/real_v1.jsonl --pred runs/B0_s001_r384/predictions.jsonl --out runs/B0_s001_r384/eval --experiment-id B0_s001_r384 --pixel-budget 384
   ```
2. 量化：用 GGUF（`artifacts/quantized/base/`）在 512² 下跑 F16 / Q8_0 / Q4_K_M：
   ```bash
   $P scripts/predict_llamacpp.py --jsonl data/annotations/real_v1.jsonl --experiment-id B0_s001_q4 \
      --model artifacts/quantized/base/base-q4_k_m.gguf --mmproj artifacts/quantized/base/mmproj-base-f16.gguf
   ```
   每个都接 `eval_predictions.py`（加 `--quantization Q4_K_M` 等）。最后 `scripts/eval_compare.py --runs-root runs --out runs/comparison`。
3. 重点观察：螺丝刀在不同分辨率下的准确率；Q4 相对 bf16 的退化（合成图上 Q4 只有 8/15 与 bf16 一致）。

### 任务 B：用框合成成对训练图

增强入口为 `scripts/data_make_augment.py`，实现位于 `src/qwen_vlm/data/augment.py`。从仓库根目录加载环境后运行以下命令。图片、样本、生成清单和抽查表保存在 Git 忽略的 `data/aug/v1/`。

```bash
source env/activate.sh
P=envs/qwen_vlm/bin/python
$P scripts/data_make_augment.py --seed 0
$P scripts/data_validate.py data/aug/v1/samples.jsonl
```

生成结果为 50 抹除对、50 遮挡对，共 200 张编辑图和 400 条 checklist。其中 33 对的真编辑抹掉万用表并保留独立线缆。每对使用同一套问法，新增困难负样本标签针对这 33 个真编辑。抽查表 `data/aug/v1/contact_sheet.jpg` 含 10 对共 20 张，曾发现第 25 张的万用表红色残影，修正后再次检查。遮罩纹理仍可辨识，需在报告中披露合成编辑与真实场景的差异。复现时参照 `data/aug/v1/manifest.json`。改种子或编辑参数时输出到另一个版本目录，以便已有 manifest 继续指向对应的图像。

- 输入：`real_v1.jsonl` + `annotations.json`。每张真实图生成若干编辑图，**继承原图的 `session_id` 和 `scene_group_id`**（保证交叉验证时合成图跟随原图所在的折，不泄漏）。
- 编辑类型：
  1. **抹除**：把某件 present 器材的框区域（适当外扩）用 `cv2.inpaint` 或白板纹理填掉 → 该类改为 `absent`。重点多做"抹掉万用表、保留线缆"（针对 B0 幻觉），打 `hard_negative_tags: ["cable_without_multimeter"]` 之类标签。
  2. **合成遮挡**：在某件器材上覆盖手形/纸张色块（覆盖大部分）→ 该类 `unknown`，`unknown_reason` 写"合成遮挡"。
  3. **诱饵编辑**（必须有，防止捷径）：在空白区域做同样的 inpaint 或色块，但不碰任何器材 → 状态不变。
- **成对**：每对 = 一张"诱饵编辑图" + 一张"真编辑图"（两张都有编辑痕迹，只差一个类别），`pair_group_id` 唯一。不要把原图放进新 pair（原图可能已在真实 pair 中，一张图只能属于一对）。
- 字段：`source: "synthetic_edit"`，`annotation_verified: false`，sample_id 形如 `s001_l008_v014e03_q01`（编号里不能有下划线）。图片缩到 1280×720 JPEG（q≈95）省磁盘。
- 问法/清单：复用 `configs/data/question_templates.yaml` 的 default 集合（参考 `src/qwen_vlm/data/convert.py` 的抽取方式）。
- 验收：`scripts/data_validate.py data/aug/v1/samples.jsonl` 0 错误；拼一张缩略图人工抽查 20 张编辑效果（抹除处不能残留明显器材，诱饵和真编辑痕迹看起来相似）；写 `tests/test_data_augment.py`（状态改变正确、诱饵不改状态、pair 恰差一类、scene_group 继承）。

### 任务 C：交叉验证划分与实验配置

1. `scripts/data_make_cv.py` 生成 `data/splits/cv3/fold{1,2,3}/`。固定划分已用真实数据和合成数据验证。三折各有 250、330、420 条混合训练样本和 50、34、16 条真实测试样本，覆盖真实金标准 100/100 条、50/50 张。每折 manifest 记录文件哈希及泄漏检查。
2. 实验（全部用同一 `max_steps`、有效 batch、LoRA 超参；**不在测试集上调参**，超参固定为 `configs/train/base.yaml`，只调 `max_steps` 使训练在 train 上收敛，三折一致）：

| ID | 训练数据 | mix 配置 |
| --- | --- | --- |
| B1 | 只用真实图 | `configs/train/B1.yaml`（无辅助、无加权） |
| M | 真实 + 合成 | `M.yaml`（pairs 辅助 + hard_negative_weight 3 + balance + unknown） |
| A1 | 真实 + 合成 | M 去掉成对辅助（`aux_presence: none`） |
| A2 | 真实 + 合成 | M 去掉困难负样本加权（`hard_negative_weight: 1.0`） |
| A3 | 真实 + 合成 | M 去掉 unknown 样本（`include_unknown: false`） |
| A4 | 只用真实图 | M 的 mix，但不加合成图（衡量合成数据本身的贡献） |

   B1 和 A4 从各折的 `train_real.jsonl` 读取 50、66、84 条真实训练样本。
3. 正式实验共用 **50 个优化步、有效 batch 16、种子 42**。两次 30 步试跑都只读取 fold1 训练集。B1 训练 loss 从 0.08168 降至约 0.0000062，且该折纯真实训练子集没有 unknown。混合 M 的三个 10 步窗口平均 loss 分别为 0.07808、0.02201、0.02199。M 三折重采样前的样本池分别为 360、468、604 条，50 步对应 800 次样本呈现，使最大折的预算比例接近 30 步试跑。试跑记录在 `work/cv3_pilot/{B1,M}_fold1/`。入口显式传 `--no-validation`，正式折外指标待测。
   ```bash
   source env/activate.sh
   P=envs/qwen_vlm/bin/python
   $P scripts/data_make_cv.py
   $P scripts/run_cv.py --max-steps 50
   $P scripts/eval_compare.py --runs-root runs --out runs/comparison
   ```
   `scripts/run_cv.py` 先核对三折固定分组、JSONL 和 manifest 哈希以及全部真实测试样本覆盖，再依次训练、预测、拼接和评测。折内记录在 `runs/cv3_folds/<ID>_fold<k>/`，汇总在 `runs/<ID>_cv3/`。可用原命令重跑并复用完成的 adapter 和预测。改变步数、配置或划分时使用新的 `--runs-root`。
4. 资源：L40 46GB，单次训练显存约 5.5GiB（512²）；磁盘约剩 9.8GB——**不要保存检查点以外的完整模型**，`save_total_limit` 设 1，跑完删 `checkpoints/` 只留 `adapter/`。

### 任务 D：汇总与文档

1. `scripts/eval_compare.py` 生成 B0/B1/M/A1–A4 对比表和优化矩阵表（`runs/comparison/`）。
2. 更新 `EXPERIMENTS.md`（加入 A4、交叉验证协议、合成数据说明）、`STATUS.md`、`reports/readiness.md`。
3. 报告必须写明的局限：只有 1 个批次、9 个摆放、50 张图；置信区间宽（标"样本不足"）；合成图只用于训练；交叉验证三折的测试组大小不均。
4. 测试状态：全量 `pytest -q tests` 111 个通过。服务器扫描的六份评测已归档。正式三折用同一命令恢复，并检查每个实验是否合并为覆盖 100 条金标准的折外预测。

### 任务 E：选定最终模型并部署（之后）

1. 将 `real_v1.jsonl` 和 `data/aug/v1/samples.jsonl` 合并为全量训练 JSONL，沿用正式实验的 M 配置和步数。用 `scripts/train.py --no-validation` 将 adapter 保存到 `runs/M_final/adapter/`。泛化指标使用前面的折外评测，最终模型用于部署演示。
2. `scripts/merge_lora.py` 合并，检查代表性输入的合并一致性，然后导出 F16/Q8_0/Q4_K_M 和 mmproj。按现有脚本的 5 GiB 空间底线，约 4.3 GiB 的合并模型和 6.8 GiB 的 GGUF 应放在预先核实有约 16 GiB 可用空间的同一文件系统上。启动本轮时项目盘约 9.8 GiB，`/tmp` 约 18 GiB，可在导出前复查并用于临时合并与转换，确认产物后清理临时合并权重。
3. 在 GGUF 所在文件系统上生成 bundle，打包脚本对模型文件使用硬链接。开发机空间不足时，先在临时盘完成转换与打包，再迁移到有足够空间的持久盘或交付给板端。
4. 用户在 Jetson 上跑 `deploy/jetson/00`–`04`，最终 bundle 的 `02`–`04` 使用 `MODEL_TAG=final`。日志回传 `runs/jetson/`。默认 `03_bench.sh` 循环使用冒烟图片，可测延迟和资源。也可以对 `real_v1.jsonl` 在板端逐条推理，按部署说明保留 `sample_id`，回传预测与对应 `tegrastats`。对全量训练的最终模型，这批预测用于检查部署后的输出和性能。

## 3. 注意事项

- 每个 shell 先 `cd /root/gpufree-data/qwen_vlm && source env/activate.sh`，Python 一律用 `envs/qwen_vlm/bin/python`（不 source 会报 cuDNN 错误）。
- 不改 `contracts/`、`src/qwen_vlm/common/`、`src/qwen_vlm/inference/base.py`。
- 不编造结果；未跑的写"待测"。合成/冒烟数字不进报告。
- 每完成一个任务跑 `pytest -q tests` 并 `git commit`；推送需用户输入 GitHub 令牌（见 `docs/git_workflow.md`）。服务器外网慢，代理常断，直连 GitHub 可用。
- 标注工具可能仍在 7861 端口运行，不影响训练。
