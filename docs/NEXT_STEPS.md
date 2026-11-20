# 下一步执行计划（交给接手的模型/同学）

更新：2026-11-20。先读 `HANDOFF.md`（项目概况、规则），本文件只写**接下来具体做什么**。

## 0. 当前状态（已完成，不要重做）

- 环境、基础模型、冻结接口、训练/评测/GGUF/演示/Jetson 脚本全部就绪，`pytest` 97 个通过。
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

新建 `src/qwen_vlm/data/augment.py` 和 `scripts/data_make_augment.py`，输出到 `data/aug/v1/`（加入 `.gitignore`，脚本固定种子可重新生成）和 `data/aug/v1/samples.jsonl`。规格：

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

1. `scripts/data_make_cv.py`：生成 `data/splits/cv3/fold{1,2,3}/{train,test}.jsonl` + `manifest.json`。train = 训练组真实样本 + 训练组的合成样本；test = 测试组**真实**样本。断言：test 中无 `synthetic_edit`，train/test 无共同 scene_group。
2. 实验（全部用同一 `max_steps`、有效 batch、LoRA 超参；**不在测试集上调参**，超参固定为 `configs/train/base.yaml`，只调 `max_steps` 使训练在 train 上收敛，三折一致）：

| ID | 训练数据 | mix 配置 |
| --- | --- | --- |
| B1 | 只用真实图 | `configs/train/B1.yaml`（无辅助、无加权） |
| M | 真实 + 合成 | `M.yaml`（pairs 辅助 + hard_negative_weight 3 + balance + unknown） |
| A1 | 真实 + 合成 | M 去掉成对辅助（`aux_presence: none`） |
| A2 | 真实 + 合成 | M 去掉困难负样本加权（`hard_negative_weight: 1.0`） |
| A3 | 真实 + 合成 | M 去掉 unknown 样本（`include_unknown: false`） |
| A4 | 只用真实图 | M 的 mix，但不加合成图（衡量合成数据本身的贡献） |

   B1 和 A4 需要一份只含真实样本的 train（可在 cv 脚本里同时输出 `train_real.jsonl`）。
3. 训练命令（每折每实验）：
   ```bash
   $P scripts/train.py --config configs/train/M.yaml --train-jsonl data/splits/cv3/fold1/train.jsonl \
      --manifest data/splits/cv3/fold1/manifest.json --output-dir runs/M_fold1 --max-steps <N>
   $P scripts/predict_transformers.py --jsonl data/splits/cv3/fold1/test.jsonl --experiment-id M_fold1 --adapter runs/M_fold1/adapter
   ```
   `--val-jsonl` 可以不给或给 train 的一小份；不要用 test。写一个 `scripts/run_cv.py` 串起"训练 → 预测 → 拼接三折 → 评测"，输出 `runs/<ID>_cv3/predictions.jsonl` 和 `runs/<ID>_cv3/eval/`。中间的单折目录命名为 `runs/cv3_folds/<ID>_fold<k>/`（不要放在 `runs/*/eval`，否则 `eval_compare.py` 会误收）。
4. 资源：L40 46GB，单次训练显存约 5.5GiB（512²）；磁盘约剩 9.8GB——**不要保存检查点以外的完整模型**，`save_total_limit` 设 1，跑完删 `checkpoints/` 只留 `adapter/`。

### 任务 D：汇总与文档

1. `scripts/eval_compare.py` 生成 B0/B1/M/A1–A4 对比表和优化矩阵表（`runs/comparison/`）。
2. 更新 `EXPERIMENTS.md`（加入 A4、交叉验证协议、合成数据说明）、`STATUS.md`、`reports/readiness.md`。
3. 报告必须写明的局限：只有 1 个批次、9 个摆放、50 张图；置信区间宽（标"样本不足"）；合成图只用于训练；交叉验证三折的测试组大小不均。

### 任务 E：选定最终模型并部署（之后）

1. 用全部 50 张真实图 + 全部合成图按 M 配置训练最终模型（不再有测试集，只用于部署演示，报告里注明）。
2. `scripts/merge_lora.py` 合并 → `scripts/export_gguf.sh artifacts/merged/final artifacts/quantized/final final` → 删除 merged 完整权重。
3. `INCLUDE_QUANTS=q4_k_m,q8_0,f16 bash scripts/make_jetson_bundle.sh artifacts/quantized/final release/jetson_bundle_final`。
4. 用户在 Jetson 上跑 `deploy/jetson/00`–`04`，日志回传 `runs/jetson/`，再用 `eval_predictions.py --tegrastats ...` 出板端速度与资源表。**板端数字只能来自真实日志。**

## 3. 注意事项

- 每个 shell 先 `cd /root/gpufree-data/qwen_vlm && source env/activate.sh`，Python 一律用 `envs/qwen_vlm/bin/python`（不 source 会报 cuDNN 错误）。
- 不改 `contracts/`、`src/qwen_vlm/common/`、`src/qwen_vlm/inference/base.py`。
- 不编造结果；未跑的写"待测"。合成/冒烟数字不进报告。
- 每完成一个任务跑 `pytest -q tests` 并 `git commit`；推送需用户输入 GitHub 令牌（见 `docs/git_workflow.md`）。服务器外网慢，代理常断，直连 GitHub 可用。
- 标注工具可能仍在 7861 端口运行，不影响训练。
