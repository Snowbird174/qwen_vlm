# 基于 Qwen3-VL-2B 的实验室器材完整性点检（Jetson Orin Nano 8GB）

固定俯视摄像头拍实验台，用户给出所需器材清单（螺丝刀、钳子、线缆、万用表、锤子中的任意子集），系统判断是否齐全、缺什么；画面不足以判断时回答"无法确认"。模型输出五类三状态 JSON，完整性结论由 `src/qwen_vlm/common/derive.py` 确定性推导。

- 课程要求和证据对应：`docs/coursework_requirements.md`
- 当前进度：`STATUS.md`；就绪情况：`reports/readiness.md`；实验登记：`EXPERIMENTS.md`
- 模型原理：`docs/model_principles.md`；冻结接口：`contracts/`

## 目录

```text
env/          环境说明、activate.sh、lock 文件       envs/       conda 环境（不入库）
contracts/    类别表、schema、指标定义、接口约定    configs/    数据/训练/推理/量化配置
src/qwen_vlm/ common 推理 data training evaluation app
scripts/      全部命令行入口                        tests/      pytest
data/         smoke/（合成冒烟）raw/（真实照片）splits/（划分）annotations/
artifacts/    base/（基础模型）adapters/ merged/ quantized/（GGUF）
deploy/jetson/ 板端脚本与说明                        runs/       每次运行的日志、预测、评测
reports/      环境、就绪、结果与报告                 release/    交付包（Jetson bundle 等）
```

## 0. 环境

每个 shell 先执行：

```bash
cd /root/gpufree-data/qwen_vlm && source env/activate.sh
```

`activate.sh` 会把所有缓存和临时文件指向项目目录，修复系统 cuDNN 冲突，设置 pip 镜像。用 `envs/qwen_vlm/bin/python` 运行一切 Python 脚本。环境的构建过程和网络情况见 `env/environment.yml` 与 `reports/environment_server.md`。

```bash
envs/qwen_vlm/bin/python -m pytest -q tests          # 全部单元测试
envs/qwen_vlm/bin/python scripts/download_base_model.py --verify-only   # 校验基础模型
envs/qwen_vlm/bin/python scripts/smoke_b0.py         # 基础模型冒烟（合成图）
```

## 1. 数据

采集与标注按 `docs/data_collection.md`、`docs/annotation_guide.md`。照片放到 `data/raw/<session>/<layout>/<variant>.jpg`，标注表模板在 `data/annotations/templates/annotation_template.csv`（编号不要含下划线）。

```bash
P=envs/qwen_vlm/bin/python
$P scripts/data_csv_to_jsonl.py --csv data/annotations/real_v1.csv --out data/annotations/real_v1.jsonl --source real_lab
$P scripts/data_validate.py data/annotations/real_v1.jsonl
$P scripts/data_split.py --input data/annotations/real_v1.jsonl --out data/splits/v1 --fail-on-leak
$P scripts/data_stats.py --split-dir data/splits/v1 --out data/splits/v1/stats.md
```

合成冒烟数据（只用于打通流程）：`$P scripts/data_make_synthetic.py`，产物在 `data/smoke/synthetic/`。

## 2. 基线 B0 与训练

```bash
# B0：原模型在测试集上推理，然后评测
$P scripts/predict_transformers.py --jsonl data/splits/v1/test.jsonl --experiment-id B0
$P scripts/eval_predictions.py --gold data/splits/v1/test.jsonl --pred runs/B0/predictions.jsonl --out runs/B0/eval --experiment-id B0

# 训练（B1 / M / A1 / A2 / A3 同一套命令，只换配置）；configs/train/base.yaml 默认指向合成数据，必须用参数覆盖
$P scripts/train.py --config configs/train/M.yaml \
   --train-jsonl data/splits/v1/train.jsonl --val-jsonl data/splits/v1/val.jsonl \
   --manifest data/splits/v1/manifest.json --max-steps <各实验相同的步数>

# 训练后的模型推理与评测（adapter 保存在 runs/<experiment_id>/adapter；配置了 export_adapter_dir 时另存一份到 artifacts/adapters/）
$P scripts/predict_transformers.py --jsonl data/splits/v1/test.jsonl --experiment-id M --adapter runs/M/adapter
$P scripts/eval_predictions.py --gold data/splits/v1/test.jsonl --pred runs/M/predictions.jsonl --out runs/M/eval --experiment-id M

# 汇总对比表
$P scripts/eval_compare.py --runs-root runs --out runs/comparison
```

冒烟训练：`$P scripts/train.py --config configs/train/smoke.yaml`（合成数据，约 30 步，显存峰值约 5.5GiB）。

## 3. 合并与转换 GGUF

```bash
# 合并 LoRA，并在一个样本上检查合并前后贪心输出一致（需要约 10GB 空闲磁盘）
$P scripts/merge_lora.py --adapter runs/M/adapter --sample-jsonl data/splits/v1/val.jsonl \
   --merged-dir artifacts/merged/M --merge-check runs/M/merge_check.json

# 服务器上构建 llama.cpp（只需一次），再转换：F16 语言 GGUF + F16 mmproj + Q8_0 + Q4_K_M
bash scripts/export_build_llamacpp_server.sh
bash scripts/export_gguf.sh artifacts/merged/M artifacts/quantized/M M

# 服务器上的部署冒烟：像素档对照、与 Transformers 的输出对照、演示界面
$P scripts/deploy_smoke.py

# 用 GGUF 对任意 JSONL 批量推理（自动启动/关闭本机 llama-server；也可用 --server 连接已运行的服务），输出格式同 predict_transformers.py
$P scripts/predict_llamacpp.py --jsonl data/splits/v1/test.jsonl --experiment-id M_q4_512 \
   --model artifacts/quantized/M/M-q4_k_m.gguf --mmproj artifacts/quantized/M/mmproj-M-f16.gguf --max-pixels 262144
```

llama.cpp 固定为 tag `b12425`（commit `18b74fa4`），依据见 `deploy/jetson/LLAMACPP_VERSION.md`。

## 4. Jetson 部署

```bash
# 打包（默认只带 Q4_K_M；做量化对比时带全三档）
INCLUDE_QUANTS=q4_k_m,q8_0,f16 bash scripts/make_jetson_bundle.sh artifacts/quantized/M release/jetson_bundle_M
```

把 bundle 拷到板子上，按 `deploy/jetson/README.md` 依次运行 `00_probe.sh` 到 `04_app.sh`，再把 `logs/` 拷回服务器的 `runs/jetson/`。板端日志评测：

```bash
$P scripts/eval_predictions.py --gold <对应 JSONL> --pred runs/jetson/<bench>/<config>.jsonl \
   --out runs/jetson/<config>/eval --quantization Q4_K_M --pixel-budget 512 --warmup 5 \
   --tegrastats runs/jetson/<bench>/tegrastats.log
```

演示界面（服务器或板端均可，需先启动 llama-server）：

```bash
$P -m qwen_vlm.app.app --server http://127.0.0.1:8080 --pixels 512 --camera 0
```

## 版本固定

- 基础模型：`Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`，逐文件 SHA256 见 `artifacts/base/manifest.json`
- llama.cpp：`b12425` / `18b74fa41dcac1f55ae1371fe58153f7225941f6`
- Python 环境：`env/environment.lock.yml`、`env/pip_freeze_qwen_vlm.txt`
- 每次运行：`runs/<experiment_id>/run_manifest.json`（git commit、数据哈希、配置、种子、步数）
