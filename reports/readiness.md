# 就绪报告

2026-11-20 更新。50 张真实照片均已标注，含 9 个摆放组和 100 条 checklist。L40 上已完成 B0 三档分辨率和三档量化评测，见 `runs/comparison/`。器材框增强和固定三折划分已用全部数据生成和校验，见 `data/splits/cv3/`。正式训练和板端实测的进度见 `STATUS.md` 与 `docs/NEXT_STEPS.md`。

日期：2026-11-16。分三类：**已验证**（有实际运行的日志或产物）、**已准备但未验证**、**等待用户材料**。本报告中的数字都来自合成数据或服务器冒烟，**不是实验结果**，不得写进结果表。

## 1. 已验证

| 项目 | 证据 | 要点 |
| --- | --- | --- |
| 项目目录与路径规则 | `env/activate.sh` | 缓存、临时文件、pip/conda 包都在项目内。修复了全局 `HF_HUB_CACHE` 覆盖问题和系统 cuDNN 9.8 冲突 |
| conda 环境 | `env/environment.lock.yml`，`reports/environment_server.md` | `envs/qwen_vlm`：py3.12，torch 2.11 cu130，transformers 5.5.4，peft 0.21.3 |
| 基础模型固定版本 | `artifacts/base/manifest.json` | revision `89644892…`，11 个文件逐一与 HF 记录的哈希一致 |
| 冻结接口 | `contracts/`，`src/qwen_vlm/common/`，`src/qwen_vlm/inference/base.py` | 类别、schema、推导规则、Runner、指标定义、训练混合规则 |
| 单元测试 | `tests/` | `91 passed`。包括手算指标核对、标签屏蔽、LoRA 不匹配视觉模块、数据泄漏检测 |
| B0 单图推理（Transformers） | `runs/smoke_b0/` | JSON 可解析；图像确实影响 logits（与纯灰图相比最大差 5.19）；三档像素对应 130/234/540 个视觉 token；显存峰值 4.2GiB |
| 数据流水线 | `data/smoke/synthetic/` | CSV 转 JSONL、校验、按场景组划分、泄漏检查全部跑通（合成 24 图 / 72 条） |
| LoRA 训练链路 | `runs/train_smoke/` | 只有语言模型 196 个投影可训练（17.4M，占 0.81%）；loss 从 0.567 降到 0.002（过拟合 4 条）；显存约 5.5GiB；adapter 保存、加载、续训正常 |
| LoRA 合并 | `runs/train_smoke/merge_check.json` | 合并前后贪心文本一致；logits 最大差 3.75（bf16 舍入，见风险 3） |
| GGUF 转换 | `artifacts/quantized/base/manifest.json` | llama.cpp `b12425`；语言 F16 3.3G / Q8_0 1.8G / Q4_K_M 1.1G；mmproj（视觉+DeepStack）F16 782M |
| 像素档跨后端一致 | `runs/deploy_smoke/resolution_check.json` | llama.cpp 与 Transformers 三档视觉 token 都是 130/234/540，说明分辨率优化在两个后端是同一个变量 |
| 端到端（两个后端） | `runs/integration_smoke/` | 预测、评测、出表在 Transformers 和 GGUF（F16/Q8_0/Q4_K_M）上都跑通；新增 `scripts/predict_llamacpp.py`（不依赖 torch，可在板上使用） |
| 演示界面（图片模式） | `runs/deploy_smoke/app_smoke.json` | Gradio + llama-server(Q4) 完整调用成功；单请求锁有单测 |
| Jetson 侧导入链 | 本报告 | 导入 `llamacpp_runner` 与 `app` 时不会引入 torch/transformers/peft |

**一个有用的早期观察**（合成图，只用于定位差异）：在 15 条合成 val 上与 Transformers bf16 逐条比较，GGUF Q8_0 完全一致（15/15），F16 为 14/15，Q4_K_M 只有 8/15（类别级 36/50）。说明 Q4 对输出的影响不可忽略，量化实验必须在真实测试集上测任务指标，不能只报速度。（`runs/integration_smoke/agreement_vs_bf16.txt`）

## 2. 已准备但未验证

| 项目 | 位置 | 缺什么才能验证 |
| --- | --- | --- |
| 正式训练 B1/M/A1–A4 | `configs/train/*.yaml`，`scripts/run_cv.py` | 根据训练损失设定统一步数。`base.yaml` 默认步数是占位值 |
| Jetson 构建与运行 | `deploy/jetson/00`–`04`，`release/jetson_bundle_base/`（1.9G，Q4_K_M + mmproj） | 用户在板上运行并传回日志 |
| 板端量化/分辨率基准 | `deploy/jetson/03_bench.sh` | 同上；包里只有 Q4，测 Q8/F16 需把对应 GGUF 拷到板上 `models/` |
| 摄像头演示 | `src/qwen_vlm/app/app.py`，`04_app.sh` | 服务器没有摄像头，只能在板上测 |
| 合并后任务分数一致性 | `scripts/merge_lora.py` | 正式训练后，在 val 集上比较合并前后的任务指标（不能只看单样本） |
| 资源与延迟统计 | `src/qwen_vlm/evaluation/latency.py`，`tegrastats.py` | 板端日志（解析器只用格式示例测过） |

## 3. 等待用户材料（按优先级）

1. 在 Jetson 上运行 `00_probe.sh` 并传回日志，确认 JetPack/CUDA、内存与 swap、Python 和摄像头后，再跑 `01`–`04`。
2. 核实截止时间、报告篇幅与排版格式、提交渠道。（板子已定为 Jetson Orin Nano 8GB；报告英文为正式版，另附中文翻译。）

## 4. 风险

1. **服务器网络很慢**（直连约 30-50KB/s）。可用路线已记录在 `env/activate.sh`：ModelScope 走代理、阿里云 PyPI 直连、gh-proxy 访问 GitHub。新增大依赖要预留时间。
2. **磁盘**：项目盘已扩容到 99 GiB（2026-11-21 剩约 59 GiB），每次合并模型约 4.3GB、每组 GGUF 约 7GB，不再紧张。剩下的风险是 `/root/gpufree-data` 在释放实例时不保存，最终 adapter、GGUF 和 bundle 需要另外备份。
3. **合并误差**：bf16 合并在 logits 上有可见误差，冒烟样本的贪心输出没有变化。正式模型要以 val 集任务分数一致作为合并通过的标准。
4. **板端内存**：F16 语言权重 3.3G、mmproj 0.8G，再加 KV cache，8GB 统一内存可能放不下。先用 Q4 跑通，F16 若 OOM 就如实记录，改用 Q8 作为板端高精度参照。
5. **Jetson 环境**：脚本假设 JetPack 6 / Python 3.10，需要 `python3-venv`，`01` 需要网络安装 gradio 等。如果板子没网，需要另外准备离线 wheel。
6. **进程内存**：tegrastats 只有整机内存。如果需要进程级内存，要在 bench 脚本里额外采样 `/proc/<pid>/status`（后续可以补）。
7. **unknown 的边界**：B0 512² 在真实图上只召回 2/18 个 unknown 标签。合成遮挡针对这一问题补充监督。
