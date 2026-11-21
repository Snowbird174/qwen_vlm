# 交接说明

课程：计算机视觉 Coursework 2（要求原文 `docs/coursework/cv_coursework2.docx`）。
项目：用 **Qwen3-VL-2B** 检查实验台上的器材是否齐全，包括螺丝刀、钳子、线缆、万用表、锤子。先做领域微调，再部署到 **Jetson Orin Nano 8GB**。
仓库：<https://github.com/zy-StarXH/qwen_vlm>

## 1. 现在做到哪了

| 部分 | 状态 |
| --- | --- |
| 环境、模型下载、接口定义 | 完成 |
| 训练 / 评测 / 转 GGUF / 演示界面 | 服务器第 1–5 步已完成，实际产物见 `STATUS.md` |
| Jetson Orin Nano 8GB | 最终部署包 SHA256 校验通过，实机、摄像头及视频待测、待录 |
| 真实照片 | `data/raw/session_001/` 共 50 张、9 个摆放组。状态与器材框已标注，用户已确认结束采集 |
| 原模型真实基线 | 512² 已评测 100 条 checklist，决策准确率 80/100，误报齐全 7/31，unknown 召回 2/18。见 `runs/B0_s001/eval/` |
| 正式三折 | B1、M、A1–A4 的 18/18 折已核对，每种方法 100 条折外预测。训练产物 `runs/cv3_folds/`，运行日志 `runs/logs/cv3_main.log` |
| 对比与模型选择 | 六种 LoRA 方法均为 2/31 误报齐全，按预定规则保留 M。详见 `runs/comparison/` 和 `EXPERIMENTS.md` |
| 鲁棒性 | 27/27 组已验收。M 在 `clean`、重度暗光、重度模糊分别判对 87/100、79/100、78/100。产物 `runs/robust/`，恢复命令见 `docs/NEXT_STEPS.md` |
| 最终 M | 全量 500 条完成 50 步。合并检查通过，已导出三档 GGUF。`runs/M_final/adapter/` 和 `release/jetson_bundle_final/` 均需在实例释放前备份 |
| 报告与演示 | `reports/technical_report.md`、`reports/technical_report.zh.md`、`reports/demo_script.md` 和 `reports/defense_outline.md` 已完成。真实图片模式证据见 `runs/M_final/demo/` |

当前里程碑见 `STATUS.md`，运行命令和恢复位置见 `docs/NEXT_STEPS.md`。接手时先核对运行目录下的 manifest 和日志。

## 2. 本轮产物与下一步

B0 的服务器分辨率和量化扫描、框引导合成训练图、固定三折划分以及 `docs/NEXT_STEPS.md` 本轮第 1–5 步均已完成。正式模型选择沿用 M。

1. 释放实例前备份最终 adapter、部署包和项目报告。部署包中 `SHA256SUMS` 已覆盖 91 个文件。
2. 用户拿到 Jetson Orin Nano 8GB 后，按 `deploy/jetson/README.md` 运行 `00`–`04`。当前课程的板端实测、现场演示和录像待完成。

## 3. 怎么开始

**在原服务器上接手**：所有东西都在 `/root/gpufree-data/qwen_vlm`，每次先执行：

```bash
cd /root/gpufree-data/qwen_vlm && source env/activate.sh
envs/qwen_vlm/bin/python -m pytest -q tests
```

**在新机器上从 GitHub 开始**：仓库里只有代码和文档，下面这些要自己准备：

| 缺的东西 | 怎么补 |
| --- | --- |
| conda 环境 | `conda env create -p envs/qwen_vlm -f env/environment.yml && envs/qwen_vlm/bin/pip install -e .` |
| 基础模型（4.3GB） | `python scripts/download_base_model.py`（会逐个文件校验哈希） |
| llama.cpp 源码 | 下载 tag `b12425` 的源码，解压到 `vendor/llama.cpp`（见 `deploy/jetson/LLAMACPP_VERSION.md`），再运行 `bash scripts/export_build_llamacpp_server.sh` |
| 真实照片 | 不进 git，找原作者要 `session_001.zip`，解压到 `data/raw/` |
| GGUF 和模型权重 | 用 `scripts/export_gguf.sh` 重新生成 |

`env/activate.sh` 里写死了 `/root/gpufree-data/qwen_vlm`，在别的路径使用时改第 3 行。

## 4. 必须知道的规则

- **不要改冻结的接口**：`contracts/`、`src/qwen_vlm/common/`、`src/qwen_vlm/inference/base.py`。改了会让前后实验没法比较；确实要改，就把版本号加一并记下来。
- **不编造结果**：合成图（`data/smoke/`）和冒烟运行（`runs/*smoke*`）的数字一律不能写进报告。没跑的实验就写"待测"。
- **测试集冻结**：划分好之后，选模型和调参只能看验证集。
- **同一摆放的照片不能拆开**：程序按"摆放编号"划分数据，标注时编号一定要填对。
- **磁盘**：项目盘已扩容到 99 GiB（2026-11-21 剩约 59 GiB），合并模型和 GGUF 直接放在 `artifacts/` 下。训练 checkpoint 只保留一个，单折预测验证后保留 adapter。`/root/gpufree-data` 在释放实例时不保存，重要产物要提醒用户下载备份。
- **网络**：服务器外网很慢，能用的下载方法写在 `env/activate.sh` 的注释里。推送 GitHub 要开本地代理，或者不走代理直连。

## 5. 文件导航

| 想做什么 | 看哪里 |
| --- | --- |
| 所有命令 | `README.md` |
| 课程要求逐条对应 | `docs/coursework_requirements.md` |
| 怎么拍、怎么标 | `docs/data_collection.md`、`docs/annotation_guide.md` |
| 实验设计（B0/B1/M/A1–A3、优化矩阵） | `EXPERIMENTS.md` |
| 指标定义 | `contracts/metrics.md` |
| 模型原理（写报告用） | `docs/model_principles.md` |
| 板端部署 | `deploy/jetson/README.md` |
| 风险和已知问题 | `reports/readiness.md` 第 4 节 |
| git 怎么用 | `docs/git_workflow.md` |
| 快速测试一张图 | `python scripts/check_image.py <图片> --items 螺丝刀 钳子` |

## 6. 还没确认的事

- 截止时间、报告篇幅与排版格式、提交方式。（已定：板子用 Jetson Orin Nano 8GB，不再向老师确认；报告英文为正式版，另附中文翻译供用户校对。）
- 拍摄时相机用的是自动曝光，以后最好改成手动锁定。
