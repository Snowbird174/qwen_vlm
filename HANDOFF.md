# 交接说明

课程：计算机视觉 Coursework 2（要求原文 `docs/coursework/cv_coursework2.docx`）。
项目：用 **Qwen3-VL-2B** 检查实验台上的器材是否齐全，包括螺丝刀、钳子、线缆、万用表、锤子。先做领域微调，再部署到 **Jetson Orin Nano 8GB**。
仓库：<https://github.com/zy-StarXH/qwen_vlm>

## 1. 现在做到哪了

| 部分 | 状态 |
| --- | --- |
| 环境、模型下载、接口定义 | 完成 |
| 训练 / 评测 / 转 GGUF / 演示界面 | 端到端冒烟已通过，真实实验产物见 `STATUS.md` |
| Jetson 部署脚本和部署包 | 写好了，**还没在板子上跑过** |
| 真实照片 | `data/raw/session_001/` 共 50 张、9 个摆放组。状态与器材框已标注，用户已确认结束采集 |
| 原模型真实基线 | 512² 已评测 100 条 checklist，决策准确率 80/100，误报齐全 7/31，unknown 召回 2/18。见 `runs/B0_s001/eval/` |

当前里程碑见 `STATUS.md`，运行命令和恢复位置见 `docs/NEXT_STEPS.md`。接手时先核对运行目录下的 manifest 和日志。

## 2. 接下来按顺序做

1. 扫描 B0 的服务器分辨率档和 GGUF 量化档，保存真实结果矩阵。
2. 用器材框合成配对训练图，抽查编辑质量，按原图摆放组切三折。
3. 用训练损失确定统一步数，运行 B1、M、A1–A4 共 18 次训练。每个方法合并三折的真实折外预测，统一评分。
4. 汇总对比和样本局限。用全量数据训练部署用 M，导出 GGUF 并打包。
5. 在 Jetson 上执行 `deploy/jetson/00`–`04`，回传真实日志生成板端表格，补齐报告与演示视频。

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
- **磁盘**：启动本轮时项目盘剩约 9.8 GiB。训练 checkpoint 只保留一个，单折预测验证后保留 adapter。最终导出需要同时容纳合并目录和各档 GGUF，空间预算见 `docs/NEXT_STEPS.md`。
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

- 老师是否接受用 Orin Nano 8GB 代替要求里写的 "Jetson Nano"。
- 截止时间、报告格式、提交方式。
- 拍摄时相机用的是自动曝光，以后最好改成手动锁定。
