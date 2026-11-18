# 交接说明

课程：计算机视觉 Coursework 2（要求原文 `docs/coursework/cv_coursework2.docx`）。
项目：用 **Qwen3-VL-2B** 检查实验台上的器材是否齐全，包括螺丝刀、钳子、线缆、万用表、锤子。先做领域微调，再部署到 **Jetson Orin Nano 8GB**。
仓库：<https://github.com/zy-StarXH/qwen_vlm>

## 1. 现在做到哪了

| 部分 | 状态 |
| --- | --- |
| 环境、模型下载、接口定义 | 完成 |
| 训练 / 评测 / 转 GGUF / 演示界面 代码 | 完成，已用合成图跑通（不是实验结果） |
| Jetson 部署脚本和部署包 | 写好了，**还没在板子上跑过** |
| 真实照片 | 第一批 50 张已拍好（`data/raw/session_001`），**还没标注** |
| 正式训练、实验结果、报告、视频 | 未开始 |

详细清单见 `STATUS.md` 和 `reports/readiness.md`。

## 2. 接下来按顺序做

1. **标注第一批照片**：`envs/qwen_vlm/bin/python scripts/data_label_tool.py data/raw/session_001`，浏览器打开 `localhost:7861`。逐张选 5 件器材的状态，标完点"检查全部标注"。结果存在 `data/annotations/s001.csv`。
2. **转格式并划分**（命令见 `README.md` 第 1 节），输出到 `data/splits/v1/`。
3. **跑 B0 基线**：用原模型在测试集上推理并评测（`README.md` 第 2 节），看模型在真实照片上错在哪。
4. **继续拍照**：按 `docs/data_collection.md`，换光照、分几天拍，总数到 200 张以上，每批都用标注工具标。
5. **训练** B1、M、A1–A3（`configs/train/`），对比结果。
6. **转 GGUF，上 Jetson**：按 `deploy/jetson/README.md` 跑 `00`–`04` 脚本，测量化（F16/Q8/Q4）和分辨率（384²/512²/768²）两项优化。
7. 写报告，录演示视频。

## 3. 怎么开始

**在原服务器上接手**：所有东西都在 `/root/gpufree-data/qwen_vlm`，每次先执行：

```bash
cd /root/gpufree-data/qwen_vlm && source env/activate.sh
envs/qwen_vlm/bin/python -m pytest -q tests      # 97 个测试应全部通过
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
- **磁盘**：服务器剩约 10GB。合并出的完整模型（4.3GB）转完 GGUF 就删掉。
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
