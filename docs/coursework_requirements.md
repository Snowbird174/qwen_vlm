# 课程要求与证据对应表

原文：`docs/coursework/cv_coursework2.docx`（Computer Vision Coursework Project, "Design and Optimization of an Intelligent Vision System Based on Jetson Nano"）。下面左列为原文引用，右侧为本项目的落实方式和证据位置。证据状态由主 agent 在 `STATUS.md` 维护。

## 1. Project Description

> You are required to design and implement an intelligent vision system for a practical application using an existing computer vision or multimodal AI model. The project should include model implementation, deployment on Jetson Nano, algorithm optimization, and experimental evaluation.

落实：实验室工作台器材完整性点检（螺丝刀、钳子、线缆、万用表、锤子）；模型使用 Qwen3-VL-2B-Instruct（属于原文列出的 VLM 类别）。

## 2. Project Requirements

| 原文要求 | 本项目落实方式 | 证据（路径） |
| --- | --- | --- |
| (1) "Select a relatively advanced computer vision or multimodal AI model" | Qwen3-VL-2B-Instruct，固定 revision | `artifacts/base/manifest.json` |
| (1) "Understand and explain the model architecture, key algorithms, and underlying principles" | 视觉编码器、DeepStack、视觉与语言融合、chat template、LoRA | `docs/model_principles.md`，技术报告 |
| (1) "Implement meaningful improvements or extensions ... model adaptation ..." | 用成对场景、困难负样本和 unknown 监督做领域 LoRA 微调，并加一个存在性辅助任务 | `src/qwen_vlm/training/`，`configs/train/` |
| (1) "You can build your own dataset to train"（可选） | 自建实验室数据集 | `docs/data_collection.md`，`data/`，data card |
| (1) "Evaluate the effectiveness of the proposed improvements through comparative experiments" | B0 / B1 / M 以及消融 A1–A4，按摆放组三折交叉验证 | `EXPERIMENTS.md`，`runs/`，`reports/` |
| (2) "Deploy the selected model or its core vision inference component on Jetson Nano and demonstrate image or video" | 在 Orin Nano 8GB 上用 llama.cpp 运行完整 VLM，接摄像头演示 | `deploy/jetson/`，`runs/jetson/` |
| (2) "encouraged to apply Jetson Nano into a real device, real sensor ..." | 固定俯视 USB/CSI 摄像头 | 演示脚本，视频 |
| (3) "Implement at least two optimization techniques ... quantization ... input resolution adjustment" | 量化（F16/Q8_0/Q4_K_M）和输入像素预算（384²/512²/768²） | `configs/quant/`，优化矩阵结果 |
| (3) "Compare the optimized system with the original baseline using quantitative experimental results" | 在同一设备、受控变量下对比 | `reports/` 中的结果表 |
| (4) "Evaluate task accuracy, inference speed, latency, resource consumption, and robustness under different conditions" | 指标见 `contracts/metrics.md`；真实条件（遮挡、有手）分开报告，光照、模糊、噪声用对真实照片的模拟干扰测试（`docs/NEXT_STEPS.md` 第 3 步）；速度和资源以板端日志为准 | `src/qwen_vlm/evaluation/`，`runs/robust/`，板端日志 |
| (4) "Analyze system limitations and propose improvements" | 失败案例和局限章节 | 技术报告 |

## 3. Deliverables

| 原文 | 落实 |
| --- | --- |
| "Complete project source code, model files, and deployment instructions" | `release/`：源码、adapter、合并后模型、GGUF 及 SHA256、`deploy/jetson/README.md` |
| "A technical report covering system design, model principles, deployment, optimization, experimental evaluation, and conclusions" | `reports/technical_report.md`，最终格式待定 |
| "A live demonstration and oral defense using Jetson Nano" | 演示脚本和答辩提纲由 AI 准备；现场演示和答辩由用户本人完成 |
| "Demonstration video" | 板端工作正常后录制；AI 提供脚本 |

## 原文没有规定的事项

截止时间、报告篇幅和格式、提交平台、视频时长、评分权重、最低准确率或帧率、模型大小、数据集规模均未规定。本项目中出现的数字都是工程目标，不是课程要求。

## 待用户确认

- 原文写的是 "Jetson Nano"，本项目使用 Jetson Orin Nano 8GB。需要用户向老师确认是否接受。
- 截止时间、报告格式和提交渠道。
