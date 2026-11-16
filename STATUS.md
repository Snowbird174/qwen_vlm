# 项目状态

更新：2026-11-16。四种状态：**已验证**（有日志或产物证明）、**已准备未验证**（代码或脚本已写好，但还没在目标条件下运行）、**等待用户**（缺用户提供的材料）、**未开始**。

| 阶段 | 内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| P0 | 课程要求核对、环境、接口冻结 | 已验证 | `docs/coursework_requirements.md`，`reports/environment_server.md`，`contracts/` |
| P1 | 基础模型单图推理与结构化输出（服务器） | 已验证（合成图） | `runs/smoke_b0/` |
| P1 | 云端转换 GGUF，与 Transformers 对照 | 已验证（合成图） | `artifacts/quantized/base/manifest.json`，`runs/deploy_smoke/` |
| P1 | 板端试跑原模型 | 等待用户 | 已打包 `release/jetson_bundle_base/`，需要在板上运行 `00`–`02` |
| P2 | 采集手册、标注模板、校验、划分、泄漏检查 | 已准备未验证（真实数据） | `docs/data_collection.md`，`src/qwen_vlm/data/`，合成集已跑通 |
| P2 | 真实照片采集与标注 | 等待用户 | — |
| P3 | LoRA 训练代码，B1/M/A1–A3 配置 | 已验证（合成冒烟） | `runs/train_smoke/` |
| P3 | 正式训练与消融 | 等待真实数据 | — |
| P3 | 评测代码（全部任务指标、bootstrap、出表） | 已验证（手算单测） | `src/qwen_vlm/evaluation/`，`tests/test_metrics.py` |
| P4 | 量化与像素档的板端实测 | 等待用户 | `deploy/jetson/03_bench.sh` |
| P5 | 摄像头演示 | 已准备未验证 | 服务器上图片模式已跑通；摄像头只能在板上测 |
| P6 | 报告、视频、答辩材料 | 未开始 | 骨架：`docs/model_principles.md`，`reports/templates/` |

**云端准备完成 ≠ 模型训练完成 ≠ 板端验证完成 ≠ 课程完成。** 目前只达到第一项。
