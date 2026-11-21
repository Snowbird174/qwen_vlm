# 项目状态

更新于 2026-11-21。正式三折和结果复核已完成，恢复命令见 [本轮计划](docs/NEXT_STEPS.md)。

| 阶段 | 内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| P0 | 课程要求核对、环境、接口冻结 | 已验证 | `docs/coursework_requirements.md`，`reports/environment_server.md`，`contracts/` |
| P1 | 基础模型单图推理与结构化输出（服务器） | 已验证（合成图） | `runs/smoke_b0/` |
| P1 | 云端转换 GGUF，与 Transformers 对照 | 已验证（合成图） | `artifacts/quantized/base/manifest.json`，`runs/deploy_smoke/` |
| P1 | 板端试跑原模型 | 等待用户 | 已打包 `release/jetson_bundle_base/`，需要在板上运行 `00`–`02` |
| P2 | 真实照片采集、状态及器材框标注 | 已验证 | 50 张、100 条 checklist，`data/annotations/` |
| P2 | 器材框增强 | 已生成 | 100 对、200 张编辑图，`data/aug/v1/manifest.json` |
| P2 | 固定三折划分及泄漏检查 | 已验证 | `data/splits/cv3/`，100 条真实测试样本恰好覆盖一次 |
| P3 | B0 真实评测 | 已验证 | `runs/B0_s001/eval/` |
| P3 | B0 服务器分辨率和量化扫描 | 已验证 | L40 上 BF16 三档、GGUF 三档，`runs/comparison/` |
| P3 | LoRA 训练与固定三折入口，B1/M/A1–A4 配置 | B1、M 的真实训练子集试跑已通过 | `scripts/run_cv.py`，`work/cv3_pilot/` |
| P3 | 正式训练与消融 | 18/18 折已复核 | 每折 50 步，B1、M、A1–A4 各 100 条真实折外预测。详见 `runs/cv3_folds/`、`runs/*_cv3/eval/` |
| P3 | 对比与部署选择 | 完成 | B0 误报齐全 7/31，六种 LoRA 方法均为 2/31。M 决策准确率 86/100，unknown 召回 1/18，按规则保留 M。见 `EXPERIMENTS.md` |
| P3 | 鲁棒性条件 | 九档数据校验及 B0 九档预测已完成，B1/M 待测 | `work/coursework/robust_runs/robust/`，`runs/logs/robust_B0_recovery.log` |
| P3 | 评测代码（全部任务指标、bootstrap、出表） | 已验证（手算单测） | `src/qwen_vlm/evaluation/`，`tests/test_metrics.py` |
| P4 | 量化与像素档的板端实测 | 等待用户 | `deploy/jetson/03_bench.sh` |
| P5 | 摄像头演示 | 已准备未验证 | 服务器上图片模式已跑通；摄像头只能在板上测 |
| P6 | 报告、视频、答辩材料 | 未开始 | 骨架：`docs/model_principles.md`，`reports/templates/` |
