# 整合冒烟（不是实验结果）

合成 val 集（15 条）上的端到端检查：训练 agent 用 `scripts/predict_transformers.py` 产出的预测，再经 `scripts/eval_predictions.py` 评测。只证明 Transformers 预测到评测出表这条链路可用。冒烟 adapter 只在 4 条样本上过拟合过，这里的数字没有任何意义。

放在 `runs/integration_smoke/` 下，是为了不被 `scripts/eval_compare.py`（扫描 `runs/*/eval`）收进正式对比表。
