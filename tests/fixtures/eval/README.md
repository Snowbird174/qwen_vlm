# 评测测试夹具

这里的 JSONL 和 tegrastats 日志都是为了检验评测代码而手写的格式示例。

- `gold.jsonl` / `pred.jsonl`：手工构造的金标准和预测，用来核对分子和分母。不是模型输出，也不是实验结果。
- `tegrastats_jp4.log`、`tegrastats_jp5.log`、`tegrastats_jp6.log`：三种日志格式的例子。里面的内存、利用率和功率数字是虚构的，不是 Jetson 实测。
