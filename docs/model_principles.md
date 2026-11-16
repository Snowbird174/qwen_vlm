# Qwen3-VL-2B-Instruct 模型原理（技术报告"模型原理"一章的骨架）

本文中的数字来自 `artifacts/base/Qwen3-VL-2B-Instruct/config.json`、`preprocessor_config.json`，以及对 `model.safetensors` 头部逐张量计数的结果。需要引用官方说法的地方都标了"见模型卡"，写报告时附上官方链接。

## 1. 总体结构

Qwen3-VL 由三部分组成：视觉编码器（ViT）、视觉到语言的投影（merger），以及仅解码器的语言模型。图像先被切成 patch 并编码成视觉 token，再与文本 token 一起送进语言模型做自回归生成。

| 模块 | 参数量 | 关键配置 |
| --- | --- | --- |
| 视觉 patch embedding + 位置嵌入 | 1.6M + 2.4M | patch 16×16，`temporal_patch_size=2`（静态图像在时间维复制一份），可学习位置嵌入 2304 个 |
| 视觉 Transformer blocks | 302.3M | 24 层，hidden 1024，16 头，MLP 4096，GELU(tanh) |
| merger（主投影） | 25.2M | 2×2 空间合并（`spatial_merge_size=2`）后投影到 2048 维 |
| DeepStack mergers | 75.5M | 从视觉第 5、11、17 层取特征，各自投影到 2048 维 |
| 语言模型 decoder layers | 1409.4M | 28 层，hidden 2048，16 个 Q 头和 8 个 KV 头（GQA），head_dim 128，MLP 6144（SwiGLU），RMSNorm |
| 词嵌入（与输出头共享） | 311.2M | 词表 151936，`tie_word_embeddings=true` |
| **合计** | **约 2.13B** | 权重为 bf16，约 4.26GB |

## 2. 图像预处理与视觉 token 数

1. **smart resize**：保持长宽比，把宽和高都调整为 32 的整数倍（32 = patch 16 × 合并 2），并使总像素数落在 `[shortest_edge, longest_edge]` 之间（本项目通过 `max_pixels` 控制上限）。归一化参数 mean = std = 0.5。
2. 每个 16×16 patch 是一个 ViT token，经过 2×2 合并后变成一个语言侧的视觉 token，所以每个视觉 token 对应 32×32 像素。
3. 视觉 token 数 = `t · h · w / 4`，其中 `(t, h, w)` 是 `image_grid_thw`。本项目对一张 1280×960 的图实测如下（`runs/smoke_b0/*/summary.json`）：

| max_pixels | grid (t,h,w) | 实际送入尺寸 | 视觉 token 数 |
| --- | --- | --- | --- |
| 384² = 147456 | (1, 20, 26) | 416×320 | 130 |
| 512² = 262144 | (1, 26, 36) | 576×416 | 234 |
| 768² = 589824 | (1, 40, 54) | 864×640 | 540 |

这就是第二项优化"输入分辨率调整"的作用方式：视觉 token 数大致随像素预算线性变化，而预填充（prefill）的计算量和 KV cache 都随 token 数增长。代价是细长物体（线缆、表笔线）和局部遮挡后的残余部分可能变得难以辨认。

## 3. 视觉与语言的融合

- **占位与替换**：chat template 在文本中插入 `<|vision_start|>`，接着放入 N 个 `<|image_pad|>`，再加 `<|vision_end|>`。前向计算时，N 个占位 token 的嵌入被替换成 merger 输出的视觉特征。B0 冒烟测试中已验证占位符数量等于 `t·h·w/4`。
- **DeepStack**：除了最后一层的视觉特征，还从 ViT 的浅、中、深层（第 5、11、17 层）各取一组特征，分别投影后**加到语言模型前几层对应视觉 token 位置的隐藏状态上**，把多层次的视觉细节注入语言模型（见模型卡/技术报告）。部署时需要确认 llama.cpp 的 mmproj 也实现了这一路径。
- **位置编码 MRoPE（interleaved）**：RoPE 的频率被分成时间、高、宽三组（`mrope_section=[24,20,20]`，交错排列）。视觉 token 使用二维（加时间）的位置 id，文本 token 的三个分量相同。这样语言模型能感知视觉 token 的空间布局。

## 4. 语言解码

标准的仅解码器 Transformer：GQA 注意力、SwiGLU MLP、RMSNorm、`rope_theta=5e6`。本项目使用贪心解码，输出上限 128 token。输出的是固定格式的 JSON（五类三状态），其余内容（完整性决策、中文回答）由 `common/derive.py` 确定性生成，这样评分不受措辞影响。

## 5. 本项目的改进：领域监督微调

- **LoRA**：把预训练权重 W 冻结，只学习低秩增量 ΔW = B·A（秩 r，缩放 α/r）。本项目只在**语言模型**的 q/k/v/o/gate/up/down 投影上加 LoRA，视觉编码器和所有 merger 都冻结。可训练参数名单由训练脚本写入每次运行的 `trainable_params.txt`。
- **监督信号**：标准交叉熵，只监督 assistant 回答中的 token。system、user 和图像占位一律屏蔽。
- **方法要点**（不是新架构，也不是对比学习）：
  1. 成对场景：同一布置只拿走或放入一件器材，由此生成"同问不同答"的存在性辅助样本，让答案的变化只能来自图像的变化；
  2. 困难负样本加权：针对容易混淆的情况（表笔线和线缆等）提高采样权重；
  3. unknown 监督：遮挡或画面不全时教模型回答 unknown，而不是凭共现习惯猜"齐全"。
- 消融 A1/A2/A3 分别去掉以上三项。混合规则见 `contracts/training_mix.md`。

## 6. 部署相关原理

- **合并**：W' = W + (α/r)·B·A，合并后结构与原模型相同，可以直接转 GGUF。
- **GGUF / llama.cpp**：语言模型和视觉投影（mmproj）分成两个文件。量化（Q8_0、Q4_K_M）主要作用在语言模型权重上，mmproj 的精度单独记录。在 Orin 上，CPU 和 GPU 共享 8GB 内存，权重体积直接决定能否装下以及内存带宽压力。
- **量化**：Q8_0 是按块（32 个权重共享一个缩放）的 8 bit 对称量化；Q4_K_M 是 K-quant 系列的 4 bit 混合量化，对部分敏感张量保留更高位宽。具体以所用 llama.cpp 版本的文档为准。

## 待补充（写报告时）

- 引用模型卡、Qwen3-VL 技术报告和 DeepStack 原论文。
- 补一张架构示意图（ViT → merger / DeepStack → LLM）。
