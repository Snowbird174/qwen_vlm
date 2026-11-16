# llama.cpp 版本

- tag: `b12425`
- commit: `18b74fa41dcac1f55ae1371fe58153f7225941f6`
- 发布日期：2026-11-15
- 源码目录：`vendor/llama.cpp`（已去掉 GitHub tarball 的顶层目录名 `llama.cpp-b12425`，里面没有 `.git`）

## 获取方式

GitHub tag tarball：

`https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12425.tar.gz`

（codeload 直链约 37 MiB）。2026-11-16 在这台服务器上，直连 GitHub 大约 15 KB/s。实际下载用了 `aria2c -x 16`，地址是

`https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12425.tar.gz`

平均约 482 KB/s。commit 用下面的命令核对，返回的 tag 对象就是这个 commit（不是带 peel 的 annotated tag）：

```bash
git ls-remote https://gh-proxy.com/https://github.com/ggml-org/llama.cpp.git refs/tags/b12425
```

没有使用 gitee 镜像，因此不存在 commit 对不齐的问题。

解压后的目录没有 `.git`。编译时设置了 `GIT_CEILING_DIRECTORIES`，避免 `git rev-parse` 走到上层课程仓库。因此 `llama-server --version` 显示 `commit unknown`。上面的 tag 和 commit 以 `git ls-remote` 为准，不以二进制里的字符串为准。

## 为什么这个版本支持 Qwen3-VL（含 DeepStack）

Qwen3-VL 最初进入 llama.cpp 是 tag `b6887`（commit `d261223d24e97f2df50220e4a5b7f0adb69bba81`，PR #16780，2025-10-30）。`b12425` 在那之后，源码里仍保留并继续维护这条路径：

1. `conversion/__init__.py` 把 Hugging Face 架构 `Qwen3VLForConditionalGeneration` 映射到转换模块 `qwen3vl`。
2. `conversion/qwen3vl.py` 的 `Qwen3VLVisionModel`：
   - 投影器类型写成 `QWEN3VL`；
   - 从 `vision_config.deepstack_visual_indexes` 生成 `is_deepstack_layers`，并写入 GGUF；
   - `visual.deepstack_merger_list.*` 映射成 DeepStack 的 norm / fc1 / fc2。
3. 同文件的 `Qwen3VLTextModel.set_gguf_parameters` 写入 `n_deepstack_layers`（语言侧要预留 DeepStack 注入层）。
4. `tools/mtmd/mtmd-model.cpp` 的 `PROJECTOR_TYPE_QWEN3VL` 分支读取 spatial merge、像素上下限，并在加载权重时取出 `deepstack_norm` / `deepstack_fc1` / `deepstack_fc2`。
5. `tools/mtmd/mtmd.cpp` 为该投影器选择 `mtmd_image_preprocessor_qwen3vl`。静止图走动态分辨率，视频帧走 Qwen3-VL 的 `smart_resize`。
6. `common/arg.cpp` 提供 `--image-min-tokens` 和 `--image-max-tokens`（PR #16921 引入，本 tag 仍在）。服务端和 `llama-mtmd-cli` 都会把它们传给 mtmd。

像素换算：patch 16、merge 2，一个视觉 token 是 32×32 像素。`--image-max-tokens N` 在加载时变成 `image_max_pixels = N * 1024`。因此 384² / 512² / 768² 分别对应 144 / 256 / 576 个 token。缺省的最小像素与 HF `preprocessor_config.json` 的 `shortest_edge=65536` 对齐，也就是 64 个 token。

## 转换环境

转换脚本跑在 `envs/qwen_vlm`（Python 3.12，torch 2.11.0+cu130，transformers 5.5.4），进程里把 `CUDA_VISIBLE_DEVICES` 置空，权重在 CPU 上读。没有把 CPU 版 torch 和 llama.cpp 钉死的 `transformers==4.57.6` 装进 `envs/gguf`：那一套大约 1.5–2 GB，会挤占本机 8 GB 的 GGUF 预算。`sentencepiece` 若缺失，只装到 `work/deploy_pydeps`。

服务器编译见 `scripts/export_build_llamacpp_server.sh`（sm_89）。板端编译见 `deploy/jetson/01_build_llamacpp.sh`（sm_87），两边用同一份源码 tarball。
