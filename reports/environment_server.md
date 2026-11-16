# 服务器环境记录

记录时间：2026-11-16（Asia/Shanghai）。所有数值都是在这台服务器上实际查询得到的。

## 硬件与系统

| 项目 | 值 |
| --- | --- |
| GPU | 1 × NVIDIA L40，46068 MiB，compute capability 8.9 |
| 驱动 | 580.126.09（支持 CUDA 13.0） |
| 系统 CUDA 工具链 | nvcc 12.8.93（`/usr/local/cuda-12.8`） |
| CPU | Intel Xeon Platinum 8358 @ 2.60GHz，128 个逻辑核 |
| 内存 | 1007 GiB，没有 swap |
| 系统 | Ubuntu 22.04.5 LTS，内核 5.15.0-168-generic |
| 构建工具 | gcc 11.4.0，cmake 3.22.1（系统自带），aria2c |
| 开始时 GPU 占用 | 122 MiB，没有其他进程 |

## 磁盘

| 挂载点 | 总量 | 项目开始前可用 | 说明 |
| --- | --- | --- | --- |
| `/root/gpufree-data` | 49G | 30G | 项目根目录所在盘，与 smolVLA 等项目共用 |
| `/`（overlay） | 30G | 19G | `/opt/conda` 所在盘；本项目不往这里写 |

装完两个环境、下完基础模型后约剩 21G。磁盘预算见执行计划第 6 节。

## 网络（会直接影响后续工作）

- 服务器外网很慢：网卡总吞吐只有约 8 KB/s 的背景流量，直连国内镜像约 30-50 KB/s，ping 丢包约 33%。
- 本地代理 `127.0.0.1:7897`（clash-verge）是全局设置。它会把 `hf-mirror.com` 重定向到 huggingface.co，而 HF 的 CDN 走代理只有约 30 KB/s。
- **ModelScope 走代理**能到 2-6 MB/s，基础模型权重就是从这里下载的。下载后用 HF 固定 revision 记录的 LFS SHA256 做了校验，见 `artifacts/base/manifest.json`。
- pip 走清华源直连（不经代理）可以用，但很慢：几个小包装了约 5 分钟，gradio 及其依赖用时更长。
- `www.baidu.com` 等域名优先解析到 IPv6，而 IPv6 不通，所以 curl 测试要加 `-4`。

## 已有环境（只读，未修改）

| 环境 | Python | torch | transformers |
| --- | --- | --- | --- |
| `/opt/conda/envs/isaaclab`（默认激活） | 3.11 | 2.7.0+cu128 | 4.56.0（不支持 Qwen3-VL） |
| `/opt/conda/envs/lerobot` | 3.12 | 2.11.0+cu130 | 5.5.4 |
| `/root/gpufree-data/conda/envs/lerobot` | 3.12 | 2.11.0+cu130 | 5.5.4 |

全局环境变量会影响本项目：`HF_HOME`、`HF_HUB_CACHE`、`HF_DATASETS_CACHE` 指向 `/root/gpufree-data/hf`，`HF_ENDPOINT=https://hf-mirror.com`，另有 `HTTP(S)_PROXY`/`ALL_PROXY` 代理设置，`LD_LIBRARY_PATH` 中包含 `/usr/lib/x86_64-linux-gnu`。

## 本项目环境

| 环境 | 路径 | 构建方式 | 关键版本 |
| --- | --- | --- | --- |
| 主环境 | `envs/qwen_vlm` | 从 `/opt/conda/envs/lerobot` 离线克隆，再用 pip 补装小包 | Python 3.12.16，torch 2.11.0+cu130，cuDNN 9.19，transformers 5.5.4，peft 0.21.3，accelerate 1.15.0 |
| 转换环境 | `envs/gguf` | 用本地 conda 包缓存离线创建 | Python 3.11.18，只有 pip；**目前未使用**：GGUF 转换实际在 `envs/qwen_vlm` 中运行（关闭 CUDA），因为给 `envs/gguf` 装 CPU 版 torch 需要约 1.5-2GB 磁盘 |

版本精确锁定见 `env/environment.lock.yml` 和 `env/pip_freeze_*.txt`。偏离原计划的地方：原计划写的是 Python 3.11 加全新安装 torch cu128，因为网络太慢改成了克隆；transformers 5.x 已原生支持 Qwen3-VL。

### `env/activate.sh` 解决的两个问题

1. 全局的 `HF_HUB_CACHE`/`HF_DATASETS_CACHE` 优先级高于 `HF_HOME`，所以三个变量都要改成指向 `cache/huggingface`。
2. 系统自带的 cuDNN 9.8（在 `/usr/lib/x86_64-linux-gnu`，同时出现在 `LD_LIBRARY_PATH` 和 `ld.so.cache` 里）会顶替 torch 自带的 cuDNN 9.19，导致 `torch.backends.cudnn` 报版本不兼容（原 lerobot 环境也有这个问题）。`activate.sh` 把系统目录移出 `LD_LIBRARY_PATH`，并把环境内 `nvidia/cudnn/lib` 放到最前面。修复后已验证 cuDNN 卷积能在 GPU 上运行。

两个环境的 `etc/conda/activate.d/qwen_vlm_paths.sh` 都会 source `env/activate.sh`。直接调用 `envs/qwen_vlm/bin/python` 时必须先手动 `source env/activate.sh`。

## 基础模型

`Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203`，本地路径 `artifacts/base/Qwen3-VL-2B-Instruct/`。逐文件 SHA256 见 `artifacts/base/manifest.json`。
