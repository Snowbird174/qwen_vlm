# Source this file (or rely on conda activate.d hooks) before any project work.
# Every cache, temp file and download for this project must stay under QWEN_VLM_ROOT.
export QWEN_VLM_ROOT=/root/gpufree-data/qwen_vlm

mkdir -p "$QWEN_VLM_ROOT"/cache/{huggingface/hub,huggingface/datasets,torch,pip,conda_pkgs,xdg} "$QWEN_VLM_ROOT"/work/tmp

# The server sets HF_HUB_CACHE/HF_DATASETS_CACHE globally to an older project; they take
# precedence over HF_HOME, so all three must be overridden.
export HF_HOME="$QWEN_VLM_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="$QWEN_VLM_ROOT/cache/torch"
export PIP_CACHE_DIR="$QWEN_VLM_ROOT/cache/pip"
export XDG_CACHE_HOME="$QWEN_VLM_ROOT/cache/xdg"
export CONDA_PKGS_DIRS="$QWEN_VLM_ROOT/cache/conda_pkgs"
export TMPDIR="$QWEN_VLM_ROOT/work/tmp"
export TRITON_CACHE_DIR="$QWEN_VLM_ROOT/cache/xdg/triton"
export TOKENIZERS_PARALLELISM=false

# Measured on this server (2026-11-16): Aliyun PyPI direct ~1 MB/s, Tsinghua ~20-100 KB/s,
# PyPI/GitHub/HF CDN through the proxy ~10-30 KB/s, ModelScope through the proxy 2-6 MB/s.
# Run pip with the proxy variables unset: `qv_noproxy pip install ...`.
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export PIP_DEFAULT_TIMEOUT=120
qv_noproxy() { env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy -u ALL_PROXY -u all_proxy "$@"; }

# The system cuDNN 9.8 in /usr/lib/x86_64-linux-gnu (global LD_LIBRARY_PATH and ld.so.cache)
# shadows the cuDNN 9.19 bundled with PyTorch, and torch then refuses to use cuDNN.
# Drop the system dirs and put the bundled cuDNN first.
_QV_CUDNN="$QWEN_VLM_ROOT/envs/qwen_vlm/lib/python3.12/site-packages/nvidia/cudnn/lib"
LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v -x -e /usr/lib/x86_64-linux-gnu -e /usr/local/cuda/lib64 -e "$_QV_CUDNN" -e '' | paste -sd: -)
export LD_LIBRARY_PATH="$_QV_CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _QV_CUDNN
