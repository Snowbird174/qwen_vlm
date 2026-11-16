#!/usr/bin/env bash
# 把一个 HF 目录转成可在 llama.cpp 上跑的 GGUF。
# 以后训练得到的合并模型也可以用同一条命令，只要目录里是 HF 权重。
# 用法：scripts/export_gguf.sh <hf_model_dir> <out_dir> <tag>
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "用法: $0 <hf_model_dir> <out_dir> <tag>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env/activate.sh"

HF_DIR="$1"
OUT_DIR="$2"
TAG="$3"

avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
echo "转换前磁盘剩余 ${avail_kb} KB"
if (( avail_kb < 5 * 1024 * 1024 )); then
  echo "剩余不足 5GB，停止" >&2
  exit 1
fi

exec "$ROOT/envs/qwen_vlm/bin/python" "$ROOT/scripts/export_gguf.py" "$HF_DIR" "$OUT_DIR" "$TAG"
