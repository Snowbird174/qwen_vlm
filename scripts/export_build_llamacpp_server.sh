#!/usr/bin/env bash
# 在服务器（L40，sm_89）上配置并编译 llama.cpp 的多模态与量化工具。
# 只编译本项目用到的目标，避免把整个 examples 树都编进来。
# 用法：scripts/export_build_llamacpp_server.sh
# 可选环境变量：JOBS（默认 16）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env/activate.sh"

SRC="$ROOT/vendor/llama.cpp"
if [[ ! -f "$SRC/CMakeLists.txt" ]]; then
  echo "找不到 $SRC/CMakeLists.txt" >&2
  exit 1
fi

avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if (( avail_kb < 5 * 1024 * 1024 )); then
  echo "磁盘剩余不足 5GB（${avail_kb} KB），停止构建" >&2
  exit 1
fi

cd "$SRC"
# 解压出来的源码没有 .git。不挡住的话，git 会走到上层课程仓库，
# llama-server --version 会印出那个仓库的 commit，而不是 llama.cpp 的。
export GIT_CEILING_DIRECTORIES="$ROOT/vendor"
# 课程要求的配置。CMAKE_BUILD_TYPE 未指定时，llama.cpp 的 CMakeLists 会默认 Release。
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF
cmake --build build -j "${JOBS:-16}" --target llama-mtmd-cli llama-server llama-quantize llama-bench

echo "构建完成："
ls -lh build/bin/llama-mtmd-cli build/bin/llama-server build/bin/llama-quantize build/bin/llama-bench
df -h "$ROOT" | tail -1
