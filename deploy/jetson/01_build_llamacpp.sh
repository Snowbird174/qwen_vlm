#!/usr/bin/env bash
# 用 bundle 里的 llama.cpp 源码在 Jetson 上编译，并建一个不依赖系统包的 venv。
# 不用 sudo。编译目标是 sm_87（Orin）。JOBS 默认 2，避免 8GB 内存上并行 nvcc 把机器打满。
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./lib.sh
jetson_begin_log "01_build_llamacpp"

fail() {
  echo "RESULT: FAIL $*"
  exit 1
}

ROOT="$(jetson_root)"
cd "$ROOT"

avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
echo "磁盘剩余 ${avail_kb} KB"
if (( avail_kb < 5 * 1024 * 1024 )); then
  fail "剩余不足 5GB"
fi

if [[ ! -f "$ROOT/llama.cpp/CMakeLists.txt" ]]; then
  tarball="$ROOT/llama.cpp-b12425.tar.gz"
  if [[ ! -f "$tarball" ]]; then
    fail "没有 llama.cpp 源码，也没有 $tarball"
  fi
  tar -xzf "$tarball" -C "$ROOT"
fi
if [[ ! -f "$ROOT/llama.cpp/CMakeLists.txt" ]]; then
  fail "解压后仍找不到 llama.cpp/CMakeLists.txt"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  fail "找不到 nvcc。确认 JetPack 的 CUDA 在 PATH 里，例如 export PATH=/usr/local/cuda/bin:\$PATH"
fi
if ! command -v cmake >/dev/null 2>&1; then
  fail "找不到 cmake"
fi

cd "$ROOT/llama.cpp"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF
cmake --build build -j "${JOBS:-2}" --target llama-mtmd-cli llama-server llama-quantize llama-bench
test -x build/bin/llama-server || fail "没有生成 llama-server"

if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv" || fail "python3 -m venv 失败。板子需要 python3-venv，但脚本不会用 sudo 安装它"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/deploy/jetson/requirements.txt"
python - <<'PY'
import gradio, cv2, PIL, yaml, jsonschema, requests
print("app deps", gradio.__version__, cv2.__version__)
PY

echo "llama-server: $ROOT/llama.cpp/build/bin/llama-server"
echo "RESULT: PASS"
