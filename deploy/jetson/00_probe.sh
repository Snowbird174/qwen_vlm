#!/usr/bin/env bash
# 只读探测 Jetson 环境。不安装、不改功耗模式、不用 sudo。
# 探测完成即 RESULT: PASS。某一项不存在会记成 MISSING，不算失败。
set -u
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./lib.sh
jetson_begin_log "00_probe"

section() {
  echo
  echo "===== $1 ====="
}

run_optional() {
  local title="$1"
  shift
  section "$title"
  if "$@" > /tmp/qwen_vlm_probe_out 2>/tmp/qwen_vlm_probe_err; then
    cat /tmp/qwen_vlm_probe_out
    echo "OK"
  else
    echo "MISSING 或命令失败：$*"
    cat /tmp/qwen_vlm_probe_err || true
  fi
}

section "型号"
if [[ -r /proc/device-tree/model ]]; then
  tr -d '\0' < /proc/device-tree/model
  echo
  echo "OK"
else
  echo "MISSING /proc/device-tree/model"
fi

section "L4T / JetPack"
if [[ -r /etc/nv_tegra_release ]]; then
  cat /etc/nv_tegra_release
else
  echo "MISSING /etc/nv_tegra_release"
fi
if [[ -r /etc/nv_tegra_release ]] && command -v dpkg >/dev/null 2>&1; then
  dpkg -l 'nvidia-jetpack*' 2>/dev/null | awk 'NR==1 || /nvidia-jetpack/' || echo "MISSING nvidia-jetpack 包信息"
fi

run_optional "nvpmodel -q" nvpmodel -q
run_optional "CUDA/nvcc" nvcc --version
section "CUDA 版本文件"
if [[ -r /usr/local/cuda/version.json ]]; then
  cat /usr/local/cuda/version.json
else
  echo "MISSING /usr/local/cuda/version.json"
fi
echo "CUDA_HOME=${CUDA_HOME:-未设置}"
ls -d /usr/local/cuda* 2>/dev/null || echo "MISSING /usr/local/cuda*"

section "内存和 swap"
free -h || echo "MISSING free"
swapon --show || echo "没有 swap 或 swapon 不可用"
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo || true

section "磁盘"
df -h || true

section "Python / cmake / gcc"
python3 --version || echo "MISSING python3"
cmake --version | head -n 1 || echo "MISSING cmake"
gcc --version | head -n 1 || echo "MISSING gcc"
g++ --version | head -n 1 || echo "MISSING g++"

section "摄像头"
ls -l /dev/video* 2>/dev/null || echo "MISSING /dev/video*"
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --list-devices || echo "v4l2-ctl 执行失败"
else
  echo "MISSING v4l2-ctl"
fi

section "用户组"
id || true
groups || true

section "网络"
if python3 - <<'PY'
import urllib.request
urllib.request.urlopen("https://pypi.org", timeout=8).read(64)
PY
then
  echo "OK 可以访问 https://pypi.org（01 安装依赖需要网络）"
else
  echo "MISSING 当前访问不了 https://pypi.org。构建 llama.cpp 仍可离线，pip 安装需要网络或离线 wheel。"
fi

echo
echo "探测只记录现状，不修改系统。"
echo "RESULT: PASS"
