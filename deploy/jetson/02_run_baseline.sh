#!/usr/bin/env bash
# 用指定 GGUF / mmproj / 像素档对一张测试图做一次推理，保存原始输出。
# 环境变量：
#   GGUF、MMPROJ、PIXELS（384/512/768，默认 512）、IMAGE、PORT
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./lib.sh
jetson_begin_log "02_run_baseline"

fail() {
  echo "RESULT: FAIL $*"
  exit 1
}

ROOT="$(jetson_root)"
MODEL_TAG="${MODEL_TAG:-base}"
GGUF="${GGUF:-$ROOT/models/$MODEL_TAG-q4_k_m.gguf}"
MMPROJ="${MMPROJ:-$ROOT/models/mmproj-$MODEL_TAG-f16.gguf}"
PIXELS="${PIXELS:-512}"
IMAGE="${IMAGE:-$ROOT/data/smoke/b0/bench_hammer_screwdriver.png}"
PORT="${PORT:-8080}"
PID=""
cleanup() { jetson_stop_pid "$PID"; }
trap cleanup EXIT

[[ -f "$GGUF" ]] || fail "找不到 GGUF：$GGUF"
[[ -f "$MMPROJ" ]] || fail "找不到 mmproj：$MMPROJ"
[[ -f "$IMAGE" ]] || fail "找不到图片：$IMAGE"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

server_log="$ROOT/logs/llama_server_baseline_$(date +%Y%m%d_%H%M%S).txt"
PID="$(jetson_start_server "$GGUF" "$MMPROJ" "$PIXELS" "$PORT" "$server_log")"
jetson_wait_health "$PORT" "$PID" || fail "llama-server 没有起来，看 $server_log"

out="$ROOT/logs/baseline_${PIXELS}_$(date +%Y%m%d_%H%M%S).jsonl"
"$PY" "$ROOT/deploy/jetson/bench_client.py" \
  --server "http://127.0.0.1:$PORT" \
  --image "$IMAGE" \
  --pixels "$PIXELS" \
  --quant "$(basename "$GGUF")" \
  --out "$out" \
  --repeat 1 \
  --warmup 0
echo "原始输出：$out"
echo "服务日志：$server_log"
echo "RESULT: PASS"
