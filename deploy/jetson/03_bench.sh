#!/usr/bin/env bash
# 量化 × 像素档的板端计时。预热 5 次，正式默认 100 次。
# 不改 nvpmodel，也不启动 jetson_clocks，只把当前状态写进日志。
# 某一档 OOM 或失败会记下来，继续下一档。
#
# 环境变量：
#   QUANTS   逗号分隔，默认 q4_k_m,q8_0,f16（文件不存在的档会记 FAIL 并跳过）
#   PIXELS   逗号分隔，默认 384,512,768
#   REPEAT   正式次数，默认 100
#   WARMUP   预热次数，默认 5
#   PORT     默认 8080
set -u
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./lib.sh
jetson_begin_log "03_bench"

ROOT="$(jetson_root)"
QUANTS="${QUANTS:-q4_k_m,q8_0,f16}"
PIXELS_LIST="${PIXELS:-384,512,768}"
REPEAT="${REPEAT:-100}"
WARMUP="${WARMUP:-5}"
PORT="${PORT:-8080}"
MODELS="${MODELS_DIR:-$ROOT/models}"
MODEL_TAG="${MODEL_TAG:-base}"
MMPROJ="${MMPROJ:-$MODELS/mmproj-$MODEL_TAG-f16.gguf}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/logs/bench_$STAMP"
mkdir -p "$OUT_DIR"

echo "nvpmodel:"
nvpmodel -q >"$OUT_DIR/nvpmodel.txt" 2>&1 || echo "MISSING nvpmodel" | tee -a "$OUT_DIR/nvpmodel.txt"
if pgrep -f jetson_clocks >/dev/null 2>&1; then
  echo "jetson_clocks: running" | tee "$OUT_DIR/jetson_clocks.txt"
else
  echo "jetson_clocks: not_running" | tee "$OUT_DIR/jetson_clocks.txt"
fi
echo "脚本不会修改功耗模式，也不会自己运行 jetson_clocks。"

TEGRA_PID=""
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 500 >"$OUT_DIR/tegrastats.txt" 2>&1 &
  TEGRA_PID=$!
  echo "tegrastats pid=$TEGRA_PID"
else
  echo "MISSING tegrastats" | tee "$OUT_DIR/tegrastats.txt"
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    jetson_stop_pid "$SERVER_PID"
  fi
  if [[ -n "$TEGRA_PID" ]]; then
    jetson_stop_pid "$TEGRA_PID"
  fi
}
trap cleanup EXIT
SERVER_PID=""

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

mapfile -t images < <(find "$ROOT/data/smoke/b0" -type f \( -name '*.png' -o -name '*.jpg' \) | sort)
if ((${#images[@]} == 0)); then
  echo "RESULT: FAIL 没有测试图片 $ROOT/data/smoke/b0"
  exit 1
fi

fail_count=0
IFS=',' read -r -a quant_arr <<<"$QUANTS"
IFS=',' read -r -a pixel_arr <<<"$PIXELS_LIST"
for quant in "${quant_arr[@]}"; do
  for pixels in "${pixel_arr[@]}"; do
    quant="${quant// /}"
    pixels="${pixels// /}"
    gguf="$MODELS/$MODEL_TAG-${quant}.gguf"
    echo
    echo "===== quant=$quant pixels=$pixels ====="
    if [[ ! -f "$gguf" || ! -f "$MMPROJ" ]]; then
      echo "FAIL 缺少文件 gguf=$gguf mmproj=$MMPROJ"
      echo "{\"quant\":\"$quant\",\"pixels\":\"$pixels\",\"error\":\"missing file\"}" >>"$OUT_DIR/errors.jsonl"
      fail_count=$((fail_count + 1))
      continue
    fi
    server_log="$OUT_DIR/server_${quant}_${pixels}.log"
    if ! SERVER_PID="$(jetson_start_server "$gguf" "$MMPROJ" "$pixels" "$PORT" "$server_log")"; then
      echo "FAIL 无法启动（像素档非法或没有 llama-server）"
      fail_count=$((fail_count + 1))
      SERVER_PID=""
      continue
    fi
    if ! jetson_wait_health "$PORT" "$SERVER_PID"; then
      echo "FAIL llama-server 没起来，可能是 OOM。日志 $server_log"
      echo "{\"quant\":\"$quant\",\"pixels\":\"$pixels\",\"error\":\"server failed to become healthy\"}" >>"$OUT_DIR/errors.jsonl"
      jetson_stop_pid "$SERVER_PID"
      SERVER_PID=""
      fail_count=$((fail_count + 1))
      continue
    fi
    jsonl="$OUT_DIR/${quant}_${pixels}.jsonl"
    image_args=()
    for image in "${images[@]}"; do
      image_args+=(--image "$image")
    done
    if ! "$PY" "$ROOT/deploy/jetson/bench_client.py" \
      --server "http://127.0.0.1:$PORT" \
      --pixels "$pixels" \
      --quant "$quant" \
      --out "$jsonl" \
      --repeat "$REPEAT" \
      --warmup "$WARMUP" \
      "${image_args[@]}"; then
      echo "FAIL 客户端中途出错 quant=$quant pixels=$pixels"
      echo "{\"quant\":\"$quant\",\"pixels\":\"$pixels\",\"error\":\"client failed\"}" >>"$OUT_DIR/errors.jsonl"
      fail_count=$((fail_count + 1))
    else
      echo "OK $jsonl"
    fi
    jetson_stop_pid "$SERVER_PID"
    SERVER_PID=""
  done
done

echo
echo "结果目录 $OUT_DIR"
if ((fail_count > 0)); then
  echo "RESULT: FAIL ${fail_count} 个配置没有跑完，细节在 $OUT_DIR/errors.jsonl 和服务日志里"
  exit 1
fi
echo "RESULT: PASS"
