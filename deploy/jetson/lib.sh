# 被 deploy/jetson/*.sh source。不要直接执行。
# 约定：bundle 根目录是本文件的上两级（.../deploy/jetson/lib.sh）。

jetson_root() {
  (cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
}

jetson_begin_log() {
  local name="$1"
  local root
  root="$(jetson_root)"
  mkdir -p "$root/logs"
  LOG_FILE="$root/logs/${name}_$(date +%Y%m%d_%H%M%S).txt"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "LOG_FILE=$LOG_FILE"
  echo "开始时间 $(date --iso-8601=seconds)"
}

jetson_pixels_to_tokens() {
  case "$1" in
    384) echo 144 ;;
    512) echo 256 ;;
    768) echo 576 ;;
    *)
      echo "未知像素档: $1（只能是 384、512、768）" >&2
      return 1
      ;;
  esac
}

jetson_llama_server() {
  local root
  root="$(jetson_root)"
  if [[ -n "${LLAMA_SERVER:-}" && -x "$LLAMA_SERVER" ]]; then
    echo "$LLAMA_SERVER"
    return
  fi
  local candidate="$root/llama.cpp/build/bin/llama-server"
  if [[ -x "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  echo "找不到 llama-server。先跑 01_build_llamacpp.sh，或设置 LLAMA_SERVER" >&2
  return 1
}

# 启动 llama-server。像素档必须和客户端一致。--fit off 避免把整块内存吃掉。
jetson_start_server() {
  local model="$1" mmproj="$2" pixels="$3" port="$4" extra_log="$5"
  local bin max_tok
  bin="$(jetson_llama_server)" || return 1
  max_tok="$(jetson_pixels_to_tokens "$pixels")" || return 1
  echo "启动 llama-server 像素档=${pixels} image-max-tokens=${max_tok} 端口=${port}" >&2
  setsid "$bin" \
    -m "$model" \
    --mmproj "$mmproj" \
    --host 127.0.0.1 \
    --port "$port" \
    --jinja \
    --temp 0 \
    -c "${N_CTX:-4096}" \
    -ngl 99 \
    -b 2048 \
    -ub 2048 \
    --image-min-tokens 64 \
    --image-max-tokens "$max_tok" \
    --no-warmup \
    --fit off \
    >"$extra_log" 2>&1 < /dev/null &
  echo $!
}

jetson_wait_health() {
  local port="$1" pid="$2"
  local i
  for i in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "llama-server 进程 $pid 已退出"
      return 1
    fi
    if python3 - "$port" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2).read()
except Exception:
    sys.exit(1)
PY
    then
      echo "llama-server 健康检查通过"
      return 0
    fi
    sleep 1
  done
  echo "等待 llama-server 超时"
  return 1
}

jetson_stop_pid() {
  local pid="$1"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
}
