#!/usr/bin/env bash
# 打包给 Jetson 的离线目录。
# 用法：scripts/make_jetson_bundle.sh <gguf_dir> <out_dir>
# 默认只放 Q4_K_M 语言 GGUF 和 F16 mmproj。需要别的档时设置 INCLUDE_QUANTS=q4_k_m,q8_0,f16。
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法: $0 <gguf_dir> <out_dir>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env/activate.sh"

GGUF_DIR="$(cd "$1" && pwd)"
OUT="$2"
INCLUDE_QUANTS="${INCLUDE_QUANTS:-q4_k_m}"

avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
echo "打包前磁盘剩余 ${avail_kb} KB"
if (( avail_kb < 5 * 1024 * 1024 )); then
  echo "剩余不足 5GB，停止打包" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT/models" "$OUT/data/smoke/b0" "$OUT/src" "$OUT/scripts"

echo "打包 llama.cpp 源码（不含 build）"
tar -C "$ROOT/vendor" -czf "$OUT/llama.cpp-b12425.tar.gz" \
  --exclude='llama.cpp/build' \
  --exclude='__pycache__' \
  --exclude='*.o' \
  llama.cpp

shopt -s nullglob
mmproj=( "$GGUF_DIR"/mmproj-*.gguf )
if ((${#mmproj[@]} == 0)); then
  echo "找不到 mmproj：$GGUF_DIR/mmproj-*.gguf" >&2
  exit 1
fi
# 同一文件系统上用硬链接，避免把 GGUF 再复制一份。拷到板子或打 tar 时会变成独立文件。
cp -al "${mmproj[@]}" "$OUT/models/"
IFS=',' read -r -a quants <<<"$INCLUDE_QUANTS"
for quant in "${quants[@]}"; do
  quant="${quant// /}"
  matches=( "$GGUF_DIR"/*-"${quant}".gguf )
  copied=0
  for file in "${matches[@]}"; do
    base="$(basename "$file")"
    if [[ "$base" == mmproj-* ]]; then
      continue
    fi
    cp -al "$file" "$OUT/models/"
    copied=1
  done
  if ((copied == 0)); then
    echo "找不到语言 GGUF 档 $quant" >&2
    exit 1
  fi
done
if [[ -f "$GGUF_DIR/manifest.json" ]]; then
  cp -a "$GGUF_DIR/manifest.json" "$OUT/models/manifest.json"
fi
shopt -u nullglob

cp -a "$ROOT/deploy" "$OUT/deploy"
cp -a "$ROOT/contracts" "$OUT/contracts"
cp -a "$ROOT/configs" "$OUT/configs"
cp -a "$ROOT/scripts/predict_llamacpp.py" "$OUT/scripts/"
cp -a "$ROOT/data/smoke/b0/." "$OUT/data/smoke/b0/"
mkdir -p "$OUT/src"
cp -a "$ROOT/src/qwen_vlm" "$OUT/src/qwen_vlm"
find "$OUT/src" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$OUT" -name '*.pyc' -delete

cat >"$OUT/README.md" <<'EOF'
# Jetson bundle

从这里开始读：`deploy/jetson/README.md`。

语言 GGUF 的精度写在文件名里。`mmproj-*-f16.gguf` 是视觉部分，保持 F16，没有跟语言权重一起量化。
EOF

(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
echo "bundle: $OUT"
du -sh "$OUT" "$OUT/models" "$OUT/llama.cpp-b12425.tar.gz"
df -h "$ROOT" | tail -1
