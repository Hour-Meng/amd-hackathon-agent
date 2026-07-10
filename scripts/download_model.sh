#!/usr/bin/env bash
# Download the Track 1 bundled GGUF model (Qwen2.5-1.5B-Instruct Q4_K_M).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${LOCAL_MODEL_DIR:-$ROOT/models}"
OUT_FILE="${LOCAL_GGUF_PATH:-$OUT_DIR/model.gguf}"
MODEL_URL="${LOCAL_GGUF_URL:-https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf}"

mkdir -p "$(dirname "$OUT_FILE")"
if [[ -f "$OUT_FILE" ]]; then
  echo "Model already present: $OUT_FILE"
  exit 0
fi

echo "Downloading GGUF to $OUT_FILE"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "$OUT_FILE" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT_FILE" "$MODEL_URL"
else
  echo "Need curl or wget to download the model" >&2
  exit 1
fi

echo "Done: $OUT_FILE ($(du -h "$OUT_FILE" | cut -f1))"
