#!/usr/bin/env bash
# Serve a fused model via mlx_lm.server (OpenAI-compatible HTTP for sft_harness/).
# Usage: serve_fused.sh <fused-dir> [--port 8080]
set -euo pipefail

MODEL="${1:?Usage: serve_fused.sh <fused-dir> [--port 8080]}"
shift

PORT=8080
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:?--port requires a value}"
      shift 2
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

exec uv run mlx_lm.server --model "$MODEL" --port "$PORT" --chat-template-args '{"enable_thinking": true}' --temp 0.9 --top-p 0.95 --max-tokens 1024
