#!/usr/bin/env bash
# Convert openbmb/MiniCPM5-1B (BF16) to MLX format for DoRA SFT.
# Conversion is lightweight (~2GB weights); OOM risk is in training, not here.
set -euo pipefail

HF_PATH="${1:-openbmb/MiniCPM5-1B}"
MLX_PATH="${2:-models/minicpm5-1b-bf16}"

uv run mlx_lm.convert --hf-path "$HF_PATH" --mlx-path "$MLX_PATH" --dtype bfloat16
echo "Converted $HF_PATH -> $MLX_PATH (chat template from $HF_PATH included)"
