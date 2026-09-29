#!/usr/bin/env bash
set -euo pipefail

HYDRA_ARGS=()
LORA_ARGS=()
SEEN_SEP=0
for arg in "$@"; do
  case "$arg" in
  --) SEEN_SEP=1 ;;
  *) if [[ "$SEEN_SEP" == "1" ]]; then LORA_ARGS+=("$arg"); else HYDRA_ARGS+=("$arg"); fi ;;
  esac
done

uv run python -m src.train ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}

# src.train already ran mlx_lm.lora; only re-run when extra lora flags were passed via `--`.
if [[ "${#LORA_ARGS[@]}" -gt 0 ]]; then
  OUT_DIR="$(
    uv run python - ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} <<'EOF'
import sys
from hydra import compose, initialize_config_dir
from pathlib import Path
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    print(compose(config_name="sft_mlx_config", overrides=sys.argv[1:]).training.output_dir)
EOF
  )"

  exec uv run mlx_lm.lora --config "$OUT_DIR/lora.yaml" "${LORA_ARGS[@]}"
fi
