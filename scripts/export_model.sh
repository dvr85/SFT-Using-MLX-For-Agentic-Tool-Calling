#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${1:?Usage: export_model.sh <adapter_path> [output_dir] [-- extra fuse flags...]}"
shift
OUTPUT_ARG=""
FUSE_ARGS=()
SEEN_SEP=0
for arg in "$@"; do
  case "$arg" in
  --) SEEN_SEP=1 ;;
  *) if [[ "$SEEN_SEP" == "1" ]]; then FUSE_ARGS+=("$arg"); elif [[ -z "$OUTPUT_ARG" ]]; then OUTPUT_ARG="$arg"; else
    echo "error: unexpected argument: $arg" >&2
    exit 1
  fi ;;
  esac
done

read -r MODEL DEFAULT_OUT <<<"$(
  uv run python - <<'EOF'
from hydra import compose, initialize_config_dir
from pathlib import Path
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    cfg = compose(config_name="sft_mlx_config")
print(f"{cfg.model.name} {cfg.training.output_dir}/export/fused")
EOF
)"
OUT_DIR="${OUTPUT_ARG:-$DEFAULT_OUT}"

exec uv run mlx_lm.fuse --model "$MODEL" --adapter-path "$ADAPTER_PATH" --save-path "$OUT_DIR" ${FUSE_ARGS[@]+"${FUSE_ARGS[@]}"}
