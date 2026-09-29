#!/usr/bin/env bash
# Start the MLflow tracking server from the Hydra mlflow group.
# MLFLOW_TRACKING_URI env overrides the configured tracking_uri.
set -euo pipefail

read -r TRACKING_URI ARTIFACT_ROOT <<<"$(
  uv run python - <<'EOF'
from hydra import compose, initialize_config_dir
from pathlib import Path
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    cfg = compose(config_name="sft_mlx_config")
print(f"{cfg.mlflow.tracking_uri} {cfg.mlflow.artifact_location}")
EOF
)"
TRACKING_URI="${MLFLOW_TRACKING_URI:-$TRACKING_URI}"
read -r HOST PORT <<<"$(TRACKING_URI="$TRACKING_URI" uv run python - <<'EOF'
import os
from urllib.parse import urlparse
u = urlparse(os.environ["TRACKING_URI"])
print(f"{u.hostname or '127.0.0.1'} {u.port or 5000}")
EOF
)"

echo "Starting MLFlow tracking server on $HOST:$PORT (artifacts: $ARTIFACT_ROOT)"
uv run mlflow server \
    --host "$HOST" \
    --port "$PORT" \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root "$ARTIFACT_ROOT"
