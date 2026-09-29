# AGENTS.md

## Commands

- **Install dependencies**: `uv sync --extra dev`
- **Lint**: `uv run ruff check .`
- **Lint fix**: `uv run ruff check . --fix`
- **Type check**: `uv run mypy src/ tests/`
- **Run tests**: `uv run pytest tests/ -v`
- **Start MLFlow**: `bash scripts/setup_mlflow.sh`
- **Convert model**: `bash scripts/convert_minicpm5_1b.sh [hf-path] [mlx-path]`
- **Train**: `uv run python -m src.train`
- **Train (bare mlx-lm CLI)**: `bash scripts/train_lora.sh [hydra overrides...] [-- extra mlx_lm.lora flags...]`
- **Export model**: `bash scripts/export_model.sh <adapters_path> [output_dir] [-- extra fuse flags...]`
- **Eval tool-calls**: `uv run python -m src.eval --model <fused-or-base> --data <test.jsonl> [--adapter-path <dir>]`
- **Serve fused model (harness)**: `bash scripts/serve_fused.sh <fused-dir> [--port 8080]`
- **Harness unit**: `cargo test -p sft-harness`
- **Harness lint/format**: `cargo clippy --all-targets -- -D warnings` / `cargo fmt --check`

## Tech Stack

- Python 3.12, uv, Apple MLX + mlx-lm (LoRA/DORA/full fine-tuning) on Apple Silicon unified memory (single CPU/GPU pool; training sized for it — attention-only DoRA, grad_checkpoint, batch 2/seq 2048)
- Dataset: Team-ACE/ToolACE exported to mlx-lm `messages` JSONL (`system` holds `<tools>` defs, optional `tools` array)
- Models: `openbmb/MiniCPM5-1B` (post-RL/OPD final, BF16; Llama 24L, 1536/4608, GQA 16/2/D128, vocab 130560, ctx 131k) converted via `mlx_lm.convert` to `models/minicpm5-1b-bf16`
- Pure-Python preprocessing in `src/data/normalize.py` (no Rust extension)
- MLflow for run params/artifacts + model registry; plots via seaborn/matplotlib from MLflow histories
- Hydra + OmegaConf for data-prep config; training itself uses generated mlx-lm `lora.yaml`
- Matplotlib, Seaborn for visualization (plotted from MLflow metric histories)
- Attention runs on MLX native `fast.scaled_dot_product_attention` (tiled/IO-aware); no custom training-time Metal kernels

## Conventions

- Use `src/` for all source modules
- Config-driven: all hyperparams in `configs/` via Hydra; `src/train.py` emits mlx-lm `lora.yaml`
- DoRA (rank 8, scale 16, 8 layers, attention-only keys q/k/v/o) is the primary method
- `num_train_epochs` is converted to mlx-lm `iters`; checkpoints/`adapter_config.json` logged as MLflow artifacts
- All Python files use type hints
- Follow ruff linting rules (E, F, I, N, W, UP)
- Data pipeline modules in `src/data/` (loader + normalize); `export_messages_jsonl` writes mlx-lm ChatDataset files
- MLflow Model Registry for model versioning (`adapters` artifact path)
- `configs/` — Hydra config tree (`sft_mlx_config.yaml`): `model/`, `training/`, `lora/`, `tokenizer/`, `dataset/`, `mlflow/` (emits mlx-lm `lora.yaml`; explicit arg > `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT` env > config defaults)
- OOM ladder (activations first, then data, then capacity): keys (attention-only already set) -> batch 2->1, seq 2048->1024, samples 2000->500, lora num_layers 8->4
- Attention work stays in `lora.keys`/`num_layers`; never add training-time Metal kernels (forward-only, no gradients)
- `lora.keys` must be read via `get_lora_keys()` (OmegaConf container) — never `config.lora.keys` attribute (clashes with `DictConfig.keys` method)
- Tests mirror upstream `mlx-lm/tests` by behavior (pytest style kept, offline-first, skip-if-no-model); any `configs/lora|training|dataset|model|mlflow` change must update README tables/ladder too
- Harness boundary: Rust `sft_harness/` talks to the model via `mlx_lm.server` HTTP only (`configs/sft_harness/default.yaml:model_url`); never reimplement tokenizer/model/sampling in Rust; any `configs/sft_harness` change must update README tables too
- Git: this is its own repo (the old Axolotl parent remote no longer applies); `models/`, `mlflow.db`, `mlartifacts/`, `outputs/`, `traces/` are git-ignored by the project `.gitignore`
