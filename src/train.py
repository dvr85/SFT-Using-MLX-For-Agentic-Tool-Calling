"""Prepare data (pure Python) then train via the canonical mlx-lm CLI.

Per https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md:
data dir holds train.jsonl (+ optional valid.jsonl), training runs as
`mlx_lm.lora --config lora.yaml`. This module owns data prep, iters math,
and YAML emission; the actual training is a subprocess call. MLflow keeps params/artifacts
only; plots are made post-hoc with seaborn/matplotlib.
"""

import hashlib
import json
import logging
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from src.data.loader import export_messages_jsonl, load_and_preprocess_dataset
from src.tracking import resolve_experiment, resolve_tracking_uri

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds to avoid randomness during training."""
    import mlx.core as mx
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def count_jsonl(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e


def compute_iters(num_epochs: int, n_train: int, batch_size: int) -> int:
    """Convert epochs to mlx-lm's fixed step count."""
    return max(1, num_epochs * n_train // batch_size)


def get_lora_keys(config: DictConfig) -> list[str]:
    """Read lora.keys via OmegaConf container (avoids DictConfig.keys method clash)."""
    container = OmegaConf.to_container(config.lora, resolve=True)
    raw: Any = container.get("keys", []) if isinstance(container, dict) else []
    return [str(k) for k in raw] if isinstance(raw, list) else []


def build_lora_config(
    config: DictConfig, data_dir: Path, adapter_path: Path, iters: int, has_heldout: bool = False
) -> dict[str, Any]:
    """Translate Hydra config into mlx-lm YAML keys (underscores, cf. CONFIG_DEFAULTS)."""
    try:
        warmup_steps = int(config.training.warmup_ratio * iters)
        learning_rate = float(config.training.learning_rate)
        dropout = float(config.lora.dropout)
        scale = float(config.lora.scale)
        test_batches = int(getattr(config.training, "test_batches", -1))
    except (ValueError, TypeError, AttributeError) as e:
        raise RuntimeError(f"Invalid training/LoRA config: {e}") from e
    lora_keys = get_lora_keys(config)
    return {
        "model": config.model.name,
        "train": True,
        "test": has_heldout,  # needs test.jsonl; export writes it only when held-out rows exist
        "test_batches": test_batches,
        "data": str(data_dir),
        "fine_tune_type": config.lora.fine_tune_type,
        "optimizer": config.training.optimizer,
        "optimizer_config": {config.training.optimizer: {}},
        "num_layers": config.lora.num_layers,
        "batch_size": config.training.batch_size,
        "iters": iters,
        "val_batches": config.training.val_batches,
        "learning_rate": learning_rate,
        "lr_schedule": {
            "name": config.training.lr_schedule,
            "arguments": [learning_rate, iters],
            "warmup": warmup_steps,
            "warmup_init": 0.0,
        },
        "steps_per_report": config.training.steps_per_report,
        "steps_per_eval": config.training.steps_per_eval,
        "adapter_path": str(adapter_path),
        "save_every": config.training.save_every,
        "max_seq_length": config.tokenizer.max_seq_length,
        "mask_prompt": bool(config.tokenizer.mask_prompt),
        "grad_checkpoint": bool(config.training.grad_checkpoint),
        "grad_accumulation_steps": config.training.grad_accumulation_steps,
        "seed": config.training.seed,
        "lora_parameters": {
            "rank": config.lora.r,
            "dropout": dropout,
            "scale": scale,
            "keys": lora_keys,
        },
    }


def write_lora_config(lora_cfg: dict[str, Any], path: Path) -> Path:
    try:
        path.write_text(OmegaConf.to_yaml(OmegaConf.create(lora_cfg)), encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to write {path}: {e}") from e
    return path


def _num(text: str | None) -> float:
    """Parse an mlx-lm number, tolerating sentence-final periods (e.g. '1.288.')."""
    if text is None:
        raise ValueError("missing number")
    return float(text.rstrip("."))


_ITER_RE = re.compile(
    r"Iter\s+(?P<step>\d+).*?Train loss\s+(?P<train>[\d.eE+-]+)"
    r"(?:.*?Learning Rate\s+(?P<lr>[\d.eE+-]+))?",
    re.IGNORECASE,
)

# trainer.py prints validation on its own line, not with the Train loss line:
# "Iter {it}: Val loss {x:.3f}, Val took {t:.3f}s".
_VAL_RE = re.compile(
    r"Iter\s+(?P<step>\d+).*?Val loss\s+(?P<val>[\d.eE+-]+)",
    re.IGNORECASE,
)

# evaluate_model() prints e.g. "Test loss 2.345, Test ppl 10.434." once, at the end.
_TEST_RE = re.compile(
    r"Test loss\s+(?P<test>[\d.eE+-]+),\s*Test ppl\s+(?P<ppl>[\d.eE+-]+)",
    re.IGNORECASE,
)


def _log_metric(key: str, value: float, step: int | None = None) -> None:
    """Log to MLflow only inside an active run; otherwise skip (standalone/test use)."""
    if mlflow.active_run() is None:
        logger.debug(f"Skipping metric {key}={value} (no active MLflow run)")
        return
    if step is None:
        mlflow.log_metric(key, value)
    else:
        mlflow.log_metric(key, value, step=step)


def get_registry_model_name(config: DictConfig) -> str | None:
    """Registry model name lives in the mlflow group; None means skip registration."""
    name = getattr(getattr(config, "mlflow", None), "model_name", None)
    return str(name) if name else None


def resolve_model_arch(config: DictConfig) -> dict[str, Any]:
    """Load the HF arch snapshot sibling to the model config (logging/validation only).

    The training path passes just ``config.model.name`` to ``mlx_lm.lora``; arch
    fields are never training inputs. This resolves ``config.model.config_file``
    (default ``minicpm5-1b.config.json``) from ``configs/model/`` and returns the
    parsed JSON plus a ``config_sha`` digest identifying which ``main`` snapshot
    the run saw. Raises a clear error telling the user to re-download from the
    HF model card when the file is absent.
    """
    from hydra.utils import get_original_cwd

    file_name = str(getattr(getattr(config, "model", None), "config_file", "minicpm5-1b.config.json"))
    try:
        base = Path(get_original_cwd())
    except Exception:
        base = Path.cwd()
    candidates = [base / "configs" / "model" / file_name, Path.cwd() / "configs" / "model" / file_name]
    path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.exists():
        hf_repo = getattr(getattr(config, "model", None), "hf_repo", "unknown")
        raise RuntimeError(
            f"Model arch snapshot not found at {path}; re-download config.json from "
            f"the HF model card ({hf_repo}) to configs/model/{file_name} or run convert first."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read model arch snapshot at {path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"Model arch snapshot at {path} must be a JSON object.")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return {**data, "config_sha": sha, "config_source": str(path)}


def run_lora(yaml_path: Path) -> None:
    """Run mlx-lm training, streaming Iter lines to stdout + MLflow metrics."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm.lora", "--config", str(yaml_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("Failed to capture mlx_lm.lora output")
    for line in proc.stdout:
        print(line, end="")
        m = _ITER_RE.search(line)
        if m:
            try:
                step = int(m.group("step"))
                _log_metric("train_loss", _num(m.group("train")), step=step)
                if m.group("lr"):
                    _log_metric("learning_rate", _num(m.group("lr")), step=step)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping unparsable Iter line: {line.strip()} ({e})")
            continue
        v = _VAL_RE.search(line)
        if v:
            try:
                _log_metric("eval_loss", _num(v.group("val")), step=int(v.group("step")))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping unparsable Val line: {line.strip()} ({e})")
            continue
        t = _TEST_RE.search(line)
        if t:
            try:
                _log_metric("test_loss", _num(t.group("test")))
                _log_metric("test_ppl", _num(t.group("ppl")))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping unparsable Test line: {line.strip()} ({e})")
    if proc.wait() != 0:
        raise RuntimeError(f"mlx_lm.lora failed (exit {proc.returncode}); see output above")


@hydra.main(config_path="../configs", config_name="sft_mlx_config", version_base=None)
def main(config: DictConfig) -> None:
    set_seed(config.training.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger.info(f"SFT config:\n{OmegaConf.to_yaml(config)}")

    try:
        arch = resolve_model_arch(config)
        logger.info(
            "Model arch snapshot: %s layers=%s hidden=%s vocab=%s sha=%s (revision=%s)",
            arch.get("config_source"),
            arch.get("num_hidden_layers"),
            arch.get("hidden_size"),
            arch.get("vocab_size"),
            arch.get("config_sha"),
            getattr(getattr(config, "model", None), "revision", "main"),
        )
    except RuntimeError as e:
        logger.warning(f"Model arch snapshot unresolved (drift check skipped): {e}")
        arch = {}

    dataset = load_and_preprocess_dataset(config)
    data_dir = Path(config.training.output_dir) / "data"
    paths = export_messages_jsonl(dataset, config, data_dir)

    n_train = count_jsonl(paths["train"])
    if n_train == 0:
        raise RuntimeError(f"No valid training rows in {paths['train']}; check dataset filters")
    iters = compute_iters(config.training.num_train_epochs, n_train, config.training.batch_size)
    logger.info(f"Training for {iters} iters ({config.training.num_train_epochs} epochs x {n_train} examples)")

    heldout_path = paths.get("heldout")
    n_heldout = count_jsonl(heldout_path) if heldout_path is not None else 0
    if n_heldout:
        logger.info(f"Held-out test set: {n_heldout} rows in {heldout_path}")

    adapter_path = Path(config.training.output_dir) / "adapters"
    yaml_path = Path(config.training.output_dir) / "lora.yaml"
    write_lora_config(build_lora_config(config, data_dir, adapter_path, iters, has_heldout=n_heldout > 0), yaml_path)
    logger.info(f"Wrote mlx-lm config to {yaml_path}")

    mlflow.set_tracking_uri(resolve_tracking_uri(getattr(config.mlflow, "tracking_uri", None)))
    mlflow.set_experiment(resolve_experiment(getattr(config.mlflow, "experiment_name", None)))
    run_prefix = getattr(config.mlflow, "run_name_prefix", "sft-mlx-lora")
    with mlflow.start_run(run_name=f"{run_prefix}_r{config.lora.r}"):
        params: dict[str, Any] = {
            "model": config.model.name,
            "dataset": config.dataset.name,
            "fine_tune_type": config.lora.fine_tune_type,
                "iters": iters,
                "n_train_messages": n_train,
                "n_test_heldout": n_heldout,
                "lora_rank": config.lora.r,
                "lora_scale": config.lora.scale,
                "lora_dropout": config.lora.dropout,
                "lora_num_layers": config.lora.num_layers,
                "lora_keys": ",".join(get_lora_keys(config)),
                "learning_rate": config.training.learning_rate,
                "batch_size": config.training.batch_size,
                "max_seq_length": config.tokenizer.max_seq_length,
                "seed": config.training.seed,
            }
        if arch:
            params["hf_repo"] = str(getattr(getattr(config, "model", None), "hf_repo", ""))
            params["model_revision"] = str(getattr(getattr(config, "model", None), "revision", "main"))
            params["config_sha"] = str(arch.get("config_sha", ""))
            for key in ("num_hidden_layers", "hidden_size", "vocab_size"):
                if arch.get(key) is not None:
                    params[f"arch_{key}"] = arch[key]
        mlflow.log_params(params)
        mlflow.log_artifact(str(yaml_path))
        run_lora(yaml_path)
        # mlx-lm outputs: adapters.safetensors + {step}_adapters.safetensors
        # checkpoints + adapter_config.json land in adapter_path.
        if not adapter_path.exists():
            raise RuntimeError(f"No adapters at {adapter_path}; training produced nothing")
        mlflow.log_artifacts(str(adapter_path), artifact_path="adapters")
        try:
            from src.visualization import plot_from_mlflow_run

            active = mlflow.active_run()
            run_id = active.info.run_id if active is not None else None
            if run_id:
                plots = plot_from_mlflow_run(run_id)
                for p in plots.values():
                    mlflow.log_artifact(str(p), artifact_path="plots")
        except Exception as e:
            logger.warning(f"Plot logging skipped: {e}")
        registry_name = get_registry_model_name(config)
        if registry_name:
            from src.tracking import register_model

            register_model(registry_name)
    logger.info("MLX SFT training complete")


if __name__ == "__main__":
    main()
