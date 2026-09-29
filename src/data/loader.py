# Import the libraries
import json
import logging
from pathlib import Path
from typing import Any, cast

# Datasets, Hydra for configuration
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig

# Pure-Python preprocessing (mlx-lm only, no Rust extension)
from src.data.normalize import (
    normalize_conversations as _normalize_conversations,
)
from src.data.normalize import (
    validate_conversation_format as _validate_conversation_format,
)

logger = logging.getLogger(__name__)

# Conversation length guard, in words.
MIN_WORDS = 10
MAX_WORDS = 2048

# Held-out rows carved off the HF test split for mlx-lm `--test` perplexity.
# The split is pre-shuffled (seed 42), so the tail is a random subset.
TEST_ROWS = 200


def load_and_preprocess_dataset(config: DictConfig) -> DatasetDict:
    logger.info(f"Loading dataset from the configuration: {config.dataset.name}")

    # Dataset Parameters from /Configs
    dataset_params: dict[str, Any] = {}
    if hasattr(config.dataset, "config") and config.dataset.config is not None:
        dataset_params["name"] = config.dataset.config
        logger.info(f"Using dataset config: {config.dataset.config}")

    # Load the HF Dataset
    logger.info(f"Fetching the {config.dataset.name} from HF.")
    hf_dataset = load_dataset(config.dataset.name, split=config.dataset.train_split, **dataset_params)

    # Test Split
    logger.info(f"Splitting train into train/test with test_size={config.dataset.test_split_ratio}")
    dataset = hf_dataset.train_test_split(test_size=config.dataset.test_split_ratio, seed=42)

    # Max samples for training and testing
    if config.dataset.max_samples is not None:
        n_train = min(config.dataset.max_samples, len(dataset["train"]))
        n_test = min(config.dataset.max_samples, len(dataset["test"]))
        dataset["train"] = dataset["train"].select(range(n_train))
        dataset["test"] = dataset["test"].select(range(n_test))
        logger.info(f"Subsampled to {n_train} train / {n_test} test examples")

    logger.info(f"Train size: {len(dataset['train']):,}")
    logger.info(f"Test size: {len(dataset['test']):,}")

    return dataset


def validate_conversation_format(conversations: list[dict[str, str]]) -> bool:
    return _validate_conversation_format(conversations)


def _example_to_messages(example: dict[str, Any], config: DictConfig) -> tuple[list[dict[str, str]], Any | None] | None:
    """Normalize one ToolACE example to messages (+ optional tools); None if invalid."""
    conversations = example.get(config.dataset.text_column, [])
    if not isinstance(conversations, list):
        return None
    normalized = cast(list[dict[str, str]], _normalize_conversations(conversations))
    system_column = getattr(config.dataset, "system_column", None)
    if system_column and example.get(system_column):
        tool_defs = str(example[system_column])
        normalized = [{"role": "system", "content": f"<tools>{tool_defs}</tools>"}, *normalized]
    messages = normalized
    if not messages or not validate_conversation_format(messages):
        return None
    tools = example.get("tools")
    return messages, tools


def _word_count_ok(messages: list[dict[str, str]]) -> bool:
    word_count = sum(len(turn["content"].split()) for turn in messages)
    return MIN_WORDS <= word_count <= MAX_WORDS


def _collect_rows(dataset_split: Any, config: DictConfig) -> list[str]:
    """Normalize + filter one HF split into mlx-lm ChatDataset JSON lines."""
    rows: list[str] = []
    for example in dataset_split:
        result = _example_to_messages(cast(dict[str, Any], example), config)
        if result is None:
            continue
        messages, tools = result
        if not _word_count_ok(messages):
            continue
        row: dict[str, Any] = {"messages": messages}
        if tools:
            row["tools"] = tools
        rows.append(json.dumps(row, ensure_ascii=False))
    return rows


def _write_lines(path: Path, rows: list[str]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            for line in rows:
                f.write(line + "\n")
    except OSError as e:
        raise RuntimeError(f"Failed to write {path}: {e}") from e


def export_messages_jsonl(dataset: DatasetDict, config: DictConfig, output_dir: str | Path) -> dict[str, Path]:
    """Write splits as mlx-lm ChatDataset JSONL: {"messages": [...], "tools": [...]?}.

    HF `train` -> `train.jsonl`; HF `test` -> `valid.jsonl` (mlx-lm validation)
    plus a `test.jsonl` tail (mlx-lm `--test` perplexity) when the kept rows
    exceed TEST_ROWS. Returns split name -> file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    if "train" in dataset:
        rows = _collect_rows(dataset["train"], config)
        path = out / "train.jsonl"
        _write_lines(path, rows)
        logger.info(f"Wrote {len(rows)}/{len(dataset['train'])} train examples to {path}")
        paths["train"] = path
    if "test" in dataset:
        kept = _collect_rows(dataset["test"], config)
        if len(kept) > TEST_ROWS:
            valid_rows, test_rows = kept[:-TEST_ROWS], kept[-TEST_ROWS:]
        else:
            valid_rows, test_rows = kept, []
        valid_path = out / "valid.jsonl"
        _write_lines(valid_path, valid_rows)
        logger.info(f"Wrote {len(valid_rows)}/{len(dataset['test'])} test examples to {valid_path}")
        paths["test"] = valid_path
        if test_rows:
            test_path = out / "test.jsonl"
            _write_lines(test_path, test_rows)
            logger.info(f"Wrote {len(test_rows)} held-out examples to {test_path}")
            paths["heldout"] = test_path
    return paths
