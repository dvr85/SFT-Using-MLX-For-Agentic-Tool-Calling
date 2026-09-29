"""Parity with mlx-lm tests/test_datsets.py::test_chat (offline, local tokenizer)."""

import json
from pathlib import Path

import pytest


def _write_jsonl(dirpath: Path, rows: list[dict], with_test: bool = False) -> None:
    names = ("train.jsonl", "valid.jsonl") + (("test.jsonl",) if with_test else ())
    for name in names:
        with open(dirpath / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class TestChatDatasetParity:
    def test_chat_dataset_loads_with_tools_system(self, tmp_path):
        try:
            from mlx_lm.tuner import datasets
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("mlx-lm not installed")
        model_path = Path("models/minicpm5-1b-bf16")
        if not model_path.exists():
            pytest.skip("converted model missing; run convert script first")
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "<tools>get_weather(city)</tools>"},
                    {"role": "user", "content": "Weather in Paris?"},
                    {"role": "assistant", "content": "Sunny."},
                ]
            }
        ]
        _write_jsonl(tmp_path, rows * 2)

        import types

        args = types.SimpleNamespace(train=True, test=False, data=str(tmp_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        assert isinstance(train, datasets.ChatDataset)
        assert len(train) == 2 and len(valid) == 2 and len(test) == 0
        assert len(train[0]) > 0

    def test_chat_dataset_loads_test_split(self, tmp_path):
        try:
            from mlx_lm.tuner import datasets
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("mlx-lm not installed")
        model_path = Path("models/minicpm5-1b-bf16")
        if not model_path.exists():
            pytest.skip("converted model missing; run convert script first")
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "<tools>get_weather(city)</tools>"},
                    {"role": "user", "content": "Weather in Paris?"},
                    {"role": "assistant", "content": "Sunny."},
                ]
            }
        ]
        _write_jsonl(tmp_path, rows * 2, with_test=True)

        import types

        args = types.SimpleNamespace(train=True, test=True, data=str(tmp_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        assert len(train) == 2 and len(valid) == 2 and len(test) == 2
