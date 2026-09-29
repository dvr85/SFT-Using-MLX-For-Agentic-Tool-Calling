import json
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from src.data.loader import (
    export_messages_jsonl,
    load_and_preprocess_dataset,
    validate_conversation_format,
)


@pytest.fixture
def data_config():
    return OmegaConf.create(
        {
            "dataset": {
                "name": "dummy-dataset",
                "train_split": "train",
                "text_column": "conversations",
                "system_column": "system",
                "max_samples": None,
                "test_split_ratio": 0.1,
                "format": "messages",
            },
        }
    )


@pytest.fixture
def sample_dataset():
    train = Dataset.from_dict(
        {
            "conversations": [
                [
                    {"from": "human", "value": "What is the weather in Paris, and should I carry an umbrella?"},
                    {"from": "gpt", "value": '{"tool": "get_weather", "args": {"city": "Paris"}}'},
                    {"from": "tool", "value": "Sunny skies and a mild twenty one degrees Celsius all afternoon long."},
                    {"from": "gpt", "value": "It is sunny and twenty one degrees in Paris, so no umbrella is needed."},
                ],
                [
                    {"from": "human", "value": "Tell me a programming joke that always makes developers laugh."},
                    {"from": "gpt", "value": "Why do programmers love dark mode? Because light attracts bugs."},
                ],
            ],
            "system": ["You are a helpful assistant with access to external tools like get_weather for forecasts.", ""],
        }
    )
    test = Dataset.from_dict(
        {
            "conversations": [[{"from": "human", "value": "Test"}]],
            "system": [""],
        }
    )
    return DatasetDict({"train": train, "test": test})


class TestLoadAndPreprocessDataset:
    @patch("src.data.loader.load_dataset")
    def test_loads_dataset_with_both_splits(self, mock_load_dataset, data_config, sample_dataset):
        mock_load_dataset.return_value = sample_dataset["train"]

        result = load_and_preprocess_dataset(data_config)

        mock_load_dataset.assert_called_once_with("dummy-dataset", split="train")
        assert "train" in result
        assert "test" in result


class TestExportMessagesJsonl:
    def test_writes_valid_messages(self, sample_dataset, data_config, tmp_path):
        paths = export_messages_jsonl(sample_dataset, data_config, tmp_path)

        assert paths["train"].exists()
        rows = [json.loads(line) for line in paths["train"].read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        first = rows[0]["messages"]
        assert first[0]["role"] == "system"
        assert "get_weather" in first[0]["content"]
        assert first[1] == {
            "role": "user",
            "content": "What is the weather in Paris, and should I carry an umbrella?",
        }
        assert first[2]["role"] == "assistant"
        assert first[3] == {"role": "tool", "content": first[3]["content"]}
        assert validate_conversation_format(first) is True

    def test_skips_invalid_examples(self, data_config, tmp_path):
        from src.data.loader import _example_to_messages

        assert _example_to_messages({"conversations": "not-a-list"}, data_config) is None
        ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "conversations": [[{"from": "human", "value": ""}]],
                        "system": [""],
                    }
                ),
            }
        )
        paths = export_messages_jsonl(ds, data_config, tmp_path)
        assert paths["train"].read_text(encoding="utf-8") == ""

    def test_skips_overlong_examples(self, data_config, tmp_path):
        long_text = " ".join(["word"] * 9000)
        ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "conversations": [[{"from": "human", "value": long_text}]],
                        "system": [""],
                    }
                ),
            }
        )
        paths = export_messages_jsonl(ds, data_config, tmp_path)
        assert paths["train"].read_text(encoding="utf-8") == ""


class TestHeldoutSplit:
    def _turn(self, i: int) -> list[dict]:
        text = f"example number {i} with more than ten words to pass the length filter easily here"
        return [{"from": "human", "value": text}]

    def test_carves_test_jsonl_tail(self, data_config, tmp_path):
        from unittest.mock import patch

        ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"conversations": [self._turn(0)], "system": [""]},
                ),
                "test": Dataset.from_dict(
                    {"conversations": [self._turn(i) for i in range(1, 4)], "system": ["", "", ""]},
                ),
            }
        )
        with patch("src.data.loader.TEST_ROWS", 2):
            paths = export_messages_jsonl(ds, data_config, tmp_path)
        valid_rows = paths["test"].read_text(encoding="utf-8").splitlines()
        held_rows = paths["heldout"].read_text(encoding="utf-8").splitlines()
        assert len(valid_rows) == 1 and len(held_rows) == 2

    def test_no_heldout_key_when_too_few_rows(self, sample_dataset, data_config, tmp_path):
        paths = export_messages_jsonl(sample_dataset, data_config, tmp_path)
        assert "heldout" not in paths
