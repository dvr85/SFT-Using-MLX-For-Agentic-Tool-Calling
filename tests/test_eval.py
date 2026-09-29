"""Offline tests for src/eval.py parsing + rate math (model calls faked)."""

import json
import sys
import types

import src.eval as eval_mod
from src.eval import _eligible, _parse_tool_calls, _prompt_messages, run_eval


def _rows():
    return [
        {
            "messages": [
                {"role": "system", "content": "<tools>get_weather(city)</tools>"},
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": '[get_weather(city="Paris")] Checking now.'},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross?"},
            ]
        },
    ]


class TestParsing:
    def test_eligible_detects_bracket_call(self):
        rows = _rows()
        assert _eligible(rows[0]["messages"]) is True
        assert _eligible(rows[1]["messages"]) is False

    def test_bracket_note_without_paren_not_eligible(self):
        messages = [{"role": "assistant", "content": "See [note] for details."}]
        assert _eligible(messages) is False

    def test_prompt_is_system_plus_first_user(self):
        prompt = _prompt_messages(_rows()[0]["messages"])
        assert [m["role"] for m in prompt] == ["system", "user"]

    def test_parse_bracket_calls(self):
        assert _parse_tool_calls('[get_weather(city="Paris")] done') == ["get_weather"]
        # multi-call list: first name captured, row still scores
        assert _parse_tool_calls("[countries/list(), Soil Data(lat=1)]")[0] == "countries/list"
        assert _parse_tool_calls("no call here") == []
        assert _parse_tool_calls("See [note] for details.") == []


def _install_fake_mlx(monkeypatch, outputs):
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.generate = lambda *a, **k: outputs.pop(0)  # noqa: E731
    utils = types.ModuleType("mlx_lm.utils")

    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            return "PROMPT"

    utils.load = lambda *a, **k: ("model", FakeTokenizer())
    mlx_lm.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)
    assert eval_mod  # keep import referenced


class TestRunEval:
    def test_rates_and_failures(self, monkeypatch, tmp_path):
        data = tmp_path / "test.jsonl"
        with open(data, "w", encoding="utf-8") as f:
            for r in _rows():
                f.write(json.dumps(r) + "\n")
        _install_fake_mlx(monkeypatch, ['[get_weather(city="London")]'])
        summary = run_eval("dummy-model", data, limit=10)
        assert summary["n_total"] == 2
        assert summary["n_eligible"] == 1
        assert summary["n_tool_call"] == 1
        assert summary["tool_call_rate"] == 1.0
        assert summary["failures"] == []

    def test_miss_recorded(self, monkeypatch, tmp_path):
        data = tmp_path / "test.jsonl"
        with open(data, "w", encoding="utf-8") as f:
            f.write(json.dumps(_rows()[0]) + "\n")
        _install_fake_mlx(monkeypatch, ["just prose, no call"])
        summary = run_eval("dummy-model", data, limit=10)
        assert summary["tool_call_rate"] == 0.0
        assert len(summary["failures"]) == 1


class TestPathPrecheck:
    def test_missing_local_model_path_raises(self, tmp_path):
        import pytest

        data = tmp_path / "test.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="run export first"):
            run_eval("models/missing/export/fused", data)

    def test_missing_adapter_path_raises(self, tmp_path):
        import pytest

        data = tmp_path / "test.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="Adapter path not found"):
            run_eval("dummy-model", data, adapter_path=str(tmp_path / "nope"))
