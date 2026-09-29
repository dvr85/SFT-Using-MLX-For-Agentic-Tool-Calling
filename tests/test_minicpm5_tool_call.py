"""Parity with mlx-lm tests/test_tool_parsing.py: MiniCPM5 XML tool-call vectors."""

import re


def parse_minicpm5_tool_call(text: str) -> dict:
    """Minimal MiniCPM5-style `<tool_call>name{json}</tool_call>` parser for tests."""
    import json

    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not m:
        raise ValueError("no tool call found")
    payload = m.group(1).strip()
    name_m = re.match(r"([A-Za-z0-9_.\-]+)\s*(\{.*\})?$", payload, re.DOTALL)
    if not name_m:
        raise ValueError("unparseable tool call")
    name, args_raw = name_m.group(1), name_m.group(2) or "{}"
    # tolerate single quotes like upstream pythonic parser tests
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        args = json.loads(args_raw.replace("'", '"'))
    return {"name": name, "arguments": args}


class TestMinicpm5ToolParsing:
    def test_basic_multiply(self):
        out = parse_minicpm5_tool_call('<tool_call>multiply{"a": 2, "b": 3}</tool_call>')
        assert out == {"name": "multiply", "arguments": {"a": 2, "b": 3}}

    def test_single_quoted_with_comma(self):
        out = parse_minicpm5_tool_call(
            "<tool_call>search{'query': 'hello, world'}</tool_call>"
        )
        assert out["arguments"]["query"] == "hello, world"

    def test_invalid_raises(self):
        import pytest

        with pytest.raises(ValueError):
            parse_minicpm5_tool_call("just prose, no call")
