"""Parity: harness parser semantics vs src/eval.py BRACKET_CALL_RE (offline, no model)."""

import subprocess
from pathlib import Path


def _eval_parse(text: str) -> list[str]:
    import sys

    sys.path.insert(0, ".")
    from src.eval import BRACKET_CALL_RE

    return [n.strip() for n in BRACKET_CALL_RE.findall(text) if n.strip()]


def test_bracket_parity() -> None:
    cases = ['[get_weather(city="Paris")]', "[multiply(a=2, b=3)]", "[see note]", "prose"]
    for c in cases:
        names = _eval_parse(c)
        if "see note" in c or c == "prose":
            assert names == []
        else:
            assert len(names) == 1


def test_cargo_parser_green() -> None:
    r = subprocess.run(
        ["cargo", "test", "-p", "sft-harness", "parser"],
        cwd=Path(__file__).resolve().parents[1] / "sft_harness",
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
