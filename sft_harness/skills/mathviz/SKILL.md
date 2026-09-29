---
name: mathviz
description: Mathematical visualizing via matplotlib — function plots, bar/scatter charts for tutor explanations. Use when the user asks for a graph, plot, chart, or visual of math. Emits a PNG path, never ASCII art.
license: BSD-3-Clause (matplotlib)
compatibility: Requires Python 3.9+ and matplotlib (uv managed).
---

# Mathviz skill

## When to use
- User asks to graph/plot/chart/visualize a function or dataset.
- Never for symbolic solving (use sympy skill) or fresh web facts.

## Workflow
1. Build JSON args: `{"kind": "function|bar|scatter", "title": "...", ...}` per `references/chart-chooser.md`.
2. Run `uv run python skills/mathviz/scripts/plot.py '<json>'` with cwd at `sft_harness/`.
3. On `{ok:false}`: apply `retry_hint` once, then answer in text.
4. Embed the returned PNG path verbatim; keep one chart per turn.

## Guardrails
- Kinds limited to function/bar/scatter (geometry deferred).
- PNG output goes under `traces/` (gitignored); script prints JSON only.
