---
name: sympy
description: Exact symbolic math via SymPy — equations, calculus, matrices, number theory, units, arithmetic eval. Use when the user asks to solve, differentiate, integrate, factor, convert units, or needs an exact result (sqrt(2), not 1.414). Prefer over mental arithmetic.
license: BSD-3-Clause (SymPy)
compatibility: Requires Python 3.9+ and SymPy 1.14+ (uv managed).
---

# SymPy skill

## When to use
- Solving equations (algebraic, systems, ODEs), calculus (diff/integral/limit), matrices, number theory (prime/factor), unit conversion, exact arithmetic eval.
- Never for fresh web facts or diagrams (use plot_chart).

## Workflow
1. Build JSON args: `{"op": "solve|eval|diff|integrate|ntheory|units", ...}` per `references/solvers.md`.
2. Run `uv run python skills/sympy/scripts/solve.py '<json>'`.
3. On `{ok:false}`: apply `retry_hint` and call once more, then answer.
4. Verify: solutions re-substituted; show LaTeX + one-line check.

## Guardrails
- Exact `Rational` over floats; `sqrt(2)` stays symbolic until the final numeric line.
- Script stdout JSON only enters context — script code never does.
- Never invent solutions; if `ok:false`, say so plainly.
