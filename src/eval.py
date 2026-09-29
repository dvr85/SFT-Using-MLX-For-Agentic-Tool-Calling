"""Tool-call spot-checks for a fused model or base + adapters.

Reads mlx-lm ChatDataset rows (test.jsonl), prompts with the system + first
user turn, generates with mlx_lm.generate, and reports how often the output
contains a well-formed ToolACE bracket call like ``[get_weather(city=...)]``
— the format the model was trained on (offline-first tests cover the
parsing/rates; only live runs touch the model).
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ToolACE bracket calls: [func(...)] or [func(...), func2(...)]. Name group
# excludes brackets so prose like "[see note]" (no paren) never matches.
BRACKET_CALL_RE = re.compile(r"\[([^\[\]]+?)\(")


def _eligible(messages: list[dict[str, str]]) -> bool:
    """A row is eligible if its reference contains an assistant bracket call."""
    return any(m.get("role") == "assistant" and BRACKET_CALL_RE.search(m.get("content", "")) for m in messages)


def _prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """System turn (tool defs) + first user turn: the tool-call trigger."""
    prompt = [m for m in messages if m.get("role") == "system"][:1]
    user = next((m for m in messages if m.get("role") == "user"), None)
    if user is not None:
        prompt.append(user)
    return prompt


def _parse_tool_calls(text: str) -> list[str]:
    """Return called function names from ToolACE bracket calls in text."""
    return [name.strip() for name in BRACKET_CALL_RE.findall(text) if name.strip()]


def _log_eval_to_mlflow(summary: dict[str, Any], params: dict[str, Any]) -> None:
    """Log eval summary to MLflow; no-op outside an active run (offline/test use)."""
    import tempfile

    import mlflow

    from src.visualization import plot_eval_summary

    if mlflow.active_run() is None:
        return
    try:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        for key in ("tool_call_rate", "n_eligible", "n_tool_call", "n_total"):
            mlflow.log_metric(key, float(summary[key]))
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "failures.json"
            fp.write_text(json.dumps(summary.get("failures", [])), encoding="utf-8")
            mlflow.log_artifact(str(fp))
            plot = plot_eval_summary(float(summary["tool_call_rate"]), save_dir=Path(tmp))
            if plot is not None:
                mlflow.log_artifact(str(plot))
    except Exception as e:
        logger.warning(f"MLflow eval logging skipped: {e}")


def run_eval(
    model: str,
    data_path: str | Path,
    adapter_path: str | None = None,
    limit: int = 50,
    max_tokens: int = 128,
    experiment_name: str | None = None,
    run_name: str | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Generate tool calls for held-out rows; return rate summary."""
    from mlx_lm import generate
    from mlx_lm.utils import load

    # Local paths must exist: otherwise load() misreads them as HF repo IDs
    # and fails with a cryptic HFValidationError. HF IDs have no local prefix.
    if ("/" in model or model.startswith(".")) and not Path(model).exists():
        raise FileNotFoundError(
            f"Model path not found: {model} — run export first "
            "(bash scripts/export_model.sh <adapters>) or check the path."
        )
    if adapter_path is not None and not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}.")

    loaded: Any = load(model, adapter_path=adapter_path) if adapter_path else load(model)
    mlx_model, tokenizer = loaded[0], loaded[1]

    rows: list[dict[str, Any]] = []
    try:
        f = open(data_path, encoding="utf-8")
    except OSError as e:
        raise FileNotFoundError(f"Eval data not found: {data_path}: {e}") from e
    with f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSONL row {i}")
                continue

    n_total = len(rows)
    n_eligible = n_tool_call = 0
    failures: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        messages = row.get("messages", [])
        if not _eligible(messages):
            continue
        n_eligible += 1
        prompt = tokenizer.apply_chat_template(_prompt_messages(messages), add_generation_prompt=True)
        text = generate(mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        text = text if isinstance(text, str) else str(text)
        if _parse_tool_calls(text):
            n_tool_call += 1
        else:
            failures.append({"row": str(idx), "output": text[:500]})
    summary: dict[str, Any] = {
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_tool_call": n_tool_call,
        "tool_call_rate": (n_tool_call / n_eligible) if n_eligible else 0.0,
        "failures": failures[:10],
    }
    logger.info(f"Eval summary: {json.dumps({k: v for k, v in summary.items() if k != 'failures'})}")
    if log_mlflow:
        try:
            import mlflow

            from src.tracking import resolve_experiment, resolve_tracking_uri

            mlflow.set_tracking_uri(resolve_tracking_uri())
            mlflow.set_experiment(resolve_experiment(experiment_name))
            params = {"model": model, "adapter_path": adapter_path, "limit": limit, "max_tokens": max_tokens}
            if mlflow.active_run() is None:
                with mlflow.start_run(run_name=run_name or "eval-tool-call"):
                    _log_eval_to_mlflow(summary, params)
            else:
                _log_eval_to_mlflow(summary, params)
        except Exception as e:
            logger.warning(f"MLflow eval logging skipped: {e}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-call spot-checks via mlx_lm.generate.")
    parser.add_argument("--model", required=True, help="Fused model dir or base repo")
    parser.add_argument("--data", required=True, help="test.jsonl from export_messages_jsonl")
    parser.add_argument("--adapter-path", default=None, help="Adapter dir (omit for fused model)")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(
        json.dumps(
            run_eval(
                args.model,
                args.data,
                args.adapter_path,
                args.limit,
                args.max_tokens,
                args.experiment,
                args.run_name,
                log_mlflow=not args.no_mlflow,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
