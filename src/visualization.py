import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 150


def _create_figure(figsize: tuple[int, int]) -> tuple[Figure, Axes]:
    """Create a seaborn-styled figure and axes."""
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _finalize_plot(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path | None,
    show: bool,
) -> None:
    """Apply labels and grid, then optionally save and display the plot."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    if show:
        plt.show()
    elif isinstance(ax.figure, Figure):
        plt.close(ax.figure)


def plot_loss_curves(
    train_losses: list[float],
    eval_losses: list[float],
    train_steps: list[int] | None = None,
    eval_steps: list[int] | None = None,
    save_path: Path | None = None,
    show: bool = False,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    train_color: str = "blue",
    eval_color: str = "red",
) -> Figure:
    """Plot training and evaluation loss curves. Raises ValueError if train_losses is empty."""
    if not train_losses:
        raise ValueError("train_losses cannot be empty")

    if train_steps is None:
        train_steps = list(range(0, len(train_losses) * 10, 10))
    if eval_steps is None:
        eval_steps = list(range(0, len(eval_losses) * 100, 100))

    fig, ax = _create_figure(figsize)
    ax.plot(train_steps, train_losses, label="Train Loss", linewidth=2, color=train_color)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Eval Loss", linewidth=2, color=eval_color, linestyle="--")
    ax.legend()
    _finalize_plot(ax, "Training Steps", "Loss", "Training and Evaluation Loss", save_path, show)
    return fig


def plot_learning_rate_schedule(
    learning_rates: list[float],
    steps: list[int] | None = None,
    save_path: Path | None = None,
    show: bool = False,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    color: str = "green",
) -> Figure:
    """Plot learning rate schedule over training steps. Raises ValueError if learning_rates is empty."""
    if not learning_rates:
        raise ValueError("learning_rates cannot be empty")

    if steps is None:
        steps = list(range(0, len(learning_rates) * 10, 10))

    fig, ax = _create_figure(figsize)
    ax.plot(steps, learning_rates, linewidth=2, color=color)
    _finalize_plot(ax, "Training Steps", "Learning Rate", "Learning Rate Schedule", save_path, show)
    return fig


def plot_eval_summary(
    tool_call_rate: float,
    save_dir: Path | None = None,
    show: bool = False,
) -> Path | None:
    """Bar-plot tool-call hit vs miss rate; returns png path or None."""
    if not 0.0 <= tool_call_rate <= 1.0:
        raise ValueError("tool_call_rate must be in [0, 1]")
    fig, ax = _create_figure(DEFAULT_FIGSIZE)
    ax.bar(["hit", "miss"], [tool_call_rate, 1.0 - tool_call_rate])
    path = (save_dir or Path("models/plots")) / "eval_summary.png"
    _finalize_plot(ax, "Outcome", "Rate", "Tool-call rate", path, show)
    return path


def plot_from_mlflow_run(
    run_id: str,
    tracking_uri: str | None = None,
    output_dir: Path | None = None,
    show: bool = False,
) -> dict[str, Path]:
    """Fetch train/eval loss + LR histories from an MLflow run and plot them.

    Per-step metrics exist only if they were logged to the run;
    returns {} when train_loss is absent.
    """
    from mlflow import MlflowClient

    from src.tracking import resolve_tracking_uri

    client = MlflowClient(tracking_uri=resolve_tracking_uri(tracking_uri))
    train = sorted((m.step, m.value) for m in client.get_metric_history(run_id, "train_loss"))
    evalu = sorted((m.step, m.value) for m in client.get_metric_history(run_id, "eval_loss"))
    lrs = sorted((m.step, m.value) for m in client.get_metric_history(run_id, "learning_rate"))

    output_dir = output_dir or Path("models/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    if not train:
        logger.warning(f"No train_loss metrics in run {run_id}; skipping plots")
        return saved

    loss_path = output_dir / "loss_curves.png"
    plot_loss_curves(
        train_losses=[v for _, v in train],
        eval_losses=[v for _, v in evalu],
        train_steps=[s for s, _ in train],
        eval_steps=[s for s, _ in evalu],
        save_path=loss_path,
        show=show,
    )
    saved["loss"] = loss_path

    if lrs:
        lr_path = output_dir / "lr_schedule.png"
        plot_learning_rate_schedule(
            learning_rates=[v for _, v in lrs],
            steps=[s for s, _ in lrs],
            save_path=lr_path,
            show=show,
        )
        saved["lr"] = lr_path
    return saved
