"""
Unit tests for src/visualization.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

from src.visualization import (
    plot_from_mlflow_run,
    plot_learning_rate_schedule,
    plot_loss_curves,
)


class TestPlotLossCurves:
    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.Path.mkdir")
    def test_returns_figure(self, mock_mkdir, mock_savefig):
        fig = plot_loss_curves([1.0, 0.8, 0.6], [0.9, 0.7], save_path=None, show=False)
        assert fig is not None

    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.Path.mkdir")
    def test_saves_when_path_provided(self, mock_mkdir, mock_savefig):
        save_path = Path("output/loss.png")
        plot_loss_curves([1.0, 0.8], [0.9], save_path=save_path, show=False)
        mock_savefig.assert_called_once()
        mock_mkdir.assert_called_once()
        assert save_path in mock_savefig.call_args[0]

    def test_rejects_empty_train_losses(self):
        try:
            plot_loss_curves([], [0.9])
        except ValueError as e:
            assert "train_losses" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestPlotLearningRateSchedule:
    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.Path.mkdir")
    def test_returns_figure(self, mock_mkdir, mock_savefig):
        fig = plot_learning_rate_schedule([0.001, 0.0008], save_path=None, show=False)
        assert fig is not None

    def test_rejects_empty(self):
        try:
            plot_learning_rate_schedule([])
        except ValueError as e:
            assert "learning_rates" in str(e)
        else:
            raise AssertionError("expected ValueError")


def _metric(step, value):
    m = MagicMock()
    m.step = step
    m.value = value
    return m


class TestPlotFromMlflowRun:
    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.Path.mkdir")
    @patch("mlflow.MlflowClient")
    def test_plots_loss_and_lr(self, mock_client_cls, mock_mkdir, mock_savefig):
        mock_client = mock_client_cls.return_value
        mock_client.get_metric_history.side_effect = lambda run, key: {
            "train_loss": [_metric(10, 2.5), _metric(20, 2.0)],
            "eval_loss": [_metric(20, 2.1)],
            "learning_rate": [_metric(10, 0.001), _metric(20, 0.0008)],
        }[key]

        result = plot_from_mlflow_run("abc123")
        assert "loss" in result
        assert "lr" in result
        assert mock_savefig.call_count == 2

    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.Path.mkdir")
    @patch("mlflow.MlflowClient")
    def test_returns_empty_without_train_metrics(self, mock_client_cls, mock_mkdir, mock_savefig):
        mock_client_cls.return_value.get_metric_history.return_value = []
        assert plot_from_mlflow_run("abc123") == {}
        mock_savefig.assert_not_called()
