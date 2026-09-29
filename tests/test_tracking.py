# Tests for MLflow registry wrappers in src/tracking.py
from unittest.mock import MagicMock, patch

from src.tracking import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    register_model,
    resolve_experiment,
    resolve_tracking_uri,
)


class TestResolve:
    def test_explicit_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
        monkeypatch.setenv("MLFLOW_EXPERIMENT", "env-exp")
        assert resolve_tracking_uri("http://arg:5000") == "http://arg:5000"
        assert resolve_experiment("arg-exp") == "arg-exp"

    def test_env_over_default(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
        monkeypatch.setenv("MLFLOW_EXPERIMENT", "env-exp")
        assert resolve_tracking_uri() == "http://env:5000"
        assert resolve_experiment() == "env-exp"

    def test_default_without_env(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT", raising=False)
        assert resolve_tracking_uri() == DEFAULT_TRACKING_URI
        assert resolve_experiment() == DEFAULT_EXPERIMENT


class TestRegisterModel:
    def _run(self, artifacts):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        mock_client.search_runs.return_value = [MagicMock(info=MagicMock(run_id="abc"))]
        mock_client.list_artifacts.return_value = [MagicMock(path=p) for p in artifacts]
        mock_client.create_model_version.return_value = MagicMock(version="3")
        with patch("src.tracking.MlflowClient", return_value=mock_client):
            assert register_model("m") == "3"
        return mock_client

    def test_prefers_adapters_artifact(self):
        client = self._run(["adapters", "mlruns.db"])
        source = client.create_model_version.call_args[1]["source"]
        assert source == "runs:/abc/adapters"

    def test_falls_back_to_run_root(self):
        client = self._run(["other"])
        source = client.create_model_version.call_args[1]["source"]
        assert source == "runs:/abc"
