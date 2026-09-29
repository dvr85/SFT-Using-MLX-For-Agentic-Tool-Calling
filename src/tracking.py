"""MLflow model registry for mlx-lm SFT adapters."""

import logging
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "sft-mlx-agentic-tool-calling"


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Precedence: explicit arg > MLFLOW_TRACKING_URI env > built-in default."""
    return tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def resolve_experiment(experiment_name: str | None = None) -> str:
    """Precedence: explicit arg > MLFLOW_EXPERIMENT env > built-in default."""
    return experiment_name or os.environ.get("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT)


def _client(tracking_uri: str | None = None) -> MlflowClient:
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    return MlflowClient()


def register_model(
    model_name: str,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:
    experiment_name = resolve_experiment(experiment_name)
    client = _client(tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"No experiment '{experiment_name}' found. Train first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time desc"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No training runs found. Train first.")

    run_id = runs[0].info.run_id

    artifacts = [f.path for f in client.list_artifacts(run_id, "")]
    source = f"runs:/{run_id}/adapters" if "adapters" in artifacts else f"runs:/{run_id}"

    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)

    mv = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
    )

    if tags:
        for k, v in tags.items():
            client.set_model_version_tag(name=model_name, version=mv.version, key=k, value=v)

    logger.info(f"Registered model: {model_name} version {mv.version}")
    return mv.version
