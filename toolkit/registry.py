"""Integracion pequeña con MLflow; concentra convenciones repetibles."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def configure_mlflow(root: Path, experiment_name: str) -> None:
    """Usa SQLite persistente y artefactos locales."""
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').resolve()}")
    mlflow.set_registry_uri(f"sqlite:///{(root / 'mlflow.db').resolve()}")
    mlflow.set_experiment(experiment_name)


def log_run(model: Any, params: Mapping[str, Any], metrics: Mapping[str, float], x_example: Any, artifact_dir: Path, approved: bool, model_name: str) -> tuple[str, str | None]:
    """Registra toda evidencia; solo crea versión si el gate aprueba."""
    with mlflow.start_run() as run:
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
        mlflow.set_tag("quality_gate", "approved" if approved else "rejected")
        mlflow.log_artifacts(str(artifact_dir), artifact_path="validation")
        signature = infer_signature(x_example, model.predict(x_example))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=x_example.head(5),
        )
        version = None
        if approved:
            mv = mlflow.register_model(model_info.model_uri, model_name)
            client = mlflow.MlflowClient()
            for _ in range(30):
                current = client.get_model_version(model_name, mv.version)
                if current.status == "READY":
                    break
                time.sleep(0.2)
            client.set_registered_model_alias(model_name, "champion", mv.version)
            version = str(mv.version)
        return run.info.run_id, version

