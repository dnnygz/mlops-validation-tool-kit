"""Carga el champion (o version concreta) y predice localmente."""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="Versión para rollback; por defecto usa @champion")
    args = parser.parse_args()
    db = (Path("mlruns") / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_registry_uri(f"sqlite:///{db}")
    uri = f"models:/fraud-classifier/{args.version}" if args.version else "models:/fraud-classifier@champion"
    model = mlflow.pyfunc.load_model(uri)
    sample = pd.DataFrame([{
        "log_amount": 5.1, "hour": 2.0, "customer_age": 34.0,
        "account_age_days": 100.0, "is_international": 1.0,
        "is_night": 1.0, "channel_index": 1.0,
    }])
    print({"model_uri": uri, "prediction": model.predict(sample).tolist()})


if __name__ == "__main__":
    main()

