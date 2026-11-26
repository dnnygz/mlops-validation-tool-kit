"""Pipeline local completo: Spark -> sklearn -> validación -> MLflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from toolkit.preprocessing import build_medallion, create_spark, join_example, sql_examples
from toolkit.registry import configure_mlflow, log_run
from toolkit.validation import classification_metrics, confusion_matrix_payload, evaluate_gate, validate_dataframe, write_validation_report

FEATURES = ["log_amount", "hour", "customer_age", "account_age_days", "is_international", "is_night", "channel_index"]


def run(config_path: Path, n_rows: int, work_dir: Path) -> bool:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts = work_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    spark = create_spark()
    try:
        gold = build_medallion(spark, work_dir / "data", n_rows)
        print(f"Particiones Gold: {gold.rdd.getNumPartitions()}")
        print("Plan físico de agregación (busca Exchange = shuffle):")
        sql_examples(gold).explain(mode="formatted")
        print("Agregación SQL:")
        sql_examples(gold).show(truncate=False)  # action
        print(f"Filas tras join broadcast: {join_example(gold).count()}")  # action
        # bien aca por volumen acotado. En produccion collect()/toPandas() pueden agotar RAM del driver.
        pdf = gold.toPandas()
    finally:
        spark.stop()

    data_report = validate_dataframe(pdf, config["data"]["required_columns"], config["data"]["min_rows"], config["data"]["max_null_fraction"])
    if not data_report["passed"]:
        write_validation_report({"data": data_report, "passed": False}, artifacts / "validation_report.json")
        return False
    # Un contrato float64 evita diferencias int32/int64 entre Spark, pandas y JSON serving.
    features = pdf[FEATURES].astype("float64")
    x_train, x_test, y_train, y_test = train_test_split(features, pdf["label"], test_size=0.30, stratify=pdf["label"], random_state=42)
    model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, pred, score)

    baseline_model = DummyClassifier(strategy="prior").fit(x_train, y_train)
    baseline_pred = baseline_model.predict(x_test)
    baseline_score = baseline_model.predict_proba(x_test)[:, 1]
    baseline = classification_metrics(y_test, baseline_pred, baseline_score)
    gate = evaluate_gate(metrics, baseline, config["model"])
    report = {"passed": gate["passed"], "data": data_report, "model": gate, "confusion_matrix": confusion_matrix_payload(y_test, pred)}
    write_validation_report(report, artifacts / "validation_report.json")
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1])
    plt.tight_layout(); plt.savefig(artifacts / "confusion_matrix.png"); plt.close()

    configure_mlflow(work_dir / "mlruns", "fraud-validation")
    run_id, version = log_run(model, {"rows": n_rows, "algorithm": "logistic_regression"}, metrics, x_train.head(20), artifacts, gate["passed"], "fraud-classifier")
    print(json.dumps({"passed": gate["passed"], "metrics": metrics, "baseline": baseline, "mlflow_run_id": run_id, "model_version": version}, indent=2))
    return bool(gate["passed"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/validation.yaml"))
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    return 0 if run(args.config, args.rows, args.work_dir) else 2


if __name__ == "__main__":
    sys.exit(main())

