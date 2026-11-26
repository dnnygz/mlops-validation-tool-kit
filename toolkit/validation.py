"""Validaciones de datos y modelos independientes del caso de fraude"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score


def validate_dataframe(df: Any, required_columns: Sequence[str], min_rows: int, max_null_fraction: float) -> dict[str, Any]:
    """Valida esquema, volumen y nulos de un DataFrame pandas."""
    missing = sorted(set(required_columns) - set(df.columns))
    null_fraction = float(df[list(required_columns)].isna().mean().max()) if not missing else 1.0
    checks = {"schema": not missing, "min_rows": len(df) >= min_rows, "nulls": null_fraction <= max_null_fraction}
    return {"passed": all(checks.values()), "checks": checks, "missing_columns": missing, "rows": len(df), "max_null_fraction": null_fraction}


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    """Calcula métricas adecuadas para clasificación desbalanceada."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def evaluate_gate(metrics: Mapping[str, float], baseline: Mapping[str, float], criteria: Mapping[str, Any]) -> dict[str, Any]:
    """Compara métricas con umbrales configurables y devuelve evidencia."""
    checks = {name: metrics[name] >= float(criteria[f"min_{name}"]) for name in ("precision", "recall", "f1", "average_precision")}
    if criteria.get("require_better_than_baseline", False):
        checks["better_than_baseline_ap"] = metrics["average_precision"] > baseline["average_precision"]
    return {"passed": all(checks.values()), "checks": checks, "metrics": dict(metrics), "baseline": dict(baseline)}


def write_validation_report(report: Mapping[str, Any], path: Path) -> None:
    """Escribe reporte JSON serializable como artefacto auditable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def confusion_matrix_payload(y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
    """Devuelve una matriz de confusión JSON serializable."""
    return np.asarray(confusion_matrix(y_true, y_pred, labels=[0, 1]), dtype=int).tolist()

