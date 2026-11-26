import pandas as pd

from toolkit.validation import classification_metrics, evaluate_gate, validate_dataframe


def test_dataframe_validation_passes():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = validate_dataframe(df, ["a", "b"], min_rows=2, max_null_fraction=0)
    assert result["passed"]


def test_classification_metrics_are_correct():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0


def test_gate_blocks_weak_model():
    metrics = {"precision": 0.2, "recall": 0.8, "f1": 0.3, "average_precision": 0.4}
    baseline = {"average_precision": 0.1}
    criteria = {"min_precision": 0.3, "min_recall": 0.5, "min_f1": 0.4, "min_average_precision": 0.3, "require_better_than_baseline": True}
    result = evaluate_gate(metrics, baseline, criteria)
    assert not result["passed"]
    assert not result["checks"]["precision"]

