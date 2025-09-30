"""Unit tests for src/dsworkflows/models/evaluate.py module.
Author: Gabriel Demetrios Lafis (@galafis)
Date: September 30, 2025
"""
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.dsworkflows.models.train import ModelTrainer, generate_synthetic_data
from src.dsworkflows.models import evaluate as eval_mod


@pytest.fixture
def small_trained_model():
    """Train a small model and return (model, artifacts) without saving to disk."""
    X, y = generate_synthetic_data(n_samples=150, n_features=6, n_classes=3, random_state=77)
    trainer = ModelTrainer(model_dir=tempfile.gettempdir(), random_state=42)
    res = trainer.train_model(X, y, test_size=0.2)
    return trainer.model, trainer.label_encoder, X, y


class TestMetricFunctions:
    """Covers basic metrics, confusion matrix, classification report, and ROC utility."""

    def test_compute_basic_metrics(self):
        y_true = np.array([0, 1, 2, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 2, 1, 0])
        m = eval_mod.compute_basic_metrics(y_true, y_pred)
        assert set(["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"]).issubset(m.keys())
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_compute_confusion_matrix_and_normalize(self):
        y_true = ["a", "b", "a", "c", "b"]
        y_pred = ["a", "b", "c", "c", "b"]
        cm = eval_mod.compute_confusion_matrix(y_true, y_pred, labels=["a", "b", "c"], normalize=None)
        assert cm["labels"] == ["a", "b", "c"]
        assert isinstance(cm["matrix"], list)
        cm_norm = eval_mod.compute_confusion_matrix(y_true, y_pred, labels=["a", "b", "c"], normalize="true")
        # rows should sum to 1 when normalized by true
        assert all(abs(sum(row) - 1.0) < 1e-6 for row in cm_norm["matrix"])

    def test_classification_report_dict(self):
        y_true = [0, 1, 1, 2]
        y_pred = [0, 1, 0, 2]
        rep = eval_mod.compute_classification_report(y_true, y_pred, target_names=["c0", "c1", "c2"], output_dict=True)
        assert isinstance(rep, dict)
        assert "accuracy" in rep
        assert "macro avg" in rep and "weighted avg" in rep

    def test_binary_and_multiclass_roc(self):
        # Binary case with direct positive prob vector
        y_true_bin = np.array([0, 1, 0, 1, 1])
        p_pos = np.array([0.1, 0.7, 0.4, 0.8, 0.6])
        roc_bin = eval_mod.compute_roc_curves(y_true_bin, p_pos)
        assert "curves" in roc_bin and len(roc_bin["curves"]) == 1
        assert "macro_auc" in roc_bin

        # Multiclass case: probs shape (n, n_classes)
        y_true_mc = np.array([0, 1, 2, 1, 0, 2])
        proba_mc = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.5, 0.3],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
        ])
        roc_mc = eval_mod.compute_roc_curves(y_true_mc, proba_mc)
        assert len(roc_mc["curves"]) == 3
        assert "macro_auc" in roc_mc and "micro_auc" in roc_mc


class TestEvaluateService:
    """Covers EvaluateService with metrics and ROC plotting."""

    def test_evaluate_end_to_end_with_plot(self, tmp_path, small_trained_model):
        model, label_encoder, X, y = small_trained_model
        # simulate a test split
        X_test = X.iloc[:40]
        y_test = y.iloc[:40]
        service = eval_mod.EvaluateService(model, label_encoder=label_encoder)
        out = service.evaluate(X_test.values, y_test.values, plot_roc_path=str(tmp_path / "roc.png"), title="ROC Test")
        # check keys
        assert set(["metrics", "confusion_matrix", "classification_report"]).issubset(out.keys())
        if hasattr(model, "predict_proba"):
            assert "roc" in out
            assert "roc_plot" in out and os.path.exists(out["roc_plot"]["path"]) 

    def test_evaluate_without_predict_proba(self, small_trained_model, monkeypatch):
        model, label_encoder, X, y = small_trained_model
        # Create a wrapper that lacks predict_proba
        class NoProba:
            def __init__(self, base):
                self.base = base
            def predict(self, X):
                return self.base.predict(X)
        no_proba_model = NoProba(model)
        service = eval_mod.EvaluateService(no_proba_model, label_encoder=label_encoder)
        out = service.evaluate(X.iloc[:20].values, y.iloc[:20].values)
        assert "metrics" in out and "roc" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
