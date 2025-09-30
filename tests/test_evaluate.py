"""Unit tests for src/dsworkflows/models/evaluate.py module.
Author: Gabriel Demetrios Lafis (@galafis)
Date: September 30, 2025

Este arquivo cobre todas as funções públicas de evaluate.py usando dados sintéticos
(binário e multiclasse) e valida os resultados esperados.
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
    """Train a small multiclass model and return (model, label_encoder, X, y).

    - Usa dados sintéticos de 3 classes
    - Não persiste em disco
    """
    X, y = generate_synthetic_data(
        n_samples=180, n_features=6, n_classes=3, random_state=77
    )
    trainer = ModelTrainer(model_dir=tempfile.gettempdir(), random_state=42)
    _ = trainer.train_model(X, y, test_size=0.25)
    return trainer.model, trainer.label_encoder, X, y


class TestMetricFunctions:
    """Covers basic metrics, confusion matrix, classification report, and ROC utility."""

    def test_compute_basic_metrics(self):
        """Docstring: valida chaves e faixa [0,1] em métricas básicas."""
        y_true = np.array([0, 1, 2, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 2, 1, 0])
        m = eval_mod.compute_basic_metrics(y_true, y_pred)
        expected_keys = {
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        }
        assert expected_keys.issubset(m.keys())
        for k in expected_keys:
            assert 0.0 <= float(m[k]) <= 1.0

    def test_compute_confusion_matrix_and_normalize(self):
        """Docstring: valida labels e normalização por linha (true)."""
        y_true = ["a", "b", "a", "c", "b"]
        y_pred = ["a", "b", "c", "c", "b"]
        cm = eval_mod.compute_confusion_matrix(
            y_true, y_pred, labels=["a", "b", "c"], normalize=None
        )
        assert cm["labels"] == ["a", "b", "c"]
        assert isinstance(cm["matrix"], list)

        cm_norm = eval_mod.compute_confusion_matrix(
            y_true, y_pred, labels=["a", "b", "c"], normalize="true"
        )
        # Linhas devem somar ~1
        assert all(abs(sum(row) - 1.0) < 1e-6 for row in cm_norm["matrix"])  # type: ignore

    def test_classification_report_dict(self):
        """Docstring: valida presença de accuracy, macro e weighted avg."""
        y_true = [0, 1, 1, 2]
        y_pred = [0, 1, 0, 2]
        rep = eval_mod.compute_classification_report(
            y_true, y_pred, target_names=["c0", "c1", "c2"], output_dict=True
        )
        assert isinstance(rep, dict)
        assert "accuracy" in rep
        assert "macro avg" in rep and "weighted avg" in rep

    def test_binary_and_multiclass_roc(self):
        """Docstring: cobre binário (vetor positivo) e multiclasse (matriz proba)."""
        # Binário com vetor de prob da classe positiva diretamente
        y_true_bin = np.array([0, 1, 0, 1, 1])
        p_pos = np.array([0.1, 0.7, 0.4, 0.8, 0.6])
        roc_bin = eval_mod.compute_roc_curves(y_true_bin, p_pos)
        assert "curves" in roc_bin and len(roc_bin["curves"]) == 1
        assert "macro_auc" in roc_bin
        c = roc_bin["curves"][0]
        assert 0.0 <= c["auc"] <= 1.0

        # Multiclasse: probs shape (n, n_classes)
        y_true_mc = np.array([0, 1, 2, 1, 0, 2])
        proba_mc = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.2, 0.7],
                [0.2, 0.5, 0.3],
                [0.8, 0.1, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )
        roc_mc = eval_mod.compute_roc_curves(y_true_mc, proba_mc)
        assert len(roc_mc["curves"]) == 3
        assert "macro_auc" in roc_mc and "micro_auc" in roc_mc
        for c in roc_mc["curves"]:
            assert 0.0 <= c["auc"] <= 1.0

    def test_plot_roc_curves_binary_and_multiclass(self, tmp_path):
        """Docstring: valida que a função gera o arquivo PNG em binário e multiclasse."""
        # Binário
        y_true_bin = np.array([0, 1, 0, 1, 1, 0, 1])
        p_pos = np.array([0.2, 0.8, 0.35, 0.7, 0.62, 0.1, 0.9])
        out_bin = eval_mod.plot_roc_curves(
            y_true_bin, p_pos, title="ROC Binary", save_path=str(tmp_path / "roc_bin.png")
        )
        assert os.path.exists(out_bin["path"]) and out_bin["path"].endswith("roc_bin.png")

        # Multiclasse
        y_true_mc = np.array([0, 1, 2, 1, 0, 2])
        proba_mc = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.2, 0.7],
                [0.2, 0.5, 0.3],
                [0.8, 0.1, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )
        out_mc = eval_mod.plot_roc_curves(
            y_true_mc, proba_mc, title="ROC MC", save_path=str(tmp_path / "roc_mc.png")
        )
        assert os.path.exists(out_mc["path"]) and out_mc["path"].endswith("roc_mc.png")


class TestEvaluateService:
    """Covers EvaluateService with metrics and ROC plotting."""

    def test_evaluate_end_to_end_with_plot(self, tmp_path, small_trained_model):
        """Docstring: fluxo e2e com plot quando predict_proba está disponível."""
        model, label_encoder, X, y = small_trained_model
        # simula um split de teste
        X_test = X.iloc[:40]
        y_test = y.iloc[:40]

        service = eval_mod.EvaluateService(model, label_encoder=label_encoder)
        out = service.evaluate(
            X_test.values,
            y_test.values,
            plot_roc_path=str(tmp_path / "roc.png"),
            title="ROC Test",
        )
        # chaves
        assert {"metrics", "confusion_matrix", "classification_report"}.issubset(out.keys())
        if hasattr(model, "predict_proba"):
            assert "roc" in out
            assert "roc_plot" in out and os.path.exists(out["roc_plot"]["path"])  # type: ignore

    def test_evaluate_without_predict_proba(self, small_trained_model):
        """Docstring: quando modelo não tem predict_proba, ROC deve ser omitido."""
        model, label_encoder, X, y = small_trained_model

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
