"""Unit tests for src/dsworkflows/models/predict.py module.
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
from src.dsworkflows.models import predict as predict_mod


@pytest.fixture
def temp_model_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def trained_model_path(temp_model_dir):
    """Train a small model and return saved model path."""
    X, y = generate_synthetic_data(n_samples=120, n_features=6, n_classes=3, random_state=123)
    trainer = ModelTrainer(model_dir=temp_model_dir, random_state=42)
    trainer.train_model(X, y)
    model_path = trainer.save_model("unit_rf_model")
    assert os.path.exists(model_path)
    return model_path


class TestPredictServiceBatchAndSingle:
    """Covers batch and single prediction, proba, and explanation flows."""

    def test_predict_batch_dataframe(self, trained_model_path):
        X_new, _ = generate_synthetic_data(n_samples=10, n_features=6, n_classes=3, random_state=321)
        service = predict_mod.PredictService(trained_model_path)
        out = service.predict_batch(X_new, return_proba=True, explain=False)
        assert "predictions" in out
        assert len(out["predictions"]) == 10
        # probabilities should be present and sum close to 1
        assert "probabilities" in out
        sums = [sum(row.values()) for row in out["probabilities"]]
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_predict_batch_array_with_feature_names(self, trained_model_path):
        X_new, _ = generate_synthetic_data(n_samples=5, n_features=6, n_classes=3, random_state=99)
        X_arr = X_new.values
        feature_names = list(X_new.columns)
        out = predict_mod.predict_batch(trained_model_path, X_arr, feature_names=feature_names, return_proba=False)
        assert "predictions" in out and len(out["predictions"]) == 5

    def test_predict_one_series_and_dict(self, trained_model_path):
        X_new, _ = generate_synthetic_data(n_samples=3, n_features=6, n_classes=3, random_state=42)
        # as Series
        one_series = X_new.iloc[0]
        out_series = predict_mod.predict_one(trained_model_path, one_series, return_proba=True)
        assert "prediction" in out_series
        assert "probabilities" in out_series
        # as dict
        one_dict = X_new.iloc[1].to_dict()
        out_dict = predict_mod.predict_one(trained_model_path, one_dict, return_proba=True)
        assert "prediction" in out_dict
        assert "probabilities" in out_dict

    def test_explanations_shap_or_fallback(self, trained_model_path):
        X_new, _ = generate_synthetic_data(n_samples=7, n_features=6, n_classes=3, random_state=7)
        service = predict_mod.load_service(trained_model_path)
        out = service.predict_batch(X_new, return_proba=False, explain=True, max_explanations=5)
        assert "explanations" in out
        # Should have max_explanations entries, and each explanation is a dict
        assert len(out["explanations"]) == 7  # first 5 via SHAP/fallback, rest padded by fallback
        assert all(isinstance(e, dict) for e in out["explanations"]) 

    def test_predict_one_with_explanation(self, trained_model_path):
        X_new, _ = generate_synthetic_data(n_samples=2, n_features=6, n_classes=3, random_state=11)
        out = predict_mod.predict_one(trained_model_path, X_new.iloc[0], return_proba=True, explain=True)
        assert "prediction" in out
        assert "probabilities" in out
        assert "explanation" in out


class TestPredictServiceErrors:
    """Covers error handling and edge cases."""

    def test_load_missing_model_file_raises(self, temp_model_dir):
        missing = Path(temp_model_dir) / "does_not_exist.joblib"
        with pytest.raises(FileNotFoundError):
            predict_mod.PredictService(missing)

    def test_predict_with_missing_values(self, trained_model_path):
        X_new, _ = generate_synthetic_data(n_samples=5, n_features=6, n_classes=3, random_state=55)
        # introduce NaNs
        X_new.iloc[0, 0] = np.nan
        X_new.iloc[3, 4] = np.nan
        service = predict_mod.PredictService(trained_model_path)
        out = service.predict_batch(X_new, return_proba=True)
        assert len(out["predictions"]) == 5
        assert "probabilities" in out

    def test_array_without_feature_names_infers(self, trained_model_path):
        X_new, _ = generate_synthetic_data(n_samples=4, n_features=6, n_classes=3, random_state=5)
        X_arr = X_new.values
        # No feature_names provided; should infer generic names
        service = predict_mod.PredictService(trained_model_path)
        out = service.predict_batch(X_arr)
        assert len(out["predictions"]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
