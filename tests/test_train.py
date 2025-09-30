"""Unit tests for src/dsworkflows/models/train.py module.

Author: Gabriel Demetrios Lafis (@galafis)
Date: September 30, 2025
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from src.dsworkflows.models.train import (
    ModelTrainer,
    generate_synthetic_data
)


class TestGenerateSyntheticData:
    """Tests for the generate_synthetic_data function."""

    def test_generate_synthetic_data_default_params(self):
        """Test synthetic data generation with default parameters."""
        X, y = generate_synthetic_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == 1000  # default n_samples
        assert X.shape[1] == 10  # default n_features
        assert len(y) == 1000
        assert y.name == 'target'

    def test_generate_synthetic_data_custom_params(self):
        """Test synthetic data generation with custom parameters."""
        X, y = generate_synthetic_data(
            n_samples=500,
            n_features=5,
            n_classes=3,
            random_state=123
        )
        
        assert X.shape[0] == 500
        assert X.shape[1] == 5
        assert len(y) == 500
        assert len(y.unique()) == 3

    def test_generate_synthetic_data_reproducibility(self):
        """Test that same random_state produces same data."""
        X1, y1 = generate_synthetic_data(random_state=42)
        X2, y2 = generate_synthetic_data(random_state=42)
        
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_generate_synthetic_data_column_names(self):
        """Test that feature columns have meaningful names."""
        X, y = generate_synthetic_data(n_features=8)
        
        expected_columns = [f'feature_{i+1}' for i in range(8)]
        assert list(X.columns) == expected_columns


class TestModelTrainer:
    """Tests for the ModelTrainer class."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return generate_synthetic_data(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )

    @pytest.fixture
    def trainer(self, temp_model_dir):
        """Create a ModelTrainer instance."""
        return ModelTrainer(model_dir=temp_model_dir, random_state=42)

    def test_modeltrainer_initialization(self, temp_model_dir):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(model_dir=temp_model_dir, random_state=42)
        
        assert trainer.model_dir == Path(temp_model_dir)
        assert trainer.random_state == 42
        assert trainer.model is None
        assert trainer.scaler is not None
        assert trainer.label_encoder is not None

    def test_modeltrainer_creates_model_dir(self):
        """Test that ModelTrainer creates model directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, 'new_models')
            trainer = ModelTrainer(model_dir=model_dir)
            
            assert os.path.exists(model_dir)

    def test_preprocess_data_fit(self, trainer, sample_data):
        """Test data preprocessing with fitting."""
        X, y = sample_data
        
        X_processed, y_encoded = trainer.preprocess_data(X, y, fit_preprocessors=True)
        
        assert isinstance(X_processed, np.ndarray)
        assert isinstance(y_encoded, np.ndarray)
        assert X_processed.shape[0] == len(y_encoded)
        assert X_processed.shape[1] == X.shape[1]

    def test_preprocess_data_transform(self, trainer, sample_data):
        """Test data preprocessing with transformation only."""
        X, y = sample_data
        
        # First fit
        trainer.preprocess_data(X[:50], y[:50], fit_preprocessors=True)
        
        # Then transform
        X_processed, y_encoded = trainer.preprocess_data(
            X[50:], y[50:], fit_preprocessors=False
        )
        
        assert X_processed.shape[0] == 50
        assert y_encoded.shape[0] == 50

    def test_train_model_returns_results(self, trainer, sample_data):
        """Test that train_model returns expected results dictionary."""
        X, y = sample_data
        
        results = trainer.train_model(X, y, test_size=0.2, cv_folds=3)
        
        assert 'model' in results
        assert 'scaler' in results
        assert 'label_encoder' in results
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        assert 'cv_scores' in results
        assert 'cv_mean' in results
        assert 'cv_std' in results
        assert 'classification_report' in results
        assert 'feature_importance' in results
        assert 'model_params' in results

    def test_train_model_accuracy_range(self, trainer, sample_data):
        """Test that model achieves reasonable accuracy."""
        X, y = sample_data
        
        results = trainer.train_model(X, y)
        
        assert 0.0 <= results['train_accuracy'] <= 1.0
        assert 0.0 <= results['test_accuracy'] <= 1.0
        # Model should perform better than random (0.5 for binary classification)
        assert results['test_accuracy'] > 0.5

    def test_train_model_with_custom_params(self, trainer, sample_data):
        """Test training with custom model parameters."""
        X, y = sample_data
        
        custom_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42
        }
        
        results = trainer.train_model(X, y, model_params=custom_params)
        
        assert results['model'].n_estimators == 50
        assert results['model'].max_depth == 5

    def test_save_model_creates_file(self, trainer, sample_data, temp_model_dir):
        """Test that save_model creates a joblib file."""
        X, y = sample_data
        
        trainer.train_model(X, y)
        model_path = trainer.save_model('test_model')
        
        assert os.path.exists(model_path)
        assert model_path.endswith('.joblib')

    def test_save_model_without_training_raises_error(self, trainer):
        """Test that saving without training raises ValueError."""
        with pytest.raises(ValueError, match="No model has been trained yet"):
            trainer.save_model('test_model')

    def test_load_model_restores_artifacts(self, trainer, sample_data, temp_model_dir):
        """Test that load_model restores model artifacts correctly."""
        X, y = sample_data
        
        # Train and save
        trainer.train_model(X, y)
        model_path = trainer.save_model('test_model')
        
        # Create new trainer and load
        new_trainer = ModelTrainer(model_dir=temp_model_dir, random_state=42)
        artifacts = new_trainer.load_model(model_path)
        
        assert 'model' in artifacts
        assert 'model_type' in artifacts
        assert 'training_timestamp' in artifacts
        assert new_trainer.model is not None

    def test_load_model_nonexistent_file_raises_error(self, trainer):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            trainer.load_model('/nonexistent/path/model.joblib')

    def test_predict_after_training(self, trainer, sample_data):
        """Test making predictions after training."""
        X, y = sample_data
        
        trainer.train_model(X, y)
        predictions = trainer.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(isinstance(pred, str) for pred in predictions)

    def test_predict_without_training_raises_error(self, trainer, sample_data):
        """Test that predicting without training raises ValueError."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="No model has been trained or loaded"):
            trainer.predict(X)

    def test_predict_after_load(self, trainer, sample_data, temp_model_dir):
        """Test making predictions after loading model."""
        X, y = sample_data
        
        # Train, save, and load
        trainer.train_model(X, y)
        model_path = trainer.save_model('test_model')
        
        new_trainer = ModelTrainer(model_dir=temp_model_dir, random_state=42)
        new_trainer.load_model(model_path)
        
        predictions = new_trainer.predict(X[:10])
        assert len(predictions) == 10

    def test_feature_importance_shape(self, trainer, sample_data):
        """Test that feature importance has correct shape."""
        X, y = sample_data
        
        results = trainer.train_model(X, y)
        feature_importance = results['feature_importance']
        
        assert len(feature_importance) == X.shape[1]
        assert 'feature' in feature_importance.columns
        assert 'importance' in feature_importance.columns

    def test_cross_validation_scores(self, trainer, sample_data):
        """Test that cross-validation produces multiple scores."""
        X, y = sample_data
        
        results = trainer.train_model(X, y, cv_folds=5)
        
        assert len(results['cv_scores']) == 5
        assert all(0.0 <= score <= 1.0 for score in results['cv_scores'])

    def test_classification_report_structure(self, trainer, sample_data):
        """Test that classification report has expected structure."""
        X, y = sample_data
        
        results = trainer.train_model(X, y)
        report = results['classification_report']
        
        assert isinstance(report, dict)
        assert 'accuracy' in report
        # Should have metrics for each class
        for class_name in trainer.label_encoder.classes_:
            assert class_name in report

    def test_model_directory_persistence(self, temp_model_dir):
        """Test that model directory persists across trainer instances."""
        # Create first trainer and train
        trainer1 = ModelTrainer(model_dir=temp_model_dir, random_state=42)
        X, y = generate_synthetic_data(n_samples=100, n_features=5, random_state=42)
        trainer1.train_model(X, y)
        model_path = trainer1.save_model('persistent_model')
        
        # Create second trainer and load
        trainer2 = ModelTrainer(model_dir=temp_model_dir, random_state=42)
        trainer2.load_model(model_path)
        
        # Both should make same predictions
        X_test, _ = generate_synthetic_data(n_samples=10, n_features=5, random_state=99)
        preds1 = trainer1.predict(X_test)
        preds2 = trainer2.predict(X_test)
        
        np.testing.assert_array_equal(preds1, preds2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
