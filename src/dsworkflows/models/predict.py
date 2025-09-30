"""Prediction utilities for trained ML models.

This module provides functions and a professional PredictService class to:
- Load trained models saved via joblib (compatible with ModelTrainer artifacts)
- Perform single and batch predictions
- Return predicted class probabilities
- Provide basic explanations using SHAP (if available) or feature importances fallback

Author: Gabriel Demetrios Lafis (@galafis)
Date: September 30, 2025
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Optional SHAP dependency
try:
    import shap  # type: ignore
    _SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    shap = None  # type: ignore
    _SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PredictService:
    """Production-ready prediction service for scikit-learn style models.

    This class expects artifacts saved by ModelTrainer.save_model, which may include:
    - 'model': Trained estimator (e.g., RandomForestClassifier)
    - 'scaler': Optional StandardScaler for features
    - 'label_encoder': Optional LabelEncoder for decoding predictions

    Attributes:
        model: The loaded model used for prediction
        scaler: Optional scaler for input features
        label_encoder: Optional encoder to inverse-transform predicted labels
        model_type: String identifier of the model type
        training_timestamp: Timestamp when model was trained
    """

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = str(model_path)
        self.model: Any = None
        self.scaler: Any = None
        self.label_encoder: Any = None
        self.model_type: Optional[str] = None
        self.training_timestamp: Optional[str] = None
        self._explainer: Any = None  # SHAP Explainer cache

        self._load(model_path)

    def _load(self, model_path: Union[str, Path]) -> None:
        """Load artifacts from a joblib file.

        Args:
            model_path: Path to joblib file saved by ModelTrainer
        """
        model_path = str(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        artifacts: Dict[str, Any] = joblib.load(model_path)
        self.model = artifacts.get('model')
        self.scaler = artifacts.get('scaler', None)
        self.label_encoder = artifacts.get('label_encoder', None)
        self.model_type = artifacts.get('model_type', None)
        self.training_timestamp = artifacts.get('training_timestamp', None)

        if self.model is None:
            raise ValueError("Loaded artifacts did not contain a 'model' object.")

        logger.info(
            "Loaded model: %s | Trained at: %s | From: %s",
            self.model_type or type(self.model).__name__,
            self.training_timestamp,
            model_path,
        )

    # ---------------------- Input utilities ----------------------
    @staticmethod
    def _to_dataframe(X: Union[pd.DataFrame, Dict[str, List[Any]], np.ndarray, List[List[Any]]],
                      feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Coerce input into a pandas DataFrame.

        Args:
            X: Input features in various formats
            feature_names: Optional column names (required if X is array-like)
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, dict):
            return pd.DataFrame(X)
        X_arr = np.asarray(X)
        if feature_names is None:
            feature_names = [f"feature_{i+1}" for i in range(X_arr.shape[1])]
        return pd.DataFrame(X_arr, columns=feature_names)

    def _preprocess(self, X_df: pd.DataFrame) -> np.ndarray:
        """Apply optional scaler if present."""
        # Fill missing with mean per numeric column
        X_proc = X_df.fillna(X_df.mean(numeric_only=True))
        return self.scaler.transform(X_proc) if self.scaler is not None else X_proc.values

    # ---------------------- Core predictions ----------------------
    def predict_batch(self, X: Union[pd.DataFrame, Dict[str, List[Any]], np.ndarray, List[List[Any]]],
                      feature_names: Optional[List[str]] = None,
                      return_proba: bool = False,
                      explain: bool = False,
                      max_explanations: int = 100) -> Dict[str, Any]:
        """Run batch prediction.

        Args:
            X: Features in DataFrame, dict-of-lists, or array-like
            feature_names: Optional feature names if X is array-like
            return_proba: If True, include class probabilities
            explain: If True, return explanations (SHAP if available, else feature importance)
            max_explanations: Cap number of rows to compute SHAP for performance

        Returns:
            Dict with keys: predictions, probabilities (optional), explanations (optional)
        """
        X_df = self._to_dataframe(X, feature_names)
        X_mat = self._preprocess(X_df)

        # Predictions
        preds = self.model.predict(X_mat)
        if self.label_encoder is not None and hasattr(self.label_encoder, 'inverse_transform'):
            preds = self.label_encoder.inverse_transform(preds)
        predictions = preds.tolist()

        result: Dict[str, Any] = {"predictions": predictions}

        # Probabilities
        if return_proba and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_mat)
            # Map probabilities to class labels if available
            if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
                class_labels = list(self.label_encoder.classes_)
            elif hasattr(self.model, 'classes_'):
                class_labels = [str(c) for c in self.model.classes_]
            else:
                class_labels = [f"class_{i}" for i in range(proba.shape[1])]
            result["probabilities"] = [
                {label: float(p) for label, p in zip(class_labels, row)} for row in proba
            ]

        # Explanations
        if explain:
            result["explanations"] = self._explain(X_df, X_mat, max_explanations=max_explanations)

        return result

    def predict_one(self, x: Union[pd.Series, Dict[str, Any], List[Any], np.ndarray],
                    feature_names: Optional[List[str]] = None,
                    return_proba: bool = False,
                    explain: bool = False) -> Dict[str, Any]:
        """Run a single prediction.

        Args:
            x: Single sample as Series, dict, list, or 1D ndarray
            feature_names: Feature names if x is array-like
            return_proba: Whether to include probabilities
            explain: Whether to include SHAP/importance explanation for the sample
        """
        if isinstance(x, (pd.Series, dict)):
            X_df = self._to_dataframe([x]) if isinstance(x, dict) else x.to_frame().T
        else:
            X_df = self._to_dataframe([x], feature_names)

        batch = self.predict_batch(X_df, return_proba=return_proba, explain=explain, max_explanations=1)
        # Unpack the single result
        out: Dict[str, Any] = {"prediction": batch["predictions"][0]}
        if "probabilities" in batch:
            out["probabilities"] = batch["probabilities"][0]
        if "explanations" in batch:
            out["explanation"] = batch["explanations"][0]
        return out

    # ---------------------- Explanations ----------------------
    def _ensure_explainer(self, X_background: np.ndarray) -> Optional[Any]:
        if not _SHAP_AVAILABLE:
            return None
        if self._explainer is not None:
            return self._explainer
        try:
            # Tree-based models -> TreeExplainer; otherwise KernelExplainer
            if hasattr(self.model, 'predict_proba') and hasattr(self.model, 'estimators_'):
                self._explainer = shap.TreeExplainer(self.model)
            else:
                # Use a small background for performance
                background = X_background
                if background.shape[0] > 200:
                    background = background[:200]
                self._explainer = shap.KernelExplainer(self.model.predict_proba, background)
            return self._explainer
        except Exception as e:  # pragma: no cover - best effort
            logger.warning("Could not initialize SHAP explainer: %s", e)
            return None

    def _fallback_feature_importance(self, X_df: pd.DataFrame) -> List[Dict[str, float]]:
        """Provide per-sample importance using model-level feature_importances_ as proxy."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = list(X_df.columns)
            ranked = [{f: float(w) for f, w in sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)}]
            # Repeat same ranking for each sample (proxy)
            return ranked * len(X_df)
        # If no importances, return empty
        return [{} for _ in range(len(X_df))]

    def _explain(self, X_df: pd.DataFrame, X_mat: np.ndarray, max_explanations: int = 100) -> List[Dict[str, float]]:
        n = min(len(X_df), max_explanations)
        X_subset = X_mat[:n]
        try:
            explainer = self._ensure_explainer(X_subset)
            if explainer is None:
                return self._fallback_feature_importance(X_df.iloc[:n])

            # Use predict_proba if available to get class probabilities explanations
            if hasattr(self.model, 'predict_proba'):
                shap_values = explainer.shap_values(X_subset)
                # Pick top class per-sample to summarize
                proba = self.model.predict_proba(X_subset)
                top_class = np.argmax(proba, axis=1)
                if isinstance(shap_values, list):
                    # shap_values[class][sample, feature]
                    sv = [shap_values[c][i] for i, c in enumerate(top_class)]
                else:
                    sv = shap_values  # shape (n_samples, n_features)
            else:
                sv = explainer.shap_values(X_subset)
                if isinstance(sv, list):
                    sv = sv[0]

            feature_names = list(X_df.columns)
            explanations: List[Dict[str, float]] = []
            for row in np.asarray(sv):
                explanations.append({f: float(v) for f, v in sorted(zip(feature_names, row), key=lambda t: abs(t[1]), reverse=True)})
            # If we truncated, pad with fallback for remaining rows
            if len(X_df) > n:
                explanations.extend(self._fallback_feature_importance(X_df.iloc[n:]))
            return explanations
        except Exception as e:  # pragma: no cover - explanation is best-effort
            logger.warning("Explanation failed, falling back to feature importance: %s", e)
            return self._fallback_feature_importance(X_df)


# ---------------------- Convenience functions ----------------------
def load_service(model_path: Union[str, Path]) -> PredictService:
    """Load and return a PredictService for a given model path."""
    return PredictService(model_path)


def predict_batch(model_path: Union[str, Path],
                  X: Union[pd.DataFrame, Dict[str, List[Any]], np.ndarray, List[List[Any]]],
                  feature_names: Optional[List[str]] = None,
                  return_proba: bool = False,
                  explain: bool = False,
                  max_explanations: int = 100) -> Dict[str, Any]:
    """Batch prediction convenience wrapper.

    See PredictService.predict_batch for parameters.
    """
    service = load_service(model_path)
    return service.predict_batch(X, feature_names=feature_names, return_proba=return_proba, explain=explain, max_explanations=max_explanations)


def predict_one(model_path: Union[str, Path],
                x: Union[pd.Series, Dict[str, Any], List[Any], np.ndarray],
                feature_names: Optional[List[str]] = None,
                return_proba: bool = False,
                explain: bool = False) -> Dict[str, Any]:
    """Single prediction convenience wrapper.

    See PredictService.predict_one for parameters.
    """
    service = load_service(model_path)
    return service.predict_one(x, feature_names=feature_names, return_proba=return_proba, explain=explain)


# ---------------------- Synthetic usage examples ----------------------
if __name__ == "__main__":
    """Demonstration with synthetic data matching train.py defaults.

    Example:
        1) Train and save a model using ModelTrainer in train.py
           e.g., trainer.save_model("random_forest_classifier")
        2) Load and predict here:
           python -m dsworkflows.models.predict
    """
    from dsworkflows.models.train import ModelTrainer, generate_synthetic_data

    logging.getLogger().setLevel(logging.INFO)
    logger.info("Running synthetic prediction demo...")

    # Train quickly on synthetic data and save
    X_train, y_train = generate_synthetic_data(n_samples=300, n_features=6, n_classes=3, random_state=7)
    trainer = ModelTrainer(model_dir="../../../models", random_state=42)
    _ = trainer.train_model(X_train, y_train)
    model_path = trainer.save_model("demo_rf_classifier")

    # Batch prediction with probabilities and explanations
    X_new, _ = generate_synthetic_data(n_samples=5, n_features=6, n_classes=3, random_state=99)
    batch_out = predict_batch(model_path, X_new, return_proba=True, explain=True)
    print("Batch predictions:\n", batch_out)

    # Single prediction
    one_out = predict_one(model_path, X_new.iloc[0].to_dict(), return_proba=True, explain=True)
    print("\nSingle prediction:\n", one_out)
