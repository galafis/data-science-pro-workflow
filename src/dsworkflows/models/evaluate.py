"""Advanced evaluation utilities for classification models.
This module provides:
- Functions to compute metrics: ROC curve(s), AUC, confusion_matrix, precision, recall, f1_score, accuracy
- Detailed classification report (macro/weighted) and per-class metrics
- Plotting utilities to save ROC curves (binary and one-vs-rest for multiclass) as PNG
- Professional EvaluateService to streamline evaluation of sklearn-style models
- Synthetic usage examples

Author: Gabriel Demetrios Lafis (@galafis)
Date: September 30, 2025
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
)
import matplotlib.pyplot as plt
from itertools import cycle

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray, List[Any]]


def _ensure_numpy(arr: ArrayLike) -> np.ndarray:
    """Coerce input array-like into numpy ndarray."""
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.values
    return np.asarray(arr)


def _infer_classes(y_true: ArrayLike, y_pred: ArrayLike, labels: Optional[List[Any]] = None) -> List[Any]:
    """Infer class labels from provided labels or from y_true and y_pred."""
    if labels is not None:
        return list(labels)
    y_true_np = _ensure_numpy(y_true).ravel()
    y_pred_np = _ensure_numpy(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true_np, y_pred_np]))
    return list(classes)


def compute_basic_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Compute basic classification metrics.

    Returns a dict with: accuracy, precision_macro, recall_macro, f1_macro, precision_weighted,
    recall_weighted, f1_weighted.
    """
    y_true_np = _ensure_numpy(y_true).ravel()
    y_pred_np = _ensure_numpy(y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision_macro": float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
    }


def compute_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, labels: Optional[List[Any]] = None,
                              normalize: Optional[str] = None) -> Dict[str, Any]:
    """Compute confusion matrix.

    Args:
        y_true: Ground-truth labels
        y_pred: Predicted labels
        labels: Optional label order
        normalize: None, 'true', 'pred', or 'all' (as in sklearn)
    Returns:
        Dict with keys: matrix (2D list), labels (list)
    """
    y_true_np = _ensure_numpy(y_true).ravel()
    y_pred_np = _ensure_numpy(y_pred).ravel()
    cls = _infer_classes(y_true_np, y_pred_np, labels)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=cls, normalize=normalize)
    return {"matrix": cm.tolist(), "labels": list(map(lambda x: x if isinstance(x, (str, int)) else str(x), cls))}


def compute_classification_report(y_true: ArrayLike, y_pred: ArrayLike, target_names: Optional[List[str]] = None,
                                  output_dict: bool = True) -> Union[str, Dict[str, Any]]:
    """Compute detailed classification report.

    Args:
        y_true: Ground-truth labels
        y_pred: Predicted labels
        target_names: Optional class names
        output_dict: If True, returns dict; else returns string report
    """
    y_true_np = _ensure_numpy(y_true).ravel()
    y_pred_np = _ensure_numpy(y_pred).ravel()
    return classification_report(y_true_np, y_pred_np, target_names=target_names, zero_division=0,
                                 output_dict=output_dict)


def compute_roc_curves(y_true: ArrayLike, y_proba: ArrayLike, labels: Optional[List[Any]] = None,
                       pos_label: Optional[Any] = None) -> Dict[str, Any]:
    """Compute ROC curves for binary or multiclass (one-vs-rest) settings.

    Args:
        y_true: Ground-truth labels of shape (n_samples,)
        y_proba: Probabilities of shape (n_samples, n_classes) for multiclass or (n_samples,) / (n_samples, 2) for binary
        labels: Optional class labels order (length n_classes)
        pos_label: For binary case, which label is considered positive (default inferred)
    Returns:
        Dict with keys:
          - curves: list of dicts per class with keys: class_label, fpr, tpr, thresholds, auc
          - macro_auc: float (multiclass) or auc value (binary)
          - micro_auc: float (multiclass only)
          - classes: class labels list
    """
    y_true_np = _ensure_numpy(y_true).ravel()
    proba_np = _ensure_numpy(y_proba)

    # Handle binary special cases
    if proba_np.ndim == 1:
        # Provided probabilities for positive class directly
        pos = pos_label
        if pos is None:
            # Infer positive label as the max label (convention) if labels are numeric/ordinal
            unique = np.unique(y_true_np)
            pos = unique.max()
        fpr, tpr, thr = roc_curve(y_true_np, proba_np, pos_label=pos)
        auc_val = auc(fpr, tpr)
        return {
            "curves": [{"class_label": pos, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist(), "auc": float(auc_val)}],
            "macro_auc": float(auc_val),
            "classes": [pos],
        }

    # Ensure 2D shape
    if proba_np.ndim == 1:
        proba_np = proba_np.reshape(-1, 1)

    # If binary with shape (n,2), take positive column as the higher class index unless labels provided
    n_classes = proba_np.shape[1]
    if n_classes == 2:
        if labels is None:
            classes = list(np.unique(y_true_np))
            if len(classes) != 2:
                # Fallback to [0,1]
                classes = [0, 1]
        else:
            classes = list(labels)
        pos = classes[-1] if pos_label is None else pos_label
        fpr, tpr, thr = roc_curve(y_true_np, proba_np[:, classes.index(pos)], pos_label=pos)
        auc_val = auc(fpr, tpr)
        return {
            "curves": [{"class_label": pos, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist(), "auc": float(auc_val)}],
            "macro_auc": float(auc_val),
            "classes": classes,
        }

    # Multiclass one-vs-rest
    if labels is None:
        classes = list(np.unique(y_true_np))
        if len(classes) != n_classes:
            classes = list(range(n_classes))
    else:
        classes = list(labels)

    # Binarize y_true for one-vs-rest
    curves = []
    fpr_dict: Dict[Any, np.ndarray] = {}
    tpr_dict: Dict[Any, np.ndarray] = {}
    auc_dict: Dict[Any, float] = {}
    for i, cls in enumerate(classes):
        y_true_bin = (y_true_np == cls).astype(int)
        fpr_i, tpr_i, thr_i = roc_curve(y_true_bin, proba_np[:, i])
        auc_i = auc(fpr_i, tpr_i)
        curves.append({
            "class_label": cls,
            "fpr": fpr_i.tolist(),
            "tpr": tpr_i.tolist(),
            "thresholds": thr_i.tolist(),
            "auc": float(auc_i),
        })
        fpr_dict[cls] = fpr_i
        tpr_dict[cls] = tpr_i
        auc_dict[cls] = float(auc_i)

    # Micro/macro AUC
    # Micro: flatten all decisions
    y_true_bin_all = np.vstack([(y_true_np == c).astype(int) for c in classes]).T
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin_all.ravel(), proba_np.ravel())
    micro_auc = auc(fpr_micro, tpr_micro)
    macro_auc = float(np.mean(list(auc_dict.values())))

    return {
        "curves": curves,
        "macro_auc": float(macro_auc),
        "micro_auc": float(micro_auc),
        "classes": classes,
    }


def plot_roc_curves(y_true: ArrayLike, y_proba: ArrayLike, labels: Optional[List[Any]] = None,
                    title: str = "ROC Curve", save_path: Union[str, Path] = "roc_curve.png",
                    dpi: int = 120) -> Dict[str, Any]:
    """Plot ROC curve(s) and save as PNG.

    Supports binary and multiclass (one-vs-rest) settings. Returns a dict with path and AUC summary.
    """
    eval_roc = compute_roc_curves(y_true, y_proba, labels=labels)
    curves = eval_roc["curves"]
    classes = eval_roc["classes"]

    plt.figure(figsize=(7, 6), dpi=dpi)

    if len(curves) == 1:
        c = curves[0]
        plt.plot(c["fpr"], c["tpr"], color="C0", lw=2, label=f"AUC = {c['auc']:.3f}")
    else:
        colors = cycle([f"C{i}" for i in range(10)])
        for c, color in zip(curves, colors):
            plt.plot(c["fpr"], c["tpr"], lw=1.8, color=color, label=f"Class {c['class_label']} (AUC={c['auc']:.3f})")
        if "micro_auc" in eval_roc:
            # Micro curve over all classes
            y_true_np = _ensure_numpy(y_true).ravel()
            proba_np = _ensure_numpy(y_proba)
            y_true_bin_all = np.vstack([(y_true_np == cls).astype(int) for cls in classes]).T
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin_all.ravel(), proba_np.ravel())
            plt.plot(fpr_micro, tpr_micro, color="black", lw=2.2, linestyle="--", label=f"micro-AUC = {eval_roc['micro_auc']:.3f}")

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    save_path = str(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    out: Dict[str, Any] = {"path": save_path, "macro_auc": float(eval_roc.get("macro_auc", np.nan))}
    if "micro_auc" in eval_roc:
        out["micro_auc"] = float(eval_roc["micro_auc"])
    return out


class EvaluateService:
    """Service for evaluating sklearn-style classifiers.

    Attributes:
        model: Fitted classifier with predict and optionally predict_proba
        label_encoder: Optional label encoder to inverse-transform labels
    """

    def __init__(self, model: Any, label_encoder: Optional[Any] = None):
        self.model = model
        self.label_encoder = label_encoder

    def _inverse_transform(self, y: ArrayLike) -> np.ndarray:
        if self.label_encoder is not None and hasattr(self.label_encoder, "inverse_transform"):
            return self.label_encoder.inverse_transform(y)
        return _ensure_numpy(y)

    def evaluate(self, X: ArrayLike, y_true: ArrayLike, labels: Optional[List[Any]] = None,
                 plot_roc_path: Optional[Union[str, Path]] = None,
                 title: str = "ROC Curve") -> Dict[str, Any]:
        """Evaluate predictions and probabilities if available.

        Returns a dict with metrics, confusion_matrix, classification_report, and ROC/AUC (and optional saved plot).
        """
        X_np = _ensure_numpy(X)
        y_true_np = _ensure_numpy(y_true).ravel()

        y_pred = self.model.predict(X_np)
        # If model predicts numeric indices but label_encoder exists
        y_pred_inv = self._inverse_transform(y_pred)
        y_true_inv = self._inverse_transform(y_true_np)

        metrics = compute_basic_metrics(y_true_inv, y_pred_inv)
        cls = _infer_classes(y_true_inv, y_pred_inv, labels)
        cm = compute_confusion_matrix(y_true_inv, y_pred_inv, labels=cls)
        report = compute_classification_report(y_true_inv, y_pred_inv, target_names=[str(c) for c in cls], output_dict=True)

        out: Dict[str, Any] = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_np)
            roc_info = compute_roc_curves(y_true_inv, proba, labels=cls)
            out["roc"] = roc_info
            if plot_roc_path is not None:
                plot_info = plot_roc_curves(y_true_inv, proba, labels=cls, title=title, save_path=plot_roc_path)
                out["roc_plot"] = plot_info
        else:
            logger.info("Model has no predict_proba; ROC/AUC skipped.")

        return out


# ---------------------- Synthetic usage examples ----------------------
if __name__ == "__main__":
    """Demonstration using synthetic data and a RandomForestClassifier.

    Example:
        python -m dsworkflows.models.evaluate
    """
    import joblib
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    logging.getLogger().setLevel(logging.INFO)
    logger.info("Running synthetic evaluation demo...")

    # Create synthetic multiclass data
    X, y = make_classification(n_samples=600, n_features=10, n_informative=6, n_redundant=2,
                               n_classes=3, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    service = EvaluateService(clf)
    results = service.evaluate(X_test, y_test, plot_roc_path="./roc_demo.png", title="ROC Curve (Synthetic Multiclass)")
