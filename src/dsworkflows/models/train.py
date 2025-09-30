"""Machine Learning Model Training Module.

This module provides functions for training machine learning models,
saving trained models, and generating synthetic data for examples.

Author: Gabriel Demetrios Lafis (@galafis)
Date: September 30, 2025
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Professional ML model trainer with best practices."""
    
    def __init__(self, model_dir: str = "models", random_state: int = 42):
        """Initialize the ModelTrainer.
        
        Args:
            model_dir: Directory to save trained models
            random_state: Random state for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, 
                       fit_preprocessors: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess features and labels.
        
        Args:
            X: Feature matrix
            y: Target vector (optional for prediction)
            fit_preprocessors: Whether to fit or just transform
            
        Returns:
            Tuple of processed features and labels
        """
        logger.info("Preprocessing data...")
        
        # Handle missing values
        X_processed = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Scale features
        if fit_preprocessors:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
            
        # Encode labels if provided
        y_encoded = None
        if y is not None:
            if fit_preprocessors:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
                
        return X_scaled, y_encoded
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_params: Optional[Dict[str, Any]] = None,
                   test_size: float = 0.2,
                   cv_folds: int = 5) -> Dict[str, Any]:
        """Train a Random Forest Classifier with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_params: Model hyperparameters
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("Starting model training...")
        
        # Set default parameters
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
        # Preprocess data
        X_processed, y_encoded = self.preprocess_data(X, y, fit_preprocessors=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Initialize and train model
        self.model = RandomForestClassifier(**model_params)
        logger.info(f"Training model with parameters: {model_params}")
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate classification report
        class_report = classification_report(
            y_test, y_pred_test, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'feature_importance': feature_importance,
            'model_params': model_params
        }
        
        logger.info("Model training completed successfully!")
        return results
    
    def save_model(self, model_name: str, include_preprocessors: bool = True) -> str:
        """Save the trained model and preprocessors to disk.
        
        Args:
            model_name: Name for the saved model file
            include_preprocessors: Whether to save scaler and label encoder
            
        Returns:
            Path to the saved model file
        """
        if self.model is None:
            raise ValueError("No model has been trained yet. Call train_model() first.")
            
        model_path = self.model_dir / f"{model_name}.joblib"
        
        model_artifacts = {
            'model': self.model,
            'model_type': 'RandomForestClassifier',
            'training_timestamp': pd.Timestamp.now().isoformat()
        }
        
        if include_preprocessors:
            model_artifacts.update({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            })
            
        joblib.dump(model_artifacts, model_path)
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a previously saved model and preprocessors.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Dictionary containing loaded model artifacts
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model_artifacts = joblib.load(model_path)
        
        self.model = model_artifacts['model']
        if 'scaler' in model_artifacts:
            self.scaler = model_artifacts['scaler']
        if 'label_encoder' in model_artifacts:
            self.label_encoder = model_artifacts['label_encoder']
            
        logger.info(f"Model loaded from: {model_path}")
        return model_artifacts
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded. Train or load a model first.")
            
        X_processed, _ = self.preprocess_data(X, fit_preprocessors=False)
        predictions = self.model.predict(X_processed)
        
        # Decode predictions if label encoder exists
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
            
        return predictions


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10, 
                           n_classes: int = 2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic classification data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of target classes
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of features DataFrame and target Series
    """
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features, {n_classes} classes")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=random_state,
        class_sep=0.8
    )
    
    # Create DataFrame with meaningful column names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Create Series with meaningful class names
    class_names = [f'Class_{chr(65+i)}' for i in range(n_classes)]  # Class_A, Class_B, etc.
    y_series = pd.Series([class_names[label] for label in y], name='target')
    
    return X_df, y_series


def main():
    """Example usage of the ModelTrainer class with synthetic data."""
    logger.info("Starting ML training pipeline example...")
    
    try:
        # Generate synthetic data
        X, y = generate_synthetic_data(n_samples=1000, n_features=8, n_classes=3)
        logger.info(f"Data shape: {X.shape}, Target distribution:\n{y.value_counts()}")
        
        # Initialize trainer
        trainer = ModelTrainer(model_dir="../../../models", random_state=42)
        
        # Define model parameters
        model_params = {
            'n_estimators': 150,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        results = trainer.train_model(X, y, model_params=model_params)
        
        # Display results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Cross-validation Mean: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
        
        print("\nTop 5 Most Important Features:")
        print(results['feature_importance'].head())
        
        # Save model
        model_path = trainer.save_model("random_forest_classifier")
        print(f"\nModel saved to: {model_path}")
        
        # Demonstrate prediction on new data
        X_new, _ = generate_synthetic_data(n_samples=10, n_features=8, n_classes=3, random_state=99)
        predictions = trainer.predict(X_new)
        
        print("\nPredictions on new data:")
        for i, pred in enumerate(predictions[:5]):
            print(f"Sample {i+1}: {pred}")
            
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
