"""
ML Model Agent

Trains and serves gradient-boosted models for scoring and classification.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from common.schemas import MLPredictionRequest, MLPredictionResponse

logger = logging.getLogger(__name__)


class MLModelAgent:
    """
    ML Model Agent for gradient-boosted predictions

    Responsibilities:
    - Train XGBoost models on labeled data
    - Serve predictions with calibrated probabilities
    - Feature importance analysis
    - Model versioning and persistence
    - Online feature engineering
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize ML Model Agent

        Args:
            model_dir: Directory for storing trained models
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / "models"

        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # In-memory model cache
        self.models: Dict[str, Dict[str, Any]] = {}

        logger.info(f"✅ ML Model Agent initialized with model dir: {model_dir}")

    def predict(
        self,
        request: MLPredictionRequest,
    ) -> MLPredictionResponse:
        """
        Make prediction using trained model

        Args:
            request: MLPredictionRequest with features and model name

        Returns:
            MLPredictionResponse with predictions and confidence
        """
        model_name = request.model_name
        features = request.features

        # Load model if not in cache
        if model_name not in self.models:
            success = self._load_model(model_name)
            if not success:
                logger.error(f"Model not found: {model_name}")
                return MLPredictionResponse(
                    case_id=request.case_id,
                    from_agent=request.to_agent or request.from_agent,
                    to_agent=None,
                    model_name=model_name,
                    prediction=None,
                    confidence=0.0,
                    probabilities={},
                    feature_importance={},
                    explanation=f"Model '{model_name}' not found",
                )

        try:
            # Get model and metadata
            model_data = self.models[model_name]
            model = model_data["model"]
            feature_names = model_data["feature_names"]
            model_type = model_data["model_type"]

            # Prepare feature vector
            feature_vector = self._prepare_features(features, feature_names)

            # Make prediction
            if model_type == "binary_classifier":
                prediction, confidence, probabilities = self._predict_binary(
                    model, feature_vector
                )
            elif model_type == "multi_classifier":
                prediction, confidence, probabilities = self._predict_multi(
                    model, feature_vector, model_data.get("class_names", [])
                )
            elif model_type == "regressor":
                prediction, confidence, probabilities = self._predict_regression(
                    model, feature_vector
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Get feature importance for this prediction
            feature_importance = self._get_feature_importance(
                model, feature_names, top_k=10
            )

            explanation = (
                f"Prediction: {prediction} (confidence: {confidence:.2%})"
            )

            logger.debug(
                f"Model {model_name} prediction for case {request.case_id}: "
                f"{prediction} (confidence: {confidence:.3f})"
            )

            return MLPredictionResponse(
                case_id=request.case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                model_name=model_name,
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                feature_importance=feature_importance,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {e}")
            return MLPredictionResponse(
                case_id=request.case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                model_name=model_name,
                prediction=None,
                confidence=0.0,
                probabilities={},
                feature_importance={},
                explanation=f"Prediction error: {str(e)}",
            )

    def train_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: str = "binary_classifier",
        hyperparameters: Optional[Dict[str, Any]] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train XGBoost model

        Args:
            model_name: Name for the model
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: List of feature names
            model_type: "binary_classifier", "multi_classifier", or "regressor"
            hyperparameters: XGBoost hyperparameters
            class_names: Class names for classifiers

        Returns:
            Dict with training metrics
        """
        logger.info(
            f"Training model '{model_name}' ({model_type}) with {len(X)} samples"
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if model_type != "regressor" else None
        )

        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }

        # Train base model
        if model_type == "binary_classifier":
            base_model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **hyperparameters,
            )
            base_model.fit(X_train, y_train)

            # Calibrate probabilities
            model = CalibratedClassifierCV(base_model, cv=3, method="sigmoid")
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

        elif model_type == "multi_classifier":
            base_model = xgb.XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                **hyperparameters,
            )
            base_model.fit(X_train, y_train)

            # Calibrate
            model = CalibratedClassifierCV(base_model, cv=3, method="sigmoid")
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }

        elif model_type == "regressor":
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                **hyperparameters,
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Save model
        model_data = {
            "model": model,
            "feature_names": feature_names,
            "model_type": model_type,
            "class_names": class_names,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "trained_at": datetime.utcnow().isoformat(),
        }

        self._save_model(model_name, model_data)
        self.models[model_name] = model_data

        logger.info(
            f"Model '{model_name}' trained successfully. Metrics: {metrics}"
        )

        return {
            "model_name": model_name,
            "metrics": metrics,
            "feature_count": len(feature_names),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def _predict_binary(
        self,
        model: Any,
        feature_vector: np.ndarray,
    ) -> Tuple[Any, float, Dict[str, float]]:
        """
        Binary classification prediction

        Args:
            model: Trained classifier
            feature_vector: Feature array

        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
        prediction = int(proba[1] > 0.5)
        confidence = float(proba[prediction])

        probabilities = {
            "negative": float(proba[0]),
            "positive": float(proba[1]),
        }

        return prediction, confidence, probabilities

    def _predict_multi(
        self,
        model: Any,
        feature_vector: np.ndarray,
        class_names: List[str],
    ) -> Tuple[Any, float, Dict[str, float]]:
        """
        Multi-class classification prediction

        Args:
            model: Trained classifier
            feature_vector: Feature array
            class_names: Class name mapping

        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
        prediction = int(np.argmax(proba))
        confidence = float(proba[prediction])

        probabilities = {
            class_names[i] if class_names else f"class_{i}": float(proba[i])
            for i in range(len(proba))
        }

        return prediction, confidence, probabilities

    def _predict_regression(
        self,
        model: Any,
        feature_vector: np.ndarray,
    ) -> Tuple[Any, float, Dict[str, float]]:
        """
        Regression prediction

        Args:
            model: Trained regressor
            feature_vector: Feature array

        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        prediction = float(model.predict(feature_vector.reshape(1, -1))[0])

        # For regression, confidence is not well-defined
        # Use a placeholder value
        confidence = 1.0

        probabilities = {"value": prediction}

        return prediction, confidence, probabilities

    def _prepare_features(
        self,
        features: Dict[str, Any],
        feature_names: List[str],
    ) -> np.ndarray:
        """
        Prepare feature vector from dictionary

        Args:
            features: Feature dict
            feature_names: Expected feature names in order

        Returns:
            Feature vector as numpy array
        """
        feature_vector = []

        for name in feature_names:
            value = features.get(name, 0.0)

            # Handle None values
            if value is None:
                value = 0.0

            # Convert to float
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0

            feature_vector.append(value)

        return np.array(feature_vector)

    def _get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Get feature importance scores

        Args:
            model: Trained model
            feature_names: Feature names
            top_k: Number of top features to return

        Returns:
            Dict mapping feature names to importance scores
        """
        try:
            # Get base estimator for calibrated classifiers
            if hasattr(model, "calibrated_classifiers_"):
                base_model = model.calibrated_classifiers_[0].estimator
            else:
                base_model = model

            # Get importance scores
            if hasattr(base_model, "feature_importances_"):
                importances = base_model.feature_importances_
            else:
                return {}

            # Sort by importance
            indices = np.argsort(importances)[::-1][:top_k]

            feature_importance = {
                feature_names[i]: float(importances[i])
                for i in indices
            }

            return feature_importance

        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return {}

    def _save_model(self, model_name: str, model_data: Dict[str, Any]):
        """
        Save model to disk

        Args:
            model_name: Model name
            model_data: Model data dict
        """
        try:
            model_path = self.model_dir / f"{model_name}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            logger.debug(f"Saved model to {model_path}")

        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")

    def _load_model(self, model_name: str) -> bool:
        """
        Load model from disk

        Args:
            model_name: Model name

        Returns:
            True if successful
        """
        try:
            model_path = self.model_dir / f"{model_name}.pkl"

            if not model_path.exists():
                return False

            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            self.models[model_name] = model_data

            logger.debug(f"Loaded model from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models

        Returns:
            List of model metadata dicts
        """
        models = []

        for model_file in self.model_dir.glob("*.pkl"):
            model_name = model_file.stem

            # Load model to get metadata
            if model_name not in self.models:
                self._load_model(model_name)

            if model_name in self.models:
                model_data = self.models[model_name]
                models.append({
                    "name": model_name,
                    "type": model_data["model_type"],
                    "feature_count": len(model_data["feature_names"]),
                    "metrics": model_data.get("metrics", {}),
                    "trained_at": model_data.get("trained_at"),
                })

        return models

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed model information

        Args:
            model_name: Model name

        Returns:
            Model metadata dict or None
        """
        if model_name not in self.models:
            success = self._load_model(model_name)
            if not success:
                return None

        model_data = self.models[model_name]

        return {
            "name": model_name,
            "type": model_data["model_type"],
            "feature_names": model_data["feature_names"],
            "class_names": model_data.get("class_names"),
            "hyperparameters": model_data.get("hyperparameters", {}),
            "metrics": model_data.get("metrics", {}),
            "trained_at": model_data.get("trained_at"),
        }


# Singleton instance
_ml_model_agent: Optional[MLModelAgent] = None


def get_ml_model_agent(model_dir: Optional[Path] = None) -> MLModelAgent:
    """
    Get or create singleton MLModelAgent instance

    Args:
        model_dir: Directory for storing models

    Returns:
        MLModelAgent instance
    """
    global _ml_model_agent

    if _ml_model_agent is None:
        _ml_model_agent = MLModelAgent(model_dir=model_dir)

    return _ml_model_agent
