#!/usr/bin/env python3
"""
ML Classifier for fault type identification.
Two-stage classification:
  Stage 1: Isolation Forest - is this anomalous?
  Stage 2: SVM - if anomalous, what fault type?

Architecture note for Phase 4:
- Stage 1 (Isolation Forest) is fast → used in cascade Layer 2
- Stage 2 (SVM) is slower → used in cascade Layer 3 (full hybrid)
- Models saved as .pkl files and loaded at runtime
"""
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


class MLClassifier:
    def __init__(self, model_dir: str):
        """
        Initialize classifier with path to saved models.

        Args:
            model_dir: directory containing .pkl model files
        """
        self.model_dir = model_dir
        self.isolation_forest = None
        self.svm_classifier = None
        self.scaler = None
        self.label_encoder = None
        self.is_loaded = False

        # Anomaly threshold for Isolation Forest
        # Negative scores = more anomalous in sklearn's IF
        self.anomaly_threshold = -0.1

    def load_models(self):
        """Load trained models from disk"""
        try:
            self.isolation_forest = joblib.load(
                os.path.join(self.model_dir, 'isolation_forest.pkl')
            )
            self.svm_classifier = joblib.load(
                os.path.join(self.model_dir, 'svm_classifier.pkl')
            )
            self.scaler = joblib.load(
                os.path.join(self.model_dir, 'scaler.pkl')
            )
            self.label_encoder = joblib.load(
                os.path.join(self.model_dir, 'label_encoder.pkl')
            )
            self.is_loaded = True
            print(f'[MLClassifier] Models loaded from {self.model_dir}')
            return True
        except FileNotFoundError as e:
            print(f'[MLClassifier] Model files not found: {e}')
            return False

    def predict(self, feature_vector: np.ndarray):
        """
        Two-stage fault classification.

        Args:
            feature_vector: extracted features from FeatureExtractor

        Returns:
            fault_type: string label
            confidence: float 0.0-1.0
            is_anomaly: bool from Isolation Forest
        """
        if not self.is_loaded:
            return 'unknown', 0.0, False

        # Normalize features
        features = self.scaler.transform(
            feature_vector.reshape(1, -1)
        )

        # Stage 1: Anomaly detection
        anomaly_score = self.isolation_forest.decision_function(features)[0]
        is_anomaly = anomaly_score < self.anomaly_threshold

        if not is_anomaly:
            return 'none', 1.0 - abs(anomaly_score), False

        # Stage 2: Fault classification (only if anomaly detected)
        svm_prediction = self.svm_classifier.predict(features)[0]
        svm_probabilities = self.svm_classifier.predict_proba(features)[0]

        fault_type = self.label_encoder.inverse_transform([svm_prediction])[0]
        confidence = float(np.max(svm_probabilities))

        return fault_type, confidence, True

    def get_isolation_forest_score(self, feature_vector: np.ndarray):
        """
        Fast anomaly score only - used by Phase 4 cascade Layer 2.
        Much faster than full predict() - skips SVM.

        Returns:
            float: anomaly score (negative = more anomalous)
        """
        if not self.is_loaded:
            return 0.0
        features = self.scaler.transform(feature_vector.reshape(1, -1))
        return float(self.isolation_forest.decision_function(features)[0])