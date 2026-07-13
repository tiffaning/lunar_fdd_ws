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

from hybrid_fdd.feature_extractor import ANOMALY_FEATURE_INDICES


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
        self.severity_regressor = None
        self.decision_tree = None        # Phase 4 cascade Layer 2
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
            # Calibrated anomaly threshold (falls back to the default if absent)
            try:
                self.anomaly_threshold = float(joblib.load(
                    os.path.join(self.model_dir, 'anomaly_threshold.pkl')
                ))
            except (FileNotFoundError, OSError):
                pass
            # Optional severity regressor (severity_estimate stays 0 if absent)
            try:
                self.severity_regressor = joblib.load(
                    os.path.join(self.model_dir, 'severity_regressor.pkl')
                )
            except (FileNotFoundError, OSError):
                pass
            # Optional Phase 4 Layer 2 decision tree
            try:
                self.decision_tree = joblib.load(
                    os.path.join(self.model_dir, 'decision_tree.pkl')
                )
            except (FileNotFoundError, OSError):
                pass
            print(f'[MLClassifier] Models loaded from {self.model_dir} '
                  f'(anomaly threshold {self.anomaly_threshold:.4f})')
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

        # Stage 1: Anomaly detection (IF uses the fault-bearing feature subset)
        anomaly_score = self.isolation_forest.decision_function(
            features[:, ANOMALY_FEATURE_INDICES]
        )[0]
        is_anomaly = anomaly_score < self.anomaly_threshold

        if not is_anomaly:
            return 'none', 1.0 - abs(anomaly_score), False

        # Stage 2: Fault classification (only if anomaly detected)
        svm_prediction = self.svm_classifier.predict(features)[0]
        svm_probabilities = self.svm_classifier.predict_proba(features)[0]

        fault_type = self.label_encoder.inverse_transform([svm_prediction])[0]
        confidence = float(np.max(svm_probabilities))

        return fault_type, confidence, True

    def classify_continuous(self, feature_vector: np.ndarray):
        """
        Standard (continuous) hybrid FDD: run BOTH stages every cycle, no gating.

        Unlike predict() -- which gates the SVM behind the anomaly check and is
        meant for the Phase 4 cascade -- this always runs the SVM so a fault is
        classified even when the (imperfect) Isolation Forest misses the anomaly.
        The IF result is returned separately as an independent anomaly signal.

        Returns:
            fault_type: string label (from SVM)
            confidence: float 0.0-1.0 (SVM max class probability)
            is_anomaly: bool (Isolation Forest)
            anomaly_score: float (IF decision function; lower = more anomalous)
            severity: float 0.0-1.0 (regressed fault severity; 0 if no regressor)
        """
        if not self.is_loaded:
            return 'unknown', 0.0, False, 0.0, 0.0

        features = self.scaler.transform(feature_vector.reshape(1, -1))

        anomaly_score = float(
            self.isolation_forest.decision_function(
                features[:, ANOMALY_FEATURE_INDICES]
            )[0]
        )
        is_anomaly = anomaly_score < self.anomaly_threshold

        svm_prediction = self.svm_classifier.predict(features)[0]
        svm_probabilities = self.svm_classifier.predict_proba(features)[0]
        fault_type = self.label_encoder.inverse_transform(
            [svm_prediction]
        )[0]
        confidence = float(np.max(svm_probabilities))

        severity = 0.0
        if self.severity_regressor is not None:
            severity = float(np.clip(
                self.severity_regressor.predict(features)[0], 0.0, 1.0
            ))

        return fault_type, confidence, is_anomaly, anomaly_score, severity

    def classify_tree(self, feature_vector: np.ndarray):
        """
        Phase 4 cascade Layer 2: fast supervised classification via the decision
        tree on the fault-bearing feature subset (effort + IMU + residual).

        Returns:
            fault_type: string label
            confidence: float 0.0-1.0 (tree class probability)
        """
        if not self.is_loaded or self.decision_tree is None:
            return 'unknown', 0.0
        features = self.scaler.transform(feature_vector.reshape(1, -1))
        sub = features[:, ANOMALY_FEATURE_INDICES]
        pred = self.decision_tree.predict(sub)[0]
        proba = self.decision_tree.predict_proba(sub)[0]
        fault_type = self.label_encoder.inverse_transform([pred])[0]
        confidence = float(np.max(proba))
        return fault_type, confidence

    def predict_severity(self, feature_vector: np.ndarray):
        """Regress fault severity (0.0-1.0). Returns 0.0 if no regressor loaded.
        Used by the cascade to attach a severity to Layer 2 fault detections."""
        if not self.is_loaded or self.severity_regressor is None:
            return 0.0
        features = self.scaler.transform(feature_vector.reshape(1, -1))
        return float(np.clip(
            self.severity_regressor.predict(features)[0], 0.0, 1.0
        ))

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
        return float(self.isolation_forest.decision_function(
            features[:, ANOMALY_FEATURE_INDICES]
        )[0])