#!/usr/bin/env python3
"""
Feature Extraction Module
Converts raw sensor snapshots into statistical feature vectors.

Architecture:
- Input: SensorSnapshot messages (from /degraded_sensor_snapshot)
- Output: numpy feature vectors → ML classifier
- Window: 1-second sliding window at 100Hz = 100 samples per vector

Phase 4 note: This module is shared between continuous and cascade modes.
The cascade controller calls this same extractor at every layer.
"""
import numpy as np
from scipy import stats
from collections import deque


# Feature names for interpretability and CSV logging
# Order must match feature vector output
FEATURE_NAMES = []
for j in range(6):
    for stat in ['mean', 'std', 'skew', 'kurt', 'rms', 'range']:
        FEATURE_NAMES.append(f'j{j}_pos_{stat}')
for j in range(6):
    for stat in ['mean', 'std', 'skew', 'kurt', 'rms', 'range']:
        FEATURE_NAMES.append(f'j{j}_vel_{stat}')
for j in range(6):
    for stat in ['mean', 'std', 'skew', 'kurt', 'rms', 'range']:
        FEATURE_NAMES.append(f'j{j}_eff_{stat}')
# IMU features
for axis in ['ax', 'ay', 'az', 'wx', 'wy', 'wz']:
    for stat in ['mean', 'std', 'skew', 'kurt']:
        FEATURE_NAMES.append(f'imu_{axis}_{stat}')
# Residual features (populated by Kalman filter)
for j in range(6):
    for stat in ['mean', 'std', 'max']:
        FEATURE_NAMES.append(f'j{j}_residual_{stat}')


class FeatureExtractor:
    def __init__(self, window_size: int = 100):
        """
        Initialize feature extractor with sliding window.

        Args:
            window_size: Number of samples per feature vector (default 100 = 1s at 100Hz)
        """
        self.window_size = window_size

        # Sliding windows for each signal
        self.pos_window = deque(maxlen=window_size)    # 6 joints
        self.vel_window = deque(maxlen=window_size)    # 6 joints
        self.eff_window = deque(maxlen=window_size)    # 6 joints
        self.imu_window = deque(maxlen=window_size)    # 6 axes
        self.residual_window = deque(maxlen=window_size)  # 6 joints

    def add_sample(self, snapshot, residuals=None):
        """
        Add one sensor snapshot to sliding windows.

        Args:
            snapshot: SensorSnapshot message
            residuals: np.array of shape (6,) from Kalman filter, or None
        """
        # Pad to 6 joints if needed
        pos = list(snapshot.joint_positions) + \
              [0.0] * (6 - len(snapshot.joint_positions))
        vel = list(snapshot.joint_velocities) + \
              [0.0] * (6 - len(snapshot.joint_velocities))
        eff = list(snapshot.joint_efforts) + \
              [0.0] * (6 - len(snapshot.joint_efforts))
        imu = [
            snapshot.imu_linear_accel_x,
            snapshot.imu_linear_accel_y,
            snapshot.imu_linear_accel_z,
            snapshot.imu_angular_vel_x,
            snapshot.imu_angular_vel_y,
            snapshot.imu_angular_vel_z
        ]

        self.pos_window.append(pos[:6])
        self.vel_window.append(vel[:6])
        self.eff_window.append(eff[:6])
        self.imu_window.append(imu)

        if residuals is not None:
            self.residual_window.append(list(residuals))
        else:
            self.residual_window.append([0.0] * 6)

    def is_ready(self) -> bool:
        """Returns True when window is full and features can be extracted"""
        return len(self.pos_window) == self.window_size

    def extract_features(self) -> np.ndarray:
        """
        Extract statistical features from current window.
        Returns feature vector of shape (n_features,).
        """
        if not self.is_ready():
            return None

        features = []

        # Convert windows to numpy arrays: shape (window_size, 6)
        pos_arr = np.array(self.pos_window)
        vel_arr = np.array(self.vel_window)
        eff_arr = np.array(self.eff_window)
        imu_arr = np.array(self.imu_window)
        res_arr = np.array(self.residual_window)

        # Statistical features per joint per signal
        for arr in [pos_arr, vel_arr, eff_arr]:
            for j in range(6):
                signal = arr[:, j]
                features.extend([
                    np.mean(signal),
                    np.std(signal),
                    float(stats.skew(signal)),
                    float(stats.kurtosis(signal)),
                    np.sqrt(np.mean(signal ** 2)),      # RMS
                    np.max(signal) - np.min(signal)     # Range
                ])

        # IMU statistical features
        for axis in range(6):
            signal = imu_arr[:, axis]
            features.extend([
                np.mean(signal),
                np.std(signal),
                float(stats.skew(signal)),
                float(stats.kurtosis(signal))
            ])

        # Residual features from Kalman filter
        for j in range(6):
            residuals = res_arr[:, j]
            features.extend([
                np.mean(np.abs(residuals)),
                np.std(residuals),
                np.max(np.abs(residuals))
            ])

        return np.array(features, dtype=np.float32)

    @staticmethod
    def extract_from_dataframe(df, window_size: int = 100):
        """
        Batch feature extraction from CSV dataframe.
        Used during ML training (offline, not real-time).

        Returns:
            X: feature matrix (n_windows, n_features)
            y: label vector (n_windows,)
            severities: severity vector (n_windows,)
        """
        extractor = FeatureExtractor(window_size)
        X, y, severities = [], [], []

        for i in range(len(df)):
            row = df.iloc[i]

            # Build mock snapshot from CSV row
            class MockSnapshot:
                pass

            snap = MockSnapshot()
            snap.joint_positions = [
                row['j0_pos'], row['j1_pos'], row['j2_pos'],
                row['j3_pos'], row['j4_pos'], row['j5_pos']
            ]
            snap.joint_velocities = [
                row['j0_vel'], row['j1_vel'], row['j2_vel'],
                row['j3_vel'], row['j4_vel'], row['j5_vel']
            ]
            snap.joint_efforts = [
                row['j0_eff'], row['j1_eff'], row['j2_eff'],
                row['j3_eff'], row['j4_eff'], row['j5_eff']
            ]
            snap.imu_linear_accel_x = row.get('imu_ax', 0.0)
            snap.imu_linear_accel_y = row.get('imu_ay', 0.0)
            snap.imu_linear_accel_z = row.get('imu_az', 0.0)
            snap.imu_angular_vel_x = row.get('imu_wx', 0.0)
            snap.imu_angular_vel_y = row.get('imu_wy', 0.0)
            snap.imu_angular_vel_z = row.get('imu_wz', 0.0)

            extractor.add_sample(snap)

            if extractor.is_ready() and i % (window_size // 2) == 0:
                features = extractor.extract_features()
                if features is not None:
                    X.append(features)
                    y.append(row['fault_type'])
                    severities.append(row.get('fault_severity', 0.0))

        return np.array(X), np.array(y), np.array(severities)