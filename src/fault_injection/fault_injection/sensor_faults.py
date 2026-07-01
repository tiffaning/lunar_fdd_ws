#!/usr/bin/env python3
"""
Sensor Faults Module
Handles sensor-level fault injection (noise, drift, bias).

Architecture note for Phase 3:
- These faults affect sensor readings but NOT physics
- Kalman filter residuals will show sensor inconsistency
- ML classifier will distinguish sensor faults from mechanical faults
"""
import numpy as np


class SensorFaults:
    def __init__(self):
        self.rng = np.random.default_rng(seed=123)
        self.drift_accumulator = {}  # Tracks cumulative drift per joint

    def apply_imu_noise(self, imu_data: dict, severity: float) -> dict:
        """
        Add noise to IMU readings.
        Simulates electromagnetic interference in lunar environment.
        """
        noisy = imu_data.copy()
        noise_std = severity * 0.05

        noisy['ax'] += self.rng.normal(0, noise_std)
        noisy['ay'] += self.rng.normal(0, noise_std)
        noisy['az'] += self.rng.normal(0, noise_std * 0.5)
        noisy['wx'] += self.rng.normal(0, noise_std * 0.1)
        noisy['wy'] += self.rng.normal(0, noise_std * 0.1)
        noisy['wz'] += self.rng.normal(0, noise_std * 0.1)

        return noisy

    def apply_encoder_drift(self, positions: list,
                            joint_idx: int, severity: float,
                            dt: float) -> list:
        """
        Apply cumulative encoder drift to joint position.
        Simulates encoder degradation from radiation/regolith.
        """
        if joint_idx not in self.drift_accumulator:
            self.drift_accumulator[joint_idx] = 0.0

        # Drift rate increases with severity
        drift_rate = severity * 0.001  # radians per second
        self.drift_accumulator[joint_idx] += drift_rate * dt

        modified = list(positions)
        modified[joint_idx] += self.drift_accumulator[joint_idx]

        return modified

    def apply_signal_dropout(self, value: float,
                             severity: float) -> float:
        """
        Randomly drop sensor signal.
        Returns last known value (zero here for simplicity).
        """
        dropout_probability = severity * 0.1
        if self.rng.random() < dropout_probability:
            return 0.0
        return value

    def reset_drift(self):
        """Reset accumulated drift (used between experiments)"""
        self.drift_accumulator = {}
