#!/usr/bin/env python3
"""
Kalman Filter for UR10 Joint State Prediction
Physics-based component of the hybrid FDD system.

Predicts expected joint states using simplified kinematics.
Residuals (predicted - actual) indicate anomalies.

Architecture note for Phase 4:
- residuals published to /sensor_residuals
- Phase 4 cascade Layer 1 uses max_residual as quick threshold check
- Full Kalman processing is Layer 3 of cascade
"""
import numpy as np


class KalmanFilter:
    def __init__(self, n_joints: int = 6):
        """
        Initialize Kalman filter for joint state estimation.

        State vector: [position, velocity] for each joint
        State dimension: 12 (6 joints × 2 states)
        Measurement dimension: 6 (joint positions only)
        """
        self.n_joints = n_joints
        self.state_dim = n_joints * 2   # position + velocity per joint
        self.meas_dim = n_joints        # observe positions only
        self.dt = 0.01                  # 100Hz sensor rate

        # State vector: [q0, q1, ..., q5, dq0, dq1, ..., dq5]
        self.x = np.zeros(self.state_dim)

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        for i in range(n_joints):
            self.F[i, i + n_joints] = self.dt  # position += velocity * dt

        # Observation matrix (observe positions only)
        self.H = np.zeros((self.meas_dim, self.state_dim))
        for i in range(n_joints):
            self.H[i, i] = 1.0

        # Process noise covariance
        # Higher = less trust in model, more trust in measurements
        self.Q = np.diag(
            [0.001] * n_joints +    # position process noise
            [0.01] * n_joints       # velocity process noise
        )

        # Measurement noise covariance
        # Values determined from baseline sensor variance
        self.R = np.diag(
            [0.005] * n_joints      # position measurement noise
        )

        # Initial covariance
        self.P = np.eye(self.state_dim) * 0.1

        # Gravity compensation for lunar environment
        # UR10 joint torques under lunar gravity (1.62 m/s²)
        # Approximated as offset to velocity predictions
        self.gravity_bias = np.zeros(n_joints)
        self.gravity_bias[1] = -0.002   # shoulder_lift most affected
        self.gravity_bias[2] = -0.001   # elbow affected

        self.initialized = False

    def initialize(self, joint_positions, joint_velocities):
        """Initialize filter state from first measurement"""
        self.x[:self.n_joints] = joint_positions[:self.n_joints]
        self.x[self.n_joints:] = joint_velocities[:self.n_joints]
        self.initialized = True

    def predict(self):
        """Prediction step: project state forward"""
        # Apply gravity bias to velocity prediction
        self.x[self.n_joints:] += self.gravity_bias * self.dt

        # State prediction
        self.x = self.F @ self.x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        Update step: correct prediction with measurement.

        Args:
            measurement: observed joint positions (6,)

        Returns:
            residuals: measurement - prediction (6,)
        """
        measurement = np.array(measurement[:self.n_joints])

        # Innovation (residual before update)
        predicted_measurement = self.H @ self.x
        residuals = measurement - predicted_measurement

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ residuals

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return residuals

    def process(self, joint_positions, joint_velocities):
        """
        Full predict+update cycle.
        Call this at every sensor sample (100Hz).

        Returns:
            residuals: np.array (6,) - large values indicate anomaly
            predicted_positions: np.array (6,) - what filter expected
        """
        if not self.initialized:
            self.initialize(joint_positions, joint_velocities)
            return np.zeros(self.n_joints), np.array(joint_positions[:self.n_joints])

        self.predict()
        residuals = self.update(joint_positions)
        predicted_positions = self.H @ self.x

        return residuals, predicted_positions

    def get_anomaly_score(self, residuals):
        """
        Convert residuals to single anomaly score.
        Used by Phase 4 cascade Layer 1 threshold check.

        Returns:
            float: 0.0 = normal, >1.0 = likely anomaly
        """
        return float(np.max(np.abs(residuals)))

    def reset(self):
        """Reset filter state (call between experiments)"""
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 0.1
        self.initialized = False