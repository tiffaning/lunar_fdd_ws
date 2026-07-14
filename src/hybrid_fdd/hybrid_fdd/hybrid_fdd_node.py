#!/usr/bin/env python3
"""
Hybrid FDD Node - standard (continuous) fault detection & diagnosis.

Pipeline, run on the incoming sensor stream (~100Hz):
  /degraded_sensor_snapshot
    -> Kalman filter (predict positions -> residuals)
    -> sliding-window feature extraction (stats + Kalman residual features)
    -> Isolation Forest (anomaly?) + SVM (which fault?)
    -> /fdd_result, /sensor_residuals, /energy_metrics

Design notes:
- This is the "continuous" baseline: the full hybrid pipeline runs every cycle
  and always classifies. Phase 4 adds a SEPARATE cascade node that gates the
  expensive SVM behind cheap layers (Kalman residual threshold -> Isolation
  Forest -> SVM), reusing these same KalmanFilter / FeatureExtractor /
  MLClassifier modules. Phase 5 compares the two on energy vs reliability.
- Feature extraction and residual computation here MUST match train_models.py
  exactly (same KalmanFilter, same FeatureExtractor, same velocity clipping),
  or the loaded scaler/models receive out-of-distribution inputs.
- The snapshot's fault_type field is ground truth; it is NOT used for detection
  (that would be cheating) - only the sensor values feed the pipeline.
"""
import rclpy
from rclpy.node import Node
from lunar_fdd_interfaces.msg import (
    SensorSnapshot, FDDResult, SensorResiduals, EnergyMetrics
)
import numpy as np
import psutil
import time

from hybrid_fdd.kalman_filter import KalmanFilter
from hybrid_fdd.feature_extractor import FeatureExtractor
from hybrid_fdd.ml_classifier import MLClassifier


# Each fault type is injected on a known joint (see fault_injection configs).
# Used only to fill FDDResult.affected_joint for operators; not a detection input.
FAULT_JOINT = {
    'bearing_wear': 'shoulder_lift_joint',
    'joint_stiffness': 'shoulder_pan_joint',
    'sensor_noise': 'elbow_joint',
    'none': 'none',
    'unknown': 'unknown',
}


class HybridFDDNode(Node):
    def __init__(self):
        super().__init__('hybrid_fdd_node')

        # Parameters
        self.declare_parameter('model_dir', '')
        self.declare_parameter('window_size', 100)
        self.declare_parameter('detection_mode', 'continuous')
        # Classify every N samples once the window is full. 50 = ~2Hz at 100Hz,
        # matching the 50% window overlap used in training.
        self.declare_parameter('classify_every', 50)

        model_dir = self.get_parameter('model_dir').value
        self.window_size = self.get_parameter('window_size').value
        self.detection_mode = self.get_parameter('detection_mode').value
        self.classify_every = self.get_parameter('classify_every').value

        # Pipeline components
        self.kalman = KalmanFilter(n_joints=6)
        self.extractor = FeatureExtractor(window_size=self.window_size)
        self.classifier = MLClassifier(model_dir)
        if not self.classifier.load_models():
            self.get_logger().error(
                f'Failed to load models from "{model_dir}". Node will run and '
                f'publish residuals, but classification will report "unknown".'
            )

        # State
        self.sample_count = 0
        self.detection_count = 0
        self.process = psutil.Process()
        self.n_cores = psutil.cpu_count() or 1

        # Subscriber: the (possibly degraded) sensor feed from the fault injector
        self.snapshot_sub = self.create_subscription(
            SensorSnapshot, '/degraded_sensor_snapshot',
            self.snapshot_callback, 10
        )

        # Publishers
        self.fdd_result_pub = self.create_publisher(
            FDDResult, '/fdd_result', 10
        )
        self.residuals_pub = self.create_publisher(
            SensorResiduals, '/sensor_residuals', 10
        )
        self.energy_pub = self.create_publisher(
            EnergyMetrics, '/energy_metrics', 10
        )

        self.get_logger().info(
            f'Hybrid FDD node started | mode: {self.detection_mode} | '
            f'window: {self.window_size} | classify every: '
            f'{self.classify_every} samples | models: '
            f'{"loaded" if self.classifier.is_loaded else "MISSING"}'
        )

    def snapshot_callback(self, msg: SensorSnapshot):
        positions = list(msg.joint_positions)
        velocities = list(msg.joint_velocities)
        if len(positions) < 6 or len(velocities) < 6:
            return  # incomplete sample, skip

        # --- Kalman: predict expected positions, get residuals ---
        residuals, predicted = self.kalman.process(positions, velocities)

        # Feed the sliding window (add_sample clips velocities to joint limits,
        # matching training) with the Kalman residuals attached.
        self.extractor.add_sample(msg, residuals)
        self._publish_residuals(residuals, predicted, positions)

        self.sample_count += 1

        # --- Classify once the window is full, at the training cadence ---
        if (self.extractor.is_ready() and
                self.sample_count % self.classify_every == 0):
            self._run_detection(residuals)

    def _run_detection(self, residuals):
        t0 = time.perf_counter()
        features = self.extractor.extract_features()
        if features is None:
            return
        fault_type, confidence, is_anomaly, anomaly_score, severity = \
            self.classifier.classify_continuous(features)
        processing_time_ms = (time.perf_counter() - t0) * 1000.0

        self.detection_count += 1
        self._publish_fdd_result(
            fault_type, confidence, is_anomaly, severity,
            residuals, processing_time_ms
        )
        self._publish_energy_metrics(processing_time_ms)

        self.get_logger().info(
            f'Samples: {self.sample_count} | '
            f'Detections: {self.detection_count} | '
            f'Fault: {fault_type} ({confidence:.2f}) | '
            f'Severity: {severity:.2f} | '
            f'Anomaly: {is_anomaly} | '
            f'Proc: {processing_time_ms:.2f}ms',
            throttle_duration_sec=2.0
        )

    def _publish_residuals(self, residuals, predicted, actual):
        """Publish Kalman residuals (position channel) for Phase 4 cascade."""
        msg = SensorResiduals()
        msg.joint_position_residuals = [float(r) for r in residuals]
        msg.predicted_positions = [float(p) for p in predicted]
        msg.actual_positions = [float(a) for a in actual[:6]]
        msg.timestamp = self.get_clock().now().to_msg()
        self.residuals_pub.publish(msg)

    def _publish_fdd_result(self, fault_type, confidence, is_anomaly,
                            severity, residuals, processing_time_ms):
        """Publish complete FDD result."""
        msg = FDDResult()
        msg.fault_type = fault_type
        msg.affected_joint = FAULT_JOINT.get(fault_type, 'unknown')
        msg.confidence = float(confidence)
        msg.severity_estimate = float(severity)
        msg.anomaly_detected = bool(is_anomaly)
        msg.max_residual = float(np.max(np.abs(residuals)))
        msg.processing_time_ms = float(processing_time_ms)
        msg.detection_mode = self.detection_mode
        msg.timestamp = self.get_clock().now().to_msg()
        self.fdd_result_pub.publish(msg)

    def _publish_energy_metrics(self, processing_time_ms):
        """Log computational cost of this detection cycle (for Phase 5)."""
        # Normalize by core count: psutil sums across cores (can exceed 100%);
        # dividing gives a 0-100% system-wide fraction so energy stays <= 15W.
        cpu = self.process.cpu_percent() / self.n_cores
        mem = self.process.memory_info().rss / 1024 / 1024
        # Energy for this cycle: CPU fraction * processor TDP (15W) * time.
        energy = (cpu / 100.0) * 15.0 * (processing_time_ms / 1000.0)

        msg = EnergyMetrics()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.cpu_usage_percent = float(cpu)
        msg.memory_usage_mb = float(mem)
        msg.estimated_energy_joules = float(energy)
        msg.processing_component = 'hybrid_fdd'
        self.energy_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HybridFDDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
