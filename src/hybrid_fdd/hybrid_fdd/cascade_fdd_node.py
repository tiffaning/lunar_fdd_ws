#!/usr/bin/env python3
"""
Model Cascade FDD Node (Phase 4) - energy-efficient staged fault detection.

Runs the same input as the continuous hybrid node (/degraded_sensor_snapshot),
but escalates through three layers of increasing cost, stopping as soon as a
layer is confident enough. Layer 3 is the full hybrid FDD (Phase 3).

  Layer 1 - statistical health screen (~0.1 ms): cheap stats on rolling buffers
            (max Kalman residual + per-joint effort RMS vs healthy baselines).
            Confident-healthy -> report 'none' and STOP; else escalate.
  Layer 2 - supervised decision tree (~0.1-2 ms): classifies on the fault-bearing
            feature subset. Confidence >= l2_threshold -> report and STOP; else escalate.
  Layer 3 - full hybrid (~15-25 ms): Kalman + IF + SVM + severity regressor.

Gating (spec): escalate Layer 1 -> 2 when normality confidence < l1_threshold;
escalate Layer 2 -> 3 when tree confidence < l2_threshold. Both are ROS params
so Phase 5 can grid-search them.

Publishes:
  /fdd_result     (FDDResult)     - same schema as the continuous node, so the
                                    Phase 5 evaluator works unchanged.
  /fault_status   (FaultStatus)   - which layer decided + per-detection compute time.
  /energy_metrics (EnergyMetrics) - processing_component='cascade'.

Design notes:
- Reuses KalmanFilter / FeatureExtractor / MLClassifier unchanged, so Layer 3 is
  bit-identical to the continuous node. Toggle continuous vs cascade by launching
  the respective node; both are directly comparable for Phase 5.
- Layer 1/2 stops do not produce a severity estimate (that is a Layer 3 output);
  severity_estimate is 0.0 unless the cascade reaches Layer 3.
"""
import rclpy
from rclpy.node import Node
from lunar_fdd_interfaces.msg import (
    SensorSnapshot, FDDResult, SensorResiduals, EnergyMetrics, FaultStatus
)
import numpy as np
import joblib
import os
import time
from collections import deque

from hybrid_fdd.kalman_filter import KalmanFilter
from hybrid_fdd.feature_extractor import FeatureExtractor
from hybrid_fdd.ml_classifier import MLClassifier


FAULT_JOINT = {
    'bearing_wear': 'shoulder_lift_joint',
    'joint_stiffness': 'shoulder_pan_joint',
    'sensor_noise': 'elbow_joint',
    'none': 'none',
    'unknown': 'unknown',
}


class CascadeFDDNode(Node):
    def __init__(self):
        super().__init__('cascade_fdd_node')

        # Parameters
        self.declare_parameter('model_dir', '')
        self.declare_parameter('window_size', 100)
        self.declare_parameter('classify_every', 50)
        # Escalation thresholds (grid-searched in Phase 5). Offline sweep shows
        # the useful Layer-1 operating point is ~0.9 (stop only when very
        # confident healthy); 0.3 stopped on nearly everything and missed faults.
        self.declare_parameter('l1_threshold', 0.9)   # normality conf < this -> L2
        self.declare_parameter('l2_threshold', 0.8)   # tree conf < this -> L3

        model_dir = self.get_parameter('model_dir').value
        self.window_size = self.get_parameter('window_size').value
        self.classify_every = self.get_parameter('classify_every').value
        self.l1_threshold = self.get_parameter('l1_threshold').value
        self.l2_threshold = self.get_parameter('l2_threshold').value

        # Pipeline components (Layer 3 == the continuous hybrid, reused verbatim)
        self.kalman = KalmanFilter(n_joints=6)
        self.extractor = FeatureExtractor(window_size=self.window_size)
        self.classifier = MLClassifier(model_dir)
        if not self.classifier.load_models():
            self.get_logger().error(
                f'Failed to load models from "{model_dir}".'
            )

        # Layer 1 healthy baselines + cheap rolling buffers
        self.layer1_baseline = None
        try:
            self.layer1_baseline = joblib.load(
                os.path.join(model_dir, 'layer1_baseline.pkl')
            )
        except (FileNotFoundError, OSError):
            self.get_logger().warn(
                'layer1_baseline.pkl not found; Layer 1 will always escalate.'
            )
        self.l1_resid = deque(maxlen=self.window_size)   # per-sample max|residual|
        self.l1_eff = deque(maxlen=self.window_size)     # per-sample effort (6,)

        # State + per-layer counters (Phase 5 accounting)
        self.sample_count = 0
        self.detection_count = 0
        self.layer_counts = {1: 0, 2: 0, 3: 0}
        try:
            import psutil
            self.process = psutil.Process()
        except Exception:
            self.process = None

        # I/O
        self.snapshot_sub = self.create_subscription(
            SensorSnapshot, '/degraded_sensor_snapshot',
            self.snapshot_callback, 10
        )
        self.fdd_result_pub = self.create_publisher(
            FDDResult, '/fdd_result', 10
        )
        self.status_pub = self.create_publisher(
            FaultStatus, '/fault_status', 10
        )
        self.residuals_pub = self.create_publisher(
            SensorResiduals, '/sensor_residuals', 10
        )
        self.energy_pub = self.create_publisher(
            EnergyMetrics, '/energy_metrics', 10
        )

        self.get_logger().info(
            f'Cascade FDD node started | L1 thr {self.l1_threshold} | '
            f'L2 thr {self.l2_threshold} | window {self.window_size} | '
            f'models: {"loaded" if self.classifier.is_loaded else "MISSING"}'
        )

    def snapshot_callback(self, msg: SensorSnapshot):
        positions = list(msg.joint_positions)
        velocities = list(msg.joint_velocities)
        efforts = list(msg.joint_efforts)
        if len(positions) < 6 or len(velocities) < 6:
            return
        efforts = (efforts + [0.0] * 6)[:6]

        residuals, predicted = self.kalman.process(positions, velocities)
        self.extractor.add_sample(msg, residuals)

        # Cheap Layer 1 rolling buffers
        self.l1_resid.append(float(np.max(np.abs(residuals))))
        self.l1_eff.append(efforts)
        self._publish_residuals(residuals, predicted, positions)

        self.sample_count += 1
        if (self.extractor.is_ready() and
                self.sample_count % self.classify_every == 0):
            self._run_cascade(residuals)

    # --- Layer 1: cheap statistical health screen ---
    def _layer1_normality_confidence(self):
        """Confidence (0-1) that the current window is healthy. Cheap: only a
        max-residual and per-joint effort-RMS comparison to healthy baselines."""
        if (self.layer1_baseline is None or
                len(self.l1_resid) < self.window_size):
            return 0.0  # cannot screen -> force escalation (safe default)
        max_resid = max(self.l1_resid)
        eff = np.asarray(self.l1_eff)                      # (window, 6)
        eff_rms = np.sqrt(np.mean(eff ** 2, axis=0))       # (6,)
        resid_term = max_resid / self.layer1_baseline['resid_scale']
        mean = np.asarray(self.layer1_baseline['eff_rms_mean'])
        std = np.asarray(self.layer1_baseline['eff_rms_std'])
        effort_term = float(np.max(np.abs(eff_rms - mean) / std) / 4.0)  # 4 sigma
        anomaly = max(resid_term, effort_term)
        return float(np.clip(1.0 - anomaly, 0.0, 1.0))

    def _run_cascade(self, residuals):
        t0 = time.perf_counter()
        layer_used = 1
        severity = 0.0
        is_anomaly = False

        # --- Layer 1 ---
        conf_none = self._layer1_normality_confidence()
        if conf_none >= self.l1_threshold:
            fault_type, confidence = 'none', conf_none
        else:
            # --- Layer 2 ---
            layer_used = 2
            features = self.extractor.extract_features()
            fault_type, confidence = self.classifier.classify_tree(features)
            is_anomaly = fault_type != 'none'
            if confidence < self.l2_threshold:
                # --- Layer 3 (full hybrid) ---
                layer_used = 3
                (fault_type, confidence, is_anomaly,
                 _anom_score, severity) = \
                    self.classifier.classify_continuous(features)
            elif fault_type != 'none':
                # Fault accepted at Layer 2: add a severity estimate so the
                # cascade reports severity (for Phase 5), not just Layer 3.
                severity = self.classifier.predict_severity(features)

        processing_time_ms = (time.perf_counter() - t0) * 1000.0
        self.detection_count += 1
        self.layer_counts[layer_used] += 1

        self._publish_fdd_result(fault_type, confidence, is_anomaly,
                                 severity, residuals, processing_time_ms)
        self._publish_status(fault_type, confidence, layer_used,
                             processing_time_ms)
        self._publish_energy_metrics(processing_time_ms)

        self.get_logger().info(
            f'Det {self.detection_count} | L{layer_used} | '
            f'{fault_type} ({confidence:.2f}) | sev {severity:.2f} | '
            f'{processing_time_ms:.2f}ms | '
            f'L1/L2/L3={self.layer_counts[1]}/{self.layer_counts[2]}/'
            f'{self.layer_counts[3]}',
            throttle_duration_sec=2.0
        )

    def _publish_residuals(self, residuals, predicted, actual):
        msg = SensorResiduals()
        msg.joint_position_residuals = [float(r) for r in residuals]
        msg.predicted_positions = [float(p) for p in predicted]
        msg.actual_positions = [float(a) for a in actual[:6]]
        msg.timestamp = self.get_clock().now().to_msg()
        self.residuals_pub.publish(msg)

    def _publish_fdd_result(self, fault_type, confidence, is_anomaly,
                            severity, residuals, processing_time_ms):
        msg = FDDResult()
        msg.fault_type = fault_type
        msg.affected_joint = FAULT_JOINT.get(fault_type, 'unknown')
        msg.confidence = float(confidence)
        msg.severity_estimate = float(severity)
        msg.anomaly_detected = bool(is_anomaly)
        msg.max_residual = float(np.max(np.abs(residuals)))
        msg.processing_time_ms = float(processing_time_ms)
        msg.detection_mode = 'cascade'
        msg.timestamp = self.get_clock().now().to_msg()
        self.fdd_result_pub.publish(msg)

    def _publish_status(self, fault_type, confidence, layer_used,
                        processing_time_ms):
        """Which layer produced the answer + its compute cost (Phase 5)."""
        msg = FaultStatus()
        msg.fault_type = fault_type
        msg.confidence_level = float(confidence)
        msg.detection_layer_used = int(layer_used)
        msg.computation_time_ms = float(processing_time_ms)
        msg.timestamp = self.get_clock().now().to_msg()
        self.status_pub.publish(msg)

    def _publish_energy_metrics(self, processing_time_ms):
        cpu = self.process.cpu_percent() if self.process else 0.0
        mem = (self.process.memory_info().rss / 1024 / 1024
               if self.process else 0.0)
        energy = (cpu / 100.0) * 15.0 * (processing_time_ms / 1000.0)
        msg = EnergyMetrics()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.cpu_usage_percent = float(cpu)
        msg.memory_usage_mb = float(mem)
        msg.estimated_energy_joules = float(energy)
        msg.processing_component = 'cascade'
        self.energy_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CascadeFDDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
