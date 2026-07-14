#!/usr/bin/env python3
"""
Phase 5 evaluation logger.

Subscribes to the detection outputs, ground-truth labels, and energy metrics and
writes ONE synchronized CSV row per detection. One file per run; the offline
analysis (evaluate_phase5.py) reads all of them.

Works for BOTH strategies:
  - continuous (standard_fdd): detection_mode='continuous', layer=0
  - cascade (cascade_fdd):     detection_mode='cascade', layer=1/2/3 from /fault_status

Ground truth is taken as the most recent /fault_label at each detection (the
labels are time-aligned with the sensor stream by construction), which realizes
the +/-1s matching the methodology calls for at the ~2Hz detection cadence.
"""
import rclpy
from rclpy.node import Node
from lunar_fdd_interfaces.msg import (
    FDDResult, FaultStatus, FaultLabel, EnergyMetrics
)
import csv
import os
import time
from datetime import datetime


class FDDEvaluatorNode(Node):
    def __init__(self):
        super().__init__('fdd_evaluator_node')
        self.declare_parameter('strategy', 'continuous')     # or 'cascade'
        self.declare_parameter('experiment_name', 'baseline')
        self.declare_parameter('log_dir', os.path.expanduser(
            '~/lunar_fdd_ws/data/phase5'))

        strategy = self.get_parameter('strategy').value
        experiment = self.get_parameter('experiment_name').value
        log_dir = self.get_parameter('log_dir').value
        os.makedirs(log_dir, exist_ok=True)

        sid = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(
            log_dir, f'eval_{strategy}_{experiment}_{sid}.csv')
        self.start_time = time.time()

        # Latest ground truth / layer status / energy (attached to each detection)
        self.gt_type = 'none'
        self.gt_sev = 0.0
        self.gt_active = False
        self.layer = 0
        self.comp_ms = 0.0
        self.cpu = 0.0
        self.mem = 0.0
        self.energy = 0.0

        with open(self.path, 'w', newline='') as f:
            csv.writer(f).writerow([
                't_rel', 'wall', 'det_fault_type', 'confidence',
                'severity_est', 'anomaly_detected', 'proc_ms', 'detection_mode',
                'layer', 'comp_ms', 'gt_fault_type', 'gt_severity',
                'gt_is_active', 'cpu_percent', 'mem_mb', 'energy_j'
            ])

        self.create_subscription(FaultLabel, '/fault_label', self.cb_gt, 10)
        self.create_subscription(
            FaultStatus, '/fault_status', self.cb_status, 10)
        self.create_subscription(
            EnergyMetrics, '/energy_metrics', self.cb_energy, 10)
        # /fdd_result is the trigger: one logged row per detection
        self.create_subscription(FDDResult, '/fdd_result', self.cb_result, 10)

        self.get_logger().info(
            f'Evaluator logging {strategy}/{experiment} -> {self.path}')

    def cb_gt(self, m):
        self.gt_type = m.fault_type
        self.gt_sev = m.severity
        self.gt_active = m.is_active

    def cb_status(self, m):
        self.layer = m.detection_layer_used
        self.comp_ms = m.computation_time_ms

    def cb_energy(self, m):
        # Only the FDD node's own energy (not the performance monitor's)
        if m.processing_component in ('hybrid_fdd', 'cascade'):
            self.cpu = m.cpu_usage_percent
            self.mem = m.memory_usage_mb
            self.energy = m.estimated_energy_joules

    def cb_result(self, m):
        t = time.time()
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow([
                round(t - self.start_time, 3), t, m.fault_type, m.confidence,
                m.severity_estimate, m.anomaly_detected, m.processing_time_ms,
                m.detection_mode, self.layer, self.comp_ms,
                self.gt_type, self.gt_sev, self.gt_active,
                self.cpu, self.mem, self.energy
            ])


def main(args=None):
    rclpy.init(args=args)
    node = FDDEvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
