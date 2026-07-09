#!/usr/bin/env python3
"""
Fault Injector Node - Main controller for fault injection experiments.

Publishes: /fault_label (FaultLabel) - ground truth for Phase 3/4
Subscribes: /joint_states - to read current robot state
Modifies: sensor data via degradation modules

Architecture note:
- /fault_label topic is the ground truth signal
- Performance monitor already subscribes to this
- Phase 3 FDD will use these labels for training/evaluation
- Phase 4 cascade will use detection_layer field in energy metrics
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from lunar_fdd_interfaces.msg import FaultLabel, SensorSnapshot
import yaml
import time
import os

from fault_injection.joint_degradation import JointDegradation
from fault_injection.sensor_faults import SensorFaults


class FaultInjectorNode(Node):
    def __init__(self):
        super().__init__('fault_injector')

        # Parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('auto_start', True)

        config_file = self.get_parameter('config_file').value

        # Load experiment config
        self.config = self._load_config(config_file)
        self.experiment_name = self.config['experiment']['name']
        self.experiment_duration = self.config['experiment']['duration_seconds']
        self.faults_enabled = self.config['faults']['enabled']
        self.fault_sequence = self.config['faults'].get('sequence', [])

        # Fault modules
        self.joint_degradation = JointDegradation(self)
        self.sensor_faults = SensorFaults()

        # State tracking
        self.experiment_start_time = None
        self.current_joint_state = None
        self.current_fault_config = None
        self.experiment_running = False
        # Latest IMU reading, degraded into the snapshot for sensor faults
        self.current_imu = {'ax': 0.0, 'ay': 0.0, 'az': 0.0,
                            'wx': 0.0, 'wy': 0.0, 'wz': 0.0}

        # Publishers
        self.fault_label_pub = self.create_publisher(
            FaultLabel, '/fault_label', 10
        )
        # Modified sensor snapshot - this is what Phase 3 FDD receives
        self.degraded_snapshot_pub = self.create_publisher(
            SensorSnapshot, '/degraded_sensor_snapshot', 10
        )

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_callback, 10
        )
        # IMU is degraded (EMI noise) for sensor faults before being logged
        self.imu_sub = self.create_subscription(
            Imu, '/lunar_robot/imu',
            self.imu_callback, 10
        )

        # Timers
        # Fault update at 10Hz (enough for progressive degradation)
        self.fault_timer = self.create_timer(0.1, self.update_fault_state)

        # Auto start experiment
        if self.get_parameter('auto_start').value:
            self.experiment_start_time = time.time()
            self.experiment_running = True
            self.get_logger().info(
                f'Experiment started: {self.experiment_name} | '
                f'Duration: {self.experiment_duration}s | '
                f'Faults enabled: {self.faults_enabled}'
            )

    def _load_config(self, config_file: str) -> dict:
        """Load experiment configuration from YAML"""
        if not config_file or not os.path.exists(config_file):
            self.get_logger().warn(
                f'Config file not found: {config_file}. '
                f'Using baseline (no faults).'
            )
            return {
                'experiment': {
                    'name': 'baseline',
                    'duration_seconds': 300.0
                },
                'faults': {
                    'enabled': False,
                    'sequence': []
                }
            }

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            self.get_logger().info(
                f'Loaded config: {config_file}'
            )
            return config

    def imu_callback(self, msg: Imu):
        """Store latest IMU reading"""
        self.current_imu = {
            'ax': msg.linear_acceleration.x,
            'ay': msg.linear_acceleration.y,
            'az': msg.linear_acceleration.z,
            'wx': msg.angular_velocity.x,
            'wy': msg.angular_velocity.y,
            'wz': msg.angular_velocity.z,
        }

    def joint_callback(self, msg: JointState):
        """Store latest joint state"""
        self.current_joint_state = msg

        # If fault is active, publish degraded sensor data
        if (self.experiment_running and
                self.current_fault_config is not None and
                self.current_fault_config['severity'] > 0.0):
            self._publish_degraded_snapshot(msg)
        else:
            # No fault - publish clean snapshot
            self._publish_clean_snapshot(msg)

    def update_fault_state(self):
        """
        Main fault state machine.
        Called at 10Hz to update fault severity and publish labels.
        """
        if not self.experiment_running:
            return

        elapsed = time.time() - self.experiment_start_time

        # Check experiment duration
        if elapsed >= self.experiment_duration:
            self._end_experiment()
            return

        # Process fault sequence
        if not self.faults_enabled or not self.fault_sequence:
            self._publish_fault_label(
                fault_type='none',
                affected_joint='none',
                severity=0.0,
                progression_rate=0.0,
                is_active=False
            )
            return

        # Find active fault in sequence
        active_fault = None
        for fault in self.fault_sequence:
            if fault['start_time'] <= elapsed <= fault['end_time']:
                active_fault = fault
                break

        if active_fault is None:
            # No fault currently scheduled
            self.current_fault_config = None
            self._publish_fault_label(
                fault_type='none',
                affected_joint='none',
                severity=0.0,
                progression_rate=0.0,
                is_active=False
            )
            return

        # Compute current severity
        severity = self.joint_degradation.compute_severity(
            active_fault, elapsed
        )

        self.current_fault_config = {
            'fault_type': active_fault['fault_type'],
            'affected_joint': active_fault['affected_joint'],
            'severity': severity,
            'start_time': active_fault['start_time'],
            'end_time': active_fault['end_time'],
            'initial_severity': active_fault['initial_severity'],
            'final_severity': active_fault['final_severity']
        }

        # Compute progression rate
        duration = active_fault['end_time'] - active_fault['start_time']
        progression_rate = (
            active_fault['final_severity'] -
            active_fault['initial_severity']
        ) / duration

        # Publish ground truth fault label
        self._publish_fault_label(
            fault_type=active_fault['fault_type'],
            affected_joint=active_fault['affected_joint'],
            severity=severity,
            progression_rate=progression_rate,
            is_active=True
        )

        self.get_logger().info(
            f'Fault: {active_fault["fault_type"]} | '
            f'Joint: {active_fault["affected_joint"]} | '
            f'Severity: {severity:.3f} | '
            f'Elapsed: {elapsed:.1f}s',
            throttle_duration_sec=5.0
        )

    def _publish_fault_label(self, fault_type, affected_joint,
                             severity, progression_rate, is_active):
        """Publish ground truth fault label"""
        msg = FaultLabel()
        msg.fault_type = fault_type
        msg.affected_joint = affected_joint
        msg.severity = severity
        msg.progression_rate = progression_rate
        msg.is_active = is_active
        msg.timestamp = self.get_clock().now().to_msg()
        self.fault_label_pub.publish(msg)

    def _publish_degraded_snapshot(self, joint_msg: JointState):
        """
        Apply fault effects and publish degraded sensor data.
        Phase 3 FDD subscribes to /degraded_sensor_snapshot.
        """
        if self.current_fault_config is None:
            return

        pos, vel, eff = self.joint_degradation.compute_degraded_state(
            joint_msg.position,
            joint_msg.velocity,
            joint_msg.effort,
            self.current_fault_config
        )

        # Degrade IMU too: sensor_noise faults inject EMI noise into the IMU.
        # Mechanical faults (bearing_wear/joint_stiffness) leave the IMU clean;
        # their signature lives on the joints. This gives each fault type a
        # distinct multi-channel signature for the Phase 3 classifier.
        imu = dict(self.current_imu)
        if self.current_fault_config['fault_type'] == 'sensor_noise':
            imu = self.sensor_faults.apply_imu_noise(
                imu, self.current_fault_config['severity']
            )

        snapshot = SensorSnapshot()
        snapshot.joint_positions = pos
        snapshot.joint_velocities = vel
        snapshot.joint_efforts = eff
        snapshot.joint_accelerations = []
        self._set_snapshot_imu(snapshot, imu)
        snapshot.fault_active = True
        snapshot.fault_type = self.current_fault_config['fault_type']
        snapshot.timestamp_sec = time.time()

        self.degraded_snapshot_pub.publish(snapshot)

    def _publish_clean_snapshot(self, joint_msg: JointState):
        """Publish unmodified sensor data"""
        snapshot = SensorSnapshot()
        snapshot.joint_positions = list(joint_msg.position)
        snapshot.joint_velocities = list(joint_msg.velocity)
        snapshot.joint_efforts = list(joint_msg.effort)
        snapshot.joint_accelerations = []
        self._set_snapshot_imu(snapshot, self.current_imu)
        snapshot.fault_active = False
        snapshot.fault_type = 'none'
        snapshot.timestamp_sec = time.time()
        self.degraded_snapshot_pub.publish(snapshot)

    @staticmethod
    def _set_snapshot_imu(snapshot: SensorSnapshot, imu: dict):
        """Copy an IMU dict (ax/ay/az/wx/wy/wz) into a SensorSnapshot"""
        snapshot.imu_linear_accel_x = imu['ax']
        snapshot.imu_linear_accel_y = imu['ay']
        snapshot.imu_linear_accel_z = imu['az']
        snapshot.imu_angular_vel_x = imu['wx']
        snapshot.imu_angular_vel_y = imu['wy']
        snapshot.imu_angular_vel_z = imu['wz']

    def _end_experiment(self):
        """Clean shutdown at experiment end"""
        self.experiment_running = False
        self._publish_fault_label(
            fault_type='none',
            affected_joint='none',
            severity=0.0,
            progression_rate=0.0,
            is_active=False
        )
        self.get_logger().info(
            f'Experiment complete: {self.experiment_name}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FaultInjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
