#!/usr/bin/env python3
"""
Performance Monitor Node
Subscribes to: /joint_states, /lunar_robot/imu, /fault_label
Publishes to:  /energy_metrics, /sensor_snapshot
Logs to:       CSV files via DataLogger

Architecture note:
- Fault label subscription syncs ground truth from fault injector
- SensorSnapshot message format is what Phase 3 FDD will consume
- EnergyMetrics format is what Phase 4 cascade will compare against
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from lunar_fdd_interfaces.msg import EnergyMetrics, FaultLabel, SensorSnapshot
import psutil
import time
import sys
import os

# Import shared data logger
sys.path.insert(0, '/home/tiffa/lunar_fdd_ws/src/data_logger')
from data_logger.data_logger import DataLogger


class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        # Declare parameters (configurable from yaml)
        self.declare_parameter('experiment_name', 'baseline')
        self.declare_parameter('log_dir', '/tmp/lunar_fdd_data')
        self.declare_parameter('energy_publish_rate', 10.0)

        experiment_name = self.get_parameter('experiment_name').value
        log_dir = self.get_parameter('log_dir').value

        # Initialize data logger
        self.logger = DataLogger(experiment_name, log_dir)

        # Current state storage
        self.current_joint_state = None
        self.current_imu = {'ax': 0.0, 'ay': 0.0, 'az': 0.0,
                            'wx': 0.0, 'wy': 0.0, 'wz': 0.0}
        self.current_fault_label = FaultLabel()
        self.current_fault_label.fault_type = 'none'
        self.current_fault_label.is_active = False

        # CPU monitoring
        self.process = psutil.Process()

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/lunar_robot/imu',
            self.imu_callback, 10
        )
        # Subscribe to fault labels from fault injector
        # Phase 3/4 will also subscribe to this topic
        self.fault_sub = self.create_subscription(
            FaultLabel, '/fault_label',
            self.fault_callback, 10
        )
        # The degraded snapshot from the fault injector is the source of truth
        # for logging: it carries the fault-modified joint AND imu data (clean
        # when no fault is active). Logging this - not raw /joint_states - is
        # what makes injected faults actually appear in the training CSVs.
        self.degraded_sub = self.create_subscription(
            SensorSnapshot, '/degraded_sensor_snapshot',
            self.degraded_snapshot_callback, 10
        )

        # Publishers
        self.energy_pub = self.create_publisher(
            EnergyMetrics, '/energy_metrics', 10
        )
        # SensorSnapshot - this is what Phase 3 FDD will subscribe to
        self.snapshot_pub = self.create_publisher(
            SensorSnapshot, '/sensor_snapshot', 10
        )

        # Timers
        rate = self.get_parameter('energy_publish_rate').value
        self.energy_timer = self.create_timer(
            1.0 / rate, self.publish_energy_metrics
        )

        self.get_logger().info(
            f'Performance Monitor started | '
            f'Experiment: {experiment_name} | '
            f'Logging to: {log_dir}'
        )

    def joint_callback(self, msg: JointState):
        """Process incoming joint state data.

        Note: logging is driven by degraded_snapshot_callback (which carries the
        fault-modified data), NOT here. This callback only republishes the raw
        snapshot for any consumer that wants the unmodified signal.
        """
        self.current_joint_state = msg

        # Publish raw SensorSnapshot (unmodified) for reference/debug
        self._publish_sensor_snapshot(msg)

    def degraded_snapshot_callback(self, msg: SensorSnapshot):
        """Log the fault-injector's (possibly degraded) snapshot to CSV.

        This is the sensor data the Phase 3 models train on: fault-modified when
        a fault is active, clean otherwise. Joint and IMU data + the fault label
        all come from this single message, so they are consistent by construction.
        """
        imu = {
            'ax': msg.imu_linear_accel_x,
            'ay': msg.imu_linear_accel_y,
            'az': msg.imu_linear_accel_z,
            'wx': msg.imu_angular_vel_x,
            'wy': msg.imu_angular_vel_y,
            'wz': msg.imu_angular_vel_z,
        }
        self.logger.log_sensor_data(
            joint_positions=list(msg.joint_positions),
            joint_velocities=list(msg.joint_velocities),
            joint_efforts=list(msg.joint_efforts),
            imu_data=imu,
            fault_active=msg.fault_active,
            fault_type=msg.fault_type
        )

    def imu_callback(self, msg: Imu):
        """Process IMU data"""
        self.current_imu = {
            'ax': msg.linear_acceleration.x,
            'ay': msg.linear_acceleration.y,
            'az': msg.linear_acceleration.z,
            'wx': msg.angular_velocity.x,
            'wy': msg.angular_velocity.y,
            'wz': msg.angular_velocity.z
        }

    def fault_callback(self, msg: FaultLabel):
        """
        Sync fault ground truth from fault injector.
        This allows monitor to label all sensor data correctly.
        """
        self.current_fault_label = msg
        self.logger.log_fault_event(
            fault_type=msg.fault_type,
            affected_joint=msg.affected_joint,
            severity=msg.severity,
            progression_rate=msg.progression_rate,
            is_active=msg.is_active
        )
        self.get_logger().info(
            f'Fault update: {msg.fault_type} | '
            f'severity: {msg.severity:.2f} | '
            f'active: {msg.is_active}'
        )

    def _publish_sensor_snapshot(self, joint_msg: JointState):
        """
        Publish complete sensor snapshot.
        Phase 3 FDD system subscribes to /sensor_snapshot.
        """
        snapshot = SensorSnapshot()
        snapshot.joint_positions = list(joint_msg.position)
        snapshot.joint_velocities = list(joint_msg.velocity)
        snapshot.joint_efforts = list(joint_msg.effort)
        snapshot.imu_linear_accel_x = self.current_imu['ax']
        snapshot.imu_linear_accel_y = self.current_imu['ay']
        snapshot.imu_linear_accel_z = self.current_imu['az']
        snapshot.imu_angular_vel_x = self.current_imu['wx']
        snapshot.imu_angular_vel_y = self.current_imu['wy']
        snapshot.imu_angular_vel_z = self.current_imu['wz']
        snapshot.timestamp_sec = time.time()
        snapshot.fault_active = self.current_fault_label.is_active
        snapshot.fault_type = self.current_fault_label.fault_type
        self.snapshot_pub.publish(snapshot)

    def publish_energy_metrics(self):
        """Publish and log energy/computational metrics"""
        cpu = self.process.cpu_percent()
        mem = self.process.memory_info().rss / 1024 / 1024
        # Energy estimate: CPU% * processor TDP (15W) * time interval
        energy = (cpu / 100.0) * 15.0 * (1.0 / 10.0)

        # Publish
        msg = EnergyMetrics()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.cpu_usage_percent = cpu
        msg.memory_usage_mb = mem
        msg.estimated_energy_joules = energy
        msg.processing_component = 'performance_monitor'
        self.energy_pub.publish(msg)

        # Log - detection_layer is 'none' until Phase 4
        self.logger.log_energy_metrics(
            cpu_percent=cpu,
            memory_mb=mem,
            energy_joules=energy,
            component='performance_monitor',
            detection_layer='none'
        )


def main(args=None):
    rclpy.init(args=args)
    monitor = PerformanceMonitor()
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
