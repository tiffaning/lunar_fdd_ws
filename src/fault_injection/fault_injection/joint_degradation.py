#!/usr/bin/env python3
"""
Joint Degradation Module
Injects physics-level faults into Gazebo simulation.
Communicates with Gazebo via gazebo_ros services.

Architecture note for Phase 3:
- Fault parameters published to /fault_label
- Physics changes reflected in /joint_states
- Phase 3 Kalman filter will detect residuals between
  predicted (healthy) and actual (degraded) joint states
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import numpy as np


# Joint name to index mapping
JOINT_INDEX = {
    'shoulder_pan_joint': 0,
    'shoulder_lift_joint': 1,
    'elbow_joint': 2,
    'wrist_1_joint': 3,
    'wrist_2_joint': 4,
    'wrist_3_joint': 5
}

# Fault effect definitions
# These modify what the joint state publisher reports
# Phase 3 will compare these against physics model predictions
FAULT_EFFECTS = {
    'bearing_wear': {
        # Increases friction, causes velocity noise
        'friction_multiplier': lambda severity: 1.0 + (severity * 4.0),
        'velocity_noise_std': lambda severity: severity * 0.05,
        'effort_bias': lambda severity: severity * 2.0
    },
    'joint_stiffness': {
        # Reduces range of motion, increases effort
        'position_noise_std': lambda severity: severity * 0.02,
        'effort_multiplier': lambda severity: 1.0 + (severity * 3.0),
        'velocity_damping': lambda severity: severity * 0.3
    },
    'sensor_noise': {
        # Adds noise to sensor readings only
        'position_noise_std': lambda severity: severity * 0.1,
        'velocity_noise_std': lambda severity: severity * 0.08,
        'effort_noise_std': lambda severity: severity * 1.5
    }
}


class JointDegradation:
    def __init__(self, node: Node):
        self.node = node
        self.active_faults = {}
        self.rng = np.random.default_rng(seed=42)  # Reproducible noise

    def compute_degraded_state(self, joint_positions, joint_velocities,
                               joint_efforts, fault_config):
        """
        Apply fault effects to sensor readings.
        Returns modified sensor data that reflects degradation.

        This is what Phase 3 will detect as anomalies:
        - Residual = healthy_prediction - degraded_actual
        """
        fault_type = fault_config['fault_type']
        joint_name = fault_config['affected_joint']
        severity = fault_config['severity']

        if fault_type not in FAULT_EFFECTS:
            return joint_positions, joint_velocities, joint_efforts

        joint_idx = JOINT_INDEX.get(joint_name, 0)
        effects = FAULT_EFFECTS[fault_type]

        # Deep copy to avoid modifying originals
        pos = list(joint_positions)
        vel = list(joint_velocities)
        eff = list(joint_efforts)

        # Apply position noise
        if 'position_noise_std' in effects:
            std = effects['position_noise_std'](severity)
            pos[joint_idx] += self.rng.normal(0, std)

        # Apply velocity noise
        if 'velocity_noise_std' in effects:
            std = effects['velocity_noise_std'](severity)
            vel[joint_idx] += self.rng.normal(0, std)

        # Apply velocity damping
        if 'velocity_damping' in effects:
            damping = effects['velocity_damping'](severity)
            vel[joint_idx] *= (1.0 - damping)

        # Apply effort bias
        if 'effort_bias' in effects:
            bias = effects['effort_bias'](severity)
            eff[joint_idx] += bias

        # Apply effort multiplier
        if 'effort_multiplier' in effects:
            mult = effects['effort_multiplier'](severity)
            eff[joint_idx] *= mult

        return pos, vel, eff

    def compute_severity(self, fault_config, elapsed_time):
        """
        Compute current fault severity based on progression.
        Linear progression from initial to final severity.
        """
        start_time = fault_config['start_time']
        end_time = fault_config['end_time']
        initial = fault_config['initial_severity']
        final = fault_config['final_severity']

        if elapsed_time < start_time:
            return 0.0

        if elapsed_time >= end_time:
            return final

        # Linear interpolation
        progress = (elapsed_time - start_time) / (end_time - start_time)
        return initial + (progress * (final - initial))
