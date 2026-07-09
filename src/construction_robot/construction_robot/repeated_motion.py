#!/usr/bin/env python3
"""
Continuous Robot Motion for Data Collection
Cycles through poses indefinitely until Ctrl+C.
Designed to exercise full workspace for meaningful FDD sensor data.

Usage:
    Terminal 1: ros2 launch fault_injection fault_experiment.launch.py experiment:=bearing_wear
    Terminal 2 (after robot appears): ros2 run construction_robot repeated_motion
    Terminate Terminal 2 after 300s with Ctrl+C
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import random
import time


# Joint order:
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]

# Home pose: safe neutral position, always return here between large moves
HOME = [0.0, -1.0, 0.5, -1.0, 0.0, 0.0]

# 5 poses that exercise full workspace
# Spread across reach, rotation, and elevation extremes
BASE_POSES = [
    # Pose 1: Extended forward reach, low
    [0.0,   -0.8,  1.2,  -1.8,  0.0,   0.0],
    # Pose 2: Rotated left, mid height
    [1.2,   -1.2,  0.8,  -1.2,  1.0,   0.5],
    # Pose 3: Extended upward reach
    [0.0,   -1.6,  0.3,  -0.5,  0.0,   0.0],
    # Pose 4: Rotated right, low sweep
    [-1.2,  -0.7,  1.4,  -1.6, -1.0,  -0.5],
    # Pose 5: Cross-body reach with wrist rotation
    [0.8,   -1.4,  1.0,  -1.0, -0.8,   1.2],
]

# Movement duration range (seconds) - randomized per move
DURATION_MIN = 2.5
DURATION_MAX = 4.5

# Position jitter: slight randomness added to each pose
# Keeps consecutive runs non-identical without large unsafe jumps
JITTER_STD = 0.04  # radians


class ContinuousMotionNode(Node):
    def __init__(self):
        super().__init__('continuous_motion')

        # Hybrid control: the 3 fault-target joints run on the effort controller,
        # the 3 wrists on the position controller. Each pose is split and sent to
        # both. /joint_states still reports all 6 joints via joint_state_broadcaster.
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_effort_controller/joint_trajectory',
            10
        )
        self.wrist_pub = self.create_publisher(
            JointTrajectory,
            '/wrist_position_controller/joint_trajectory',
            10
        )

        # Joint order for the full 6-element pose arrays below.
        self.arm_joints = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint'
        ]
        self.wrist_joints = [
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.pose_index = 0
        self.controller_ready = False
        self.move_in_progress = False
        self.last_move_time = time.time()
        self.current_duration = 3.0

        # Check controller readiness at 2Hz
        self.ready_timer = self.create_timer(0.5, self.check_controller)

        # Motion timer runs at 10Hz to check if move is complete
        self.motion_timer = self.create_timer(0.1, self.motion_loop)

        self.get_logger().info(
            'Continuous motion node started. '
            'Waiting for trajectory controller...'
        )

    def check_controller(self):
        """Wait for controller to become available"""
        if not self.controller_ready:
            if (self.arm_pub.get_subscription_count() > 0 and
                    self.wrist_pub.get_subscription_count() > 0):
                self.controller_ready = True
                self.get_logger().info(
                    'Controller ready. Moving to home position...'
                )
                # Always start from home
                self.send_pose(HOME, duration_sec=4.0)
                self.last_move_time = time.time()
                self.current_duration = 4.0
                self.ready_timer.cancel()

    def motion_loop(self):
        """
        Check if current move is complete, then send next pose.
        No pause between moves - continuous motion.
        """
        if not self.controller_ready:
            return

        elapsed = time.time() - self.last_move_time

        # Move is complete when duration has passed
        if elapsed >= self.current_duration:
            self.send_next_pose()

    def send_next_pose(self):
        """Send next pose in sequence with jitter and random duration"""
        base_pose = BASE_POSES[self.pose_index]

        # Add slight random jitter to each joint position
        jitter = [
            random.gauss(0, JITTER_STD) for _ in range(6)
        ]
        target_pose = [
            base_pose[i] + jitter[i] for i in range(6)
        ]

        # Random duration within range
        duration = random.uniform(DURATION_MIN, DURATION_MAX)

        self.send_pose(target_pose, duration)

        self.get_logger().info(
            f'Pose {self.pose_index + 1}/5 | '
            f'Duration: {duration:.2f}s | '
            f'Pan: {target_pose[0]:.2f} | '
            f'Lift: {target_pose[1]:.2f} | '
            f'Elbow: {target_pose[2]:.2f}'
        )

        # Advance pose index (cycles 0→1→2→3→4→0→...)
        self.pose_index = (self.pose_index + 1) % len(BASE_POSES)
        self.last_move_time = time.time()
        self.current_duration = duration

    def send_pose(self, positions, duration_sec):
        """Send joint trajectory command.

        The 6-element pose is split: joints 0-2 go to the arm effort controller,
        joints 3-5 go to the wrist position controller, both with the same
        time_from_start so the whole arm moves together.
        """
        # Convert float duration to sec/nanosec (shared by both messages)
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        time_from_start = Duration(sec=sec, nanosec=nanosec)

        arm_point = JointTrajectoryPoint()
        arm_point.positions = list(positions[0:3])
        arm_point.time_from_start = time_from_start
        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joints
        arm_msg.points = [arm_point]
        self.arm_pub.publish(arm_msg)

        wrist_point = JointTrajectoryPoint()
        wrist_point.positions = list(positions[3:6])
        wrist_point.time_from_start = time_from_start
        wrist_msg = JointTrajectory()
        wrist_msg.joint_names = self.wrist_joints
        wrist_msg.points = [wrist_point]
        self.wrist_pub.publish(wrist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ContinuousMotionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()