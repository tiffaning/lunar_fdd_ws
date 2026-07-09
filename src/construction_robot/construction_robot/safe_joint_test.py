#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import sys

class SafeJointTester(Node):
    def __init__(self):
        super().__init__('safe_joint_tester')
        
        # Hybrid control: arm joints on the effort controller, wrists on the
        # position controller (see ros2_controllers.yaml).
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

        # Wait for controller to be ready
        self.timer = self.create_timer(1.0, self.check_controller)
        self.controller_ready = False

    def check_controller(self):
        if not self.controller_ready:
            self.get_logger().info("Waiting for trajectory controllers...")
            if (self.arm_pub.get_subscription_count() > 0 and
                    self.wrist_pub.get_subscription_count() > 0):
                self.controller_ready = True
                self.get_logger().info("Controller ready! Moving to safe position...")
                self.move_to_safe_position()
                self.timer.cancel()

    def move_to_safe_position(self):
        """Move robot to a safe, known position"""
        # Safe "home" position (slight bend to avoid singularities)
        # Full pose: [pan, lift, elbow, wrist_1, wrist_2, wrist_3]
        home = [0.0, -1.0, 0.5, -1.0, 0.0, 0.0]
        time_from_start = Duration(sec=5, nanosec=0)  # Slow movement

        arm_point = JointTrajectoryPoint()
        arm_point.positions = home[0:3]
        arm_point.time_from_start = time_from_start
        arm_msg = JointTrajectory()
        arm_msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                               'elbow_joint']
        arm_msg.points = [arm_point]
        self.arm_pub.publish(arm_msg)

        wrist_point = JointTrajectoryPoint()
        wrist_point.positions = home[3:6]
        wrist_point.time_from_start = time_from_start
        wrist_msg = JointTrajectory()
        wrist_msg.joint_names = ['wrist_1_joint', 'wrist_2_joint',
                                 'wrist_3_joint']
        wrist_msg.points = [wrist_point]
        self.wrist_pub.publish(wrist_msg)
        self.get_logger().info("Moving to safe home position...")
        
        # Shutdown after sending command
        self.create_timer(1.0, lambda: rclpy.shutdown())

def main(args=None):
    rclpy.init(args=args)
    tester = SafeJointTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
