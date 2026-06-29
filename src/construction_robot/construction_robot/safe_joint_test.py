#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import sys

class SafeJointTester(Node):
    def __init__(self):
        super().__init__('safe_joint_tester')
        
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Wait for controller to be ready
        self.timer = self.create_timer(1.0, self.check_controller)
        self.controller_ready = False
        
    def check_controller(self):
        if not self.controller_ready:
            self.get_logger().info("Waiting for trajectory controller...")
            if self.publisher.get_subscription_count() > 0:
                self.controller_ready = True
                self.get_logger().info("Controller ready! Moving to safe position...")
                self.move_to_safe_position()
                self.timer.cancel()

    def move_to_safe_position(self):
        """Move robot to a safe, known position"""
        msg = JointTrajectory()
        msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Safe "home" position (slight bend to avoid singularities)
        point = JointTrajectoryPoint()
        point.positions = [0.0, -1.0, 0.5, -1.0, 0.0, 0.0]
        point.time_from_start = Duration(sec=5, nanosec=0)  # Slow movement
        
        msg.points = [point]
        
        self.publisher.publish(msg)
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
