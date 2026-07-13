import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    experiment_arg = DeclareLaunchArgument(
        'experiment',
        default_value='baseline',
        description='Experiment config name (baseline, bearing_wear, etc.)'
    )
    l1_arg = DeclareLaunchArgument('l1_threshold', default_value='0.9')
    l2_arg = DeclareLaunchArgument('l2_threshold', default_value='0.8')
    experiment = LaunchConfiguration('experiment')
    l1_threshold = LaunchConfiguration('l1_threshold')
    l2_threshold = LaunchConfiguration('l2_threshold')

    hybrid_fdd_pkg = get_package_share_directory('hybrid_fdd')
    fault_injection_pkg = get_package_share_directory('fault_injection')

    robot = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory('construction_robot'),
            'launch', 'lunar_robot.launch.py'
        )
    )

    # Performance monitor (energy baseline + CSV logging for Phase 5)
    monitor = TimerAction(period=6.0, actions=[
        Node(
            package='performance_monitor',
            executable='monitor_node',
            name='performance_monitor',
            parameters=[{
                'experiment_name': experiment,
                'log_dir': os.path.expanduser('~/lunar_fdd_ws/data/phase4')
            }],
            output='screen'
        )
    ])

    # Fault injector
    fault_injector = TimerAction(period=8.0, actions=[
        Node(
            package='fault_injection',
            executable='fault_injector',
            name='fault_injector',
            parameters=[{
                'config_file': [
                    fault_injection_pkg, '/config/',
                    experiment, '_experiment.yaml'
                ],
                'auto_start': True
            }],
            output='screen'
        )
    ])

    # Model cascade - staged detection (Layer 1 -> 2 -> 3)
    cascade = TimerAction(period=10.0, actions=[
        Node(
            package='hybrid_fdd',
            executable='cascade_fdd_node',
            name='cascade_fdd_node',
            parameters=[{
                'model_dir': os.path.join(hybrid_fdd_pkg, 'models'),
                'window_size': 100,
                'l1_threshold': l1_threshold,
                'l2_threshold': l2_threshold
            }],
            output='screen'
        )
    ])

    return LaunchDescription([
        experiment_arg,
        l1_arg,
        l2_arg,
        robot,
        monitor,
        fault_injector,
        cascade,
    ])
