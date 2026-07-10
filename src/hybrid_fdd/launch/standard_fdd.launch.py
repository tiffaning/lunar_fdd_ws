import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Which experiment config the fault injector should run
    experiment_arg = DeclareLaunchArgument(
        'experiment',
        default_value='baseline',
        description='Experiment config name (baseline, bearing_wear, etc.)'
    )
    experiment = LaunchConfiguration('experiment')

    hybrid_fdd_pkg = get_package_share_directory('hybrid_fdd')
    fault_injection_pkg = get_package_share_directory('fault_injection')

    # Robot + Gazebo environment
    robot = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory('construction_robot'),
            'launch', 'lunar_robot.launch.py'
        )
    )

    # Performance monitor (energy baseline + CSV logging for Phase 5 comparison)
    monitor = TimerAction(period=6.0, actions=[
        Node(
            package='performance_monitor',
            executable='monitor_node',
            name='performance_monitor',
            parameters=[{
                'experiment_name': experiment,
                'log_dir': os.path.expanduser(
                    '~/lunar_fdd_ws/data/phase3'
                )
            }],
            output='screen'
        )
    ])

    # Fault injector - publishes /degraded_sensor_snapshot + /fault_label
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

    # Standard hybrid FDD - continuous mode (full pipeline every cycle)
    hybrid_fdd = TimerAction(period=10.0, actions=[
        Node(
            package='hybrid_fdd',
            executable='hybrid_fdd_node',
            name='hybrid_fdd_node',
            parameters=[{
                'model_dir': os.path.join(hybrid_fdd_pkg, 'models'),
                'window_size': 100,
                'detection_mode': 'continuous'
            }],
            output='screen'
        )
    ])

    return LaunchDescription([
        experiment_arg,
        robot,
        monitor,
        fault_injector,
        hybrid_fdd,
    ])
