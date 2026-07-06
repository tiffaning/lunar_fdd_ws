import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Launch argument: which experiment config to use
    experiment_arg = DeclareLaunchArgument(
        'experiment',
        default_value='baseline',
        description='Experiment config name (baseline, bearing_wear, etc.)'
    )

    experiment = LaunchConfiguration('experiment')

    fault_injection_pkg = get_package_share_directory('fault_injection')
    performance_monitor_pkg = get_package_share_directory('performance_monitor')

    # Include robot launch
    robot_launch = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory('construction_robot'),
            'launch',
            'lunar_robot.launch.py'
        )
    )

    # Performance monitor - starts after robot
    monitor_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='performance_monitor',
                executable='monitor_node',
                name='performance_monitor',
                parameters=[{
                    'experiment_name': experiment,
		    'log_dir': '/home/tiffa/lunar_fdd_ws/data/raw'
                }],
                output='screen'
            )
        ]
    )

    # Fault injector - starts after monitor
    fault_injector_node = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='fault_injection',
                executable='fault_injector',
                name='fault_injector',
                parameters=[{
                    'config_file': [
                        fault_injection_pkg,
                        '/config/',
                        experiment,
                        '_experiment.yaml'
                    ],
                    'auto_start': True
                }],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        experiment_arg,
        robot_launch,
        monitor_node,
        fault_injector_node,
    ])
