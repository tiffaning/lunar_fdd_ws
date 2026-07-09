import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Get package directories
    construction_robot_pkg = get_package_share_directory('construction_robot')
    lunar_env_pkg = get_package_share_directory('lunar_environment')

    # Path to xacro file
    urdf_xacro_path = os.path.join(
        construction_robot_pkg, 'urdf', 'lunar_ur10.urdf.xacro'
    )

    # Path to world file
    world_path = os.path.join(
        lunar_env_pkg, 'worlds', 'lunar_flat.world'
    )

    # Process xacro into robot_description string
    robot_description = ParameterValue(
        Command(['xacro ', urdf_xacro_path]),
        value_type=str
    )

    # Launch Gazebo with lunar world
    gazebo = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('gazebo_ros'),
            'launch',
            'gazebo.launch.py'
        ]),
        launch_arguments={
            'world': world_path,
            'verbose': 'true'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }],
        output='screen'
    )

    # Spawn robot with delay
    spawn_robot = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', 'lunar_ur10',
                    '-topic', 'robot_description',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.05'
                ],
                output='screen'
            )
        ]
    )

    # Controller spawners (with delay to ensure robot is spawned first)
    controller_spawners = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen'
            ),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['arm_effort_controller'],
                output='screen'
            ),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['wrist_position_controller'],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        controller_spawners,
    ])
