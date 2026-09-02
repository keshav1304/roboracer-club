"""
Mapping localization: map_server + AMCL + lifecycle manager.

  ros2 launch autodrive_roboracer localize.launch.py \
      map:=/data/maps/<name>/map.yaml \
      params_file:=.../amcl_params.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('autodrive_roboracer')
    default_params = os.path.join(pkg, 'config', 'amcl_params.yaml')
    # Source-tree fallback when share/config is not installed yet
    if not os.path.isfile(default_params):
        default_params = os.path.join(
            os.path.dirname(__file__), '..', 'amcl_params.yaml')

    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map', description='Full path to map.yaml'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='AMCL / map_server / lifecycle params'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false'),

        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                params_file,
                {'yaml_filename': map_yaml, 'use_sim_time': use_sim_time},
            ],
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            remappings=[
                ('scan', '/autodrive/roboracer_1/lidar'),
            ],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[
                params_file,
                {
                    'use_sim_time': use_sim_time,
                    'autostart': True,
                    'node_names': ['map_server', 'amcl'],
                },
            ],
        ),
    ])
