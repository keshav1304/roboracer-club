from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # AutoDRIVE Bridge
        Node(
            package='autodrive_roboracer',
            executable='autodrive_bridge',
            name='autodrive_bridge',
            emulate_tty=True,
            output='screen',
        ),
        # Pure Pursuit racing algorithm
        Node(
            package='autodrive_roboracer',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            emulate_tty=True,
            output='screen',
        ),
        # # Waypoint visualizer
        # Node(
        #     package='autodrive_roboracer',
        #     executable='waypoint_visualizer',
        #     name='waypoint_visualizer',
        #     emulate_tty=True,
        #     output='screen',
        # ),
    ])
