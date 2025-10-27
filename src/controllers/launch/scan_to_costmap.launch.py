#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for real scan-to-costmap conversion with dummy goal.
    This launches the costmap processor node (expects real LiDAR data on /scan)
    and a dummy goal publisher for testing the controller.
    """
    # Launch arguments
    config_file_path = LaunchConfiguration('config_file_path', default='')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'config_file_path',
            default_value=config_file_path,
            description='Path to the experiment configuration file'),

        # Launch the costmap processor node (expects real LiDAR data)
        Node(
            package='controllers',
            executable='costmap_processor_node',
            name='costmap_processor_node',
            output='screen',
            parameters=[{
                'config_file_path': config_file_path,
            }]
        ),

        # Launch the dummy goal publisher for controller testing
        Node(
            package='controllers',
            executable='dummy_goal_publisher',
            name='dummy_goal_publisher',
            output='screen'
        ),
    ])
