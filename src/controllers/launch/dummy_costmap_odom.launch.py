#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for complete dummy testing environment.
    This launches dummy costmap, dummy odometry, and dummy goal publishers.
    Use this when you want to test the controller without any real hardware.
    """
    # Launch arguments
    config_file_path = LaunchConfiguration('config_file_path', default='')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'config_file_path',
            default_value=config_file_path,
            description='Path to the experiment configuration file'),

        # Launch the dummy local costmap publisher (replaces real LiDAR processing)
        Node(
            package='controllers',
            executable='dummy_local_costmap_publisher',
            name='dummy_local_costmap_publisher',
            output='screen',
            parameters=[{
                'config_file_path': config_file_path,
            }]
        ),

        # Launch the dummy odometry publisher (replaces real odometry)
        Node(
            package='controllers',
            executable='dummy_odom_publisher',
            name='dummy_odom_publisher',
            output='screen'
        ),

        # Launch the dummy goal publisher (for controller testing)
        Node(
            package='controllers',
            executable='dummy_goal_publisher',
            name='dummy_goal_publisher',
            output='screen'
        ),
    ])
