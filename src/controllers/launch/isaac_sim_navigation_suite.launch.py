#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    controllers_pkg_share = get_package_share_directory('controllers')
    default_config_path = os.path.join(controllers_pkg_share, 'config', 'experiment_config.yaml')
    
    config_file_path = LaunchConfiguration('config_file_path')
    controller_type = LaunchConfiguration('controller_type')
    control_frequency = LaunchConfiguration('control_frequency')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file_path',
            default_value=default_config_path,
            description='Path to experiment config file'),
        
        DeclareLaunchArgument(
            'controller_type',
            default_value='mppi_pytorch',
            description='Controller type'),
        
        DeclareLaunchArgument(
            'control_frequency',
            default_value='20.0',
            description='Control frequency (Hz)'),

        Node(
            package='controllers',
            executable='costmap_processor_node',
            name='costmap_processor_node',
            output='screen',
            parameters=[{'config_file_path': config_file_path}]
        ),

        Node(
            package='controllers',
            executable='dummy_goal_publisher',
            name='dummy_goal_publisher',
            output='screen'
        ),

        Node(
            package='controllers',
            executable='local_planner_node',
            name='local_planner_node',
            output='screen',
            parameters=[{
                'controller_type': controller_type,
                'config_file_path': config_file_path,
                'control_frequency': control_frequency,
            }]
        ),

        Node(
            package='cmdvel_to_ackermann',
            executable='cmdvel_to_ackermann.py',
            name='cmdvel_to_ackermann',
            output='screen'
        ),
    ])

