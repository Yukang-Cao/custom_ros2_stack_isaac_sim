#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for complete dummy testing environment with MPPI controller.
    This launches dummy costmap, dummy odometry, dummy goal publishers, and local_planner_node with MPPI.
    Use this when you want to test the MPPI controller without any real hardware.
    """
    # Get the default config path
    controllers_share_dir = get_package_share_directory('controllers')
    default_config_path = os.path.join(controllers_share_dir, 'config', 'experiment_config.yaml')
    
    # Launch arguments
    config_file_path = LaunchConfiguration('config_file_path', default=default_config_path)
    controller_type = LaunchConfiguration('controller_type', default='mppi_pytorch')
    control_frequency = LaunchConfiguration('control_frequency', default='20.0')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'config_file_path',
            default_value=default_config_path,
            description='Path to the experiment configuration file'),
        
        DeclareLaunchArgument(
            'controller_type',
            default_value='mppi_pytorch',
            description='Type of controller to use (mppi_pytorch, log_mppi_pytorch, etc.)'),
        
        DeclareLaunchArgument(
            'control_frequency',
            default_value='20.0',
            description='Control loop frequency in Hz'),

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

        # Launch the local planner node with MPPI controller
        Node(
            package='controllers',
            executable='local_planner_node',
            name='local_planner_node',
            output='screen',
            parameters=[{
                'controller_type': controller_type,
                'config_file_path': config_file_path,
                'control_frequency': control_frequency,
                'map_frame': 'map',
                'base_link_frame': 'base_link',
                'use_external_sdf': False,  # Set to False for dummy testing without real SDF
                'seed': 2025,
            }]
        ),
    ])
