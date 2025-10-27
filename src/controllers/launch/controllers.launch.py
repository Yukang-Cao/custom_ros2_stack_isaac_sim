from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',  # Use default from node
        description='Path to configuration file (optional)'
    )
    
    controller_type_arg = DeclareLaunchArgument(
        'controller_type',
        default_value='mppi_pytorch',
        description='Controller type: cu_mppi_unsupervised_std, mppi_pytorch, uge_mpc_pytorch, etc.'
    )
    
    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='10.0',
        description='Control loop frequency (Hz)'
    )
    
    # Controllers node
    # Note: We intentionally do NOT pass the config_file_path parameter when it's empty
    # This allows the node to use its built-in default configuration file path
    controllers_node = Node(
        package='controllers',
        executable='local_planner_node',
        name='local_planner_node',
        output='screen',
        parameters=[
            {'controller_type': LaunchConfiguration('controller_type')},
            {'control_frequency': LaunchConfiguration('control_frequency')}
            # config_file_path is intentionally omitted to let the node use its default
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        controller_type_arg,
        control_frequency_arg,
        controllers_node,
    ])
