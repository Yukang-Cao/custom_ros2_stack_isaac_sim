# File: src/controllers/controllers/local_planner_node.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import yaml
import os
import sys
import time
import traceback
import math
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# ROS Message Types
from geometry_msgs.msg import Twist, PoseStamped, Point, TwistStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray, Marker
from ament_index_python.packages import get_package_share_directory

# TF2
import tf2_ros
from tf2_ros import TransformException
# Explicit import needed for do_transform_pose
import tf2_geometry_msgs 

import torch

# Import the controller library
# NOTE: This assumes 'lib' is correctly set up as a Python module within the ROS workspace.
# If using the provided directory structure, ensure your setup.py includes the 'lib' directory.
from controllers.lib.mppi_pytorch_controller import MPPIPyTorchController
from controllers.lib.nn_cuniform_controller import CUniformController
from controllers.lib.uge_mpc_controller import UGEMPCController
from controllers.lib.cu_mppi_controller import CUMPPiController
from controllers.lib.base_controller import PlannerInput

class LocalPlannerNode(Node):
    def __init__(self):
        super().__init__('local_planner_node')
        
        # --- Publishers (Create immediately to avoid AttributeError) ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.setup_visualization_publishers()
        
        # --- Parameters ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('controller_type', 'mppi_pytorch'),
                ('config_file_path', os.path.join(get_package_share_directory('controllers'), 'config', 'experiment_config.yaml')),
                ('control_frequency', 20.0),
                ('map_frame', 'map'),
                ('base_link_frame', 'base_link'),
                ('use_external_sdf', True), # Use SDF from scan_to_costmap package
                ('seed', 2025)
            ]
        )
        
        self.controller_type = self.get_parameter('controller_type').value
        self.config_path = self.get_parameter('config_file_path').value
        self.control_dt = 1.0 / self.get_parameter('control_frequency').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_link_frame = self.get_parameter('base_link_frame').value
        self.use_external_sdf = self.get_parameter('use_external_sdf').value
        self.seed = self.get_parameter('seed').value

        # --- Configuration Loading ---
        self.config = self.load_configuration(self.config_path)
        if self.config is None:
            # Handle error appropriately in a real ROS node (e.g., shutdown)
            self.get_logger().fatal(f"Configuration file not found at: {self.config_path}")
            return
        
        # --- Visualization Configuration ---
        self.viz_config = self.config.get('visualization', {})
        if not self.viz_config.get('enabled', True):
            self.get_logger().info("Visualizations are disabled by configuration.")

        # PyTorch device
        assert torch.cuda.is_available(), "CUDA is not available"
        self.device = torch.device("cuda")

        # --- Controller Initialization ---
        self.controller = None
        self.initialize_controller()
        if self.controller is None:
            self.get_logger().fatal(f"Controller initialization failed")
            return

        # Determine if SDF is needed
        self.requires_sdf = False
        # Check if it's a C-Uniform based controller (includes CU-MPPI) and if it's map-conditioned (type 1)
        if isinstance(self.controller, CUniformController) and self.controller.controller_type == 1:
            self.requires_sdf = True
            if self.use_external_sdf:
                self.get_logger().info("Controller will use SDF from scan_to_costmap package.")
            else: # Fail if configuration is inconsistent
                self.get_logger().fatal("Configuration Error: Controller requires SDF (Map-Conditioned), but 'use_external_sdf' is False. Terminating.")
                self.destroy_node()
                rclpy.try_shutdown()
                return

        # --- State Variables ---
        self.current_velocity = 0.0
        self.global_goal = None
        self.latest_costmap_msg = None
        self.latest_sdf_msg = None
        self.goal_received_once = False  # Flag to prevent repeated goal resets

        # --- TF2 Buffer and Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Subscribers ---
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # NOTE: select one of the following two subscriptions, Vicon in indoor and ekf odom in outdoor
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, sensor_qos) 
        # self.create_subscription(TwistStamped, '/vrpn_mocap/titan_alphatruck/twist', self.twist_callback, sensor_qos)

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap_inflated', self.costmap_callback, sensor_qos)
        
        # Subscribe to SDF if needed
        if self.requires_sdf and self.use_external_sdf:
            self.create_subscription(OccupancyGrid, '/local_costmap_sdf', self.sdf_callback, sensor_qos)
        
        # --- Main Control Loop Timer ---
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)

        self.get_logger().info(f"LocalPlannerNode initialized successfully. Controller: {self.controller_type} on {self.device}.")

    def load_configuration(self, path):
        """Loads the YAML configuration and detects velocity mode."""
        if not os.path.exists(path):
            self.get_logger().fatal(f"Configuration file not found at: {path}")
            return None
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            # Determine velocity mode automatically (required by the library initialization)
            vrange = config.get('vrange', [])
            config['variable_velocity_mode'] = (len(vrange) == 2 and vrange[0] != vrange[1])
            return config
        except yaml.YAMLError as e:
            self.get_logger().fatal(f"Error parsing YAML file: {e}")
            return None
    
    def resolve_resource_paths(self, config_section):
        """Resolves relative resource paths in the configuration."""
        if not config_section:
            return

        resource_pkg = config_section.get('resource_package')
        if not resource_pkg:
            return

        try:
            # Assuming the 'resource' directory structure is preserved during installation.
            package_share_directory = get_package_share_directory(resource_pkg)
            # Assuming resources are installed under share/<package_name>/resource/
            resource_base_path = os.path.join(package_share_directory, 'resource')

        except Exception as e:
            self.get_logger().error(f"Could not find resource package '{resource_pkg}' or 'resource' directory: {e}. Check installation.")
            return

        # List of keys that contain paths to resolve
        path_keys = ['map_conditioned_model_path', 'feature_extractor_path', 'unsupervised_cuniform_model_path']
        
        for key in path_keys:
            if key in config_section and config_section[key]:
                config_section[key] = os.path.join(resource_base_path, config_section[key])

    def initialize_controller(self):
        """Dynamically instantiates the controller based on the ROS parameter."""
        
        # Define the factory mapping
        mppi_cfg = self.config.get("mppi_controller", {})
        CONTROLLER_FACTORY = {
            "mppi_pytorch": (MPPIPyTorchController, "mppi_controller", {"type_override": 0}),
            "log_mppi_pytorch": (MPPIPyTorchController, "mppi_controller", {"type_override": 1}),
            "cuniform_map_conditioned": (CUniformController, "cuniform_controller", {"type_override": 1}),
            "cuniform_unsupervised_openspace": (CUniformController, "cuniform_controller", {"type_override": 0}),
            "cu_mppi_unsupervised_std": (
                CUMPPiController, "cuniform_controller",
                {"type_override": 0, "mppi_type_override": 0, "mppi_config": mppi_cfg}
            ),
            "cu_mppi_unsupervised_log": (
                CUMPPiController, "cuniform_controller",
                {"type_override": 0, "mppi_type_override": 1, "mppi_config": mppi_cfg}
            ),
            "cu_mppi_map_conditioned_std": (
                CUMPPiController, "cuniform_controller",
                {"type_override": 1, "mppi_type_override": 0, "mppi_config": mppi_cfg}
            ),
            "cu_mppi_map_conditioned_log": (
                CUMPPiController, "cuniform_controller",
                {"type_override": 1, "mppi_type_override": 1, "mppi_config": mppi_cfg}
            ),
            "uge_mpc_pytorch": (UGEMPCController, "uge_mpc_controller", {"mppi_config": mppi_cfg}),
        }

        if self.controller_type not in CONTROLLER_FACTORY:
            self.get_logger().fatal(f"Unknown controller type: {self.controller_type}")
            return

        try:
            ControllerClass, config_section, overrides = CONTROLLER_FACTORY[self.controller_type]
            controller_config = self.config.get(config_section)
            
            if controller_config is None:
                self.get_logger().error(f"Config section '{config_section}' missing for {self.controller_type}")
                return

            # Resolve paths before passing to the controller
            self.resolve_resource_paths(controller_config)

            self.controller = ControllerClass(
                controller_config=controller_config,
                experiment_config=self.config,
                seed=self.seed,
                **overrides
            )
            self.get_logger().info(f"Successfully instantiated {self.controller_type}")

        except Exception as e:
            self.get_logger().fatal(f"Error initializing controller {self.controller_type}: {e}")
            traceback.print_exc()
            self.controller = None
        
    def setup_visualization_publishers(self):
        """Sets up publishers for visualization."""
        vis_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.traj_marker_pub = self.create_publisher(MarkerArray, '/visualization/sampled_trajectories', vis_qos)
        self.local_goal_pub = self.create_publisher(Marker, '/visualization/local_goal', vis_qos)
        # self.nominal_path_pub = self.create_publisher(Path, '/visualization/nominal_trajectory', vis_qos)

    # --- Callbacks ---
    def odom_callback(self, msg: Odometry):
        # Assuming velocity is provided in the child frame (base_link)
        self.current_velocity = msg.twist.twist.linear.x

    def twist_callback(self, msg: TwistStamped):
        # The TwistStamped message provides velocity in the frame specified in its header 
        self.current_velocity = msg.twist.linear.x

    def goal_callback(self, msg: PoseStamped):
        # Only accept the first goal to test if frequent resets cause oscillations
        if not self.goal_received_once:
            self.global_goal = msg
            if self.controller:
                self.controller.reset()
            self.goal_received_once = True
            self.get_logger().info(f"\033[91mFIRST goal received and accepted at ({self.global_goal.pose.position.x:.2f}, {self.global_goal.pose.position.y:.2f})\033[0m")
            self.get_logger().info("\033[91mSubsequent goals will be IGNORED for testing purposes\033[0m")
        else:
            self.get_logger().info(f"Ignoring subsequent goal at ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) - using first goal only")

    def costmap_callback(self, msg: OccupancyGrid):
        self.latest_costmap_msg = msg
    
    def sdf_callback(self, msg: OccupancyGrid):
        """Callback for SDF costmap from scan_to_costmap package."""
        self.latest_sdf_msg = msg

    # --- Main Control Loop ---
    def control_loop(self):
        start_time = time.monotonic()
        if not self.is_ready():
            # Waiting for data, do not necessarily stop the robot yet.
            # Detailed status is already logged in is_ready()
            return

        # 1. Transform Goal
        local_goal_np = self.transform_goal_to_local()
        if local_goal_np is None:
            self.publish_stop_command()
            self.get_logger().error("\033[91mFailed to transform goal. Stopping robot.\033[0m", throttle_duration_sec=1.0)
            return

        # 2. Prepare Inputs (Costmap and SDF)
        planner_input = self.prepare_planner_input(local_goal_np)
        if planner_input is None:
            self.get_logger().error("\033[91mFailed to prepare planner input (Costmap/SDF invalid or mismatch). Stopping robot.\033[0m", throttle_duration_sec=1.0)
            self.publish_stop_command()
            return

        # 3. Execute Planning
        planning_start = time.monotonic()
        try:
            control_action, info = self.controller.get_control_action(planner_input)
        except Exception as e:
            self.get_logger().error(f"\033[91mRuntime error during planning step: {e}\033[0m", throttle_duration_sec=1.0)
            traceback.print_exc()
            self.publish_stop_command()
            return
        planning_time = (time.monotonic() - planning_start) * 1000

        # 4. Publish Control
        # self.get_logger().info(f"control_action: {control_action[0]}, {control_action[1]}")
        self.publish_control_command(control_action)

        # 5. Visualization
        viz_start = time.monotonic()
        if self.viz_config['enabled']:
            # Visualize Trajectories
            if self.viz_config['visualize_trajectories']:
                if 'state_rollouts_robot_frame' in info and info['state_rollouts_robot_frame'] is not None:
                    self.visualize_trajectories(info['state_rollouts_robot_frame'], info.get('is_hybrid', False))
                else: # This might happen if the controller skipped preparation
                    pass
            
            # Visualize Goal
            if self.viz_config['visualize_goal']:
                self.visualize_local_goal(local_goal_np)

        viz_time = (time.monotonic() - viz_start) * 1000
        
        # 6. Detailed Timing feedback
        total_time = time.monotonic() - start_time
        total_time_ms = total_time * 1000

        if total_time > self.control_dt:
            log_msg = (f"Control loop missed deadline! Total: {total_time_ms:.2f}ms | "
                       f"Planning: {planning_time:.2f}ms | Viz: {viz_time:.2f}ms")
            self.get_logger().warn(log_msg)
        else:
            log_msg = (f"Control loop timing - Total: {total_time_ms:.2f}ms | "
                       f"Planning: {planning_time:.2f}ms")
            self.get_logger().info(log_msg, throttle_duration_sec=2.0)

    def is_ready(self):
        """Checks if all necessary inputs are available."""
        missing_components = []
        
        if self.global_goal is None:
            missing_components.append("goal (/goal_pose)")
        
        if self.latest_costmap_msg is None:
            missing_components.append("local costmap (/local_costmap_inflated)")
        
        # Check SDF requirement based on controller type
        sdf_needed = self.requires_sdf and self.use_external_sdf
        if sdf_needed and self.latest_sdf_msg is None:
            missing_components.append("SDF from scan_to_costmap (/local_costmap_sdf)")
        
        if missing_components:
            # Create detailed status message
            status_parts = []
            status_parts.append(f"Controller: {self.controller_type}")
            status_parts.append(f"Requires SDF: {self.requires_sdf}")
            status_parts.append(f"Use external SDF: {self.use_external_sdf}")
            
            # Add received status for each component
            status_parts.append(f"Goal received: {'✓' if self.global_goal is not None else '✗'}")
            status_parts.append(f"Costmap received: {'✓' if self.latest_costmap_msg is not None else '✗'}")
            if sdf_needed:
                status_parts.append(f"SDF received: {'✓' if self.latest_sdf_msg is not None else '✗'}")
            
            missing_str = ", ".join(missing_components)
            status_str = " | ".join(status_parts)
            
            self.get_logger().info(
                f"\033[91mControl loop not ready - Missing: {missing_str}\033[0m\n"
                f"\033[93mStatus: {status_str}\033[0m", 
                throttle_duration_sec=2.0
            )
            return False
        
        return True

    def transform_goal_to_local(self):
        """Transforms the global goal into the robot's local frame (base_link) using TF2."""
        if self.global_goal is None:
            return None
            
        try:
            # transform from the goal's frame to the robot frame
            transform = self.tf_buffer.lookup_transform(
                self.base_link_frame,             # Target frame (robot)
                self.global_goal.header.frame_id, # Source frame (e.g., map)
                rclpy.time.Time()                 # Use the latest available transform
            )
            
            # Apply the transform to the goal pose
            local_goal_pose = tf2_geometry_msgs.do_transform_pose(self.global_goal.pose, transform)
            
            return np.array([local_goal_pose.position.x, local_goal_pose.position.y], dtype=np.float32)

        except TransformException as e:
            self.get_logger().warn(f"Could not transform goal to robot frame: {e}", throttle_duration_sec=1.0)
            return None

    def prepare_planner_input(self, local_goal_np):
        """Converts ROS data into the PlannerInput structure."""
        
        # 1. Process Costmap
        costmap_np = self.process_costmap(self.latest_costmap_msg)
        if costmap_np is None:
            return None
        
        # ensure input dimensions match controller expectations
        expected_size = self.controller.local_costmap_size
        if costmap_np.shape != (expected_size, expected_size):
            self.get_logger().error(
                f"Costmap dimension mismatch! Expected ({expected_size}, {expected_size}), but received {costmap_np.shape}."
                " Check scan_to_costmap node output and configuration file.",
                throttle_duration_sec=1.0
            )
            return None

        # 2. Process SDF (if required and available from scan_to_costmap)
        sdf_tensor = None
        if self.requires_sdf and self.use_external_sdf:
            if self.latest_sdf_msg is not None:
                sdf_np = self.process_sdf(self.latest_sdf_msg)
                if sdf_np is not None:
                    if sdf_np.shape != costmap_np.shape: # ensure SDF dimensions match costmap
                        self.get_logger().error(
                            f"SDF dimension mismatch with Costmap! SDF: {sdf_np.shape}, Costmap: {costmap_np.shape}.",
                            throttle_duration_sec=1.0
                        )
                        return None
                    # Convert to tensor (B, C, H, W)
                    sdf_tensor = torch.from_numpy(sdf_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                self.get_logger().error("SDF required but not received from scan_to_costmap package", throttle_duration_sec=1.0)
                return None # Cannot proceed without required SDF

        return PlannerInput(
            local_goal=local_goal_np,
            current_velocity=self.current_velocity,
            inflated_costmap=costmap_np,
            sdf_tensor=sdf_tensor
        )

    def process_costmap(self, msg: OccupancyGrid):
        """Converts OccupancyGrid to numpy array, handling normalization and orientation."""
        try:
            width = msg.info.width
            height = msg.info.height
            
            data = np.array(msg.data, dtype=np.int8).reshape(height, width)
            
            # Convert ROS standard (0-100) to controller's expectation (0.0-1.0).
            # Handle unknown space (-1). Treat as free space (0.0).
            costmap_np = np.clip(data, 0, 100).astype(np.float32) / 100.0
            return costmap_np
        except Exception as e:
            self.get_logger().error(f"Error processing costmap: {e}")
            return None

    def process_sdf(self, msg: OccupancyGrid):
        """Converts SDF OccupancyGrid to numpy array."""
        try:
            width = msg.info.width
            height = msg.info.height
            
            data = np.array(msg.data, dtype=np.int8).reshape(height, width)
            
            # Convert back from scaled int8 to float32 SDF values
            # The scan_to_costmap node scales SDF by 10 and clips to int8 range
            sdf_np = data.astype(np.float32) / 10.0

            return sdf_np
        except Exception as e:
            self.get_logger().error(f"Error processing SDF: {e}")
            return None

    def publish_stop_command(self):
        """Publishes a zero velocity command."""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
    
    def publish_control_command(self, control_action: np.ndarray):
        """Publishes the control action, interpreting the output based on the dynamics model."""
        
        # Ensure the controller and its dynamics interface are initialized
        if self.controller is None or not hasattr(self.controller, 'dynamics'):
            self.get_logger().error("Controller or dynamics interface not initialized. Stopping robot.", throttle_duration_sec=1.0)
            self.publish_stop_command()
            return

        twist = Twist()
        v = float(control_action[0])
        # The interpretation of the second control input depends on the dynamics model
        control_input_2 = float(control_action[1])
        
        # rely on the controller instance to tell us how to interpret the output
        control_type = self.controller.dynamics.control_type # initialized in TorchPlannerBase
        if control_type == "steering_angle":
            # KST Model: Input is steering angle (delta)
            delta = control_input_2
            omega = -delta 
            # # Convert steering angle to angular velocity (omega) for the Twist message
            # # omega = v * tan(delta) / L
            # L = self.controller.wheelbase
            # if L > 0 and not math.isnan(delta):
            #     omega = v * np.tan(delta) / L
            # else:
            #     omega = 0.0
        elif control_type == "angular_velocity":
            # Differential Drive/Boat Model: Input is angular velocity (omega)
            omega = control_input_2
        else: # fail if the control type is unknown
            self.get_logger().error(f"Unknown control type '{control_type}' from controller. Stopping robot.")
            self.publish_stop_command()
            return

        twist.linear.x = v
        twist.angular.z = omega
        self.cmd_vel_pub.publish(twist)
    
    def visualize_local_goal(self, local_goal_np):
        """Visualizes the local goal in the robot frame."""
        marker = Marker()
        marker.header.frame_id = self.base_link_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "local_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(local_goal_np[0])
        marker.pose.position.y = float(local_goal_np[1])
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        # Use a visible scale (e.g., 0.3m diameter sphere)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        # Magenta color
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.0, 1.0, 0.8)
        self.local_goal_pub.publish(marker)

    def visualize_trajectories(self, rollouts_data, is_hybrid):
        """Visualizes trajectories in the robot frame using MarkerArrays."""
        # Visualization is published in the robot frame (base_link_frame) 
        # as the trajectories are generated locally. RViz handles the transformation.
        
        self.get_logger().debug(f"Visualizing trajectories - Hybrid: {is_hybrid}, Data type: {type(rollouts_data)}")
        
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = self.base_link_frame
        clear_marker.header.stamp = now
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        if is_hybrid:
            # Hybrid Visualization (CU-MPPI) - 4 colors
            self.get_logger().debug(f"Hybrid mode - Keys available: {list(rollouts_data.keys()) if isinstance(rollouts_data, dict) else 'Not a dict'}")
            self._add_trajectory_marker(markers, rollouts_data.get('cu_samples'), "cu_samples", now, color=(0.5, 0.5, 0.5, 0.2), width=0.01) # Grey
            self._add_trajectory_marker(markers, rollouts_data.get('cu_best'), "cu_best", now, color=(0.0, 0.0, 1.0, 0.5), width=0.03)       # Blue
            self._add_trajectory_marker(markers, rollouts_data.get('mppi_samples'), "mppi_samples", now, color=(1.0, 0.65, 0.0, 0.3), width=0.01) # Orange
            self._add_trajectory_marker(markers, rollouts_data.get('mppi_nominal'), "mppi_nominal", now, color=(0.0, 1.0, 0.0, 0.8), width=0.05) # Green
        else:
            # Standard Visualization (MPPI/C-Uniform)
            self.get_logger().debug(f"Standard mode - Rollouts shape: {rollouts_data.shape if hasattr(rollouts_data, 'shape') else 'No shape attr'}")
            if hasattr(rollouts_data, 'shape') and len(rollouts_data.shape) >= 1 and rollouts_data.shape[0] > 0:
                # Best trajectory (Index 0)
                self._add_trajectory_marker(markers, rollouts_data[0:1], "best_trajectory", now, color=(0.0, 1.0, 0.0, 0.8), width=0.05) # Green
                # Samples (Index 1+)
                if rollouts_data.shape[0] > 1:
                    self._add_trajectory_marker(markers, rollouts_data[1:], "samples", now, color=(0.5, 0.5, 0.5, 0.3), width=0.01) # Grey

        self.get_logger().debug(f"Publishing {len(markers.markers)} markers")
        self.traj_marker_pub.publish(markers)

    def _add_trajectory_marker(self, markers, trajectories, ns, stamp, color, width):
        """Helper to add trajectories (Numpy arrays) to a MarkerArray."""
        if trajectories is None or (isinstance(trajectories, np.ndarray) and trajectories.size == 0):
            return
        
        # Ensure trajectories are 3D (N, T, 3) even if N=1
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, ...]

        for i in range(trajectories.shape[0]):
            marker = Marker()
            marker.header.frame_id = self.base_link_frame
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0 # Identity pose (points are already in the correct frame)
            marker.scale.x = width
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            
            # Add points
            for t in range(trajectories.shape[1]):
                p = Point()
                p.x = float(trajectories[i, t, 0])
                p.y = float(trajectories[i, t, 1])
                p.z = 0.05 # Visualize slightly above the ground
                marker.points.append(p)
            
            markers.markers.append(marker)

def main(args=None):
    # Initialize PyTorch CUDA context eagerly if available
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            print("INFO: CUDA initialized successfully by PyTorch.")
        except Exception as e:
            print(f"WARNING: Failed to initialize CUDA: {e}")

    rclpy.init(args=args)
    try:
        node = LocalPlannerNode()
        # Check if initialization was successful before spinning
        if node and hasattr(node, 'controller') and node.controller:
             rclpy.spin(node)
        else:
             print("LocalPlannerNode failed initialization.")
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"Node shutting down due to: {e}")
    except Exception as e:
        print(f"Node crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            if 'node' in locals() and node:
                node.publish_stop_command() # Attempt to stop the robot
                node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
