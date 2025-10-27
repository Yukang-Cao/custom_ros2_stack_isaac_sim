#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import yaml

# ROS Message Types
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

# For SDF and inflation processing
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation
import traceback

# TF2
import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped

class CostmapProcessorNode(Node):
    def __init__(self):
        super().__init__('costmap_processor_node')

        # --- Declare Parameter for Config Path ---
        # This should be set by the launch file.
        self.declare_parameter(
            'config_file_path', ''
        )
        
        # --- Load Configuration ---
        self.load_config()
        
        # --- Publishers ---
        self.binary_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_binary_raw', 10)
        self.inflated_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_inflated', 10)
        self.sdf_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_sdf', 10)
        
        # --- Subscribers ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile)
        
        # --- TF2 Buffer and Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- Processing Parameters loaded from config ---
        self.get_logger().info("Costmap Processor Node initialized")
        self.get_logger().info(f"Costmap size: {self.local_costmap_size} cells")
        self.get_logger().info(f"Resolution: {self.resolution} m/cell")
        self.get_logger().info(f"Inflation radius: {self.inflation_radius} cells")
        self.get_logger().info(f"SDF inflation: {self.sdf_inflation_cells} cells")
        self.get_logger().info(f"LiDAR max range: {self.max_range} m")
        self.get_logger().info(f"Robot body filter radius: {self.robot_body_radius} m")

    def load_config(self):
        """Load configuration from the centralized experiment config file."""
        # Get the path from the ROS parameter
        config_path = self.get_parameter('config_file_path').value
        if not config_path:
            self.get_logger().fatal("config_file_path parameter not set! Cannot load configuration.")
            rclpy.try_shutdown()
            return

        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except (FileNotFoundError, yaml.YAMLError) as e:
            self.get_logger().fatal(f"Error loading or parsing config file at {config_path}: {e}")
            rclpy.try_shutdown()
            return
        
        # Extract costmap parameters
        self.local_costmap_size = config['local_costmap_size']
        self.resolution = config['local_costmap_resolution']
        self.inflation_radius = config['inflation_radius']
        self.max_inflation_value = config['max_inflation_value']
        self.sdf_inflation_cells = config['sdf_inflation_cells']
        self.max_range = config['lidar_max_range']
        
        # Robot body filtering parameters
        self.robot_body_radius = config['robot_body_filter_radius']
        
        self.get_logger().info(f"Successfully loaded config from {config_path}")

    def lidar_callback(self, msg: LaserScan):
        """Process LiDAR scan and generate all required costmaps."""
        try:
            # Step 1: Create binary costmap from LiDAR scan
            binary_costmap = self.create_binary_costmap(msg)
            
            # Step 2: Create inflated costmap
            inflated_costmap = self.create_inflated_costmap(binary_costmap)
            
            # Step 3: Create SDF costmap
            sdf_costmap = self.create_sdf_costmap(binary_costmap)
            
            # Step 4: Publish all costmaps
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "base_link"  # Local costmap is in robot frame
            
            # Use the standard publisher for 0-1 costmaps
            # self.publish_occupancy_grid(binary_costmap, header, self.binary_costmap_pub)
            self.publish_occupancy_grid(inflated_costmap, header, self.inflated_costmap_pub)
            
            # Use the specific SDF publisher which handles encoding
            self.publish_sdf_grid(sdf_costmap, header, self.sdf_costmap_pub)
            
        except Exception as e:
            self.get_logger().error(f"Error in lidar_callback: {e}")
            traceback.print_exc()

    def create_binary_costmap(self, lidar_msg: LaserScan):
        """Convert LiDAR scan to binary occupancy grid
        NOTE: This function generates the map in ROS coordinates (bottom-left logical origin, Cartesian)
        using grid_y = center + y / resolution.
        """
        # Initialize binary costmap
        binary_costmap = np.zeros((self.local_costmap_size, self.local_costmap_size), dtype=np.float32)
        center = self.local_costmap_size // 2
        
        # Extract ranges and calculate angles
        lidar_ranges = np.array(lidar_msg.ranges)
        angles = np.linspace(lidar_msg.angle_min, lidar_msg.angle_max, len(lidar_ranges))
        
        # Process each LiDAR point
        for i, range_val in enumerate(lidar_ranges):
            if range_val >= self.max_range or range_val <= lidar_msg.range_min:
                continue
            
            # Obstacle position in the robot's frame (+X is forward)
            obs_x_robot = range_val * np.cos(angles[i])
            obs_y_robot = range_val * np.sin(angles[i])
            
            # Filter out robot body: ignore any obstacles within robot_body_radius of robot center
            distance_from_robot = np.sqrt(obs_x_robot**2 + obs_y_robot**2)
            if distance_from_robot < self.robot_body_radius:
                continue

            # # Apply 90-degree clockwise rotation to align with visualization
            # # NOTE: ideally this should be done in URDF, now it's a hack
            # # New X = Old Y, New Y = -Old X
            rotated_x = obs_y_robot
            rotated_y = -obs_x_robot
            
            # Use standard ROS coordinate frames (REP-103: X-forward, Y-left)
            # grid_x = int(center + obs_x_robot / self.resolution)
            # grid_y = int(center + obs_y_robot / self.resolution) # ROS Coordinates
            grid_x = int(center + rotated_x / self.resolution)
            grid_y = int(center + rotated_y / self.resolution) # ROS Coordinates
            
            if 0 <= grid_x < self.local_costmap_size and 0 <= grid_y < self.local_costmap_size:
                binary_costmap[grid_y, grid_x] = 1.0
        
        return binary_costmap

    def create_inflated_costmap(self, binary_costmap):
        """Create inflated costmap with smooth cost gradients."""
        # Apply distance transform to the binary costmap
        obstacle_mask = binary_costmap > 0.5
        distance_map = distance_transform_edt(~obstacle_mask)
        
        # Apply linear decay to create smooth cost gradients
        if self.inflation_radius > 0:
            inflated_costmap = np.clip(
                (self.inflation_radius - distance_map) / self.inflation_radius, 
                0, 1
            ) * self.max_inflation_value
        else:
            inflated_costmap = np.zeros_like(distance_map)
        
        # Combine with original obstacles to ensure they have the max cost
        inflated_costmap = np.maximum(binary_costmap, inflated_costmap).astype(np.float32)
        return inflated_costmap

    def create_sdf_costmap(self, binary_costmap):
        """Create Signed Distance Field (SDF) costmap."""
        # Create the binary obstacle map for the SDF
        sdf_obstacle_map = binary_costmap > 0.5
        
        # Apply binary dilation if configured
        if self.sdf_inflation_cells > 0:
            sdf_obstacle_map = binary_dilation(sdf_obstacle_map, iterations=self.sdf_inflation_cells)
        
        # Generate the SDF
        dist_out = distance_transform_edt(~sdf_obstacle_map) * self.resolution
        dist_in = distance_transform_edt(sdf_obstacle_map) * self.resolution
        sdf = dist_out - dist_in
        
        # WORKAROUND: Flip the SDF along its vertical axis (axis=0) to correct for system orientation mismatch
        sdf_np = np.flip(sdf, axis=0).copy()
        return sdf_np.astype(np.float32)

    def _set_occupancy_grid_info(self, occupancy_grid):
        """Helper to set metadata for OccupancyGrid according to REP-103."""
        occupancy_grid.info.resolution = self.resolution
        occupancy_grid.info.width = self.local_costmap_size
        occupancy_grid.info.height = self.local_costmap_size
        
        # Origin defines the pose of the bottom-left cell (0,0) in the message data.
        # Since the robot is at the center of the map (size/2), the origin is offset negatively.
        map_extent = self.local_costmap_size * self.resolution / 2.0
        occupancy_grid.info.origin.position.x = -map_extent
        occupancy_grid.info.origin.position.y = -map_extent
        occupancy_grid.info.origin.orientation.w = 1.0

    def publish_occupancy_grid(self, costmap_data, header, publisher):
        """Publish standard costmap (0.0-1.0) as OccupancyGrid message (0-100)."""
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header = header
        self._set_occupancy_grid_info(occupancy_grid)
        
        # Convert float (0.0-1.0) to int8 (0-100)
        data = (costmap_data * 100).astype(np.int8)
        occupancy_grid.data = data.flatten().tolist()
        
        publisher.publish(occupancy_grid)
    
    def publish_sdf_grid(self, sdf_data, header, publisher):
        """Publish SDF (float) as OccupancyGrid message using scaled int8 encoding."""
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header = header
        self._set_occupancy_grid_info(occupancy_grid)

        # Encoding: Scale SDF (meters) by 10. This matches decoding in local_planner_node.
        SDF_SCALE = 10.0
        scaled_data = (sdf_data * SDF_SCALE)
        
        # Clip and cast to ensure data fits within int8 range (-128 to 127)
        clipped_data = np.clip(scaled_data, -128, 127).astype(np.int8)

        occupancy_grid.data = clipped_data.flatten().tolist()
        publisher.publish(occupancy_grid)
        
def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CostmapProcessorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
