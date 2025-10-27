#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import yaml

# ROS Message Types
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

# For SDF and inflation processing
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation


class DummyLocalCostmapPublisher(Node):
    def __init__(self):
        super().__init__('dummy_local_costmap_publisher')

        # --- Declare Parameter for Config Path ---
        self.declare_parameter('config_file_path', '')
        
        # --- Load Configuration ---
        self.load_config()
        
        # --- Publishers ---
        self.binary_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_binary_raw', 10)
        self.inflated_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_inflated', 10)
        self.sdf_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_sdf', 10)
        
        # --- Timer for publishing dummy costmaps ---
        self.timer = self.create_timer(0.1, self.publish_dummy_costmaps)  # 10 Hz
        
        # Create the dummy costmaps once
        self.create_dummy_costmaps()
        
        self.get_logger().info("Dummy Local Costmap Publisher initialized")
        self.get_logger().info(f"Publishing circular obstacle 2m ahead of robot")
        self.get_logger().info(f"Costmap size: {self.local_costmap_size} cells")
        self.get_logger().info(f"Resolution: {self.resolution} m/cell")

    def load_config(self):
        """Load configuration from the centralized experiment config file."""
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
        
        self.get_logger().info(f"Successfully loaded config from {config_path}")

    def create_dummy_costmaps(self):
        """Create dummy costmaps with a circular obstacle 2 meters ahead."""
        # Initialize binary costmap
        self.binary_costmap = np.zeros((self.local_costmap_size, self.local_costmap_size), dtype=np.float32)
        center = self.local_costmap_size // 2
        
        # Create circular obstacle 2 meters ahead of robot
        obstacle_distance = 2.0  # 2 meters ahead
        obstacle_radius = 0.3    # 30cm radius obstacle
        
        # In robot frame: +X is forward, +Y is left
        # Robot is at center of costmap
        obstacle_x_robot = obstacle_distance  # 2m forward
        obstacle_y_robot = 0.0               # centered laterally
        
        # Apply the same 90-degree clockwise rotation as in the real costmap processor
        # to match the coordinate system
        rotated_x = obstacle_y_robot
        rotated_y = -obstacle_x_robot
        
        # Convert to grid coordinates
        obstacle_grid_x = int(center + rotated_x / self.resolution)
        obstacle_grid_y = int(center + rotated_y / self.resolution)
        obstacle_radius_cells = int(obstacle_radius / self.resolution)
        
        # Create circular obstacle
        y_coords, x_coords = np.ogrid[:self.local_costmap_size, :self.local_costmap_size]
        distance_from_obstacle = np.sqrt((x_coords - obstacle_grid_x)**2 + (y_coords - obstacle_grid_y)**2)
        
        # Set obstacle cells
        self.binary_costmap[distance_from_obstacle <= obstacle_radius_cells] = 1.0
        
        # Create inflated costmap
        self.inflated_costmap = self.create_inflated_costmap(self.binary_costmap)
        
        # Create SDF costmap
        self.sdf_costmap = self.create_sdf_costmap(self.binary_costmap)
        
        self.get_logger().info(f"Created circular obstacle at grid position ({obstacle_grid_x}, {obstacle_grid_y}) with radius {obstacle_radius_cells} cells")

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
        
        # Apply the same flip as in the real costmap processor
        sdf_np = np.flip(sdf, axis=0).copy()
        return sdf_np.astype(np.float32)

    def publish_dummy_costmaps(self):
        """Publish all dummy costmaps."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"  # Local costmap is in robot frame
        
        # Publish all costmaps
        self.publish_occupancy_grid(self.binary_costmap, header, self.binary_costmap_pub)
        self.publish_occupancy_grid(self.inflated_costmap, header, self.inflated_costmap_pub)
        self.publish_sdf_grid(self.sdf_costmap, header, self.sdf_costmap_pub)

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
        node = DummyLocalCostmapPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
