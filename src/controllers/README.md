# Controllers Package

Complete control pipeline: sensor preprocessing, trajectory planning, and testing utilities.

## Design Philosophy

**Single Package = Single Purpose**: All controller-related functionality (perception preprocessing, planning algorithms, testing tools) lives here to avoid cross-package dependencies and maintain cohesion.

## Nodes

### Planning
**`local_planner_node`** - Main controller  
- **In**: `/goal_pose`, `/local_costmap_inflated`, `/odometry/filtered`  
- **Out**: `/cmd_vel`

### Perception Preprocessing  
**`costmap_processor_node`** - Scan to costmap conversion  
- **In**: `/laser_scan`  
- **Out**: `/local_costmap_binary_raw`, `/local_costmap_inflated`, `/local_costmap_sdf`

### Testing Utilities
- **`dummy_local_costmap_publisher`**: Fake costmap data
- **`dummy_odom_publisher`**: Fake odometry data
- **`dummy_goal_publisher`**: Fake goal poses

## Launch Files

**`isaac_sim_navigation_suite.launch.py`** - Complete stack for Isaac Sim  
**`controllers.launch.py`** - Controller only  
**`scan_to_costmap.launch.py`** - LiDAR processing + dummy goal  
**`dummy_costmap_odom.launch.py`** - Full dummy environment

## Usage

Isaac Sim:
```bash
ros2 launch controllers isaac_sim_navigation_suite.launch.py
ros2 launch controllers isaac_sim_navigation_suite.launch.py controller_type:=uge_mpc_pytorch
```

## Configuration

Edit `config/experiment_config.yaml` to customize:
- Controller type (`cu_mppi_unsupervised_std`, `mppi_pytorch`, etc.)
- Vehicle dynamics parameters
- Control limits and constraints
- Visualization settings

## Controller Types

- **MPPI**: Model Predictive Path Integral control, Log-MPPI also included
- **C-Uniform**: Neural network-based controller
- **CU-MPPI**: C-Uniform/C-Free-Uniform + MPPI/Log-MPPI, 4 combinations here
- **UGE-MPC**: Uncertainty Guided Exploration MPC