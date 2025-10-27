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
- **In**: `/scan`  
- **Out**: `/local_costmap_binary_raw`, `/local_costmap_inflated`, `/local_costmap_sdf`

### Testing Utilities
- **`dummy_local_costmap_publisher`**: Fake costmap data
- **`dummy_odom_publisher`**: Fake odometry data
- **`dummy_goal_publisher`**: Fake goal poses

## Launch Files

**`controllers.launch.py`**: Controller only  
**`scan_to_costmap.launch.py`**: Real LiDAR processing + dummy goal  
**`dummy_costmap_odom.launch.py`**: Dummy costmap + dummy odom + dummy goal

## Usage

Via tmuxinator (recommended):
```bash
tmuxinator start -p tmux/controller/.tmuxinator.yaml
```

Edit `tmux/controller/.tmuxinator.yaml` to switch between:
- Real LiDAR mode: `scan_to_costmap.launch.py`
- Dummy mode: `dummy_costmap_odom.launch.py` (see commented line)

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