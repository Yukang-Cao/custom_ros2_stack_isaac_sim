# Isaac Sim ROS 2 Workspace

**Requirements:** Ubuntu 24.04 | ROS 2 Jazzy | NVIDIA RTX 5090 (Requires PyTorch Nightly/CUDA 12.8+)

## Build
```bash
conda activate isaac_sim_ros2
cd ~/isaac_sim_ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Run

Navigation Stack:
``` Bash
ros2 launch controllers isaac_sim_navigation_suite.launch.py
```
