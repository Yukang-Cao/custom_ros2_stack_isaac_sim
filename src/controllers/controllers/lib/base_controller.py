# lib/base_controller.py
"""Abstract base controller class."""
from abc import ABC, abstractmethod
import numpy as np
import yaml
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PlannerInput:
    """Standardized input structure for the planner."""
    # Goal in robot frame [x, y]
    local_goal: np.ndarray 
    # Current forward velocity of the robot
    current_velocity: float 
    
    # Perception Data (all robot-centric/local)
    # Inflated costmap (numpy array) for cost evaluation
    inflated_costmap: Optional[np.ndarray] = None
    # SDF tensor (PyTorch tensor) for model input (if applicable)
    sdf_tensor: Optional[Any] = None # Use Any to avoid importing torch here


class BaseController(ABC):
    """Abstract base class for all navigation controllers."""
    
    def __init__(self, controller_config: dict, experiment_config: dict, seed: int):
        """Initialize controller with provided configuration and shared parameters."""
        # Store controller-specific config
        self.config = controller_config
        
        # Store experiment config
        self.experiment_config = experiment_config
        self.seed = seed
        
        # Initialize shared parameters from experiment config
        self._init_shared_parameters()
    
    def _init_shared_parameters(self):
        """Initialize parameters shared across all controllers from experiment config."""
        # Planning parameters
        self.T = self.experiment_config['horizon_T']
        self.dt = self.experiment_config['dt']
        self.num_steps = int(self.T / self.dt)
        
        # Vehicle parameters
        self.vehicle_length = self.experiment_config['vehicle_length']
        self.vehicle_width = self.experiment_config['vehicle_width']
        self.wheelbase = self.experiment_config['vehicle_wheelbase']
        
        # --- Dynamics Model Selection ---
        # Read the dynamics model name from configuration.
        self.dynamics_model_name = self.experiment_config['dynamics_model']

        # Control constraints
        self.vrange = np.array(self.experiment_config['vrange'], dtype=np.float32)
        # self.wrange = np.array(self.experiment_config['wrange'], dtype=np.float32)
        
        # Read specific constraints. If not present, fall back to 'wrange'.
        steering_config = self.experiment_config.get('steering_angle_range')
        angular_vel_config = self.experiment_config.get('angular_velocity_range')

        # Convert to numpy arrays if configuration exists
        self.steering_angle_range = np.array(steering_config, dtype=np.float32) if steering_config is not None else None
        self.angular_velocity_range = np.array(angular_vel_config, dtype=np.float32) if angular_vel_config is not None else None

        # Ensure at least one constraint definition was provided.
        if self.steering_angle_range is None and self.angular_velocity_range is None:
            raise ValueError("Configuration error: Missing control constraints. Must provide 'wrange', 'steering_angle_range', or 'angular_velocity_range'.")
        
        # Cost weights
        self.obs_penalty = self.experiment_config['obs_penalty']
        self.dist_weight = self.experiment_config['dist_weight']
        self.action_cost_weight = self.experiment_config['action_cost_weight']
        self.terminal_weight = self.experiment_config['terminal_weight']
        self.goal_tolerance = self.experiment_config['goal_tolerance']

        self.collision_occupancy_ratio = self.experiment_config['collision_occupancy_ratio']
        self.robot_footprint_size = self.experiment_config['robot_footprint_size']
        self.robot_footprint_area = self.robot_footprint_size * self.robot_footprint_size

        # Local costmap parameters (Required for interpreting the input maps)
        self.local_costmap_size = self.experiment_config['local_costmap_size']
        self.local_costmap_resolution = self.experiment_config['local_costmap_resolution']
        
        # Sampling parameters (shared across controllers)
        self.num_rollouts = self.experiment_config['num_rollouts']
        if self.num_rollouts <= 0:
            raise ValueError(f"Configuration error: 'num_rollouts' must be > 0, but got {self.num_rollouts}.")
        # Visualization settings (shared so controllers know whether to prepare data)
        self.viz_config = self.experiment_config.get('visualization', {})
        self.num_vis_rollouts = self.experiment_config['num_vis_rollouts']
        self.initialization_budget_ratio = self.experiment_config.get('initialization_budget_ratio', 0.5)

    @abstractmethod
    def get_control_action(self, planner_input: PlannerInput) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute control action.
        
        Args:
            planner_input: Standardized input containing local goal, velocity, and perception data.

            
        Returns:
            Tuple of:
                - Control action [velocity, angular_velocity]
                - Info dictionary for visualization and debugging
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset controller state for new environment"""
        pass 