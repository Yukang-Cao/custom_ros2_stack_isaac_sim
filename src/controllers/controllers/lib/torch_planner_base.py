# lib/torch_planner_base.py
"""
Intermediate base class for PyTorch-based trajectory planners.
Centralizes dynamics, cost evaluation, and perception handling to ensure consistency
across different sampling strategies (C-Uniform, MPPI, etc.).
"""

import numpy as np
import torch
import os
import time
import math
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

from .base_controller import BaseController, PlannerInput

from .utils.dynamics import (
    get_dynamics_model,
    DynamicsModel,
    JitStepFn, # Import the type hint
    # Import legacy functions for compatibility until CUniformController is fully refactored
    cuda_dynamics_KS_3d_scalar_v_batched as dynamics_scalar_torch,
    cuda_dynamics_KS_3d_variable_v_batched as dynamics_variable_torch
)

def _rollout_controls_functional(
    controls_full: torch.Tensor, 
    initial_state: torch.Tensor, 
    dt: float, 
    # pass the specific dynamics function and its parameters
    dynamics_step_fn: JitStepFn, 
    dynamics_params: Dict[str, float]
) -> torch.Tensor:
    """
    Functional rollout loop. Relies on a pre-compiled dynamics_step_fn.
    
    We do not JIT compile this function itself. The Python loop overhead is generally 
    acceptable when the core dynamics (dynamics_step_fn) is compiled.
    This allows dynamic selection of the dynamics model.
    """
    T, K, _ = controls_full.shape
    if initial_state.dim() == 1: # handle initial_state dimensions
        state_dim = initial_state.shape[0]
        batch_current_states = initial_state.unsqueeze(0).repeat(K, 1)
    elif initial_state.dim() == 2:
        state_dim = initial_state.shape[1]
        if initial_state.shape[0] == 1:
            batch_current_states = initial_state.repeat(K, 1)
        else:
            batch_current_states = initial_state
    else:
        raise ValueError("Invalid initial_state dimensions")


    trajectory_states = torch.empty((T + 1, K, state_dim), dtype=torch.float32, device=initial_state.device)
    trajectory_states[0] = batch_current_states

    # Clone to ensure we don't modify the input states if they are reused
    states = batch_current_states.clone() 

    # The loop itself is Python, but the core dynamics computation is JIT-compiled.
    for step in range(T):
        current_controls = controls_full[step] # (K, 2)
        
        # Call the JIT-compiled dynamics step function
        # The dynamics function handles state updates (including wrapping angles) internally.
        states = dynamics_step_fn(states, current_controls, dt, dynamics_params)
        
        trajectory_states[step + 1] = states
    return trajectory_states

class TorchPlannerBase(BaseController, ABC):
    """ Provides shared PyTorch functionality for trajectory planning """
    def __init__(self, controller_config: dict, experiment_config: dict, seed: int):
        super().__init__(controller_config, experiment_config, seed)

        # Setup device
        assert torch.cuda.is_available(), "CUDA is not available"
        self.device = torch.device('cuda')

        self.variable_velocity_mode = self.experiment_config.get('variable_velocity_mode', False)
        self._init_dynamics()

        # Costmap handling: Store the inputs for the current planning cycle.
        self.current_costmap_np = None
        # Tensors derived from the current costmap (cached during the cycle)
        self.current_costmap_tensor = None
        self.current_convolved_costmap = None

        # General planning parameters (from BaseController)
        self.T_horizon = self.num_steps # Use T_horizon for clarity in planning context

        self._init_footprint_kernel()
        self.set_seeds(self.seed)

    def _init_dynamics(self):
        """Initializes the dynamics model based on configuration."""
        # Prepare parameters for the dynamics model
        # pass relevant vehicle parameters from the experiment config
        dynamics_params = {
            'wheelbase': self.wheelbase,
            # Add other parameters here if needed by future models
        }
        
        # Instantiate the dynamics model using the factory
        # self.dynamics_model_name is initialized in BaseController
        try:
            self.dynamics: DynamicsModel = get_dynamics_model(self.dynamics_model_name, dynamics_params)
            print(f"Initialized dynamics model: {self.dynamics_model_name}")
            print(f"  Control type: {self.dynamics.control_type}")
        except ValueError as e:
            # Brittle: Fail immediately if dynamics initialization fails (e.g., missing parameters)
            raise RuntimeError(f"Failed to initialize dynamics model '{self.dynamics_model_name}': {e}")

        # Get the JIT-compiled step function and parameters
        self.dynamics_step_fn = self.dynamics.step
        self.dynamics_params_jit = self.dynamics.params # This is Dict[str, float]
        self._select_active_wrange()

    def _select_active_wrange(self):
        """Determines the appropriate control limits based on the dynamics model's control type."""
        if self.dynamics.control_type == 'angular_velocity':
            # Use the limits loaded from 'angular_velocity_range'
            self.active_wrange = self.angular_velocity_range
            source = 'angular_velocity_range'
        elif self.dynamics.control_type == 'steering_angle':
            # Use the limits loaded from 'steering_angle_range'
            self.active_wrange = self.steering_angle_range
            source = 'steering_angle_range'
        else: # fail if the dynamics model reports an unknown control type.
            raise RuntimeError(f"Unknown control type '{self.dynamics.control_type}' reported by dynamics model.")

        # Ensure the selected range is valid and configured.
        # check if the specific range (e.g., self.angular_velocity_range) was successfully loaded in BaseController.
        if self.active_wrange is None or self.active_wrange.size != 2:
            raise ValueError(
                f"Configuration error: Invalid or missing control limits for dynamics model '{self.dynamics_model_name}'.\n"
                f"Required configuration '{source}' (or the fallback 'wrange') must be defined as a list of 2 elements (e.g., [-1.0, 1.0])."
            )
        print(f"  Active control limits ({self.dynamics.control_type}): {self.active_wrange}")
    
    def _init_footprint_kernel(self):
        """Initializes the kernel for costmap convolution optimization"""
        self.effective_footprint_size = 2 * (self.robot_footprint_size // 2) + 1

        # The kernel is a square of ones matching the effective footprint size.
        # Shape: (out_channels, in_channels, H, W) = (1, 1, size, size)
        self.footprint_kernel = torch.ones(
            (1, 1, self.effective_footprint_size, self.effective_footprint_size),
            dtype=torch.float32,
            device=self.device
        )
        # Padding required for the convolution (S_eff is always odd)
        self.footprint_padding = self.effective_footprint_size // 2

    def set_seeds(self, seed):
        """Set all random seeds for deterministic execution."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic CUDA operations if possible
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def reset(self):
        """Reset shared planner state."""
        self.current_costmap_np = None
        self.current_costmap_tensor = None
        self.current_convolved_costmap = None
        self.set_seeds(self.seed)

    # =================================================================================
    # Input Processing
    # =================================================================================
    def _process_planner_input(self, planner_input: PlannerInput):
        """
        Processes the input provided by the ROS node and prepares internal state for the planning cycle.
        This replaces the simulation's _prepare_perception_inputs.
        """
        self.current_costmap_np = planner_input.inflated_costmap
        # Invalidate tensors derived from the previous costmap
        self.current_costmap_tensor = None
        self.current_convolved_costmap = None

    # =================================================================================
    # Dynamics and Rollouts
    # =================================================================================
    def _rollout_full_controls_torch(self, controls_full, initial_state):
        """
        General rollout using PyTorch dynamics. Calls the JIT-compiled implementation.
        controls_full shape: (T, K, 2) [v, w]
        initial_state shape: (3,) [x, y, theta] (must be a Tensor)
        Returns shape: (T+1, K, 3)
        """
        return _rollout_controls_functional(
            controls_full, 
            initial_state, 
            self.dt, 
            self.dynamics_step_fn,
            self.dynamics_params_jit
        )

    def _rollout_cuniform_controls_torch(self, controls_steer, initial_state, v_const):
        """
        Specialized rollout for C-Uniform initialization (constant velocity).
        controls_steer shape: (T, K) [w]
        initial_state shape: (3,) [x, y, theta] (must be a Tensor)
        Returns shape: (T+1, K, 3)
        """
        # check: This specialized rollout inherently assumes KST dynamics, 
        # because C-Uniform controllers are trained specifically for KST.
        if self.dynamics_model_name != 'kinematic_single_track':
            # This should ideally be caught during controller initialization, but added here for safety.
            raise RuntimeError(
                f"Attempted to use C-Uniform specialized rollout with non-KST dynamics ('{self.dynamics_model_name}'). "
                "C-Uniform controllers require 'kinematic_single_track'."
            )

        T, K = controls_steer.shape
        state_dim = initial_state.shape[0]

        batch_current_states = initial_state.unsqueeze(0).repeat(K, 1)
        trajectory_states = torch.empty((T + 1, K, state_dim), dtype=torch.float32, device=self.device)
        trajectory_states[0] = batch_current_states

        for step in range(T):
            # Dynamics expects (K, 1) shape for steering.
            current_steerings = controls_steer[step].unsqueeze(1)

            batch_current_states = dynamics_scalar_torch(
                batch_current_states, current_steerings, self.dt, v_const, self.wheelbase
            )
            trajectory_states[step + 1] = batch_current_states
        return trajectory_states

    # =================================================================================
    # Cost Evaluation
    # =================================================================================

    def _calculate_trajectory_costs(self, robot_frame_trajectories: torch.Tensor,
        goal_tensor: torch.Tensor, perturbed_controls_clamped: torch.Tensor = None) -> torch.Tensor:
        """
        (Fully Vectorized) Calculates the total cost for a batch of trajectories in the robot frame.
        Implements the logic of early exit (cost accumulation stops after goal reach or collision).
        robot_frame_trajectories shape: (T+1, K, 3)
        goal_tensor shape: (2,)
        Returns shape: (K,)
        """
        # Input shape: (T+1, K, 3)
        positions = robot_frame_trajectories[..., :2] # (T+1, K, 2)

        # --- Profiling Setup ---
        t_start = time.monotonic()
        profile_data = {}

        # 1. Calculate costs at every timestep (Stateless calculations)
        # 1a. Distance costs (T+1, K)
        # Calculate squared distance to goal for all trajectories at all timesteps.
        t0 = time.monotonic()
        dist_to_goal_squared = torch.sum((positions - goal_tensor)**2, dim=2)
        distance_costs = dist_to_goal_squared * self.dist_weight
        profile_data['cost_dist'] = (time.monotonic() - t0) * 1000

        # 1b. Obstacle costs (T+1, K)
        # Use the optimized, vectorized function which utilizes the pre-convolved costmap.
        t0 = time.monotonic()
        obstacle_costs_raw = self._calculate_robot_frame_costmap_cost(positions)
        profile_data['cost_obs_lookup'] = (time.monotonic() - t0) * 1000

        # 2. Determine stateful masks (Collision and Goal Reached)
        # 2a. Collision detection at each timestep (T+1, K)
        t0 = time.monotonic()
        collision_threshold = self.robot_footprint_area * self.collision_occupancy_ratio
        # Boolean mask indicating if a collision occurred exactly at time t.
        collided_at_t = obstacle_costs_raw > collision_threshold

        # 2b. Goal reached detection at each timestep (T+1, K)
        goal_tolerance_squared = self.goal_tolerance ** 2
        # Boolean mask indicating if the goal was reached exactly at time t.
        reached_at_t = dist_to_goal_squared <= goal_tolerance_squared

        # 3. Propagate masks forward in time (Stateful logic vectorization)
        # If an event occurs at time t, the mask must remain true for all t' > t.
        # use cumulative maximum (cummax) along the time dimension (dim=0)
        # This acts as a cumulative OR operation. Using .byte() is efficient.

        # (T+1, K). True if collided at or before t
        # .values extracts the tensor from the (values, indices) tuple returned by cummax
        collision_mask = torch.cummax(collided_at_t.byte(), dim=0).values.bool()
        # (T+1, K). True if reached goal at or before t
        goal_reached_mask = torch.cummax(reached_at_t.byte(), dim=0).values.bool()
        profile_data['cost_mask_prop'] = (time.monotonic() - t0) * 1000

        # 4. Apply masks to costs
        t0 = time.monotonic()
        # Combined termination mask (T+1, K)
        is_terminated = collision_mask | goal_reached_mask

        # 4a. Apply termination mask to distance costs
        # If terminated (collided or reached goal) at or before t, the distance cost at t is zeroed out
        distance_costs_masked = torch.where(
            is_terminated,
            torch.zeros_like(distance_costs),
            distance_costs
        )

        # 4b. Process and apply masks to obstacle costs
        # Calculate the scaled obstacle cost (used when not in a hard collision)
        scaled_obstacle_costs = obstacle_costs_raw * (self.obs_penalty / self.robot_footprint_area)

        # Apply collision mask: If collided at or before t, use full penalty, otherwise use scaled cost
        obstacle_costs_penalized = torch.where(
            collision_mask,
            torch.full_like(scaled_obstacle_costs, self.obs_penalty),
            scaled_obstacle_costs
        )

        # Apply goal reached mask: If goal reached at or before t, zero out obstacle cost (early exit)
        obstacle_costs_masked = torch.where(
            goal_reached_mask,
            torch.zeros_like(obstacle_costs_penalized),
            obstacle_costs_penalized
        )
        profile_data['cost_mask_apply'] = (time.monotonic() - t0) * 1000

        # 5. Calculate Total Cost (Sum over time dimension)
        # (T+1, K) -> (K,)
        t0 = time.monotonic()
        total_costs = torch.sum(distance_costs_masked + obstacle_costs_masked, dim=0)

        perturbed_controls_clamped = None
        if perturbed_controls_clamped is not None:
            # Ensure the weights are a tensor on the correct device.
            # `self.action_cost_weight` is expected to be a list or array like [weight_v, weight_steer].
            action_weights = torch.tensor(self.action_cost_weight, dtype=torch.float32, device=self.device)
            # `perturbed_controls_clamped` has shape (T, K, 2).
            # `torch.diff` computes u_t - u_{t-1}, result `action_diffs` has shape (T-1, K, 2).
            action_diffs = torch.diff(perturbed_controls_clamped, n=1, dim=0)
            # Calculate the squared difference for each action dimension. Shape: (T-1, K, 2)
            action_diffs_squared = action_diffs**2
            # Apply the respective weights to each action dimension via broadcasting.
            # (T-1, K, 2) * (2,) -> (T-1, K, 2)
            weighted_action_diffs_squared = action_diffs_squared * action_weights
            # Sum the weighted penalties across the action dimensions for each time step. Result shape: (T-1, K)
            per_step_action_cost = torch.sum(weighted_action_diffs_squared, dim=2)
            # Sum the costs over the entire time horizon for each trajectory. Result shape: (K,)
            action_costs = torch.sum(per_step_action_cost, dim=0)
            total_costs += action_costs # add this smoothness cost to the total cost.

        # 6. Terminal costs
        # Terminal costs apply only for trajectories that never reached the goal AND were always safe.

        # The final state of the masks (at T) tells us if the event ever happened.
        ever_collided = collision_mask[-1] # (K,)
        ever_reached_goal = goal_reached_mask[-1] # (K,)

        is_safe_and_unsuccessful = ~(ever_reached_goal | ever_collided)

        # Calculate terminal distance based on the very last position (already calculated)
        terminal_distances_squared = dist_to_goal_squared[-1]
        terminal_costs = is_safe_and_unsuccessful.float() * terminal_distances_squared * self.terminal_weight
        total_costs += terminal_costs
        profile_data['cost_sum_terminal'] = (time.monotonic() - t0) * 1000
        profile_data['cost_total'] = (time.monotonic() - t_start) * 1000
        # Attach profile data to the costs tensor itself for access in the calling function
        # This avoids changing the function signature.
        if not hasattr(total_costs, '_profile_data'):
            setattr(total_costs, '_profile_data', {})
        total_costs._profile_data.update(profile_data)
        return total_costs

    def _ensure_convolved_costmap(self):
        """ Ensures the costmap tensor is loaded and pre-convolved with the robot footprint """
        # Check if the raw costmap needs updating or loading
        if self.current_costmap_tensor is None and self.current_costmap_np is not None:
            self.current_costmap_tensor = torch.from_numpy(self.current_costmap_np).float().to(self.device)
            # Perform 2D convolution
            # Input shape: (H, W) -> (1, 1, H, W)
            input_map = self.current_costmap_tensor.unsqueeze(0).unsqueeze(0)

            # --- Boundary Conditions ---
            # used index clamping when the footprint extended beyond the map bounds, equivalent to 'replication' padding
            # manually pad the input map before convolution
            P = self.footprint_padding
            padded_input_map = F.pad(input_map, (P, P, P, P), mode='replicate')

            # Apply the convolution using the pre-initialized kernel with padding=0 ('valid')
            convolved_output = F.conv2d(
                padded_input_map,
                self.footprint_kernel,
                padding=0
            )

            # (1, 1, H, W) -> (H, W)
            self.current_convolved_costmap = convolved_output.squeeze(0).squeeze(0)
        
        # Handle the case where no map is available
        if self.current_costmap_np is None:
            self.current_costmap_tensor = None
            self.current_convolved_costmap = None

    def _calculate_robot_frame_costmap_cost(self, robot_frame_positions: torch.Tensor) -> torch.Tensor:
        """
        (Vectorized) Calculate obstacle costs using the pre-convolved local costmap.
        Input shape: (..., 2) (e.g., (T+1, K, 2) or (K, 2))
        Output shape: (...)
        """
        # Ensure the costmap is loaded and convolved
        self._ensure_convolved_costmap()

        if self.current_convolved_costmap is None:
            # Return zero costs if no costmap loaded
            return torch.zeros(robot_frame_positions.shape[:-1], device=self.device)

        # The convolved costmap holds the total footprint cost at each cell
        # perform a fast lookup using vectorized indexing
        resolution = self.local_costmap_resolution

        # Robot frame: robot at center of the grid, facing +X direction
        costmap_size = self.current_convolved_costmap.shape[0]
        center = costmap_size // 2

        # Convert robot frame coordinates (..., 2) to grid indices (...), use .long() for truncation
        grid_x = (center + robot_frame_positions[..., 0] / resolution).long()
        grid_y = (center + robot_frame_positions[..., 1] / resolution).long()
        # grid_y = (center - robot_frame_positions[..., 1] / resolution).long()  # Y-flip for image coordinates

        # Clamp indices to valid range (Conservative approach)
        grid_x = torch.clamp(grid_x, 0, costmap_size - 1)
        grid_y = torch.clamp(grid_y, 0, costmap_size - 1)
    
        # Index the convolved costmap (Vectorized gathering - single GPU operation)
        costs = self.current_convolved_costmap[grid_y, grid_x]
        return costs