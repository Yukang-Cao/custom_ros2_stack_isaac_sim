# lib/cu_mppi_controller.py
"""
Hybrid C-Uniform MPPI (CU-MPPI) Controller.
Combines C-Uniform sampling for initialization with MPPI for refinement.
(Pure PyTorch Implementation)
"""

import numpy as np
import torch
import time
from typing import Tuple, Dict, Any

# Inherit from CUniformController to reuse its methods
from .nn_cuniform_controller import CUniformController
from .mppi_pytorch_controller import MPPIPyTorchController, PlannerInput

class CUMPPiController(CUniformController):
    """
    Manages the hybrid CU-MPPI pipeline entirely within PyTorch.
    """
    def __init__(self, controller_config: dict, experiment_config: dict = None,
                 type_override: int = None, mppi_config: dict = None,
                 mppi_type_override: int = None, seed: int = None):

        # Initialize the C-Uniform part (models, parameters, etc.)
        super().__init__(controller_config, experiment_config, type_override, seed)

        if mppi_config is None:
            raise ValueError("mppi_config must be provided via CONTROLLER_FACTORY overrides.")

        # Initialize the MPPI Refinement part using composition
        # We use the MPPIPyTorchController instance to handle the refinement logic.
        self.mppi_refiner = MPPIPyTorchController(
            controller_config=mppi_config,
            experiment_config=experiment_config,
            type_override=mppi_type_override,
            seed=seed
        )
        
        self.mppi_type = self.mppi_refiner.mppi_type

        # Split sampling budget between initialization and refinement
        total_budget = self.num_rollouts
        # initialization_budget_ratio is initialized in BaseController
        init_budget = int(total_budget * self.initialization_budget_ratio)
        mppi_budget = total_budget - init_budget

        # Brittle check: Validate budgets to prevent runtime errors (e.g., torch.argmin on empty tensor)
        if init_budget <= 0: # This occurs if the total budget is too low or the ratio is too close to 0.0
            raise ValueError(f"CU-MPPI Initialization budget is {init_budget}. Increase total budget ({total_budget}) or initialization_budget_ratio ({self.initialization_budget_ratio:.2f}).")
        if mppi_budget <= 0:
            raise ValueError(f"CU-MPPI Refinement budget is {mppi_budget}. Increase total budget ({total_budget}) or decrease initialization_budget_ratio ({self.initialization_budget_ratio:.2f}).")
        
        # Configure C-Uniform budget
        self.num_trajectories_init = init_budget
        
        # Override parent class num_trajectories for initialization phase
        self.num_trajectories = self.num_trajectories_init
        
        # Configure MPPI budget
        self.num_trajectories_mppi = mppi_budget
        self.mppi_refiner.K = mppi_budget # Set the sample size (K) for the refiner

        # Helper tensor for initialization phase
        self.steering_actions_tensor = torch.tensor(self.actions[:, 0], dtype=torch.float32, device=self.device)
        print(f"CU-MPPI Controller (All-PyTorch) initialized. CU-Type: {self.controller_type}, MPPI-Type: {self.mppi_type}")
        print(f"Budget allocation - Total: {total_budget}, CU Init: {init_budget}, MPPI: {mppi_budget}")
    
    def reset(self):
        """Reset controller state for new environment."""
        super().reset() # Reset C-Uniform part
        self.mppi_refiner.reset() # Reset MPPI part


    def get_control_action(self, planner_input: PlannerInput) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute control action using the hybrid CU-MPPI approach."""
        # Initialize profiling dictionary
        # Ensure previous cycle's GPU work is finished before starting measurement
        # 1. Process inputs for the C-Uniform part (self)
        self._process_planner_input(planner_input)

        # Determine velocity for initialization
        current_velocity = planner_input.current_velocity
        if current_velocity < self.vrange[0] or np.isnan(current_velocity) or current_velocity < 0:
            init_velocity = self.vrange[0]
        else:
            # Clamp velocity to the defined range (safety measure)
            init_velocity = max(self.vrange[0], min(self.vrange[1], current_velocity))

        # Extract model inputs
        sdf_tensor = planner_input.sdf_tensor
        feature_extractor = self.feature_extractor if self.controller_type == 1 else None

        # Validation for supervised mode
        if self.controller_type == 1 and sdf_tensor is None:
            raise ValueError("SDF tensor missing from planner input for supervised mode.")

        # 2. C-Uniform Initialization Phase (Exploration)
        cu_trajectories_robot, cu_action_indices = self._sample_trajectories_robot_frame(
            sdf_tensor, feature_extractor, init_velocity
        )

        # Evaluate C-Uniform samples and find the best initial nominal control sequence
        u_nominal_initial, best_idx_cu, cu_cost_profile = self._find_best_initialization(
            cu_trajectories_robot, cu_action_indices, planner_input.local_goal, init_velocity
        ) # NOTE: take ~3ms, cost evaluation is very fast

        # 3. MPPI Refinement Phase (Exploitation) - Delegated to MPPIPyTorchController
        # Ensure the MPPI refiner uses the same perception data.
        self.mppi_refiner._process_planner_input(planner_input)

        # Run the optimization routine from the dedicated MPPI instance
        u_nominal_final, mppi_trajectories = self.mppi_refiner._run_mppi_optimization(
            u_nominal_initial, planner_input.local_goal
        )

        # 4. Final Control Selection
        # The result of MPPI is the weighted average (the updated nominal trajectory)
        control_action_np = u_nominal_final[0].cpu().numpy()

        # 5. Visualization
        # Check if visualization preparation is needed
        if self.viz_config.get('enabled', True) and self.viz_config.get('visualize_trajectories', True):
             info = self._prepare_visualization_info_hybrid(
                cu_trajectories_robot, best_idx_cu,
                mppi_trajectories, u_nominal_final
            )
        else:
            # Minimal info if visualization is disabled
            info = {'state_rollouts_robot_frame': None, 'is_hybrid': True}
        return control_action_np, info

    def _find_best_initialization(self, trajectories, action_indices, local_goal, current_velocity):
        """
        Evaluates pre-computed C-Uniform trajectories, finds the best one,
        and constructs the corresponding full nominal control sequence.
        """
        # 1. Evaluate costs (using the provided trajectories)
        goal_tensor = torch.from_numpy(local_goal).float().to(self.device)

        # Calculate costs (Reused from parent CUniformController)
        total_costs = self._calculate_trajectory_costs(trajectories, goal_tensor)

        # Extract profiling data attached to the tensor by TorchPlannerBase
        cost_profile = getattr(total_costs, '_profile_data', {})
        best_idx = torch.argmin(total_costs).item()

        # 2. Extract the best action sequence (Steering angles)
        # action_indices shape: (T, K). Get the sequence for the best index.
        best_action_indices_sequence = action_indices[:, best_idx] # (T,)
        # Efficiently convert indices to values for the single best trajectory.
        u_nominal_steer = self.steering_actions_tensor[best_action_indices_sequence] # (T,)

        # 3. Combine with the CURRENT VELOCITY to form the full nominal control sequence [T, 2]

        constant_vel = torch.full_like(u_nominal_steer, current_velocity)
        u_nominal_full = torch.stack((constant_vel, u_nominal_steer), dim=1)

        return u_nominal_full, best_idx, cost_profile

    # === Visualization ===
    # Visualization remains specific to the hybrid approach, combining outputs 
    # from C-Uniform initialization and MPPI refinement.
    def _prepare_visualization_info_hybrid(self, cu_trajectories_robot, best_idx_cu, 
                                           mppi_trajectories_robot, u_nominal_final):
        """
        Prepare visualization data for the hybrid approach (4-color visualization)
        Packages all trajectories (in ROBOT FRAME) into a dictionary
        """
        # --- 1. C-Uniform Samples (All) ---
        # Transform (T+1, N, 3) -> (N, T+1, 3)
        cu_traj_robot_formatted = cu_trajectories_robot.permute(1, 0, 2)
        
        # Limit visualization count
        num_vis = min(self.num_vis_rollouts, cu_traj_robot_formatted.shape[0])
        cu_samples_np = cu_traj_robot_formatted[:num_vis].cpu().numpy()

        # --- 2. C-Uniform Best Sample ---
        cu_best_np = cu_traj_robot_formatted[best_idx_cu].cpu().numpy()

        # --- 3. MPPI Samples (Last iteration) ---
        # Transform (T+1, N, 3) -> (N, T+1, 3)
        mppi_traj_robot_formatted = mppi_trajectories_robot.permute(1, 0, 2) 

        # Limit visualization count
        num_vis_mppi = min(self.num_vis_rollouts, mppi_traj_robot_formatted.shape[0])
        mppi_samples_np = mppi_traj_robot_formatted[:num_vis_mppi].cpu().numpy()

        # --- 4. MPPI Final (Weighted Average) ---
        # Rollout the final nominal control sequence
        robot_frame_initial_state = torch.zeros(3, device=self.device, dtype=torch.float32)
        u_nominal_reshaped = u_nominal_final.unsqueeze(1)
        # Use the refiner's rollout function as it handles variable velocity correctly
        nominal_traj_robot = self.mppi_refiner._rollout_full_controls_torch(u_nominal_reshaped, robot_frame_initial_state)
        
        # (T+1, 1, 3) -> (T+1, 3)
        mppi_nominal_np = nominal_traj_robot.squeeze(1).cpu().numpy()

        # --- 5. Package Data ---
        # Package into a dictionary structure for the visualizer
        trajectory_data = {
            'cu_samples': cu_samples_np,
            'cu_best': cu_best_np,
            'mppi_samples': mppi_samples_np,
            'mppi_nominal': mppi_nominal_np,
        }
        
        vis_data = {
            # The ROS node uses these keys for visualization
            'state_rollouts_robot_frame': trajectory_data,
            'is_hybrid': True,
        }

        # Apply angle wrapping to all trajectory components
        for key, traj_array in trajectory_data.items():
            if isinstance(traj_array, np.ndarray) and traj_array.size > 0:
                if traj_array.ndim == 3: # Samples (N, T, 3)
                    traj_array[:, :, 2] = np.arctan2(np.sin(traj_array[:, :, 2]), np.cos(traj_array[:, :, 2]))
                elif traj_array.ndim == 2: # Best/Final (T, 3)
                    traj_array[:, 2] = np.arctan2(np.sin(traj_array[:, 2]), np.cos(traj_array[:, 2]))
        
        return {
            'local_costmap': self.current_costmap_np,
            'visualize_costmap': True,
            'visualize_all_trajectories': True,
            'state_rollouts_robot_frame': trajectory_data, # Pass the trajectory data directly
            'is_hybrid': True # Indicate this is hybrid visualization
        }