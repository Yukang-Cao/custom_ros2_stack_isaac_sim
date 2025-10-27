# lib/nn_cuniform_controller.py
"""
Neural C-Uniform controllers using PyTorch.
   0: Unsupervised NN C-Uniform, 
   1: Supervised Map conditioned NN C-Uniform
Relies on TorchPlannerBase for standardized dynamics and cost evaluation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import sys
import time
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from .torch_planner_base import TorchPlannerBase, PlannerInput
from .utils.dynamics import cuda_dynamics_KS_3d_scalar_v_batched as dynamics_scalar_torch
from .utils.model_parts import MapPixelFeature, MapAct_PixelInterpolated, MapAct

TENSORRT_AVAILABLE = False
try:# assuming tensorrt_utils.py is placed in lib/utils/
    from .utils.tensorrt_utils import TensorRTEngineWrapper, convert_pytorch_to_onnx, build_tensorrt_engine, trt
    if trt:
        TENSORRT_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: TensorRT or its utilities not found ({e}). Falling back to PyTorch.")


# Debug timestamp counter
_debug_timestamp = 0

def load_compiled_state_dict(model, path):
    """Loads a state_dict saved from a torch.compile'd model."""
    state_dict = torch.load(path, map_location='cuda', weights_only=True)
    
    # Use a dictionary comprehension to strip the '_orig_mod.' prefix
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    print(f"Successfully loaded compiled weights into {model.__class__.__name__} from {path}")

class CUniformController(TorchPlannerBase):
    """C-Uniform controller with neural network guidance."""
    
    def __init__(self, controller_config: dict, experiment_config: dict = None, type_override: int = None, seed: int = None):
        """Initialize C-Uniform controller."""
        if seed is None:
            seed = experiment_config.get('seed', 2025) if experiment_config else 2025
        super().__init__(controller_config, experiment_config, seed)

        # Configuration flags for TensorRT
        self.use_tensorrt = controller_config.get('use_tensorrt', False) and TENSORRT_AVAILABLE
        self.use_fp16 = controller_config.get('tensorrt_fp16', True)

        # 0: Unsupervised, 1: Supervised and map conditioned
        # Allow controller_type to be overridden, otherwise use the value from the config file
        self.controller_type = type_override if type_override is not None else self.config['controller_type']
        
        self.num_a = self.config['num_a']
        self.num_steering_angle = self.config['num_steering_angle']
        self.arange = self.config['arange'] 

        self.feature_extractor_path = self.config['feature_extractor_path'] if self.controller_type == 1 else None
        self.model_path = (self.config['map_conditioned_model_path'] if self.controller_type == 1 
                          else self.config['unsupervised_cuniform_model_path'])

        self.rng = np.random.RandomState(self.seed)
        self.actions = self.generate_actions(
            self.arange, self.num_a, self.active_wrange, self.num_steering_angle, False
        )
        self.actions_tensor = torch.tensor(self.actions[:, 0], dtype=torch.float32, device=self.device)
        
        # Load models if paths are provided - instantiate classes first, then load state dicts
        self.model = None
        self.feature_extractor = None

        # Define Model Architecture Parameters (Must match training configuration)
        self.feature_dim = 64 # Default for MapPixelFeature
        self.state_dim = 4    # [x, y, sin(t), cos(t)]

        if self.model_path and os.path.exists(self.model_path):
            if self.use_tensorrt and self.controller_type == 1:
                # Attempt to load or build TensorRT engines for map-conditioned models
                self._load_or_build_tensorrt_engines()
            else: # Load PyTorch models if TensorRT is disabled or not applicable (e.g., type 0)
                self._load_pytorch_models()
        else:
            if self.model_path:
                print(f"WARNING: Model path not found: {self.model_path}")
            
        # Planning parameters - get from experiment config
        self.num_trajectories = self.num_rollouts  # Use shared parameter from experiment config
        self.trajectory_length = self.num_steps     # Use shared parameter from experiment config
        self.num_vis_trajectories = min(self.num_vis_rollouts, self.num_trajectories)  # Limit visualization like MPPI
        self._validate_model_action_space()
        
        # --- Status Report ---
        # We check the type of the loaded model to determine if TRT is active
        is_actor_trt = TENSORRT_AVAILABLE and isinstance(self.model, TensorRTEngineWrapper)
        is_fe_trt = TENSORRT_AVAILABLE and isinstance(self.feature_extractor, TensorRTEngineWrapper)

        print(f"C-Uniform Controller initialized (type: {self.controller_type}).")
        print(f"  Actor Optimization (TensorRT): {is_actor_trt}")

        if self.controller_type == 1:
            assert is_actor_trt, "\033[91mActor model must be optimized with TensorRT\033[0m"
            print(f"  Feature Extractor Optimization (TensorRT): {is_fe_trt}")
            assert is_fe_trt, "\033[91mFeature Extractor model must be optimized with TensorRT\033[0m"

    # =================================================================================
    # Model Loading and Optimization Helpers
    # =================================================================================
    def _load_pytorch_models(self):
        """Loads the PyTorch models (Feature Extractor and Actor). Used for conversion and fallback."""
        print("Loading PyTorch models (for fallback or TensorRT conversion)...")
        
        # --- Load Actor Model ---
        if self.controller_type == 0:
            # Unsupervised model - load full model object
            utils_path = os.path.join(os.path.dirname(__file__), 'utils')
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            
            if 'model_parts' not in sys.modules:
                from .utils import model_parts
                sys.modules['model_parts'] = model_parts
            
            self.model = torch.load(self.model_path, map_location='cuda', weights_only=False)
        else:
            # Map-conditioned model - instantiate and load state dict
            num_actions = len(self.actions)
            # Instantiate the model architecture (MapAct_PixelInterpolated)
            self.model = MapAct_PixelInterpolated(
                num_actions=num_actions, 
                state_dim=self.state_dim, 
                feature_channels=self.feature_dim
            )
            load_compiled_state_dict(self.model, self.model_path)
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded PyTorch Actor model from: {self.model_path}")
        
        # --- Load Feature Extractor (only for type 1) ---
        if self.controller_type == 1:
            if self.feature_extractor_path and os.path.exists(self.feature_extractor_path):
                # Instantiate the model architecture (MapPixelFeature)
                self.feature_extractor = MapPixelFeature(feature_dim=self.feature_dim)
                load_compiled_state_dict(self.feature_extractor, self.feature_extractor_path)
                self.feature_extractor.to(self.device)
                self.feature_extractor.eval()
                print(f"Loaded PyTorch Feature Extractor from: {self.feature_extractor_path}")
            else:
                print(f"ERROR: Feature extractor path missing or invalid: {self.feature_extractor_path}")

    def _load_or_build_tensorrt_engines(self):
        """Attempts to load TensorRT engines for both Actor and Feature Extractor, building them if necessary."""
        
        # 1. Ensure PyTorch models are loaded first (needed for conversion and as fallback)
        self._load_pytorch_models()
        
        if self.model is None or self.feature_extractor is None:
            print("ERROR: PyTorch models could not be loaded. Cannot proceed with TensorRT optimization.")
            return

        # Calculate the typical batch size for the Actor model based on the initialization budget ratio
        # This is used as the 'optimal' batch size (N_opt) for the TRT optimization profile.
        N_opt = max(1, int(self.num_rollouts * self.initialization_budget_ratio))
        # The maximum batch size (N_max) is the total number of rollouts.
        N_max = self.num_rollouts

        # 2. Optimize Actor Model (MapAct_PixelInterpolated) - The main bottleneck.
        # Inputs: state (B, 4), interpolated_features (B, 64)
        actor_engine = self._optimize_model_with_trt(
            model=self.model,
            model_path=self.model_path,
            input_shapes={'state': (self.state_dim,), 'interpolated_features': (self.feature_dim,)},
            input_names=['state', 'interpolated_features'],
            output_names=['probabilities'],
            optimization_batch_size=N_opt,
            max_batch_size=N_max
        )

        # 3. Optimize Feature Extractor Model (MapPixelFeature)
        # Input: SDF tensor (1, 1, H, W). Batch size is always 1 (Fixed).
        map_size = self.local_costmap_size
        feature_extractor_engine = self._optimize_model_with_trt(
            model=self.feature_extractor,
            model_path=self.feature_extractor_path,
            # Note: Input shape includes channel dimension (1)
            input_shapes={'sdf_input': (1, map_size, map_size)}, 
            input_names=['sdf_input'],
            output_names=['dense_features'],
            optimization_batch_size=1,
            max_batch_size=1
        )

        # 4. Replace PyTorch models with TRT engines if successful
        if actor_engine:
            # Free up memory from the original PyTorch model
            del self.model
            torch.cuda.empty_cache()
            self.model = actor_engine
        
        if feature_extractor_engine:
            del self.feature_extractor
            torch.cuda.empty_cache()
            self.feature_extractor = feature_extractor_engine

    def _optimize_model_with_trt(self, model, model_path, input_shapes, input_names, output_names, optimization_batch_size, max_batch_size):
        """Helper function to manage the TRT optimization pipeline for a specific model."""
        
        # Define paths for ONNX and TensorRT engine
        base_path = os.path.splitext(model_path)[0]
        # Use a descriptive naming convention for ONNX and Engine files
        onnx_path = f"{base_path}.onnx"
        
        # Include key parameters in the engine name for caching (Max Batch Size and Precision)
        precision_str = "fp16" if self.use_fp16 else "fp32"
        engine_path = f"{base_path}_maxB{max_batch_size}_{precision_str}.engine"

        print(f"\n--- TensorRT Optimization for: {os.path.basename(model_path)} ---")

        # 1. Try loading the existing engine
        try:
            # The wrapper's constructor handles the loading logic.
            engine_wrapper = TensorRTEngineWrapper(engine_path, self.device)
            # If the engine file didn't exist, the wrapper's engine attribute will be None.
            if engine_wrapper.engine is not None:
                print(f"Successfully loaded existing TensorRT engine: {engine_path}")
                return engine_wrapper
        except Exception as e:
            # If loading fails due to corruption or incompatibility, proceed to rebuild.
            print(f"Could not load TensorRT engine {engine_path} (Error: {e}). Proceeding to rebuild.")

        print(f"Engine not found or failed to load. Starting build process...")

        # 2. Prepare dummy inputs for ONNX tracing
        dummy_inputs_list = []
        try:
            # Use the optimization batch size for tracing
            for name in input_names:
                shape = input_shapes[name]
                # Prepend batch dimension
                full_shape = (optimization_batch_size,) + shape
                # Use float32 for tracing, conversion handled later
                dummy_input = torch.randn(full_shape, dtype=torch.float32, device=self.device)
                dummy_inputs_list.append(dummy_input)
            dummy_inputs = tuple(dummy_inputs_list)
        except Exception as e:
            print(f"ERROR: Failed to create dummy inputs for tracing: {e}")
            return None

        # 3. Convert to ONNX
        try:
            # Ensure we use the underlying PyTorch module if it was torch.compiled
            target_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            convert_pytorch_to_onnx(target_model, dummy_inputs, onnx_path, input_names, output_names)
        except Exception as e:
            print(f"ERROR: Failed to convert model to ONNX: {e}")
            # Print full traceback for debugging ONNX conversion issues
            import traceback
            traceback.print_exc()
            return None

        # 4. Build TensorRT engine
        try:
            # Enable FP16 optimization if configured.
            build_tensorrt_engine(
                onnx_path, engine_path, 
                optimization_batch_size=optimization_batch_size,
                max_batch_size=max_batch_size, 
                fp16_mode=self.use_fp16
            )
        except Exception as e:
            print(f"ERROR: Failed to build TensorRT engine: {e}")
            return None

        # 5. Load the newly built engine
        try:
            engine_wrapper = TensorRTEngineWrapper(engine_path, self.device)
            if engine_wrapper.engine is None:
                 raise RuntimeError("Engine built but failed to load.")
            return engine_wrapper
        except Exception as e:
            print(f"ERROR: Failed to load newly built TensorRT engine: {e}")
            return None

    def _validate_model_action_space(self):
        """Checks if the loaded model output matches the configured action space."""
        # If using TensorRT, the check is slightly different
        if TENSORRT_AVAILABLE and isinstance(self.model, TensorRTEngineWrapper):
            expected_num_actions = len(self.actions)
            # Assuming the last output defines the action probabilities
            # We check the shape definition (which may contain -1)
            output_shape = self.model.outputs_meta[-1]['shape_definition']
            # The last dimension is the number of actions
            actual_num_actions = output_shape[-1]
            
            if actual_num_actions != expected_num_actions:
                error_msg = (
                    f"CRITICAL ERROR: Action space mismatch in TensorRT Engine!\n"
                    f"Configuration defines {expected_num_actions} actions, but Engine outputs {actual_num_actions}.\n"
                    f"Ensure the PyTorch model used for conversion matched the configuration."
                )
                print(error_msg)
                raise RuntimeError(error_msg)
            return

        if self.model is None:
            return

        # Determine the expected number of actions from the configuration
        expected_num_actions = len(self.actions)
        
        # Inspect the model's final layer. Access the underlying module if compiled.
        target_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        # Assuming the final layer is named 'out' (as defined in model_parts.py)
        actual_num_actions = None
        if hasattr(target_model, 'out') and isinstance(target_model.out, torch.nn.Linear):
             actual_num_actions = target_model.out.out_features
        
        if actual_num_actions is not None and actual_num_actions != expected_num_actions:
            error_msg = (
                f"CRITICAL ERROR: Action space mismatch!\n"
                f"Configuration defines {expected_num_actions} actions.\n"
                f"The loaded model ('{self.model_path}') outputs {actual_num_actions} actions.\n"
                f"This causes 'index out of bounds' errors. Ensure runtime config matches the training config."
            )
            print(error_msg)
            raise RuntimeError(error_msg)
        elif actual_num_actions is None:
             print("WARNING: Could not automatically verify model output size. Assuming it matches the configuration.")
    
    def reset(self):
        """Reset controller state for new environment."""
        super().reset()
        self.current_state = np.zeros(3, dtype=np.float32)  # reset current state
        self.rng = np.random.RandomState(self.seed)  # Reset generator for reproducibility
    
    def generate_actions(self, arange, num_a, steering_angle_range, num_steering_angle, deg2rad_conversion):
        """
        Generates a NumPy array of all possible (steering, velocity) action pairs
        based on the given ranges and discretization parameters.
        Returns: np.ndarray: Array of shape (num_steering_angle * num_v, 2) where each row is [steering, acceleration]
        """
        if deg2rad_conversion:
            steering_values = np.deg2rad(np.linspace(steering_angle_range[0], steering_angle_range[1], num_steering_angle))
        else:
            steering_values = np.linspace(steering_angle_range[0], steering_angle_range[1], num_steering_angle)
        a_value = np.linspace(arange[0], arange[1], num_a)
        actions = np.array([[s, a] for s in steering_values for a in a_value], dtype=np.float32)
        return actions
    
    def get_control_action(self, planner_input: PlannerInput) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute control action using C-Uniform sampling.
        
        Args:
            planner_input: Standardized input containing local goal, velocity, and perception data.
            
        Returns:
            Tuple of:
                - Control action [velocity, angular_velocity]
                - Info dictionary for visualization and debugging
        """
        # 1. Process inputs (handles costmap preparation internally)
        self._process_planner_input(planner_input)

        # 2. Prepare model inputs
        sdf_tensor = planner_input.sdf_tensor
        feature_extractor = self.feature_extractor if self.controller_type == 1 else None

        # Validation for supervised mode
        if self.controller_type == 1:
            if sdf_tensor is None:
                raise ValueError("SDF tensor missing from planner input for supervised mode.")
            if feature_extractor is None:
                raise ValueError("Feature extractor not loaded for supervised mode.")

        # 3. Sample trajectories
        # Pure C-Uniform uses the configured fixed velocity.
        velocity = self.vrange[0]

        robot_frame_trajectories, trajectory_actions = self._sample_trajectories_robot_frame(
            sdf_tensor, feature_extractor, velocity
        )
        
        # 4. Evaluate trajectories
        control_action, best_trajectory_idx = self._evaluate_robot_frame_trajectories(
            robot_frame_trajectories, trajectory_actions, planner_input.local_goal
        )
        
        # 5. Prepare visualization data (Robot Frame)
        # Skip preparation if visualization is disabled
        if self.viz_config.get('enabled', True) and self.viz_config.get('visualize_trajectories', True):
            info = self._prepare_visualization_info(robot_frame_trajectories, trajectory_actions, best_trajectory_idx)
        else:
            info = {
                'state_rollouts_robot_frame': None,
                'is_hybrid': False
            }

        return control_action, info
    
    def _sample_trajectories_robot_frame(self, sdf_tensor: torch.Tensor, 
                                        feature_extractor, velocity: float) -> Tuple:
        """Sample trajectories in robot frame."""
        # Validate models are loaded
        if self.model is None:
            mode = "unsupervised" if self.controller_type == 0 else "supervised map-conditioned"
            raise ValueError(f"Model not loaded for {mode} mode")
        
        if self.controller_type == 1 and feature_extractor is None:
            raise ValueError("Feature extractor not loaded for supervised mode")
        
        # Generate action space
        steering_actions = self.actions[:, 0]  # Extract steering angles from pre-generated actions
        
        # Sample trajectories (always start from robot frame origin)
        robot_frame_initial_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Robot frame origin
        robot_frame_trajectories, trajectory_actions = self.sample_trajectories_cuniform(
            initial_state=robot_frame_initial_state,
            actions=steering_actions,
            dynamics_cuda=dynamics_scalar_torch,
            num_trajectories=self.num_trajectories,
            trajectory_length=self.trajectory_length,
            model=self.model,
            feature_extractor=feature_extractor,
            sdf_tensor=sdf_tensor,
            wheelbase=self.wheelbase,
            uniform_sampling=False,
            velocity_override=velocity, 
        )
        return robot_frame_trajectories, trajectory_actions
    
    def _evaluate_robot_frame_trajectories(self, robot_frame_trajectories: torch.Tensor, 
                                          trajectory_actions: torch.Tensor, 
                                          local_goal: np.ndarray) -> Tuple[np.ndarray, int]:
        """Evaluate robot frame trajectories against robot frame costmap and select best action.
        Cost calculation exits early when goal is reached - we don't care about what happens after goal."""
        goal_tensor = torch.from_numpy(local_goal).float().to(self.device)


        total_costs = self._calculate_trajectory_costs(robot_frame_trajectories, goal_tensor)
        # Find best trajectory (minimum cost)
        best_idx = torch.argmin(total_costs).item()

        # Get the first action from the best trajectory
        steering_actions = self.actions[:, 0]
        best_action_idx = trajectory_actions[0, best_idx].item()
        best_action = steering_actions[best_action_idx]

        return np.array([self.vrange[0], best_action], dtype=np.float32), best_idx
    
    def _prepare_visualization_info(self, trajectories: torch.Tensor, trajectory_actions: torch.Tensor, best_trajectory_idx: int) -> Dict[str, Any]:
        """Prepare visualization data in the robot frame."""
        # The ROS node will handle the actual visualization (RViz) and transformation if needed.
        assert trajectories is not None, "trajectories should not be None"
        
        # Convert trajectories to MPPI format: (num_trajectories, timesteps, 3)
        all_trajectories = trajectories.cpu().numpy()  # Shape: (timesteps, num_trajectories, 3)
        all_trajectories_mppi_format = all_trajectories.transpose(1, 0, 2)
        
        # Create visualization array with best trajectory at index 0
        vis_rollouts = np.zeros((self.num_vis_trajectories, self.trajectory_length + 1, 3), dtype=np.float32)
        
        # Put best trajectory at index 0 (green line in visualization)
        vis_rollouts[0] = all_trajectories_mppi_format[best_trajectory_idx]
        
        # Add remaining trajectories (black lines in visualization)
        remaining_indices = [i for i in range(min(self.num_vis_trajectories-1, all_trajectories_mppi_format.shape[0])) 
                            if i != best_trajectory_idx][:self.num_vis_trajectories-1]
        for i, idx in enumerate(remaining_indices):
            vis_rollouts[i+1] = all_trajectories_mppi_format[idx]
        
        # Apply angle wrapping to prevent visualization artifacts
        vis_rollouts[:, :, 2] = np.arctan2(np.sin(vis_rollouts[:, :, 2]), np.cos(vis_rollouts[:, :, 2]))
        state_rollouts = vis_rollouts
        
        return {
            'state_rollouts_robot_frame': state_rollouts,
            'is_hybrid': False
        }
    
    def batch_states_to_grid(self, states, sdf_shape, resolution):
        """
        Converts robot states to normalized grid coordinates for F.grid_sample.

        Args:
            states (np.ndarray or torch.Tensor): Array of shape (N, D) where D>=2.
                Only the first two dimensions (x,y) are used.
            sdf_shape (tuple): The shape (H, W) of the SDF grid.
            resolution (float): The size of each grid cell (meters per cell).

        Returns:
            torch.Tensor: A grid of shape (1, N, 1, 2) with values in [-1, 1],
                        ready to be used as the grid input for F.grid_sample.
        """
        if isinstance(states, np.ndarray): # If input is numpy, convert to torch tensor.
            states = torch.from_numpy(states).float()
        xy = states[:, :2]  # shape: (N, 2)
        H, W = sdf_shape
        center_x = (W - 1) / 2.0
        center_y = (H - 1) / 2.0

        # Convert world coordinates (in meters) to pixel indices.
        # Column index: center_x + (x / resolution)
        # Row index: center_y + (y / resolution) (ROS coordinates/Cartesian)
        cols = center_x + xy[:, 0] / resolution
        rows = center_y + xy[:, 1] / resolution

        # Normalize to the interval [0, 1] relative to grid dimensions.
        cols_norm = cols / (W - 1)
        rows_norm = rows / (H - 1)

        # Convert to grid_sample coordinates in the interval [-1, 1].
        x_grid = 2 * cols_norm - 1
        y_grid = 2 * rows_norm - 1

        grid = torch.stack([x_grid, y_grid], axis=1) # Stack into (N, 2) 
        grid = grid.unsqueeze(0).unsqueeze(2) # reshape to (1, N, 1, 2) required by grid_sample.
        return grid

    def bilinear_sample_sdf_features(self, sdf_features, states, resolution):
        """
        sdf_features: (B, C, H, W)
        states (np.ndarray or torch.Tensor): Robot states with shape (B, N, D)
                (only the first two coordinates are used).
        Returns: (B, N, C)
        """
        B, C, H, W = sdf_features.shape
        assert B == 1, "lets use multiple states to query the same feature map, batch operation not supported"
        grid = self.batch_states_to_grid(states, (H, W), resolution)   # → [1, N, 1, 2]

        # grid_sample expects (1, H_out, W_out, 2), so this is (1, N, 1, 2)
        sampled = F.grid_sample(sdf_features, grid, align_corners=False, mode='bilinear')  # returns (1, C, N, 1)
        sampled = sampled.squeeze(-1).squeeze(0)       # → [C, N]
        sampled = sampled.transpose(0, 1).contiguous() # → [N, C]
        return sampled

    def sample_trajectories_cuniform(
            self, initial_state, actions, dynamics_cuda,
            num_trajectories, trajectory_length,
            model, feature_extractor, sdf_tensor, wheelbase,
            uniform_sampling=False, velocity_override=None,
        ):
        """Sample trajectories using C-Uniform approach.
        
        This function only handles trajectory sampling and returns the raw trajectories.
        Evaluation should be done separately to keep the function modular.
        """
        # Set to False for deployment to maximize performance by removing synchronization overhead.
        ENABLE_DETAILED_PROFILING = False

        t_step = self.dt
        if velocity_override is not None:
            v = velocity_override
        else: # Default behavior for standalone C-Uniform
            v = self.vrange[0]

        # Pre-convert actions to GPU tensor for efficient indexing
        actions_tensor = self.actions_tensor

        # Shape after repeat: (num_trajectories, state_dim)
        batch_current_states = (
            torch.tensor(initial_state, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(num_trajectories, 1)
        )
        state_dim = batch_current_states.shape[1]

        # Preallocate a tensor to store states for each time step.
        # Shape: (trajectory_length+1, num_trajectories, state_dim)
        trajectory_states = torch.empty((trajectory_length + 1, num_trajectories, state_dim),
                                        dtype=torch.float32, device=self.device)
        trajectory_states[0] = batch_current_states

        # Preallocate a tensor to record the chosen action indices at each step.
        # Shape: (trajectory_length, num_trajectories)
        trajectory_actions = torch.empty((trajectory_length, num_trajectories),
                                        dtype=torch.int64, device=self.device)

        # Determine if models are TRT wrappers (if TRT is available)
        is_actor_trt = TENSORRT_AVAILABLE and isinstance(model, TensorRTEngineWrapper)
        is_fe_trt = TENSORRT_AVAILABLE and isinstance(feature_extractor, TensorRTEngineWrapper)

        # Define a helper context manager. We use torch.inference_mode for PyTorch models. 
        # It generally doesn't affect TRT wrappers, which manage their own context.

        # --- Feature Extraction (Executed once before the loop) ---
        dense_features = None

        if self.controller_type == 1 and feature_extractor is not None:
            # For pixel-interpolated model, use the dense feature map and bilinear interpolation
            with torch.inference_mode(True):
                if ENABLE_DETAILED_PROFILING:
                    torch.cuda.synchronize()
                    t_start_feat = time.monotonic()
                # Feature Extraction (Works seamlessly for both PyTorch models and TRT Wrappers)
                # The TRT wrapper handles async execution and dtype conversion internally.
                dense_features = feature_extractor(sdf_tensor)  # [1, feature_dim, H, W]
                if ENABLE_DETAILED_PROFILING:
                    torch.cuda.synchronize()
                    print(f"    Feature extraction time: {(time.monotonic() - t_start_feat)*1000:.2f} ms ({'TRT' if is_fe_trt else 'PyTorch'})")
                # NOTE: this feature extraction take around 10-20ms before TRT optimization

        total_bilinear_sampling_time = 0
        total_inference_time = 0
        t_start_sampling_loop = time.monotonic()
        with torch.inference_mode(True):
            for step in range(trajectory_length):
                theta = batch_current_states[:, 2]

                network_current_states = torch.cat((
                    batch_current_states[:, 0:2],
                    torch.sin(theta).unsqueeze(1),
                    torch.cos(theta).unsqueeze(1)
                ), dim=1)

                # Predict action probabilities in batch using the model
                if self.controller_type == 1: # Map-conditioned
                    # Ensure dense_features was calculated
                    if dense_features is None:
                         raise RuntimeError("dense_features not computed for map-conditioned model.")

                    # Bilinear sampling (Stays in PyTorch, efficient)
                    t0 = time.monotonic()
                    interpolated_features = self.bilinear_sample_sdf_features(
                        sdf_features=dense_features, 
                        states=network_current_states, 
                        resolution=self.local_costmap_resolution
                    ) #NOTE: this part takes around <1 ms per time step
                    if ENABLE_DETAILED_PROFILING:
                        torch.cuda.synchronize()
                        total_bilinear_sampling_time += (time.monotonic() - t0)

                    # --- Inference (Action Model) ---
                    if ENABLE_DETAILED_PROFILING:
                        torch.cuda.synchronize()
                        start_time = time.monotonic()
                    # Inference Call (Works seamlessly for both PyTorch models and TRT Wrappers)
                    # The TRT wrapper handles async execution and dtype conversion internally.
                    current_probabilities = model(network_current_states, interpolated_features)
                    if ENABLE_DETAILED_PROFILING:
                        torch.cuda.synchronize()
                        end_time = time.monotonic()
                        total_inference_time += (end_time - start_time)
                        print(f"Inference at step {step} took {(end_time - start_time)*1000:.2f} ms")

                elif self.controller_type == 0: # Unsupervised
                    if uniform_sampling:
                        # Use uniform probabilities instead of model predictions
                        batch_size = network_current_states.shape[0]
                        num_actions = len(actions)
                        current_probabilities = torch.ones((batch_size, num_actions), 
                                                        device=network_current_states.device, 
                                                        dtype=torch.float32) / num_actions
                    else:
                        if ENABLE_DETAILED_PROFILING:
                            torch.cuda.synchronize()
                            start_time = time.monotonic()
                        current_probabilities = model(network_current_states)
                        if ENABLE_DETAILED_PROFILING:
                            torch.cuda.synchronize()
                            total_inference_time += (time.monotonic() - start_time)
                            print(f"Inference at step {step} took {(time.monotonic() - start_time)*1000:.2f} ms")
                else:
                    raise ValueError(f"Unsupported controller type: {self.controller_type}")
            
                # Use torch.multinomial to efficiently sample one action index per trajectory.
                # chosen_action_indices = torch.multinomial(current_probabilities, num_samples=1).squeeze(1)

                ####################Gumbel-Max trick####################
                # Use Gumbel-Max trick instead of torch.multinomial
                # 1. Convert probabilities to log-probabilities
                # CRITICAL: Ensure input to log is float32. 
                # This is essential if the TensorRT engine outputs FP16, as torch.log benefits from higher precision.
                log_probs = torch.log(current_probabilities.float() + 1e-9)
                
                # 2. Generate Gumbel noise
                u = torch.rand_like(log_probs)
                gumbel_noise = -torch.log(-torch.log(u + 1e-9) + 1e-9)
                # 3. Gumbel-Max trick
                chosen_action_indices = torch.argmax(log_probs + gumbel_noise, dim=1)
                ####################Gumbel-Max trick####################

                # Save the chosen action indices into the preallocated tensor.
                trajectory_actions[step] = chosen_action_indices.to(self.device, torch.int64)

                # Convert the chosen actions using efficient tensor indexing (OPTIMIZED & Verified)
                chosen_actions_tensor = actions_tensor[chosen_action_indices]

                # Propagate states in parallel.
                batch_current_states = dynamics_cuda(batch_current_states, chosen_actions_tensor, t_step, v, wheelbase)

                # Record the newly updated states.
                trajectory_states[step + 1] = batch_current_states

            if ENABLE_DETAILED_PROFILING:
                torch.cuda.synchronize()
                print(f"    Total sampling loop time: {(time.monotonic() - t_start_sampling_loop)*1000:.2f} ms")
                print(f"    Total bilinear sampling time: {total_bilinear_sampling_time*1000:.2f} ms")
                print(f"    Total action model inference time: {total_inference_time*1000:.2f} ms ({'TRT' if is_actor_trt else 'PyTorch'})")
            return trajectory_states, trajectory_actions
