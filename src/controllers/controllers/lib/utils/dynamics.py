"""
Dynamics utilities for vehicle dynamic simulation.
Provides an abstraction layer for different dynamic models (KST, DiffDrive/Boat).
"""
import numpy as np
import torch
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

# Define the type hint for the JIT-compiled step function
# Inputs: states (Tensor), controls (Tensor), dt (float), params (Dict[str, float])
# Output: next_states (Tensor)
JitStepFn = Callable[[torch.Tensor, torch.Tensor, float, Dict[str, float]], torch.Tensor]

# =================================================================================
# Abstract Base Class
# =================================================================================
class DynamicsModel(ABC):
    """Abstract base class for vehicle dynamics models."""
    
    def __init__(self, params: Dict[str, float]):
        # Ensure params are Dict[str, float] for JIT compatibility
        self.params = params
        self._validate_params()

    @abstractmethod
    def _validate_params(self):
        """Validate required parameters for the specific model."""
        pass

    @property
    @abstractmethod
    def control_type(self) -> str:
        """Returns the type of control inputs (e.g., 'steering_angle', 'angular_velocity')."""
        pass

    # The step attribute will hold the corresponding JIT-compiled function
    step: JitStepFn = None 

# =================================================================================
# Kinematic Single Track (KST) Model
# =================================================================================
@torch.jit.script
def _step_kst(states: torch.Tensor, controls: torch.Tensor, dt: float, params: Dict[str, float]) -> torch.Tensor:
    """JIT-compatible implementation of KST dynamics (Semi-Implicit Euler)."""
    # Fail if assumptions violated (checked by TorchScript)
    assert states.dim() == 2 and states.shape[1] >= 3, "states must be (B, 3+)"
    assert controls.dim() == 2 and controls.shape[1] == 2, "controls must be (B, 2) [v, delta]"
    # We rely on Python-side validation for parameter presence.
    
    wheelbase = params['wheelbase']
    
    # Extract state components (views, no copy)
    x, y, theta = states[:, 0], states[:, 1], states[:, 2]
    
    # Extract control components
    v_cmd, steer_angle = controls[:, 0], controls[:, 1]
    
    # Compute the new orientation (Semi-Implicit Euler)
    theta_new = theta + (v_cmd / wheelbase) * torch.tan(steer_angle) * dt
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [-pi, pi]
    
    # Compute the new positions
    x_new = x + v_cmd * torch.cos(theta_new) * dt
    y_new = y + v_cmd * torch.sin(theta_new) * dt
    
    # Update states (create new tensor)
    next_states = states.clone()
    next_states[:, 0] = x_new
    next_states[:, 1] = y_new
    next_states[:, 2] = theta_new
    return next_states

class KinematicSingleTrackDynamics(DynamicsModel):
    """
    Kinematic Single Track (Bicycle) Model.
    Control: [v (velocity), delta (steering angle)]
    """
    control_type = "steering_angle"

    def _validate_params(self):
        if 'wheelbase' not in self.params or self.params['wheelbase'] <= 0:
            raise ValueError(f"KST Dynamics requires 'wheelbase' > 0, but got {self.params.get('wheelbase')}")

    # Expose the compiled JIT function
    step = _step_kst

# =================================================================================
# Differential Drive / Simple Boat Model
# =================================================================================
@torch.jit.script
def _step_diff_drive_exact(states: torch.Tensor, controls: torch.Tensor, dt: float, params: Dict[str, float]) -> torch.Tensor:
    """
    JIT-compatible implementation of Differential Drive dynamics.
    Uses exact integration assuming constant control inputs (v, omega) over dt.
    """
    # Fail if assumptions violated
    assert states.dim() == 2 and states.shape[1] >= 3, "states must be (B, 3+)"
    assert controls.dim() == 2 and controls.shape[1] == 2, "controls must be (B, 2) [v, omega]"
    
    # Extract state components
    x, y, theta = states[:, 0], states[:, 1], states[:, 2]
    
    # Extract control components
    v_cmd = controls[:, 0]
    omega_cmd = controls[:, 1] # Angular velocity
    
    # Exact integration method
    eps = 1e-6
    # Mask for trajectories moving in a straight line (omega near zero)
    is_straight = torch.abs(omega_cmd) < eps
    
    # Calculate Radius of curvature R = v / omega
    # Use a large number for R when moving straight to avoid division by zero.
    R = torch.where(is_straight, torch.full_like(v_cmd, 1e9), v_cmd / omega_cmd)
    d_theta = omega_cmd * dt

    # Pre-calculate trigonometry
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_theta_new = torch.sin(theta + d_theta)
    cos_theta_new = torch.cos(theta + d_theta)

    # Calculate coordinate changes based on arc motion
    # Δx = R * (sin(θ + Δθ) - sin(θ))
    # Δy = R * (cos(θ) - cos(θ + Δθ))
    delta_x_arc = R * (sin_theta_new - sin_theta)
    delta_y_arc = R * (cos_theta - cos_theta_new)
    
    # Calculate coordinate changes for straight motion
    delta_x_straight = v_cmd * cos_theta * dt
    delta_y_straight = v_cmd * sin_theta * dt

    # Combine results using the mask
    x_new = x + torch.where(is_straight, delta_x_straight, delta_x_arc)
    y_new = y + torch.where(is_straight, delta_y_straight, delta_y_arc)
    theta_new = theta + d_theta
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [-pi, pi]
    
    # Update states
    next_states = states.clone()
    next_states[:, 0] = x_new
    next_states[:, 1] = y_new
    next_states[:, 2] = theta_new
    return next_states

class DifferentialDriveDynamics(DynamicsModel):
    """
    Differential Drive Model (Also suitable as a simple boat model).
    Control: [v (linear velocity), omega (angular velocity)]
    """
    control_type = "angular_velocity"

    def _validate_params(self):
        # Diff drive has no specific parameters required in this implementation
        pass

    # Expose the compiled JIT function
    step = _step_diff_drive_exact

# =================================================================================
# Factory and Utilities
# =================================================================================

DYNAMICS_FACTORY = {
    "kinematic_single_track": KinematicSingleTrackDynamics,
    "differential_drive": DifferentialDriveDynamics,
    # Alias for the boat model
    "simple_boat": DifferentialDriveDynamics,
}

def get_dynamics_model(model_name: str, params: Dict[str, Any]) -> DynamicsModel:
    """Factory function to instantiate dynamics models."""
    if model_name not in DYNAMICS_FACTORY:
        # Fail if model name is unrecognized.
        raise ValueError(f"Unknown dynamics model: {model_name}. Available: {list(DYNAMICS_FACTORY.keys())}")
    
    if params is None:
        params = {}

    # Convert parameters to a format suitable for JIT (Dict[str, float])
    # We filter and cast here to ensure compatibility before passing to JIT.
    jit_params = {k: float(v) for k, v in params.items() if isinstance(v, (int, float)) and v is not None}

    ModelClass = DYNAMICS_FACTORY[model_name]
    # Instantiate the class (this runs validation)
    model_instance = ModelClass(jit_params)
    
    # Ensure the JIT function was correctly assigned
    if model_instance.step is None:
        raise RuntimeError(f"JIT-compiled step function missing for model {model_name}")
    return model_instance


# =================================================================================
# Legacy Functions (DEPRECATED)
# =================================================================================
# These functions are kept only for compatibility with the highly optimized 
# CUniformController.sample_trajectories_cuniform loop, which currently relies on them.
@torch.jit.script
def cuda_dynamics_KS_3d_scalar_v_batched(states: torch.Tensor, actions: torch.Tensor, dt: float, v: float, wheelbase: float) -> torch.Tensor:
    """Legacy KST dynamics with scalar velocity. DEPRECATED."""

    """
    Vectorized dynamic propagation for a batch of state-action pairs.
    
    Instead of propagating each state with all possible actions,
    this function assumes that for each state in the batch a corresponding 
    action is provided. Thus, the function computes one propagated state 
    per input state-action pair.
    
    Args:
        states (torch.Tensor): Tensor of shape (batch_size, 3) representing
            the current states, where each state is [x, y, theta].
        actions (torch.Tensor): Tensor of shape (batch_size,) or (batch_size,1)
            representing the steering angle for each corresponding state.
        dt (float): Time step for propagation.
        v (float): Velocity, assumed constant across the batch.
        wheelbase (float): Vehicle wheelbase (distance between axles).
        
    Returns:
        next_states (torch.Tensor): Tensor of shape (batch_size, 3) containing
            the propagated states for each state-action pair.
    """
    assert states.dim() == 2 and states.shape[1] == 3, "states must be of shape (batch_size, 3)"
    assert states.shape[0] == actions.shape[0], "Batch size of states and actions must match"
    assert actions.dim() in [1, 2] and (actions.dim() == 1 or actions.shape[1] == 1), \
        "actions must be of shape (batch_size,) or (batch_size, 1)"
    
    # Extract state components: x, y positions and orientation theta.
    # Shape: (batch_size,)
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    
    # Compute the new orientation (theta_new) for each state-action pair.
    # each element in 'actions' corresponds to the desired steering angle for the respective state.
    theta_new = theta + (v / wheelbase) * torch.tan(actions.squeeze()) * dt
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [−π, π]: 
    
    # Compute the new x position using the updated orientation.
    x_new = x + v * torch.cos(theta_new) * dt
    
    # Compute the new y position using the updated orientation.
    y_new = y + v * torch.sin(theta_new) * dt
    
    # Stack the propagated components to form the new states tensor.
    # Each state is now [x_new, y_new, theta_new].
    next_states = torch.stack([x_new, y_new, theta_new], dim=1)
    return next_states

@torch.jit.script
def cuda_dynamics_KS_3d_variable_v_batched(states: torch.Tensor, controls: torch.Tensor, dt: float, wheelbase: float) -> torch.Tensor:
    """Legacy KST dynamics with variable velocity. DEPRECATED."""

    """
    Vectorized dynamic propagation for variable velocity inputs.
    
    Args:
        states (torch.Tensor): (batch_size, 3) [x, y, theta].
        controls (torch.Tensor): (batch_size, 2) [v_cmd, steer_angle].
        dt (float): Time step.
        wheelbase (float): Vehicle wheelbase.
        
    Returns:
        next_states (torch.Tensor): (batch_size, 3).
    """
    assert states.dim() == 2 and states.shape[1] == 3
    assert controls.dim() == 2 and controls.shape[1] == 2
    assert states.shape[0] == controls.shape[0]
    
    # Extract state components
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    
    # Extract control components
    v_cmd = controls[:, 0]
    steer_angle = controls[:, 1]
    
    # Compute the new orientation (theta_new) - Semi-Implicit Euler
    theta_new = theta + (v_cmd / wheelbase) * torch.tan(steer_angle) * dt
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [−π, π]
    
    # Compute the new position using the updated orientation.
    x_new = x + v_cmd * torch.cos(theta_new) * dt
    y_new = y + v_cmd * torch.sin(theta_new) * dt
    
    next_states = torch.stack([x_new, y_new, theta_new], dim=1)
    return next_states