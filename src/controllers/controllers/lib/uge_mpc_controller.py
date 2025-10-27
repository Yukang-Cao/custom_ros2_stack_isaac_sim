# controllers/lib/uge_mpc_controller.py
import torch
import logging
import math
import numpy as np
from typing import Tuple, Dict, Any
import time

# Import necessary components from the framework
# Note: Assuming these relative imports work in the user's environment
from .torch_planner_base import TorchPlannerBase, PlannerInput
from .mppi_pytorch_controller import MPPIPyTorchController

# Import Numba CUDA for optimized kernels
try:
    from numba import cuda
    import numpy as anp # Used for types/constants in Numba kernels
except ImportError:
    print("WARNING: Numba not found or CUDA not supported. UGE-MPC optimized CUDA kernels will be disabled. Performance will be degraded.")
    cuda = None

# =================================================================================
# JIT Compiled Optimization Kernels (Replicating Numba optimizations)
# =================================================================================

# ---------------------------------------------------------------------------------
# (NEW) Custom Numba CUDA Kernel for Fused Propagation
# ---------------------------------------------------------------------------------
if cuda is not None:
    @cuda.jit(fastmath=True)
    def propagation_kernel_numba_cuda(
        initial_state_batch,  # (B, nx)
        action_seqs_cont,     # (B, T, nu)
        Sigma0_batch,         # (B, nx, nx)
        Q_diag_cont,          # (nu,)
        trajs_out,            # (B, T+1, nx)
        covs_out,             # (B, T+1, nx, nx)
        dt, L, T, B, nx, nu
    ):
        """
        (OPTIMIZED) Custom CUDA kernel for fused Mean and Covariance propagation.
        Structure: Parallel(B) -> Sequential(T).
        This eliminates the overhead of sequentially launching small PyTorch kernels
        and exactly mirrors the parallelism and dynamics (Standard Euler) of the original Numba CPU implementation.
        """
        # Parallelize over the batch dimension B
        b = cuda.grid(1)
        
        if b >= B:
            return

        # Define constants
        # pi = 3.141592653589793
        two_pi = 6.283185307179586

        # Load Q_diag into registers (assuming nu=2)
        q0 = Q_diag_cont[0]
        q1 = Q_diag_cont[1]

        # Use local arrays (registers/fast memory) for the state and covariance being propagated
        M = cuda.local.array(3, dtype=anp.float32)
        S = cuda.local.array((3, 3), dtype=anp.float32)
        
        # Initialize M and S from global memory
        for i in range(nx):
            M[i] = initial_state_batch[b, i]
        
        for i in range(nx):
            for j in range(nx):
                S[i, j] = Sigma0_batch[b, i, j]

        # Store initial state and covariance in output tensors
        for i in range(nx):
            trajs_out[b, 0, i] = M[i]
        for i in range(nx):
            for j in range(nx):
                covs_out[b, 0, i, j] = S[i, j]

        # Sequential Update Loop (T)
        for t in range(T):
            # --- 1. Extract inputs and current state ---
            V = action_seqs_cont[b, t, 0]
            Delta = action_seqs_cont[b, t, 1]
            
            X = M[0]; Y = M[1]; Theta = M[2]

            # --- 2. Mean Propagation (Mirroring propagate_batch_numba: Standard Euler) ---
            CosTh = math.cos(Theta)
            SinTh = math.sin(Theta)
            TanDelta = math.tan(Delta)

            # Standard Euler Update
            X_new = X + dt * V * CosTh
            Y_new = Y + dt * V * SinTh
            Theta_new = Theta + dt * V / L * TanDelta
            
            # Normalize theta (Mirroring propagate_batch_numba: np.fmod(th, 2*np.pi))
            # fmod ensures the result has the same sign as the dividend.
            Theta_norm = math.fmod(Theta_new, two_pi)

            M[0] = X_new
            M[1] = Y_new
            M[2] = Theta_norm
            
            # Store trajectory
            for i in range(nx):
                trajs_out[b, t+1, i] = M[i]

            # --- 3. Covariance Propagation (Mirroring propagate_uncertainty_batch_numba_fast) ---
            
            # Calculate Jacobians (Standard Euler Linearization)
            A02_t = -dt * V * SinTh
            A12_t = dt * V * CosTh

            b00 = dt * CosTh
            b10 = dt * SinTh
            b20 = dt * TanDelta / L
            
            CosDelta = math.cos(Delta)
            # Numba CPU implementation does not clamp CosDeltaSq.
            CosDeltaSq = CosDelta * CosDelta
            # Add a small epsilon for numerical stability if CosDeltaSq is close to zero
            if abs(CosDeltaSq) < 1e-9:
                CosDeltaSq = 1e-9 if CosDeltaSq >= 0 else -1e-9
                
            b21 = dt * V / (L * CosDeltaSq)

            # --- EKF Update (Element-wise, optimized) ---

            # Load S components from local array S
            S00 = S[0, 0]; S01 = S[0, 1]; S02 = S[0, 2]
            S10 = S[1, 0]; S11 = S[1, 1]; S12 = S[1, 2]
            S20 = S[2, 0]; S21 = S[2, 1]; S22 = S[2, 2]

            # Calculate N components (N = S + ES + SE^T + ESE^T + BQB^T)

            # Initialize N = S
            N00 = S00; N01 = S01; N02 = S02
            N10 = S10; N11 = S11; N12 = S12
            N20 = S20; N21 = S21; N22 = S22

            # E S
            N00 += A02_t * S20;  N01 += A02_t * S21;  N02 += A02_t * S22
            N10 += A12_t * S20;  N11 += A12_t * S21;  N12 += A12_t * S22

            # S E^T
            N00 += A02_t * S02;  N10 += A02_t * S12;  N20 += A02_t * S22
            N01 += A12_t * S02;  N11 += A12_t * S12;  N21 += A12_t * S22

            # E S E^T (only top-left 2x2)
            add = S22
            N00 += A02_t * A02_t * add
            N01 += A02_t * A12_t * add
            N10 += A12_t * A02_t * add
            N11 += A12_t * A12_t * add

            # B Q B^T
            qb00 = q0 * b00 * b00; qb01 = q0 * b00 * b10; qb02 = q0 * b00 * b20
            qb10 = q0 * b10 * b00; qb11 = q0 * b10 * b10; qb12 = q0 * b10 * b20
            qb20 = q0 * b20 * b00; qb21 = q0 * b20 * b10; qb22 = q0 * b20 * b20
            N00 += qb00; N01 += qb01; N02 += qb02
            N10 += qb10; N11 += qb11; N12 += qb12
            N20 += qb20; N21 += qb21; N22 += qb22
            N22 += q1 * b21 * b21  # q1 term

            # Write back to local array S
            S[0, 0] = N00; S[0, 1] = N01; S[0, 2] = N02
            S[1, 0] = N10; S[1, 1] = N11; S[1, 2] = N12
            S[2, 0] = N20; S[2, 1] = N21; S[2, 2] = N22

            # Store covariance in output tensor
            covs_out[b, t+1, 0, 0] = N00; covs_out[b, t+1, 0, 1] = N01; covs_out[b, t+1, 0, 2] = N02
            covs_out[b, t+1, 1, 0] = N10; covs_out[b, t+1, 1, 1] = N11; covs_out[b, t+1, 1, 2] = N12
            covs_out[b, t+1, 2, 0] = N20; covs_out[b, t+1, 2, 1] = N21; covs_out[b, t+1, 2, 2] = N22

    def _propagate_mean_and_covariance_custom_cuda(
        initial_state: torch.Tensor, 
        action_seqs: torch.Tensor,
        Sigma0_cov: torch.Tensor,
        Q_diag: torch.Tensor,
        dt: float, L: float, nx: int, nu: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper function to launch the Numba CUDA kernel from PyTorch."""
        
        if cuda is None:
            # This should be caught by the initialization logic, but acts as a safeguard.
            raise RuntimeError("Numba CUDA not available but custom kernel was called.")

        B, T, _ = action_seqs.shape

        # Ensure inputs are contiguous and in the correct shape for the kernel
        # initial_state needs to be expanded to batch size B if it's (nx,) or (1, nx)
        if initial_state.dim() == 1:
            initial_state_batch = initial_state.unsqueeze(0).expand(B, -1).contiguous()
        elif initial_state.dim() == 2:
            if initial_state.shape[0] == 1 and B > 1:
                 initial_state_batch = initial_state.expand(B, -1).contiguous()
            elif initial_state.shape[0] == B:
                 initial_state_batch = initial_state.contiguous()
            else:
                 raise ValueError(f"initial_state shape {initial_state.shape} incompatible with batch size {B}")
        else:
            raise ValueError("initial_state must be 1D or 2D tensor")

        # Sigma0_cov needs to be expanded to batch size B
        if B > 1:
             Sigma0_batch = Sigma0_cov.unsqueeze(0).expand(B, -1, -1).contiguous()
        else:
             # Ensure it is 3D (1, nx, nx)
             Sigma0_batch = Sigma0_cov.unsqueeze(0).contiguous()

        action_seqs_cont = action_seqs.contiguous()
        Q_diag_cont = Q_diag.contiguous()

        # Initialize output tensors
        trajs_out = torch.empty(B, T + 1, nx, device=device, dtype=torch.float32)
        covs_out = torch.empty(B, T + 1, nx, nx, device=device, dtype=torch.float32)

        # Configure the kernel launch
        threads_per_block = 128 # Optimized for typical GPU architectures
        blocks_per_grid = (B + threads_per_block - 1) // threads_per_block

        # Launch the kernel
        # Numba automatically handles the conversion of PyTorch tensors via __cuda_array_interface__
        propagation_kernel_numba_cuda[blocks_per_grid, threads_per_block](
            initial_state_batch,
            action_seqs_cont,
            Sigma0_batch,
            Q_diag_cont,
            trajs_out,
            covs_out,
            dt, L, T, B, nx, nu
        )
        
        # No explicit synchronization needed, PyTorch manages it.
        return trajs_out, covs_out

# ---------------------------------------------------------------------------------
# (Existing) JIT Kernels (Hellinger, Scoring, Fallback Propagation)
# ---------------------------------------------------------------------------------
@torch.jit.script
def _hellinger_3d_kernel_internal_jit(mu_c: torch.Tensor, Sig_c: torch.Tensor, mu_o: torch.Tensor, Sig_o: torch.Tensor) -> torch.Tensor:
    """
    Fast, JIT-compiled 3D Hellinger^2 calculation using manual 3x3 determinant/inverse.
    """
    # Epsilon matching Numba implementation's float32 precision considerations
    eps = torch.tensor(1e-9, dtype=torch.float32, device=mu_c.device) 

    # Expand dimensions for broadcasting (C, 1, S, ...) vs (1, K, S, ...)
    mu_c_exp = mu_c.unsqueeze(1)
    mu_o_exp = mu_o.unsqueeze(0)
    Sig_c_exp = Sig_c.unsqueeze(1)
    Sig_o_exp = Sig_o.unsqueeze(0)

    # 1. Mean differences (Delta_mu) (C, K, S, nx)
    delta_mu = mu_c_exp - mu_o_exp
    
    # NOTE: We intentionally skip explicit angle wrapping (atan2) here to exactly match 
    # the behavior of the Numba ground truth (uae_method_3d_TO.py), which also omits it.

    # 2. Average Covariance (A = (C+O)/2) (C, K, S, nx, nx)
    Avg_Sig = 0.5 * (Sig_c_exp + Sig_o_exp)

    # 3. Determinant and Inverse of A (Manually unrolled 3x3)
    # Extract elements of A
    a00 = Avg_Sig[..., 0, 0]; a01 = Avg_Sig[..., 0, 1]; a02 = Avg_Sig[..., 0, 2]
    a10 = Avg_Sig[..., 1, 0]; a11 = Avg_Sig[..., 1, 1]; a12 = Avg_Sig[..., 1, 2]
    a20 = Avg_Sig[..., 2, 0]; a21 = Avg_Sig[..., 2, 1]; a22 = Avg_Sig[..., 2, 2]

    # det(A)
    det_avg = (a00*(a11*a22 - a12*a21)
              -a01*(a10*a22 - a12*a20)
              +a02*(a10*a21 - a11*a20))

    # Mask for invalid determinants (det <= eps)
    invalid_mask = det_avg <= eps

    # Cofactors for Inverse(A) = Adjugate(A)^T / det(A)
    cof00 =  (a11*a22 - a12*a21)
    cof01 = -(a10*a22 - a12*a20)
    cof02 =  (a10*a21 - a11*a20)

    cof10 = -(a01*a22 - a02*a21)
    cof11 =  (a00*a22 - a02*a20)
    cof12 = -(a00*a21 - a01*a20)

    cof20 =  (a01*a12 - a02*a11)
    cof21 = -(a00*a12 - a02*a10)
    cof22 =  (a00*a11 - a01*a10)

    # Handle division by zero: replace det_avg with 1.0 where invalid (masked later)
    det_avg_safe = torch.where(invalid_mask, torch.ones_like(det_avg), det_avg)

    # Inverse components (Transposed Adjugate / det)
    inv00 = cof00 / det_avg_safe; inv01 = cof10 / det_avg_safe; inv02 = cof20 / det_avg_safe
    inv10 = cof01 / det_avg_safe; inv11 = cof11 / det_avg_safe; inv12 = cof21 / det_avg_safe
    inv20 = cof02 / det_avg_safe; inv21 = cof12 / det_avg_safe; inv22 = cof22 / det_avg_safe

    # 4. Quadratic form: delta_mu^T @ A^-1 @ delta_mu
    dx = delta_mu[..., 0]; dy = delta_mu[..., 1]; dth = delta_mu[..., 2]

    # A^-1 @ delta_mu
    solx = inv00*dx + inv01*dy + inv02*dth
    soly = inv10*dx + inv11*dy + inv12*dth
    solz = inv20*dx + inv21*dy + inv22*dth
    
    # delta_mu^T @ (A^-1 @ delta_mu)
    quad = dx*solx + dy*soly + dth*solz

    # 5. Exponent
    exponent = -0.125 * quad

    # 6. Determinants of C and O (Manually unrolled 3x3)
    # Extract elements of C
    c00 = Sig_c_exp[..., 0, 0]; c01 = Sig_c_exp[..., 0, 1]; c02 = Sig_c_exp[..., 0, 2]
    c10 = Sig_c_exp[..., 1, 0]; c11 = Sig_c_exp[..., 1, 1]; c12 = Sig_c_exp[..., 1, 2]
    c20 = Sig_c_exp[..., 2, 0]; c21 = Sig_c_exp[..., 2, 1]; c22 = Sig_c_exp[..., 2, 2]
    det_c = (c00*(c11*c22 - c12*c21)
            -c01*(c10*c22 - c12*c20)
            +c02*(c10*c21 - c11*c20))

    # Extract elements of O
    o00 = Sig_o_exp[..., 0, 0]; o01 = Sig_o_exp[..., 0, 1]; o02 = Sig_o_exp[..., 0, 2]
    o10 = Sig_o_exp[..., 1, 0]; o11 = Sig_o_exp[..., 1, 1]; o12 = Sig_o_exp[..., 1, 2]
    o20 = Sig_o_exp[..., 2, 0]; o21 = Sig_o_exp[..., 2, 1]; o22 = Sig_o_exp[..., 2, 2]
    det_o = (o00*(o11*o22 - o12*o21)
            -o01*(o10*o22 - o12*o20)
            +o02*(o10*o21 - o11*o20))

    # 7. Calculate Bhattacharyya Coefficient (BC)
    # Clamp exponent for stability (matching Numba behavior)
    exponent_clamped = torch.clamp(exponent, min=-60.0)
    
    # Pre-factor: ((|C|*|O|)^0.25) / sqrt(|A|)
    # Ensure determinants are non-negative before root operations
    det_c_safe = torch.clamp(det_c, min=0.0)
    det_o_safe = torch.clamp(det_o, min=0.0)
    
    # Use det_avg_safe (which is 1.0 where invalid) for the denominator sqrt
    pref = torch.pow(det_c_safe * det_o_safe, 0.25) / torch.sqrt(det_avg_safe)
    
    BC = pref * torch.exp(exponent_clamped)

    # 8. Hellinger Distance Squared (H^2 = 1 - BC)
    H_sq = 1.0 - BC

    # 9. Apply masks and clamp
    # If det_avg was invalid (mask=True), H^2 must be 1.0
    # We use torch.where for JIT compatibility
    H_sq = torch.where(invalid_mask, torch.ones_like(H_sq), H_sq)
    
    # Clamp for numerical stability (H^2 must be in [0, 1])
    H_sq = torch.clamp(H_sq, min=0.0, max=1.0)

    # 10. Return H_sq (C, K, S)
    return H_sq

@torch.jit.script
def _score_and_select_vectorized_jit(
    mu_c_all: torch.Tensor, Si_c_all: torch.Tensor,
    means3d: torch.Tensor, covs3d: torch.Tensor,
    cand_all_flat: torch.Tensor,
    N: int, M: int,
    mask: torch.Tensor
) -> torch.Tensor:
    """ Fully vectorized and JIT-compiled scoring and selection process.  """
    # N: Total number of base trajectories
    # M: Candidates per trajectory

    # 1. Calculate Hellinger distance between all candidates (C_total) and all base trajectories (N).
    # Input mu_c: (C_total, S, 3), Input mu_o: (N, S, 3)
    # Output H_sq shape: (C_total, N, S)
    H_sq = _hellinger_3d_kernel_internal_jit(mu_c_all, Si_c_all, means3d, covs3d)

    # 2. Reshape H_sq to group by the originating trajectory.
    # ( (N-1)*M, N, S ) -> (N-1, M, N, S)
    N_minus_1 = N - 1
    S = H_sq.shape[2]
    H_sq_reshaped = H_sq.view(N_minus_1, M, N, S)

    # 3. Apply the mask to exclude self-comparisons.
    # Mask shape: (N-1, N). Broadcast to (N-1, 1, N, 1).
    # Multiplying zeroes out the H^2 values that should be excluded from the sum.
    H_sq_masked = H_sq_reshaped * mask.unsqueeze(1).unsqueeze(3)

    # 4. Sum over N (others) and S (steps) to get the final scores.
    # Shape: (N-1, M)
    scores = torch.sum(H_sq_masked, dim=(2, 3))

    # 5. Select the candidate that maximizes the score (most diverse) within each group (N-1).
    # Shape: (N-1,)
    best_m_indices_local = torch.argmax(scores, dim=1)

    # 6. Gather the selected actions.
    # Calculate the global index in cand_all_flat: Global index = group_index * M + local_index
    group_indices = torch.arange(N_minus_1, device=mu_c_all.device)
    best_m_indices_global = group_indices * M + best_m_indices_local

    # Gather the corresponding action sequences.
    # Shape: (N-1, T, nu)
    selected_actions = cand_all_flat[best_m_indices_global]
    return selected_actions

# (Fallback/Legacy) PyTorch JIT Propagation Kernel
# This is kept only as a fallback if Numba CUDA is unavailable.
# (REVISED) Updated to use Standard Euler dynamics and fmod wrapping for consistency.
@torch.jit.script
def _propagate_unified_JIT_fallback(
    T: int, B: int, nx: int, nu: int,
    initial_state: torch.Tensor, # (nx,) or (1, nx)
    action_seqs_T: torch.Tensor, # (T, B, nu)
    Sigma0_batch: torch.Tensor,  # (B, nx, nx)
    Q_diag: torch.Tensor,        # (nu,) diagonal elements of Q
    dt: float,
    L: float, # wheelbase
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (FALLBACK PyTorch JIT) Fused Mean (Standard Euler) and EKF Covariance propagation.
    This implementation is slower than the custom Numba CUDA kernel for small B and large T due to sequential kernel launches.
    """
    # Initialize storage (T+1, B, ...)
    trajs = torch.empty(T + 1, B, nx, device=device, dtype=torch.float32)
    # Use coalesced layout for covariance (nx, nx, T+1, B) to optimize memory access
    covs_coalesced = torch.empty(nx, nx, T + 1, B, device=device, dtype=torch.float32)

    # Prepare initial states M (B, nx)
    if initial_state.dim() == 1:
        M = initial_state.unsqueeze(0).expand(B, -1)
    elif initial_state.dim() == 2:
        if initial_state.shape[0] == 1 and B > 1:
             M = initial_state.expand(B, -1)
        else:
             M = initial_state
    else:
        M = torch.empty(B, nx, device=device, dtype=torch.float32)
        
    M = M.clone()

    # Initialize S using Coalesced Layout (nx, nx, B)
    S = Sigma0_batch.permute(1, 2, 0).contiguous()

    trajs[0] = M
    covs_coalesced[:, :, 0, :] = S 

    # Setup constants
    q0 = Q_diag[0]; q1 = Q_diag[1]
    eps = torch.tensor(1e-9, device=device, dtype=torch.float32) # Increased epsilon for stability
    # pi = math.pi
    two_pi = 2 * math.pi

    # Fused Update Loop
    for t in range(T):
        # --- 1. Extract inputs and current state ---
        U_t = action_seqs_T[t] # (B, nu)
        V = U_t[:, 0]; Delta = U_t[:, 1]
        X = M[:, 0]; Y = M[:, 1]; Theta = M[:, 2]

        # Pre-calculate trigonometric functions
        TanDelta = torch.tan(Delta)
        CosDelta = torch.cos(Delta)
        # Use clamp for stability in JIT version matching the CUDA kernel stability check.
        CosDeltaSq = torch.clamp(CosDelta**2, min=eps) 
        SinTh = torch.sin(Theta); CosTh = torch.cos(Theta)

        # --- 2. Mean Propagation (Standard Euler - Matching Numba) ---
        
        # Update X, Y using the OLD theta
        M[:, 0] = X + V * CosTh * dt
        M[:, 1] = Y + V * SinTh * dt
        
        # Update Theta
        Theta_new = Theta + (V / L) * TanDelta * dt
        # Angle wrapping: Matching Numba's wrap (fmod(th, 2*pi))
        Theta_wrapped = torch.fmod(Theta_new, two_pi)
        M[:, 2] = Theta_wrapped
        
        trajs[t+1] = M

        # --- 3. Covariance Propagation (EKF - Consistent Standard Euler) ---
        
        # Calculate Jacobians (Standard Euler Linearization) using OLD Theta
        A02_t = -dt * V * SinTh
        A12_t = dt * V * CosTh

        b00 = dt * CosTh
        b10 = dt * SinTh
        b20 = dt * TanDelta / L
        b21 = dt * V / (L * CosDeltaSq)

        # --- EKF Update (Optimized Element-wise with Coalesced Access) ---

        # Load S components.
        S00 = S[0, 0, :]; S01 = S[0, 1, :]; S02 = S[0, 2, :]
        S11 = S[1, 1, :]; S12 = S[1, 2, :]
        S22 = S[2, 2, :]

        # ES terms (using symmetry S02=S20, S12=S21)
        ES00 = A02_t * S02; ES01 = A02_t * S12; ES02 = A02_t * S22
        ES11 = A12_t * S12; ES12 = A12_t * S22

        # SE^T terms
        SET01 = S02 * A12_t

        # ESE^T terms
        A02_S22 = A02_t * S22
        A12_S22 = A12_t * S22

        ESET00 = A02_t * A02_S22
        ESET01 = A12_t * A02_S22
        ESET11 = A12_t * A12_S22

        # --- Combine N and Add Noise BQB^T ---
        q0_b00 = q0 * b00; q0_b10 = q0 * b10; q0_b20 = q0 * b20

        # N00 (Optimization: 2*ES00 because ES00=SET00)
        N00 = S00 + 2*ES00 + ESET00 + q0_b00 * b00
        # N01
        N01 = S01 + ES01 + SET01 + ESET01 + q0_b00 * b10
        # N02
        N02 = S02 + ES02 + q0_b00 * b20
        
        # N11 (Optimization: 2*ES11)
        N11 = S11 + 2*ES11 + ESET11 + q0_b10 * b10
        # N12
        N12 = S12 + ES12 + q0_b10 * b20
        
        # N22
        N22_BQB = q0_b20 * b20 + q1 * b21 * b21
        N22 = S22 + N22_BQB

        # N20 (N20 = S20 + SET20 + BQB_20). SET20 = A02_S22. S20=S02 (symmetry).
        N20 = S02 + A02_S22 + q0_b20 * b00
        # N21
        N21 = S12 + A12_S22 + q0_b20 * b10

        # Write back to S (Coalesced Writes!)
        S[0, 0, :] = N00; S[0, 1, :] = N01; S[0, 2, :] = N02
        # Enforce symmetry explicitly on write-back
        S[1, 0, :] = N01; S[1, 1, :] = N11; S[1, 2, :] = N12
        S[2, 0, :] = N20; S[2, 1, :] = N21; S[2, 2, :] = N22

        # Store result for time t+1 directly.
        covs_coalesced[:, :, t+1, :] = S
        
    # Transpose outputs back to (B, T+1, ...)
    trajs_out = trajs.permute(1, 0, 2)
    # (nx, nx, T+1, B) -> (B, T+1, nx, nx)
    covs_out = covs_coalesced.permute(3, 2, 0, 1)
    
    return trajs_out, covs_out


# =================================================================================
# UGEMPCController Class Definition
# =================================================================================
class UGEMPCController(TorchPlannerBase):
    """
    Uncertainty-Guided Exploratory MPC (UGE-MPC) Controller (PyTorch Implementation).
    Implements Algorithm 2 from the paper: UGE-TO initialization followed by MPPI refinement.
    This replicates the exact behavior of the original Numba implementation (uae_method_3d_TO.py).
    """
    
    def __init__(self, controller_config: dict, experiment_config: dict, seed: int = None, mppi_config: dict = None, **kwargs):
        
        if seed is None:
            seed = experiment_config.get('seed', 2025) if experiment_config else 2025

        # Initialize base class (TorchPlannerBase and BaseController)
        super().__init__(controller_config, experiment_config, seed)
        
        self.logger = logging.getLogger(self.__class__.__name__)


        # Standardized dimensions
        self.T = self.T_horizon
        self.nu = 2 # Control dimension [v, delta]
        self.nx = 3 # State dimension [x, y, theta]
        
        # Load UGE-MPC specific parameters and initialize components
        self._load_uge_params()
        self._initialize_components(experiment_config, seed, mppi_config)

        # (NEW) Determine propagation implementation
        self._initialize_propagation_method()

        # Initialize the nominal control sequence (maintained across MPC steps)
        self.U_nominal = torch.zeros((self.T, self.nu), dtype=torch.float32, device=self.device)
        self.U_nominal[:, 0] = float(self.vrange[0])

        self.logger.info(f"UGEMPCController initialized. UGE-TO (N={self.N}, M={self.M}, Iters={self.iters}), MPPI (L={self.L})")

    def _initialize_propagation_method(self):
        """Determines whether to use the optimized Numba CUDA kernel or the fallback JIT implementation."""
        # Default to True if Numba CUDA is available, allow override via config if needed
        use_custom_cuda_default = (cuda is not None)
        # Allow user to disable custom kernel via config if needed (e.g., for debugging)
        self.use_custom_cuda_kernel = self.config.get("use_custom_cuda_kernel", use_custom_cuda_default)

        if self.use_custom_cuda_kernel:
            if cuda is None:
                 self.logger.warning("Custom CUDA kernel requested but Numba CUDA is not available. Falling back to PyTorch JIT implementation.")
                 self.use_custom_cuda_kernel = False
            else:
                self.logger.info("Using optimized Numba CUDA kernel for propagation.")
        else:
            self.logger.info("Using PyTorch JIT implementation for propagation (Fallback).")



    def _load_uge_params(self):
        """
        Loads parameters from the config, defining UGE-TO, MPPI, and Noise models.
        """
        try:
            # --- UGE-TO Parameters (Algorithm 1) ---
            uge_to_cfg = self.config["uge_to"]
            self.N = uge_to_cfg["num_trajectories"]
            self.M = uge_to_cfg["candidates_per_traj"]
            self.iters = uge_to_cfg["iterations"]
            self.step_interval = uge_to_cfg.get("step_interval", 5)
            self.decay_sharpness = uge_to_cfg.get("decay_sharpness", 2.0)
            
            # Pre-calculate Hellinger indices
            self.hellinger_indices = torch.arange(0, self.T + 1, self.step_interval, dtype=torch.long, device=self.device)
            self.S = len(self.hellinger_indices)

            # Pre-calculate decay coefficients
            if self.iters > 0:
                log_start, log_end = np.log(2.0), np.log(1.0)
                linspace = torch.linspace(log_start, log_end, self.iters, device=self.device, dtype=torch.float32)
                self.decay_coeffs = torch.exp(torch.pow(linspace, self.decay_sharpness))
            else:
                self.decay_coeffs = torch.tensor([], device=self.device)

            # OPTIMIZATION: Pre-calculate the selection mask for vectorized scoring
            self._initialize_selection_mask()

            noise_cfg = self.config["noise"]
            
            # R (Sigma_u): Covariance for sampling perturbations
            R_std = np.array(noise_cfg["R_std"], dtype=np.float32)
            self.R_cov = torch.diag(torch.tensor(R_std**2, device=self.device, dtype=torch.float32))
            
            # Q: Covariance for EKF propagation (Input noise model BQB^T)
            Q_std = np.array(noise_cfg["Q_std"], dtype=np.float32)
            self.Q_cov = torch.diag(torch.tensor(Q_std**2, device=self.device, dtype=torch.float32))
            # OPTIMIZATION: Prepare Q diagonal for fused kernel
            self.Q_diag = torch.diagonal(self.Q_cov).contiguous()

            Sigma0_std = np.array(noise_cfg["Sigma0_std"], dtype=np.float32)
            self.Sigma0_cov = torch.diag(torch.tensor(Sigma0_std**2, device=self.device, dtype=torch.float32))

            mppi_cfg = self.config["mppi"]
            self.L = mppi_cfg["num_rollouts"]
            self.mppi_refinement_config = mppi_cfg.get("refinement_config")

        except (KeyError, ValueError) as e:
            self.logger.error(f"CRITICAL ERROR during parameter loading: {e}")
            raise RuntimeError(f"Configuration invalid or incomplete. Error: {e}.")

    # OPTIMIZATION: initialization helper
    def _initialize_selection_mask(self):
        """
        Pre-calculates the mask used in vectorized scoring to exclude self-comparisons.
        Mask shape: (N-1, N)
        """
        if self.N <= 1:
            self.selection_mask = torch.empty(0, self.N, device=self.device, dtype=torch.float32)
            return

        N_minus_1 = self.N - 1
        # Initialize mask to all True (1.0)
        mask = torch.ones(N_minus_1, self.N, device=self.device, dtype=torch.float32)

        # We want to set mask[g, j] = 0 where j = g + 1 (the self-comparison).
        row_indices = torch.arange(N_minus_1, device=self.device)
        col_indices = row_indices + 1

        # Use index_put_ to efficiently set the specific elements to 0.0
        mask.index_put_((row_indices, col_indices), torch.tensor(0.0, device=self.device, dtype=torch.float32))

        self.selection_mask = mask

    def _initialize_components(self, experiment_config, seed, mppi_config_override):
        """Initialize the MPPI refiner and pre-calculate Cholesky decompositions."""
        
        # (MPPI Initialization remains the same)
        config_to_use = self.mppi_refinement_config if self.mppi_refinement_config is not None else mppi_config_override
        
        if config_to_use is None:
             raise ValueError("MPPI configuration missing. Ensure 'refinement_config' is in YAML or 'mppi_config' is passed via the ROS node factory.")

        if 'num_rollouts' not in config_to_use:
            config_to_use['num_rollouts'] = self.L

        # Initialize MPPI Refiner
        self.mppi_refiner = MPPIPyTorchController(
            controller_config=config_to_use,
            experiment_config=experiment_config,
            type_override=0, # UGE-MPC uses standard Gaussian MPPI (Type 0)
            seed=seed
        )
        self.mppi_refiner.K = self.L # Ensure budget matches L

        # Pre-calculate Cholesky decompositions for efficient sampling
        try:
            # Use double precision for stability, then cast back to float.
            self.chol_R = torch.linalg.cholesky(self.R_cov.double()).float().to(self.device)
            # Numba implementation uses 3*R specifically for the initial sampling in optimize3D
            self.chol_3R = torch.linalg.cholesky((3 * self.R_cov).double()).float().to(self.device)
            # self.chol_3R = self.chol_R
        except torch.linalg.LinAlgError as e:
            self.logger.error(f"Cholesky decomposition failed for R_cov. Ensure R_std values create a positive definite matrix. Error: {e}")
            raise
    
    def get_control_action(self, planner_input: PlannerInput) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal control action using UGE-MPC (Algorithm 2)."""
        # 
        # 
        # 0. Setup and Warm-start
        self._process_planner_input(planner_input)
        self._shift_nominal_trajectory()
        U_start = self.U_nominal.clone()

        # Robot frame setup
        x0_robot_frame = torch.zeros(self.nx, device=self.device, dtype=torch.float32)
        goal_tensor = torch.from_numpy(planner_input.local_goal).float().to(self.device)

        # 1. UGE-TO Initialization
        # Runs Alg. 1 and selects the best trajectory i*
        U_nominal_ugto, uge_to_trajs_T, best_idx_ugto = self._run_uge_to_initialization(
            U_start, x0_robot_frame, goal_tensor
        )

        # 2. MPPI Refinement
        # Ensure the MPPI refiner uses the same perception data.
        self.mppi_refiner._process_planner_input(planner_input)

        # Run the optimization routine from the MPPI instance, starting from the UGE-TO result.
        # mppi_trajectories_T shape: (T+1, L, nx)
        U_nominal_final, mppi_trajectories_T = self.mppi_refiner._run_mppi_optimization(
            U_nominal_ugto, planner_input.local_goal
        )

        # Update the internal state for the next iteration
        self.U_nominal = U_nominal_final

        # 3. Final Control Selection
        control_action_np = U_nominal_final[0].cpu().numpy()

        # 4. Visualization
        if self.viz_config.get('enabled', True) and self.viz_config.get('visualize_trajectories', True):
            info = self._prepare_visualization_info_hybrid(
                uge_to_trajs_T, best_idx_ugto,
                mppi_trajectories_T, U_nominal_final
            )
        else:
            info = {
                'state_rollouts_robot_frame': None,
                'is_hybrid': True, 
            }
        return control_action_np, info

    def reset(self):
        """Reset controller state."""
        super().reset() 
        self.mppi_refiner.reset()
        self.U_nominal = torch.zeros((self.T, self.nu), dtype=torch.float32, device=self.device)
        self.U_nominal[:, 0] = float(self.vrange[0])

    def _shift_nominal_trajectory(self):
        """Shift the nominal trajectory forward one step (Warm-starting)."""
        self.U_nominal = torch.roll(self.U_nominal, shifts=-1, dims=0)

    # =================================================================================
    # Core UGE-TO Logic (Algorithm 1 Implementation)
    # =================================================================================
    def _run_uge_to_initialization(self, U_nominal_initial, initial_state, goal_tensor):
        """
        Executes UGE-TO (Algorithm 1) and selects the best trajectory (Algorithm 2, Stage 1 & 2).
        (OPTIMIZED) Uses combined batch propagation and vectorized scoring.
        """
        
        # 1. Initialization
        # Initialize N trajectories: 1 base + (N-1) perturbed.
        action_seqs = torch.empty(self.N, self.T, self.nu, device=self.device, dtype=torch.float32)
        action_seqs[0] = U_nominal_initial
        
        if self.N > 1:
            # Sample noise using chol_3R (matching original implementation)
            noise = self._sample_noise_uge_to(self.N - 1, self.T, self.chol_3R)
            action_seqs[1:] = U_nominal_initial.unsqueeze(0) + noise
        
        # Clamp initial actions
        action_seqs = self._clamp_controls(action_seqs)

        # 2. Iterative Distributional Separation (Optimized Loop)
        
        for iter_idx in range(self.iters):
            
            N_minus_1 = self.N - 1
            if N_minus_1 == 0:
                break

            C_total = N_minus_1 * self.M

            # --- OPTIMIZATION: Batched Propagation Strategy ---

            # 2a. Generate M Candidates (C_total)
            decay_coeff = self.decay_coeffs[iter_idx]
            
            # Sample noise using chol_R (standard R used here) and apply decay
            noise_candidates = self._sample_noise_uge_to(C_total, self.T, self.chol_R) * decay_coeff
            
            # Repeat base action sequences (i=1 to N-1) M times
            action_seqs_repeated = action_seqs[1:].repeat_interleave(self.M, dim=0)
            
            cand_all_flat = action_seqs_repeated + noise_candidates
            cand_all_flat = self._clamp_controls(cand_all_flat)

            # 2b. Combine Batches (N + C_total)
            combined_actions = torch.cat([action_seqs, cand_all_flat], dim=0)

            # 2c. Propagate Combined Batch (N + C_total)
            # This now calls the optimized implementation (Numba CUDA or Fallback JIT).
            combined_trajs, combined_covs = self._propagate_mean_and_covariance(initial_state, combined_actions)

            # 2d. Extract Gaussians at specific intervals (S)
            # Extract base trajectories (first N)
            means3d, covs3d = self._extract_gaussians_at_idx3d(
                combined_trajs[:self.N], combined_covs[:self.N], self.hellinger_indices
            )
            
            # Extract candidates (rest C_total)
            mu_c_all, Si_c_all = self._extract_gaussians_at_idx3d(
                combined_trajs[self.N:], combined_covs[self.N:], self.hellinger_indices
            )

            # 2e. Score and Select (Vectorized)
            
            # Call the optimized JIT kernel
            selected_actions_N_minus_1 = _score_and_select_vectorized_jit(
                mu_c_all, Si_c_all, means3d, covs3d, cand_all_flat,
                self.N, self.M, self.selection_mask
            )
            
            
            # Update the action sequences (keeping index 0 unchanged)
            action_seqs[1:] = selected_actions_N_minus_1
            
        # 3. Final Evaluation and Selection (Algorithm 2, Stage 2)
        # Re-propagate the final diverse set.
        final_trajs, _ = self._propagate_mean_and_covariance(initial_state, action_seqs)

        # Calculate costs
        # Transpose final_trajs: (N, T+1, nx) -> (T+1, N, nx) for cost calculation
        final_trajs_transposed = final_trajs.permute(1, 0, 2)
        task_costs = self._calculate_trajectory_costs(final_trajs_transposed, goal_tensor, action_seqs.permute(1, 0, 2))
        
        # Select the best trajectory (i*)
        best_idx = torch.argmin(task_costs).item()
        U_nominal_ugto = action_seqs[best_idx]

        # Return the selected nominal, the set of final trajectories (T+1, N, nx), and the best index
        return U_nominal_ugto, final_trajs_transposed, best_idx

    def _sample_noise_uge_to(self, batch_size, T, chol_R):
        """Samples noise using Cholesky decomposition (Z @ L^T)."""
        Z = torch.randn(batch_size, T, self.nu, device=self.device, dtype=torch.float32)
        # (B, T, k) @ (l, k) -> (B, T, l)
        noise = torch.einsum('btk,lk->btl', Z, chol_R.T)
        return noise

    def _propagate_mean_and_covariance(self, initial_state, action_seqs):
        """ (OPTIMIZED) Fused propagation of the mean state and the covariance (EKF style).
        Selects between the optimized Numba CUDA kernel and the fallback PyTorch JIT implementation.
        """
        
        if self.use_custom_cuda_kernel:
            # Use the new, optimized Numba CUDA implementation
            trajs, covs = _propagate_mean_and_covariance_custom_cuda(
                initial_state, 
                action_seqs,
                self.Sigma0_cov,
                self.Q_diag,
                self.dt, self.wheelbase, self.nx, self.nu, self.device
            )
        else:
            # Use the PyTorch JIT implementation (Fallback)
            B, T, nu = action_seqs.shape
            nx = self.nx

            # Transpose action_seqs: (B, T, nu) -> (T, B, nu) for efficient access in the loop
            action_seqs_T = action_seqs.permute(1, 0, 2).contiguous()
            
            # Prepare initial Sigma batch (B, nx, nx)
            if B > 1:
                 Sigma0_batch = self.Sigma0_cov.unsqueeze(0).expand(B, -1, -1)
            else:
                 Sigma0_batch = self.Sigma0_cov.unsqueeze(0)

            # This fallback now also uses Standard Euler for consistency.
            trajs, covs = _propagate_unified_JIT_fallback(
                T, B, nx, nu,
                initial_state,
                action_seqs_T,
                Sigma0_batch,
                self.Q_diag,
                self.dt,
                self.wheelbase,
                self.device
            )
        
        
        # The outputs (trajs, covs) are in the expected (B, T+1, ...) format.
        return trajs, covs

    def _extract_gaussians_at_idx3d(self, trajs, covs, idx):
        """Extracts means and covariances at specific time indices."""
        # Use torch.index_select along the time dimension (dim=1)
        means3d = torch.index_select(trajs, 1, idx) # (B, S, nx)
        covs3d = torch.index_select(covs, 1, idx)   # (B, S, nx, nx)
        return means3d, covs3d

    def _clamp_controls(self, controls):
        """Clamps controls (B, T, nu) to the vehicle limits."""
        if not hasattr(self, 'min_ctrl_tensor'):
            self.min_ctrl_tensor = torch.tensor([float(self.vrange[0]), float(self.active_wrange[0])], device=self.device).view(1, 1, 2)
            self.max_ctrl_tensor = torch.tensor([float(self.vrange[1]), float(self.active_wrange[1])], device=self.device).view(1, 1, 2)
        
        return torch.max(torch.min(controls, self.max_ctrl_tensor), self.min_ctrl_tensor)

    # =================================================================================
    # Visualization
    # =================================================================================
    
    def _prepare_visualization_info_hybrid(self, uge_to_trajectories_T, best_idx_ugto, 
                                           mppi_trajectories_T, u_nominal_final):
        """
        Prepare visualization data for the hybrid approach (4-color visualization).
        (Implementation remains the same as the original file)
        """
        
        # Define the target visualization size based on the configuration
        TARGET_VIS_SIZE = self.num_vis_rollouts
        
        # Transform inputs to (Batch, T+1, nx) format and ensure contiguous memory layout
        uge_to_trajectories = uge_to_trajectories_T.permute(1, 0, 2).contiguous()
        mppi_trajectories = mppi_trajectories_T.permute(1, 0, 2).contiguous()
        N_total = uge_to_trajectories.shape[0]

        # --- 1. UGE-TO Best Sample - (Blue) ---
        if N_total > 0:
            ugto_best_np = uge_to_trajectories[best_idx_ugto].cpu().numpy()
        else:
            ugto_best_np = np.zeros((self.T + 1, self.nx), dtype=np.float32)

        # --- 2. UGE-TO Samples (Excluding Best) - (Grey) ---
        ugto_samples_np = np.tile(ugto_best_np, (TARGET_VIS_SIZE, 1, 1))

        mask = torch.ones(N_total, dtype=torch.bool, device=self.device)
        if N_total > 0:
            mask[best_idx_ugto] = False
        
        uge_to_non_best = uge_to_trajectories[mask]

        num_actual_non_best = uge_to_non_best.shape[0]
        num_to_copy = min(TARGET_VIS_SIZE, num_actual_non_best)
        
        if num_to_copy > 0:
            ugto_samples_np[:num_to_copy] = uge_to_non_best[:num_to_copy].cpu().numpy()

        # --- 3. MPPI Final (Weighted Average) - (Green) --- 
        # Rollout the final nominal control sequence
        robot_frame_initial_state = torch.zeros(self.nx, device=self.device, dtype=torch.float32)
        U_nominal_reshaped = u_nominal_final.unsqueeze(1) # (T, 1, nu)
        
        # Use the refiner's rollout function
        nominal_traj_robot = self.mppi_refiner._rollout_full_controls_torch(U_nominal_reshaped, robot_frame_initial_state)
        
        # (T+1, 1, nx) -> (T+1, nx)
        mppi_nominal_np = nominal_traj_robot.squeeze(1).cpu().numpy()

        # --- 4. MPPI Samples - (Orange) ---
        # Check if we should skip MPPI samples for performance
        skip_mppi_samples = self.viz_config.get('skip_mppi_samples', True)
        
        if skip_mppi_samples:
            # Skip MPPI samples - use empty array to reduce computation
            mppi_samples_np = np.empty((0, self.T + 1, self.nx), dtype=np.float32)
        else:
            # Full visualization - show MPPI samples
            mppi_samples_np = np.tile(mppi_nominal_np, (TARGET_VIS_SIZE, 1, 1))
            num_actual_mppi_samples = mppi_trajectories.shape[0]
            num_to_copy_mppi = min(TARGET_VIS_SIZE, num_actual_mppi_samples)
            if num_to_copy_mppi > 0:
                 mppi_samples_np[:num_to_copy_mppi] = mppi_trajectories[:num_to_copy_mppi].cpu().numpy()

        # --- 5. Package Data ---
        trajectory_data = {
            'cu_samples': ugto_samples_np,
            'cu_best': ugto_best_np,
            'mppi_samples': mppi_samples_np,
            'mppi_nominal': mppi_nominal_np,
        }
        
        # Apply angle wrapping and ensure contiguity
        for key in trajectory_data:
            traj_array = trajectory_data[key]
            if traj_array is not None:
                if not traj_array.flags['C_CONTIGUOUS']:
                    traj_array = np.ascontiguousarray(traj_array)
                
                # Apply wrapping to the theta dimension (index 2) to [-pi, pi] for visualization.
                # We use arctan2(sin, cos) which correctly handles angles regardless of their input range (e.g. [0, 2pi] or [-pi, pi]).
                traj_array[..., 2] = np.arctan2(np.sin(traj_array[..., 2]), np.cos(traj_array[..., 2]))
                
                trajectory_data[key] = traj_array

        vis_data = {
            'state_rollouts_robot_frame': trajectory_data,
            'is_hybrid': True,
        }
        return vis_data