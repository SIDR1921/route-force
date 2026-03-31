"""
A.E.G.I.S. — Utility Functions
═══════════════════════════════
Rotation matrices, Bernstein basis, and helper functions for
the equivariant trajectory prediction pipeline.
"""

import math
import torch
import numpy as np


def rotation_matrix_2d(angle_deg: float) -> torch.Tensor:
    """
    Create a 2D rotation matrix for a given angle in degrees.
    
    Args:
        angle_deg: Rotation angle in degrees.
    
    Returns:
        Tensor of shape [2, 2] representing the rotation.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def inverse_rotation_matrix_2d(angle_deg: float) -> torch.Tensor:
    """
    Create the inverse 2D rotation matrix (just transpose / negate angle).
    
    Args:
        angle_deg: Original rotation angle in degrees.
    
    Returns:
        Tensor of shape [2, 2] representing the inverse rotation.
    """
    return rotation_matrix_2d(-angle_deg)


def bernstein_basis(n: int, i: int, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the i-th Bernstein basis polynomial of degree n at parameter t.
    
    B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
    
    Args:
        n: Degree of the Bernstein polynomial (3 for cubic).
        i: Index of the basis function (0..n).
        t: Parameter values in [0, 1], shape [T].
    
    Returns:
        Tensor of shape [T] with basis values.
    """
    comb = math.comb(n, i)
    return comb * (t ** i) * ((1 - t) ** (n - i))


def compute_bernstein_matrix(num_timesteps: int, degree: int = 3) -> torch.Tensor:
    """
    Pre-compute the Bernstein basis matrix for cubic Bézier evaluation.
    
    For a cubic Bézier curve (degree=3), we have 4 control points.
    The matrix has shape [num_timesteps, degree+1] = [T, 4].
    
    Each row contains [B_{0,3}(t), B_{1,3}(t), B_{2,3}(t), B_{3,3}(t)]
    for t uniformly sampled in [0, 1].
    
    Args:
        num_timesteps: Number of timesteps to evaluate (e.g. 30).
        degree: Bézier curve degree (default: 3 for cubic).
    
    Returns:
        Tensor of shape [num_timesteps, degree+1].
    """
    t = torch.linspace(0, 1, num_timesteps)
    matrix = torch.stack([bernstein_basis(degree, i, t) for i in range(degree + 1)], dim=1)
    return matrix


def ego_normalize(coords: torch.Tensor) -> tuple:
    """
    Perform ego-centric normalization: subtract the last observed position
    so that each agent's current position becomes the origin (0, 0).
    
    Args:
        coords: Tensor of shape [..., T, 2] where T is the number of timesteps.
    
    Returns:
        Tuple of (normalized_coords, offsets) where:
        - normalized_coords has the same shape as coords
        - offsets has shape [..., 1, 2] (the subtracted reference positions)
    """
    # Last observed position as reference
    offsets = coords[..., -1:, :]  # [..., 1, 2]
    normalized = coords - offsets
    return normalized, offsets


def de_normalize(predicted: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Add back the ego-centric offset to recover global coordinates.
    
    Args:
        predicted: Predicted trajectories, shape [N, num_modes, T, 2].
        offsets: Original position offsets, shape [N, 1, 2] or broadcastable.
    
    Returns:
        Global-coordinate trajectories with the same shape as predicted.
    """
    if offsets.dim() == 2:
        offsets = offsets.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 2]
    elif offsets.dim() == 3:
        offsets = offsets.unsqueeze(1)  # [N, 1, 1, 2]
    return predicted + offsets


def rotate_coords(coords: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Rotate 2D coordinates by a given angle.
    
    Args:
        coords: Tensor of shape [..., 2].
        angle_deg: Rotation angle in degrees.
    
    Returns:
        Rotated coordinates with the same shape.
    """
    rot = rotation_matrix_2d(angle_deg).to(coords.device)
    return torch.einsum('...d, dc -> ...c', coords, rot.T)


def compute_ade(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Average Displacement Error: mean L2 distance across all timesteps.
    
    Args:
        predicted: Shape [N, T, 2].
        ground_truth: Shape [N, T, 2].
    
    Returns:
        ADE per sample, shape [N].
    """
    return torch.norm(predicted - ground_truth, p=2, dim=-1).mean(dim=-1)


def compute_fde(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Final Displacement Error: L2 distance at the final timestep.
    
    Args:
        predicted: Shape [N, T, 2].
        ground_truth: Shape [N, T, 2].
    
    Returns:
        FDE per sample, shape [N].
    """
    return torch.norm(predicted[:, -1] - ground_truth[:, -1], p=2, dim=-1)


def compute_best_of_k_ade(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Best-of-K ADE: for each sample, select the mode with lowest ADE.
    
    Args:
        predicted: Shape [N, K, T, 2] (K modes).
        ground_truth: Shape [N, T, 2].
    
    Returns:
        Best ADE per sample, shape [N].
    """
    gt_expanded = ground_truth.unsqueeze(1)  # [N, 1, T, 2]
    errors = torch.norm(predicted - gt_expanded, p=2, dim=-1).mean(dim=-1)  # [N, K]
    return errors.min(dim=1).values  # [N]


def compute_best_of_k_fde(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Best-of-K FDE: for each sample, select the mode closest at final timestep.
    
    Args:
        predicted: Shape [N, K, T, 2].
        ground_truth: Shape [N, T, 2].
    
    Returns:
        Best FDE per sample, shape [N].
    """
    gt_final = ground_truth[:, -1:, :]  # [N, 1, 2]
    pred_final = predicted[:, :, -1, :]  # [N, K, 2]
    errors = torch.norm(pred_final - gt_final, p=2, dim=-1)  # [N, K]
    return errors.min(dim=1).values  # [N]
