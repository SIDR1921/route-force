"""
A.E.G.I.S. — Continuous Bézier Decoder
═══════════════════════════════════════
Predicts multi-modal trajectories by outputting cubic Bézier
control points and evaluating them via pre-computed Bernstein
basis matrix multiplication.

Architecture:
  MLP: Linear(hidden, 64) → SiLU → Linear(64, 3×4×2=24)
  Reshape → [N, 3_modes, 4_ctrl_pts, 2_coords]
  einsum with Bernstein matrix → [N, 3, T, 2] smooth trajectories
"""

import torch
import torch.nn as nn

from aegis.utils import compute_bernstein_matrix


class BezierDecoder(nn.Module):
    """
    Decodes encoded node embeddings into multi-modal Bézier trajectories.
    
    The key insight: instead of predicting T discrete (x,y) points directly
    (which produces jittery outputs), we predict 4 control points per mode
    and smoothly interpolate using the Bernstein polynomial basis.
    
    This is:
    - Mathematically guaranteed to be smooth (C∞ continuity)
    - Parameter-efficient (24 outputs vs 60 for 30 timesteps)
    - Fast at inference (single matrix multiply)
    """
    
    def __init__(self, hidden_dim: int = 128, num_modes: int = 3,
                 num_control_points: int = 4, prediction_steps: int = 30):
        """
        Args:
            hidden_dim: Dimension of input encoded features.
            num_modes: Number of multi-modal predictions (K=3).
            num_control_points: Bézier control points per mode (4 for cubic).
            prediction_steps: Number of output timesteps (T=30).
        """
        super().__init__()
        
        self.num_modes = num_modes
        self.num_control_points = num_control_points
        self.prediction_steps = prediction_steps
        
        # Total output: K modes × C control points × 2 coordinates
        output_dim = num_modes * num_control_points * 2
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
        )
        
        # Mode confidence head (for selecting best mode at inference)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, num_modes),
        )
        
        # Pre-compute Bernstein basis matrix [T, C]
        # This is a constant — register as buffer so it moves to GPU automatically
        bernstein = compute_bernstein_matrix(prediction_steps, degree=num_control_points - 1)
        self.register_buffer('bernstein_matrix', bernstein)
    
    def forward(self, encoded: torch.Tensor) -> dict:
        """
        Forward pass: encoded features → multi-modal trajectories.
        
        Args:
            encoded: Node embeddings [N, hidden_dim].
        
        Returns:
            Dictionary with:
            - 'trajectories': [N, K, T, 2] smooth predicted paths
            - 'control_points': [N, K, C, 2] raw control points
            - 'confidences': [N, K] mode confidence scores
        """
        N = encoded.shape[0]
        
        # Predict control points
        raw = self.mlp(encoded)  # [N, K*C*2]
        control_points = raw.view(N, self.num_modes, self.num_control_points, 2)
        # Shape: [N, K=3, C=4, 2]
        
        # Predict mode confidences
        confidences = self.confidence_head(encoded)  # [N, K]
        confidences = torch.softmax(confidences, dim=-1)
        
        # Evaluate Bézier curves via matrix multiplication
        # bernstein_matrix: [T, C]
        # control_points:   [N, K, C, 2]
        # Result:           [N, K, T, 2]
        #
        # Using einsum for clarity and efficiency:
        # 'tc' = timesteps × control_points
        # 'nmcp' = batch × modes × control_points × coords
        # → 'nmtp' = batch × modes × timesteps × coords
        trajectories = torch.einsum(
            'tc, nmcp -> nmtp',
            self.bernstein_matrix,
            control_points
        )
        
        return {
            'trajectories': trajectories,       # [N, K, T, 2]
            'control_points': control_points,   # [N, K, C, 2]
            'confidences': confidences,          # [N, K]
        }
