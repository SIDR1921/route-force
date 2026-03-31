"""
A.E.G.I.S. — Minimum-of-K (MoK) Loss Function
════════════════════════════════════════════════
Winner-Takes-All loss that forces multi-modal diversity without
averaging predicted trajectories into a single mean path.

The Logic:
  1. Compare GT against all K predicted modes
  2. Find the mode with lowest Huber loss
  3. Backpropagate ONLY through the best mode
  4. Other modes receive zero penalty → free to explore

This prevents the classic "mode collapse" where all K predictions
converge to the mean trajectory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimumOfKLoss(nn.Module):
    """
    Minimum-of-K Huber (Smooth L1) Loss.
    
    For each sample, selects the best-matching mode and only
    backpropagates through that mode's loss. This is the key
    mechanism for achieving true multi-modal diversity.
    
    Why Huber instead of MSE?
    - MSE (L2) squares large errors → overly sensitive to outliers
    - MAE (L1) has discontinuous gradient at zero → training instability
    - Huber combines the best of both: smooth near zero, linear for outliers
    """
    
    def __init__(self, beta: float = 1.0, diversity_weight: float = 0.1):
        """
        Args:
            beta: Huber loss threshold. Below beta, behaves like MSE.
                  Above beta, behaves like MAE. Default 1.0.
            diversity_weight: Weight for the optional diversity regularizer
                            that pushes modes apart.
        """
        super().__init__()
        self.beta = beta
        self.diversity_weight = diversity_weight
    
    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor,
                confidences: torch.Tensor = None) -> dict:
        """
        Compute the Minimum-of-K loss.
        
        Args:
            predicted: Predicted trajectories [N, K, T, 2].
            ground_truth: Ground truth trajectory [N, T, 2].
            confidences: Optional mode confidence scores [N, K].
        
        Returns:
            Dictionary with:
            - 'loss': Scalar total loss (for backward())
            - 'mok_loss': The minimum-of-K Huber loss component
            - 'diversity_loss': Mode diversity regularization
            - 'best_modes': Indices of selected best modes [N]
            - 'per_mode_losses': Loss for each mode [N, K]
        """
        N, K, T, D = predicted.shape
        
        # Expand ground truth to match predicted shape
        gt_expanded = ground_truth.unsqueeze(1).expand_as(predicted)  # [N, K, T, 2]
        
        # Compute per-mode Huber loss
        # F.smooth_l1_loss with beta parameter = Huber loss
        per_mode_losses = torch.zeros(N, K, device=predicted.device)
        
        for k in range(K):
            mode_loss = F.smooth_l1_loss(
                predicted[:, k],        # [N, T, 2]
                gt_expanded[:, k],      # [N, T, 2]
                beta=self.beta,
                reduction='none'
            )
            per_mode_losses[:, k] = mode_loss.mean(dim=(-2, -1))  # [N]
        
        # === THE TRICK ===
        # Find the best mode for each sample
        best_modes = torch.argmin(per_mode_losses, dim=1)  # [N]
        
        # Select only the best mode's loss
        # Gather the loss at the best mode index
        mok_loss = per_mode_losses.gather(1, best_modes.unsqueeze(1)).squeeze(1)  # [N]
        mok_loss = mok_loss.mean()  # Scalar
        
        # Diversity regularizer: push different modes apart
        diversity_loss = self._diversity_loss(predicted) if self.diversity_weight > 0 else torch.tensor(0.0, device=predicted.device)
        
        # Confidence loss: encourage the model to correctly predict which mode is best
        conf_loss = torch.tensor(0.0, device=predicted.device)
        if confidences is not None:
            # Create one-hot target from best_modes
            target = F.one_hot(best_modes, K).float()
            conf_loss = F.cross_entropy(confidences, best_modes)
        
        # Total loss
        total_loss = mok_loss + self.diversity_weight * diversity_loss + 0.1 * conf_loss
        
        return {
            'loss': total_loss,
            'mok_loss': mok_loss,
            'diversity_loss': diversity_loss,
            'confidence_loss': conf_loss,
            'best_modes': best_modes,
            'per_mode_losses': per_mode_losses,
        }
    
    def _diversity_loss(self, predicted: torch.Tensor) -> torch.Tensor:
        """
        Encourage diverse modes by penalizing pairwise similarity.
        
        Compute the negative average pairwise L2 distance between mode
        endpoints. This pushes modes to predict different destinations.
        
        Args:
            predicted: [N, K, T, 2]
        
        Returns:
            Scalar diversity loss (negative = modes should be far apart).
        """
        # Use final positions of each mode
        endpoints = predicted[:, :, -1, :]  # [N, K, 2]
        
        K = endpoints.shape[1]
        div_loss = torch.tensor(0.0, device=predicted.device)
        count = 0
        
        for i in range(K):
            for j in range(i + 1, K):
                # Negative distance → minimizing this pushes modes apart
                dist = torch.norm(endpoints[:, i] - endpoints[:, j], dim=-1)
                div_loss -= dist.mean()
                count += 1
        
        if count > 0:
            div_loss /= count
        
        return div_loss
