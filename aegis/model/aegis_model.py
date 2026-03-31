"""
A.E.G.I.S. — Full Model Assembly
═════════════════════════════════
Combines the E(2) Equivariant Encoder and Bézier Decoder into
the complete A.E.G.I.S. architecture.

Data Flow:
  PyG Data → EGNNEncoder → BezierDecoder → Multi-modal Trajectories
  
  Shape: [N, feat] + [N, 2] + [2, E]
       → [N, 128]
       → {trajectories: [N, 3, 30, 2], control_points: [N, 3, 4, 2]}
"""

import torch
import torch.nn as nn

from aegis.model.encoder import EGNNEncoder
from aegis.model.decoder import BezierDecoder


class AEGIS(nn.Module):
    """
    A.E.G.I.S. — Autonomous Ego-centric Graph Intelligence System
    
    Complete model combining:
    1. E(2) Equivariant GNN Encoder — processes social interaction graphs
    2. Bézier Curve Decoder — outputs smooth multi-modal trajectories
    
    The equivariance guarantee means the model inherently generalizes
    to unseen orientations, dramatically reducing overfitting.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Model configuration dictionary with keys:
                - node_feat_dim: Input feature dimension
                - hidden_dim: EGNN hidden dimension (128)
                - num_egnn_layers: Number of EGNN layers (3)
                - num_modes: Multi-modal predictions (3)
                - num_control_points: Bézier control points (4)
                - dropout: Dropout rate (0.1)
            And data config with:
                - prediction_steps: Output timesteps (30)
        """
        super().__init__()
        
        model_cfg = config['model']
        data_cfg = config['data']
        
        self.encoder = EGNNEncoder(
            input_dim=model_cfg['node_feat_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_egnn_layers'],
            dropout=model_cfg.get('dropout', 0.1),
        )
        
        self.decoder = BezierDecoder(
            hidden_dim=model_cfg['hidden_dim'],
            num_modes=model_cfg['num_modes'],
            num_control_points=model_cfg['num_control_points'],
            prediction_steps=data_cfg['prediction_steps'],
        )
    
    def forward(self, data) -> dict:
        """
        Full forward pass.
        
        Args:
            data: PyG Data object (or Batch) with:
                - x: Node features [N, feat_dim]
                - pos: Node positions [N, 2]
                - edge_index: Graph edges [2, E]
        
        Returns:
            Dictionary with:
            - 'trajectories': [N, K, T, 2] predicted paths
            - 'control_points': [N, K, C, 2] Bézier control points
            - 'confidences': [N, K] mode confidence scores
        """
        # Encode: social graph → node embeddings
        encoded = self.encoder(data.x, data.pos, data.edge_index)
        # [N, hidden_dim]
        
        # Decode: embeddings → multi-modal Bézier trajectories
        output = self.decoder(encoded)
        # trajectories: [N, K, T, 2]
        
        return output
    
    def predict(self, data, return_best: bool = False) -> torch.Tensor:
        """
        Convenience method for inference.
        
        Args:
            data: PyG Data object.
            return_best: If True, return only the highest-confidence mode.
        
        Returns:
            If return_best: [N, T, 2] best-mode trajectories
            Else: [N, K, T, 2] all modes
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(data)
            
            if return_best:
                confidences = output['confidences']  # [N, K]
                best_mode = confidences.argmax(dim=1)  # [N]
                
                # Gather best mode trajectories
                N = best_mode.shape[0]
                trajectories = output['trajectories']  # [N, K, T, 2]
                best_traj = trajectories[torch.arange(N), best_mode]  # [N, T, 2]
                return best_traj
            
            return output['trajectories']
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (f"AEGIS(\n"
                f"  encoder={self.encoder},\n"
                f"  decoder={self.decoder},\n"
                f"  params={self.num_parameters:,}\n"
                f")")
