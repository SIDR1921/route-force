"""
A.E.G.I.S. — Test-Time Augmentation (TTA) Inference
════════════════════════════════════════════════════
Spatial ensemble over 4 rotations (0°, 90°, 180°, 270°) to
squeeze the last ~0.05m out of ADE scores.

Pipeline:
  1. Load trained model
  2. For each rotation angle:
     a. Rotate input coordinates
     b. Run forward pass → get control points
     c. Inverse-rotate the control points
  3. Average the 4 sets of un-rotated control points
  4. Generate final trajectories via Bernstein matrix
  5. De-normalize (add back original offsets)
"""

import argparse
import yaml
from pathlib import Path

import torch
import numpy as np

from aegis.model.aegis_model import AEGIS
from aegis.utils import (
    rotation_matrix_2d, inverse_rotation_matrix_2d,
    compute_bernstein_matrix, de_normalize,
    compute_ade, compute_fde,
    compute_best_of_k_ade, compute_best_of_k_fde,
)


class AEGISInference:
    """
    Test-time augmentation inference engine for A.E.G.I.S.
    
    Uses 4× rotation ensemble (0°, 90°, 180°, 270°) to cancel
    directional biases and improve prediction accuracy.
    """
    
    def __init__(self, config: dict, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            config: Configuration dictionary.
            checkpoint_path: Path to trained model checkpoint.
            device: Computation device.
        """
        self.config = config
        self.device = device
        
        # Load model
        self.model = AEGIS(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # TTA rotation angles
        self.tta_angles = config['inference'].get('tta_rotations', [0, 90, 180, 270])
        
        # Pre-compute rotation matrices
        self.rot_matrices = {}
        self.inv_rot_matrices = {}
        for angle in self.tta_angles:
            self.rot_matrices[angle] = rotation_matrix_2d(angle).to(device)
            self.inv_rot_matrices[angle] = inverse_rotation_matrix_2d(angle).to(device)
        
        # Bernstein matrix
        pred_steps = config['data']['prediction_steps']
        ctrl_pts = config['model']['num_control_points']
        self.bernstein = compute_bernstein_matrix(pred_steps, ctrl_pts - 1).to(device)
        
        print(f"🔮 A.E.G.I.S. Inference Engine loaded")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   TTA angles: {self.tta_angles}")
        print(f"   Parameters: {self.model.num_parameters:,}")
    
    @torch.no_grad()
    def predict_with_tta(self, data) -> dict:
        """
        Run prediction with test-time augmentation.
        
        Args:
            data: PyG Data object with x, pos, edge_index.
        
        Returns:
            Dictionary with:
            - 'trajectories': [N, K, T, 2] averaged multi-modal predictions
            - 'control_points': [N, K, C, 2] averaged control points
            - 'confidences': [N, K] averaged confidence scores
        """
        data = data.to(self.device)
        
        all_control_points = []
        all_confidences = []
        
        for angle in self.tta_angles:
            rot = self.rot_matrices[angle]
            inv_rot = self.inv_rot_matrices[angle]
            
            # Step 1: Rotate input
            rotated_data = data.clone()
            rotated_data.x = self._rotate_features(data.x, rot)
            rotated_data.pos = torch.einsum('nd, dc -> nc', data.pos, rot.T)
            
            # Step 2: Forward pass
            output = self.model(rotated_data)
            ctrl_pts = output['control_points']    # [N, K, C, 2]
            confs = output['confidences']          # [N, K]
            
            # Step 3: Inverse-rotate control points
            ctrl_pts_unrotated = torch.einsum('nkcd, dc -> nkcd', ctrl_pts, inv_rot.T)
            
            all_control_points.append(ctrl_pts_unrotated)
            all_confidences.append(confs)
        
        # Step 4: Average across all rotations
        avg_ctrl_pts = torch.stack(all_control_points).mean(dim=0)  # [N, K, C, 2]
        avg_confs = torch.stack(all_confidences).mean(dim=0)         # [N, K]
        
        # Step 5: Generate trajectories from averaged control points
        trajectories = torch.einsum('tc, nmcp -> nmtp', self.bernstein, avg_ctrl_pts)
        
        return {
            'trajectories': trajectories,
            'control_points': avg_ctrl_pts,
            'confidences': avg_confs,
        }
    
    @torch.no_grad()
    def predict_simple(self, data) -> dict:
        """
        Run prediction without TTA (for speed during development).
        """
        data = data.to(self.device)
        return self.model(data)
    
    def _rotate_features(self, features: torch.Tensor,
                          rot: torch.Tensor) -> torch.Tensor:
        """
        Rotate the position components within node features.
        
        Node features contain flattened (x,y) pairs from history.
        We need to rotate each (x,y) pair individually.
        
        Args:
            features: [N, feat_dim] where feat_dim = (H+1) * 2
            rot: [2, 2] rotation matrix
        
        Returns:
            Rotated features [N, feat_dim]
        """
        N, D = features.shape
        # Reshape to pairs: [N, num_pairs, 2]
        pairs = features.view(N, -1, 2)
        # Rotate each pair
        rotated_pairs = torch.einsum('npd, dc -> npc', pairs, rot.T)
        # Flatten back
        return rotated_pairs.view(N, D)
    
    def evaluate(self, loader) -> dict:
        """
        Evaluate model on a data loader with TTA.
        
        Args:
            loader: PyG DataLoader.
        
        Returns:
            Dictionary with average ADE, FDE, best-of-K ADE/FDE.
        """
        all_ade = []
        all_fde = []
        
        for batch in loader:
            output = self.predict_with_tta(batch)
            trajectories = output['trajectories']  # [N, K, T, 2]
            gt = batch.y.to(self.device)           # [N, T, 2]
            
            ade = compute_best_of_k_ade(trajectories, gt)
            fde = compute_best_of_k_fde(trajectories, gt)
            
            all_ade.append(ade)
            all_fde.append(fde)
        
        all_ade = torch.cat(all_ade)
        all_fde = torch.cat(all_fde)
        
        return {
            'ade': all_ade.mean().item(),
            'fde': all_fde.mean().item(),
            'ade_std': all_ade.std().item(),
            'fde_std': all_fde.std().item(),
            'num_agents': len(all_ade),
        }


def main():
    """Entry point for inference."""
    parser = argparse.ArgumentParser(description="A.E.G.I.S. Inference")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = args.checkpoint or config['inference']['checkpoint']
    
    engine = AEGISInference(config, checkpoint_path)
    
    # Build validation loader
    from aegis.data.pipeline import build_dataloaders
    _, val_loader, _, _ = build_dataloaders(config)
    
    print("\n📏 Evaluating on validation set...")
    results = engine.evaluate(val_loader)
    
    print(f"\n{'='*40}")
    print(f"  Results ({'with TTA' if not args.no_tta else 'no TTA'})")
    print(f"  ADE: {results['ade']:.4f} ± {results['ade_std']:.4f}")
    print(f"  FDE: {results['fde']:.4f} ± {results['fde_std']:.4f}")
    print(f"  Agents evaluated: {results['num_agents']}")
    print(f"{'='*40}\n")


if __name__ == '__main__':
    main()
