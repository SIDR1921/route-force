"""
A.E.G.I.S. — Training Pipeline
════════════════════════════════
Complete training loop with:
  - AdamW optimizer with weight decay
  - OneCycleLR scheduler (warm-up + cosine anneal)
  - Gradient clipping (max_norm=2.0) 
  - Minimum-of-K loss
  - ADE/FDE metric tracking
  - Checkpoint saving
"""

import os
import time
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from aegis.model.aegis_model import AEGIS
from aegis.model.loss import MinimumOfKLoss
from aegis.data.pipeline import build_dataloaders
from aegis.utils import compute_best_of_k_ade, compute_best_of_k_fde


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    max_grad_norm: float, device: str) -> dict:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with average loss, ADE, FDE for the epoch.
    """
    model.train()
    
    total_loss = 0.0
    total_mok = 0.0
    total_div = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0
    num_agents = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch)
        trajectories = output['trajectories']   # [N, K, T, 2]
        confidences = output['confidences']     # [N, K]
        
        # Ground truth
        gt = batch.y  # [N, T, 2]
        
        # Compute loss
        loss_dict = criterion(trajectories, gt, confidences)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping — crucial for equivariant networks
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        # Metrics
        with torch.no_grad():
            N = gt.shape[0]
            ade = compute_best_of_k_ade(trajectories, gt).sum().item()
            fde = compute_best_of_k_fde(trajectories, gt).sum().item()
        
        total_loss += loss.item() * N
        total_mok += loss_dict['mok_loss'].item() * N
        total_div += loss_dict['diversity_loss'].item() * N
        total_ade += ade
        total_fde += fde
        num_agents += N
        num_batches += 1
    
    if num_agents == 0:
        return {'loss': 0, 'mok_loss': 0, 'div_loss': 0, 'ade': 0, 'fde': 0, 'lr': 0}
    
    return {
        'loss': total_loss / num_agents,
        'mok_loss': total_mok / num_agents,
        'div_loss': total_div / num_agents,
        'ade': total_ade / num_agents,
        'fde': total_fde / num_agents,
        'lr': scheduler.get_last_lr()[0],
    }


@torch.no_grad()
def validate(model, loader, criterion, device: str) -> dict:
    """
    Validate the model.
    
    Returns:
        Dictionary with average loss, ADE, FDE for validation set.
    """
    model.eval()
    
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_agents = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        output = model(batch)
        trajectories = output['trajectories']
        confidences = output['confidences']
        gt = batch.y
        
        loss_dict = criterion(trajectories, gt, confidences)
        
        N = gt.shape[0]
        ade = compute_best_of_k_ade(trajectories, gt).sum().item()
        fde = compute_best_of_k_fde(trajectories, gt).sum().item()
        
        total_loss += loss_dict['loss'].item() * N
        total_ade += ade
        total_fde += fde
        num_agents += N
    
    if num_agents == 0:
        return {'loss': 0, 'ade': 0, 'fde': 0}
    
    return {
        'loss': total_loss / num_agents,
        'ade': total_ade / num_agents,
        'fde': total_fde / num_agents,
    }


def train(config: dict):
    """
    Full training pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML.
    """
    device = 'cpu'
    print(f"\n{'='*60}")
    print(f"  A.E.G.I.S. — Training Pipeline")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    # ═══════════════════════════════════════════
    #  Build data loaders
    # ═══════════════════════════════════════════
    print("📊 Building data loaders...")
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(config)
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(val_ds)}")
    
    if len(train_ds) == 0:
        print("⚠️  No training data found! Check your data path and configuration.")
        return
    
    # ═══════════════════════════════════════════
    #  Build model
    # ═══════════════════════════════════════════
    model = AEGIS(config).to(device)
    print(f"\n🧠 Model: {model.num_parameters:,} parameters")
    
    # ═══════════════════════════════════════════
    #  Loss, optimizer, scheduler
    # ═══════════════════════════════════════════
    loss_cfg = config.get('loss', {})
    criterion = MinimumOfKLoss(
        beta=loss_cfg.get('beta', 1.0),
        diversity_weight=loss_cfg.get('diversity_weight', 0.01),
    )
    
    train_cfg = config['training']
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )
    
    total_steps = train_cfg['epochs'] * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=train_cfg['lr'],
        total_steps=max(total_steps, 1),
        pct_start=train_cfg.get('pct_start', 0.3),
        anneal_strategy='cos',
    )
    
    # ═══════════════════════════════════════════
    #  Training loop
    # ═══════════════════════════════════════════
    checkpoint_dir = Path(train_cfg['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_ade = float('inf')
    
    print(f"\n🚀 Starting training for {train_cfg['epochs']} epochs...\n")
    print(f"{'Epoch':>6} │ {'Loss':>8} │ {'MoK':>8} │ {'ADE':>8} │ {'FDE':>8} │ {'V.Loss':>8} │ {'V.ADE':>8} │ {'V.FDE':>8} │ {'LR':>10}")
    print(f"{'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*10}")
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        t0 = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            train_cfg['max_grad_norm'], device
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        elapsed = time.time() - t0
        
        # Log
        if epoch % train_cfg.get('log_every', 1) == 0:
            print(f"{epoch:>6} │ {train_metrics['loss']:>8.4f} │ {train_metrics['mok_loss']:>8.4f} │ "
                  f"{train_metrics['ade']:>8.4f} │ {train_metrics['fde']:>8.4f} │ "
                  f"{val_metrics['loss']:>8.4f} │ {val_metrics['ade']:>8.4f} │ "
                  f"{val_metrics['fde']:>8.4f} │ {train_metrics['lr']:>10.6f}")
        
        # Save best checkpoint
        if val_metrics['ade'] < best_val_ade:
            best_val_ade = val_metrics['ade']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ade': best_val_ade,
                'val_fde': val_metrics['fde'],
                'config': config,
            }, checkpoint_dir / "best.pt")
            print(f"       └── 🏆 New best! Val ADE: {best_val_ade:.4f}")
        
        # Periodic checkpoint
        if epoch % train_cfg.get('save_every', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_dir / f"epoch_{epoch}.pt")
    
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val ADE: {best_val_ade:.4f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*60}\n")


def main():
    """Entry point for training."""
    parser = argparse.ArgumentParser(description="A.E.G.I.S. Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)


if __name__ == '__main__':
    main()
