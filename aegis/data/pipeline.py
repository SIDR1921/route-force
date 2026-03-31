"""
A.E.G.I.S. — Data Pipeline
═══════════════════════════
Transforms raw nuScenes trajectory data into ego-centric PyG graphs.

Pipeline:
  1. Parse nuScenes JSON metadata → extract pedestrian/vehicle trajectories
  2. Ego-centric normalization (current position → origin)
  3. Build social proximity graph via radius_graph
  4. Package as PyTorch Geometric Data objects
"""

import json
import os
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from torch_geometric.data import Data, Dataset, InMemoryDataset

from aegis.utils import ego_normalize


def pure_radius_graph(pos: torch.Tensor, r: float, loop: bool = False) -> torch.Tensor:
    """
    Pure-Python radius graph construction (no torch-cluster needed).
    
    Connects all pairs of nodes within Euclidean distance r.
    
    Args:
        pos: Node positions [N, 2].
        r: Radius threshold in meters.
        loop: Whether to include self-loops.
    
    Returns:
        edge_index: [2, E] tensor of edges.
    """
    # Compute pairwise distances
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 2]
    dist = torch.norm(diff, dim=-1)  # [N, N]
    
    # Find pairs within radius
    mask = dist < r
    if not loop:
        mask.fill_diagonal_(False)
    
    # Convert to edge index
    src, dst = torch.where(mask)
    edge_index = torch.stack([src, dst], dim=0)  # [2, E]
    
    return edge_index


class NuScenesTrajectoryExtractor:
    """
    Extracts agent trajectories from nuScenes JSON metadata.
    
    nuScenes stores annotations as linked lists (prev/next tokens).
    We reconstruct continuous trajectories per instance per scene.
    """
    
    def __init__(self, dataroot: str, version: str = "v1.0-mini",
                 categories: Optional[List[str]] = None):
        """
        Args:
            dataroot: Path to the extracted nuScenes data directory.
            version: nuScenes version string.
            categories: List of category names to include.
        """
        self.dataroot = dataroot
        self.version = version
        self.meta_dir = os.path.join(dataroot, version)
        
        self.categories = categories or [
            "human.pedestrian.adult",
            "human.pedestrian.child",
            "human.pedestrian.construction_worker",
            "human.pedestrian.police_officer",
            "vehicle.car",
            "vehicle.truck",
            "vehicle.bus.rigid",
        ]
        
        # Load all JSON tables
        self._load_tables()
        
    def _load_json(self, filename: str) -> list:
        path = os.path.join(self.meta_dir, filename)
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_tables(self):
        """Load and index all nuScenes metadata tables."""
        self.scenes = self._load_json("scene.json")
        self.samples = {s["token"]: s for s in self._load_json("sample.json")}
        self.annotations = {a["token"]: a for a in self._load_json("sample_annotation.json")}
        self.instances = {i["token"]: i for i in self._load_json("instance.json")}
        self.categories_table = {c["token"]: c for c in self._load_json("category.json")}
        
        # Build sample → annotations mapping
        self.sample_to_anns = defaultdict(list)
        for ann in self.annotations.values():
            self.sample_to_anns[ann["sample_token"]].append(ann)
        
        # Build category filter set
        self.valid_category_tokens = set()
        for token, cat in self.categories_table.items():
            if cat["name"] in self.categories:
                self.valid_category_tokens.add(token)
    
    def get_scene_samples(self, scene_idx: int) -> List[dict]:
        """Get all samples for a scene in temporal order."""
        scene = self.scenes[scene_idx]
        samples = []
        current_token = scene["first_sample_token"]
        while current_token:
            sample = self.samples[current_token]
            samples.append(sample)
            current_token = sample["next"] if sample["next"] != "" else None
        return samples
    
    def get_agent_position(self, ann: dict) -> np.ndarray:
        """Extract 2D (x, y) position from annotation."""
        return np.array(ann["translation"][:2], dtype=np.float32)
    
    def extract_scene_snapshots(self, scene_idx: int,
                                 history_steps: int = 4,
                                 future_steps: int = 12) -> List[dict]:
        """
        Extract trajectory snapshots (history + future) for all agents in a scene.
        
        For each valid timestep, we gather:
        - Agent positions at (t - history_steps) ... t (history)
        - Agent positions at t+1 ... t+future_steps (future / ground truth)
        
        Only agents visible in ALL required timesteps are included.
        
        Args:
            scene_idx: Index of the scene.
            history_steps: Number of past keyframes to include.
            future_steps: Number of future keyframes to predict.
        
        Returns:
            List of snapshot dicts, each containing:
            - 'history': np.ndarray [N, history_steps+1, 2]
            - 'future': np.ndarray [N, future_steps, 2]
            - 'instance_tokens': list of instance token strings
        """
        samples = self.get_scene_samples(scene_idx)
        total_needed = history_steps + 1 + future_steps
        
        if len(samples) < total_needed:
            return []
        
        # For each sample, gather annotations by instance
        sample_agents = []
        for sample in samples:
            agents = {}
            for ann in self.sample_to_anns[sample["token"]]:
                inst = self.instances[ann["instance_token"]]
                cat_token = inst["category_token"]
                if cat_token in self.valid_category_tokens:
                    agents[ann["instance_token"]] = self.get_agent_position(ann)
            sample_agents.append(agents)
        
        snapshots = []
        
        # Slide window across the scene
        for t in range(history_steps, len(samples) - future_steps):
            # Find agents present in ALL timesteps of the window
            window_range = range(t - history_steps, t + future_steps + 1)
            
            # Start with agents at current timestep
            common_agents = set(sample_agents[t].keys())
            for w in window_range:
                common_agents &= set(sample_agents[w].keys())
            
            if len(common_agents) < 1:
                continue
            
            common_agents = sorted(common_agents)  # Deterministic ordering
            
            # Build trajectory arrays
            history = np.zeros((len(common_agents), history_steps + 1, 2), dtype=np.float32)
            future = np.zeros((len(common_agents), future_steps, 2), dtype=np.float32)
            
            for i, inst_token in enumerate(common_agents):
                for j, w in enumerate(range(t - history_steps, t + 1)):
                    history[i, j] = sample_agents[w][inst_token]
                for j, w in enumerate(range(t + 1, t + future_steps + 1)):
                    future[i, j] = sample_agents[w][inst_token]
            
            snapshots.append({
                'history': history,
                'future': future,
                'instance_tokens': common_agents,
                'scene_idx': scene_idx,
                'timestep': t,
            })
        
        return snapshots


class AEGISDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for A.E.G.I.S.
    
    Converts nuScenes trajectory snapshots into ego-centric social graphs.
    Each Data object represents one scene snapshot with:
    - x: Node features [N, feat_dim] (flattened normalized history)
    - pos: Current normalized positions [N, 2]
    - edge_index: Social proximity edges [2, E]
    - y: Ground truth future trajectories [N, future_steps, 2]
    - offsets: Original positions for de-normalization [N, 1, 2]
    """
    
    def __init__(self, dataroot: str, version: str = "v1.0-mini",
                 scene_indices: Optional[List[int]] = None,
                 history_steps: int = 4, future_steps: int = 12,
                 prediction_steps: int = 30,
                 radius: float = 10.0,
                 categories: Optional[List[str]] = None,
                 transform=None, pre_transform=None):
        
        self.dataroot = dataroot
        self.version = version
        self.scene_indices = scene_indices
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.prediction_steps = prediction_steps
        self.radius = radius
        self.categories = categories
        
        # Process data without relying on InMemoryDataset file caching
        self._data_list = []
        self._process_data()
        
        # Don't call super().__init__ with root since we handle everything manually
    
    def _process_data(self):
        """Process nuScenes data into PyG Data objects."""
        extractor = NuScenesTrajectoryExtractor(
            self.dataroot, self.version, self.categories
        )
        
        scene_indices = self.scene_indices
        if scene_indices is None:
            scene_indices = list(range(len(extractor.scenes)))
        
        for scene_idx in scene_indices:
            snapshots = extractor.extract_scene_snapshots(
                scene_idx, self.history_steps, self.future_steps
            )
            
            for snap in snapshots:
                data = self._snapshot_to_pyg(snap)
                if data is not None:
                    self._data_list.append(data)
    
    def _snapshot_to_pyg(self, snapshot: dict) -> Optional[Data]:
        """
        Convert a trajectory snapshot to a PyG Data object.
        
        Steps:
        1. Ego-centric normalize: subtract each agent's current position
        2. Build node features from flattened normalized history
        3. Construct social proximity graph
        4. Interpolate future to prediction_steps if needed
        """
        history = torch.tensor(snapshot['history'], dtype=torch.float32)  # [N, H+1, 2]
        future = torch.tensor(snapshot['future'], dtype=torch.float32)    # [N, F, 2]
        
        N = history.shape[0]
        if N < 1:
            return None
        
        # Step 1: Ego-centric normalization
        # Reference: each agent's last observed position
        offsets = history[:, -1:, :].clone()  # [N, 1, 2]
        
        history_norm = history - offsets  # [N, H+1, 2]
        future_norm = future - offsets    # [N, F, 2]
        
        # Step 2: Node features — flatten normalized history
        # Shape: [N, (H+1) * 2]
        node_features = history_norm.reshape(N, -1)  # [N, (H+1)*2]
        
        # Step 3: Current positions for graph construction (normalized = at origin for ego)
        # Use the last observed normalized position (which is [0, 0] for ego)
        # But for social graph, we need relative positions between agents
        # Use the ORIGINAL positions at current timestep for distance computation
        current_pos = history[:, -1, :]  # [N, 2] — original coords
        
        # Build social proximity graph
        edge_index = pure_radius_graph(current_pos, r=self.radius, loop=False)
        
        # Step 4: Interpolate future if prediction_steps != future_steps
        if self.prediction_steps != self.future_steps:
            future_interp = self._interpolate_trajectory(
                future_norm, self.future_steps, self.prediction_steps
            )
        else:
            future_interp = future_norm
        
        data = Data(
            x=node_features,
            pos=history_norm[:, -1, :],  # [N, 2] normalized current pos (zeros)
            edge_index=edge_index,
            y=future_interp,             # [N, pred_steps, 2]
            offsets=offsets,              # [N, 1, 2] for de-normalization
            num_agents=torch.tensor([N]),
            history=history_norm,         # [N, H+1, 2] keep for reference
        )
        
        return data
    
    def _interpolate_trajectory(self, traj: torch.Tensor,
                                 src_steps: int, dst_steps: int) -> torch.Tensor:
        """
        Linearly interpolate trajectories from src_steps to dst_steps.
        
        Args:
            traj: [N, src_steps, 2]
            src_steps: Original number of steps
            dst_steps: Target number of steps
        
        Returns:
            Interpolated trajectory [N, dst_steps, 2]
        """
        N = traj.shape[0]
        # Transpose for F.interpolate: [N, 2, src_steps]
        traj_t = traj.permute(0, 2, 1)
        traj_interp = torch.nn.functional.interpolate(
            traj_t, size=dst_steps, mode='linear', align_corners=True
        )
        return traj_interp.permute(0, 2, 1)  # [N, dst_steps, 2]
    
    def len(self) -> int:
        return len(self._data_list)
    
    def get(self, idx: int) -> Data:
        return self._data_list[idx]
    
    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, idx):
        return self._data_list[idx]


def build_dataloaders(config: dict) -> Tuple:
    """
    Build training and validation DataLoaders from config.
    
    Args:
        config: Configuration dictionary from YAML.
    
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset).
    """
    from torch_geometric.loader import DataLoader
    
    data_cfg = config['data']
    
    train_dataset = AEGISDataset(
        dataroot=data_cfg['dataroot'],
        version=data_cfg['version'],
        scene_indices=data_cfg.get('train_scenes'),
        history_steps=data_cfg['history_steps'],
        future_steps=data_cfg['future_steps'],
        prediction_steps=data_cfg['prediction_steps'],
        radius=data_cfg['radius'],
        categories=data_cfg.get('categories'),
    )
    
    val_dataset = AEGISDataset(
        dataroot=data_cfg['dataroot'],
        version=data_cfg['version'],
        scene_indices=data_cfg.get('val_scenes'),
        history_steps=data_cfg['history_steps'],
        future_steps=data_cfg['future_steps'],
        prediction_steps=data_cfg['prediction_steps'],
        radius=data_cfg['radius'],
        categories=data_cfg.get('categories'),
    )
    
    train_cfg = config['training']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset
