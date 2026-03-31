"""
A.E.G.I.S. — E(2) Equivariant Graph Neural Network Encoder
═══════════════════════════════════════════════════════════
SE(2)-equivariant message passing that guarantees rotation and
translation invariance. Processes social graphs where nodes are
agents and edges represent spatial proximity.

Architecture:
  3 × EGNNLayer with SiLU activations
  Input: [N, feat_dim] node features + [N, 2] positions
  Output: [N, hidden_dim] encoded node embeddings
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EGNNLayer(MessagePassing):
    """
    E(n) Equivariant Graph Neural Network layer (Satorras et al., 2021).
    
    Adapted for E(2) — 2D coordinates for trajectory prediction.
    
    The layer updates both node features h and node coordinates x:
    - Messages depend on invariant quantities (distances, features)
    - Coordinate updates use equivariant displacement vectors
    - Node feature updates aggregate invariant messages
    
    This guarantees: if you rotate/translate all input coordinates,
    output coordinates rotate/translate identically, and features
    remain invariant.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 update_coords: bool = True):
        """
        Args:
            in_dim: Input node feature dimension.
            hidden_dim: Hidden layer dimension in MLPs.
            out_dim: Output node feature dimension.
            update_coords: Whether to update coordinates (disable in last layer
                          if only features are needed downstream).
        """
        super().__init__(aggr='mean')  # Mean aggregation for stability
        
        self.update_coords = update_coords
        
        # Edge message MLP: φ_e
        # Input: h_i || h_j || ||x_i - x_j||^2  → hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Coordinate update MLP: φ_x
        # Maps edge messages → scalar weight for displacement
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        
        # Node update MLP: φ_h
        # Input: h_i || aggregated_messages → out_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            h: Node features [N, in_dim].
            x: Node coordinates [N, 2].
            edge_index: Edge connectivity [2, E].
        
        Returns:
            Tuple of (h_new, x_new):
            - h_new: Updated features [N, out_dim]
            - x_new: Updated coordinates [N, 2]
        """
        # Compute messages and coordinate updates
        row, col = edge_index
        
        # Compute edge features (invariant quantities)
        diff = x[row] - x[col]  # [E, 2]
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]
        
        # Edge messages
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)  # [E, 2*in + 1]
        edge_msg = self.edge_mlp(edge_input)  # [E, hidden]
        
        # Coordinate updates (equivariant)
        if self.update_coords:
            coord_weights = self.coord_mlp(edge_msg)  # [E, 1]
            coord_updates = diff * coord_weights  # [E, 2]
            
            # Aggregate coordinate updates
            x_agg = torch.zeros_like(x)
            x_agg.index_add_(0, row, coord_updates)
            
            # Count neighbors for normalization
            counts = torch.zeros(x.size(0), 1, device=x.device)
            counts.index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
            counts = counts.clamp(min=1)
            
            x_new = x + x_agg / counts
        else:
            x_new = x
        
        # Aggregate messages for node update
        msg_agg = torch.zeros(h.size(0), edge_msg.size(1), device=h.device)
        msg_agg.index_add_(0, row, edge_msg)
        
        # Node feature update
        node_input = torch.cat([h, msg_agg], dim=-1)
        h_new = self.node_mlp(node_input)
        
        # Residual connection + layer norm
        if h.shape[-1] == h_new.shape[-1]:
            h_new = self.layer_norm(h_new + h)
        else:
            h_new = self.layer_norm(h_new)
        
        return h_new, x_new


class EGNNEncoder(nn.Module):
    """
    Multi-layer E(2) Equivariant GNN Encoder.
    
    Stacks multiple EGNNLayers to progressively refine node
    embeddings while maintaining equivariance guarantees.
    
    The first layer projects input features to hidden_dim.
    Subsequent layers maintain hidden_dim.
    The final layer optionally stops updating coordinates.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input node features.
            hidden_dim: Hidden dimension for all EGNN layers.
            num_layers: Number of EGNN layers to stack.
            dropout: Dropout probability between layers.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # EGNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Don't update coords in the last layer (we only need features)
            update_coords = (i < num_layers - 1)
            self.layers.append(
                EGNNLayer(hidden_dim, hidden_dim, hidden_dim,
                         update_coords=update_coords)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder stack.
        
        Args:
            x: Node features [N, input_dim].
            pos: Node positions [N, 2].
            edge_index: Graph connectivity [2, E].
        
        Returns:
            Encoded node features [N, hidden_dim].
        """
        # Project input features
        h = self.input_proj(x)  # [N, hidden_dim]
        coords = pos.clone()
        
        # Pass through EGNN layers
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)
            h = self.dropout(h)
        
        return h  # [N, hidden_dim]
