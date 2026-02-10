"""
Custom GNN Policy using PyTorch Geometric for SFC Placement.

This module provides a topology-agnostic GNN feature extractor that consumes
Dict observations (node_features, global_context, placement_mask, node_mask).
The GNN processes only real nodes (determined by node_mask), making it
portable across topologies with different num_nodes (up to max_nodes).
"""

from typing import Callable, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# PyTorch Geometric imports
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
)
from torch_geometric.utils import dense_to_sparse


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Graph Neural Network Feature Extractor using PyTorch Geometric.

    Topology-agnostic: infers max_nodes from the Dict observation space and
    uses node_mask to determine real vs padded nodes at runtime. Edge topology
    is fetched dynamically via edge_getter.

    Processes the Dict observation by:
    1. Extracting node features, global context, placement mask, and node mask
    2. Slicing to real nodes only (no GNN compute on padding)
    3. Applying GNN layers (configurable: GCN, GAT, GraphSAGE)
    4. Attention-weighted aggregation of node embeddings (masked)
    5. Fusion with global context MLP output
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        edge_getter: Callable[[], torch.Tensor],
        node_feat_dim: int = 6,
        hidden_dim: int = 64,
        features_dim: int = 256,
        gnn_type: str = "gcn",
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize the GNN Feature Extractor.

        Args:
            observation_space: Dict observation space from the environment
            edge_getter: Callable that returns current edge_index (2, num_edges)
            node_feat_dim: Dimension of per-node features (default: 6)
            hidden_dim: Hidden dimension for GNN layers
            features_dim: Output dimension of the feature extractor
            gnn_type: Type of GNN layer ("gcn", "gat", "sage")
            num_gnn_layers: Number of GNN layers
            dropout: Dropout rate for GNN layers
        """
        super().__init__(observation_space, features_dim)

        # Infer max_nodes from the observation space (topology-agnostic)
        self.max_nodes = observation_space["node_features"].shape[0]
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.edge_getter = edge_getter
        self.dropout = dropout

        # Input dimension: node features + placement mask
        input_dim = self.node_feat_dim + 1

        # Build GNN layers with LayerNorm (avoids padding-induced BatchNorm skew)
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            if gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif gnn_type == "gat":
                # GAT with 4 attention heads
                heads = 4 if i < num_gnn_layers - 1 else 1
                concat = i < num_gnn_layers - 1
                self.gnn_layers.append(
                    GATConv(
                        in_dim,
                        out_dim // heads if concat else out_dim,
                        heads=heads,
                        concat=concat,
                    )
                )
            elif gnn_type == "sage":
                self.gnn_layers.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.layer_norms.append(nn.LayerNorm(out_dim))

        # Global context processing: 3 (vnf) + 3 (sfc) + 1 (progress) = 7
        self.global_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Attention-based node aggregation
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Fusion: aggregated node embeddings (hidden_dim) + global embeddings (64)
        fusion_input_dim = hidden_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def _get_edge_index(self, device: torch.device) -> torch.Tensor:
        """Get current edge index from edge_getter."""
        if self.edge_getter is not None:
            try:
                edge_index = self.edge_getter()
                if edge_index.device != device:
                    edge_index = edge_index.to(device)
                return edge_index
            except Exception:
                pass
        # Fallback: no edges (isolated nodes)
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the GNN feature extractor.

        Args:
            observations: Dict with keys:
                - node_features:  (batch, max_nodes, 6)
                - global_context: (batch, 7)
                - placement_mask: (batch, max_nodes)
                - node_mask:      (batch, max_nodes)

        Returns:
            Feature embeddings (batch, features_dim)
        """
        device = observations["node_features"].device
        batch_size = observations["node_features"].shape[0]

        node_feats = observations["node_features"]  # (B, M, 6)
        global_context = observations["global_context"]  # (B, 7)
        placement_mask = observations["placement_mask"]  # (B, M)
        node_mask = observations["node_mask"]  # (B, M)

        # Determine N_real from node_mask (same for all batch items within a run)
        N_real = int(node_mask[0].sum().item())
        if N_real == 0:
            # Edge case: no real nodes — return zeros
            global_emb = self.global_mlp(global_context)
            zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
            combined = torch.cat([zeros, global_emb], dim=1)
            return self.fusion(combined)

        # Slice to real nodes only — no GNN compute on padding
        node_feats_real = node_feats[:, :N_real, :]  # (B, N_real, 6)
        placement_real = placement_mask[:, :N_real].unsqueeze(2)  # (B, N_real, 1)

        # Combine node features with placement mask → (B, N_real, 7)
        x = torch.cat([node_feats_real, placement_real], dim=2)

        # Get edge index (for real nodes, indices are in [0, N_real))
        edge_index = self._get_edge_index(device)

        # Build PyG batched graph
        batch_node_features = []
        batch_list = []

        for b in range(batch_size):
            batch_node_features.append(x[b])  # (N_real, input_dim)
            batch_list.append(torch.full((N_real,), b, dtype=torch.long, device=device))

        # Stack: (batch_size * N_real, input_dim)
        x_batched = torch.cat(batch_node_features, dim=0)
        batch_indices = torch.cat(batch_list, dim=0)

        # Expand edge index for the batch (offset node indices per sample)
        edge_indices_list = []
        for b in range(batch_size):
            offset = b * N_real
            edge_indices_list.append(edge_index + offset)
        edge_index_batched = torch.cat(edge_indices_list, dim=1)

        # Apply GNN layers with LayerNorm
        h = x_batched
        for i, (gnn_layer, ln) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h = gnn_layer(h, edge_index_batched)
            h = ln(h)
            h = F.relu(h)
            if i < len(self.gnn_layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Reshape back: (B, N_real, hidden_dim)
        h_reshaped = h.view(batch_size, N_real, self.hidden_dim)

        # Attention-weighted aggregation over real nodes
        attention_scores = self.attention_weights(h)  # (B * N_real, 1)
        attention_scores = attention_scores.view(batch_size, N_real, 1)

        # Softmax across real nodes only (no padding in this slice)
        attention_w = F.softmax(attention_scores, dim=1)  # (B, N_real, 1)

        # Weighted sum → graph embedding: (B, hidden_dim)
        graph_embedding = (attention_w * h_reshaped).sum(dim=1)

        # Process global context
        global_embedding = self.global_mlp(global_context)  # (B, 64)

        # Fuse graph and global embeddings
        combined = torch.cat([graph_embedding, global_embedding], dim=1)
        output = self.fusion(combined)  # (B, features_dim)

        return output


def get_edge_index_from_adj(adjacency_matrix: np.ndarray) -> torch.Tensor:
    """
    Convert an adjacency matrix to edge index format for PyTorch Geometric.

    Args:
        adjacency_matrix: Numpy adjacency matrix (num_nodes, num_nodes)

    Returns:
        Edge index tensor (2, num_edges)
    """
    adj_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(adj_tensor)
    return edge_index


def create_edge_getter(env) -> Callable[[], torch.Tensor]:
    """
    Create a callable that returns the current edge index from the environment.

    This is useful for dynamic topologies where the graph changes between episodes.

    Args:
        env: The SFCPlacementEnv or wrapped environment

    Returns:
        Callable that returns edge index tensor
    """
    import networkx as nx

    def edge_getter() -> torch.Tensor:
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(base_env, "substrate") and hasattr(base_env.substrate, "graph"):
            graph = base_env.substrate.graph
            adj = nx.to_numpy_array(graph)
            return get_edge_index_from_adj(adj)
        # Return empty edge index as fallback
        return torch.zeros((2, 0), dtype=torch.long)

    return edge_getter
