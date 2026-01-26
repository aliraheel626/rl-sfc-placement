"""
Custom GNN Policy using PyTorch Geometric for SFC Placement.

This module provides a custom Actor-Critic policy that uses Graph Neural Networks
to process the substrate network topology for better VNF placement decisions.
"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

# PyTorch Geometric imports
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx, dense_to_sparse


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Graph Neural Network Feature Extractor using PyTorch Geometric.

    Processes the observation by:
    1. Reconstructing node features from the flattened observation
    2. Building a graph using the edge index from the substrate network
    3. Applying GNN layers (configurable: GCN, GAT, GraphSAGE)
    4. Aggregating node embeddings and concatenating with global context
    """

    def __init__(
        self,
        observation_space: gym.Space,
        num_nodes: int,
        edge_index: torch.Tensor,
        node_feat_dim: int = 6,
        hidden_dim: int = 64,
        features_dim: int = 256,
        gnn_type: str = "gcn",
        num_gnn_layers: int = 3,
        use_edge_attr: bool = False,
        edge_getter: Optional[Callable] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize the GNN Feature Extractor.

        Args:
            observation_space: The observation space of the environment
            num_nodes: Number of nodes in the substrate network
            edge_index: Edge index tensor (2, num_edges) for the graph
            node_feat_dim: Dimension of node features (default: 6)
            hidden_dim: Hidden dimension for GNN layers
            features_dim: Output dimension of the feature extractor
            gnn_type: Type of GNN layer ("gcn", "gat", "sage")
            num_gnn_layers: Number of GNN layers
            use_edge_attr: Whether to use edge attributes
            edge_getter: Callable to get current edge index (for dynamic topology)
            dropout: Dropout rate for GNN layers
        """
        super().__init__(observation_space, features_dim)

        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.use_edge_attr = use_edge_attr
        self.edge_getter = edge_getter
        self.dropout = dropout

        # Register edge index as buffer (for static topology)
        self.register_buffer("edge_index", edge_index)

        # Input dimension: node features + placement mask
        input_dim = self.node_feat_dim + 1

        # Build GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

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

            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Global context processing
        # Global dims: 3 (vnf) + 3 (sfc) + 1 (progress) = 7
        global_input_dim = 7
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Attention-based node aggregation (instead of simple mean pooling)
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Fusion layer
        # Input: Aggregated node embeddings (hidden_dim) + Global embeddings (64)
        fusion_input_dim = hidden_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def _parse_observation(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse the flattened observation into components.

        Args:
            observations: Flattened observations (batch_size, obs_dim)

        Returns:
            Tuple of (node_features, global_context, placement_mask)
        """
        batch_size = observations.shape[0]
        N = self.num_nodes
        num_feats = self.node_feat_dim

        # 1. Extract Node Features [0 : N*6]
        node_feats_flat = observations[:, : N * num_feats]
        node_feats = node_feats_flat.view(batch_size, N, num_feats)

        # 2. Extract Global Context [N*6 : N*6 + 7]
        global_context_start = N * num_feats
        global_context_end = global_context_start + 7
        global_context = observations[:, global_context_start:global_context_end]

        # 3. Extract Placement Mask [N*6 + 7 : ]
        placement_mask = observations[:, global_context_end:]
        placement_mask = placement_mask.view(batch_size, N, 1)

        return node_feats, global_context, placement_mask

    def _get_edge_index(self, device: torch.device) -> torch.Tensor:
        """
        Get the current edge index, either from the getter or the buffer.

        Args:
            device: Device to place the tensor on

        Returns:
            Edge index tensor (2, num_edges)
        """
        if self.edge_getter is not None:
            try:
                edge_index = self.edge_getter()
                if edge_index.device != device:
                    edge_index = edge_index.to(device)
                return edge_index
            except Exception:
                pass

        return self.edge_index

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN feature extractor.

        Args:
            observations: Flattened observations (batch_size, obs_dim)

        Returns:
            Feature embeddings (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        device = observations.device

        # Parse observation
        node_feats, global_context, placement_mask = self._parse_observation(
            observations
        )

        # Combine node features with placement mask -> (batch_size, N, node_feat_dim + 1)
        x = torch.cat([node_feats, placement_mask], dim=2)

        # Get edge index
        edge_index = self._get_edge_index(device)

        # Process each sample in the batch through GNN
        # We need to create a batched graph for PyG
        batch_node_features = []
        batch_list = []

        for b in range(batch_size):
            # Node features for this sample: (N, input_dim)
            node_feat_b = x[b]  # (N, input_dim)
            batch_node_features.append(node_feat_b)
            batch_list.append(
                torch.full((self.num_nodes,), b, dtype=torch.long, device=device)
            )

        # Stack all node features: (batch_size * N, input_dim)
        x_batched = torch.cat(batch_node_features, dim=0)
        batch_indices = torch.cat(batch_list, dim=0)

        # Expand edge index for the batch
        # For PyG batching, we need to offset node indices
        edge_indices_list = []
        for b in range(batch_size):
            offset = b * self.num_nodes
            edge_indices_list.append(edge_index + offset)
        edge_index_batched = torch.cat(edge_indices_list, dim=1)

        # Apply GNN layers
        h = x_batched
        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            h = gnn_layer(h, edge_index_batched)
            h = bn(h)
            h = F.relu(h)
            if i < len(self.gnn_layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Attention-weighted aggregation
        # h shape: (batch_size * N, hidden_dim)
        attention_scores = self.attention_weights(h)  # (batch_size * N, 1)
        attention_scores = attention_scores.view(batch_size, self.num_nodes, 1)

        # Apply softmax across nodes for each batch
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, N, 1)

        # Reshape h back to (batch_size, N, hidden_dim)
        h_reshaped = h.view(batch_size, self.num_nodes, self.hidden_dim)

        # Weighted sum: (batch_size, hidden_dim)
        graph_embedding = (attention_weights * h_reshaped).sum(dim=1)

        # Process global context
        global_embedding = self.global_mlp(global_context)  # (batch_size, 64)

        # Fuse graph and global embeddings
        combined = torch.cat([graph_embedding, global_embedding], dim=1)
        output = self.fusion(combined)  # (batch_size, features_dim)

        return output


class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy using GNN for SFC Placement.

    This policy uses a GNN-based feature extractor to process the substrate
    network topology and make informed VNF placement decisions.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        # GNN specific parameters
        num_nodes: int = 200,
        edge_index: Optional[torch.Tensor] = None,
        node_feat_dim: int = 6,
        hidden_dim: int = 64,
        gnn_features_dim: int = 256,
        gnn_type: str = "gcn",
        num_gnn_layers: int = 3,
        edge_getter: Optional[Callable] = None,
        dropout: float = 0.1,
        # Standard policy parameters
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = GNNFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GNN Actor-Critic Policy.
        """
        # Prepare features extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        # Add GNN-specific parameters
        features_extractor_kwargs.update(
            {
                "num_nodes": num_nodes,
                "edge_index": edge_index
                if edge_index is not None
                else torch.zeros((2, 0), dtype=torch.long),
                "node_feat_dim": node_feat_dim,
                "hidden_dim": hidden_dim,
                "features_dim": gnn_features_dim,
                "gnn_type": gnn_type,
                "num_gnn_layers": num_gnn_layers,
                "edge_getter": edge_getter,
                "dropout": dropout,
            }
        )

        # Default network architecture for actor and critic
        if net_arch is None:
            net_arch = [128, 128]

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


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
