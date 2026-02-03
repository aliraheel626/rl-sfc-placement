"""
Substrate Network for RL-based SFC Placement.

This module provides the SubstrateNetwork class which manages the physical
network topology and resources.
"""

import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.requests import VNF, SFCRequest


class SubstrateNetwork:
    """
    Manages the substrate network topology and resources.

    The network is represented as a random mesh topology where each node
    has RAM, CPU, Storage, and Security Score attributes, and each link
    has Bandwidth and Latency attributes.
    """

    def __init__(self, config: dict):
        """
        Initialize the substrate network from configuration.

        Args:
            config: Dictionary containing substrate network parameters
        """
        self.config = config
        self.num_nodes = config["num_nodes"]
        self.mesh_connectivity = config["mesh_connectivity"]
        self.resource_config = config["resources"]
        self.link_config = config["links"]

        # Initialize the network
        self.graph: nx.Graph = None
        self.node_resources: dict[int, dict] = {}
        self.original_resources: dict[int, dict] = {}
        self.link_bandwidth: dict[tuple, float] = {}
        self.original_bandwidth: dict[tuple, float] = {}

        # Track active placements for TTL management
        self.active_placements: dict[int, dict] = {}  # request_id -> placement info

        self._generate_topology()

    def _generate_topology(self):
        """Generate a random mesh topology with resource attributes."""
        # Create random mesh graph
        self.graph = nx.erdos_renyi_graph(self.num_nodes, self.mesh_connectivity)

        # Ensure the graph is connected
        while not nx.is_connected(self.graph):
            # Add edges to connect components
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                self.graph.add_edge(node1, node2)

        # Initialize node resources
        for node in self.graph.nodes():
            resources = {
                "ram": random.uniform(
                    self.resource_config["ram"]["min"],
                    self.resource_config["ram"]["max"],
                ),
                "cpu": random.uniform(
                    self.resource_config["cpu"]["min"],
                    self.resource_config["cpu"]["max"],
                ),
                "storage": random.uniform(
                    self.resource_config["storage"]["min"],
                    self.resource_config["storage"]["max"],
                ),
                "security_score": random.uniform(
                    self.resource_config["security_score"]["min"],
                    self.resource_config["security_score"]["max"],
                ),
            }
            self.node_resources[node] = resources.copy()
            self.original_resources[node] = resources.copy()

        # Initialize resource matrix for fast vectorized checks
        # Columns: [RAM, CPU, Storage, Security]
        self.resource_matrix = np.zeros((self.num_nodes, 4), dtype=np.float32)
        for node in self.graph.nodes():
            res = self.node_resources[node]
            self.resource_matrix[node] = [
                res["ram"],
                res["cpu"],
                res["storage"],
                res["security_score"],
            ]

        # Initialize link attributes
        for u, v in self.graph.edges():
            bandwidth = random.uniform(
                self.link_config["bandwidth"]["min"],
                self.link_config["bandwidth"]["max"],
            )
            latency = random.uniform(
                self.link_config["latency"]["min"], self.link_config["latency"]["max"]
            )
            self.graph[u][v]["bandwidth"] = bandwidth
            self.graph[u][v]["latency"] = latency

            edge_key = tuple(sorted([u, v]))
            self.link_bandwidth[edge_key] = bandwidth
            self.original_bandwidth[edge_key] = bandwidth

        # Initialize bandwidth matrix for fast lookups
        self.bandwidth_matrix = np.zeros(
            (self.num_nodes, self.num_nodes), dtype=np.float32
        )
        for (u, v), bw in self.link_bandwidth.items():
            self.bandwidth_matrix[u, v] = bw
            self.bandwidth_matrix[v, u] = bw

        self._precompute_latencies()

    def _precompute_latencies(self):
        """Precompute paths and latencies for all pairs of nodes."""
        # Use a numpy matrix for O(1) vectorized lookups
        self.latency_matrix = np.full(
            (self.num_nodes, self.num_nodes), float("inf"), dtype=np.float32
        )

        # Store paths for bandwidth checks
        self.all_paths = {}

        # 1. Compute all-pairs shortest paths (min-hop)
        try:
            # This returns a dict of dicts: source -> target -> path (list of nodes)
            self.all_paths = dict(nx.all_pairs_shortest_path(self.graph))
        except Exception:
            return

        # 2. Calculate latency for each path
        for u in self.graph.nodes():
            self.latency_matrix[u, u] = 0.0

            if u not in self.all_paths:
                continue

            for v, path in self.all_paths[u].items():
                if u == v:
                    continue

                # Sum latency along the path
                path_latency = 0.0
                for i in range(len(path) - 1):
                    # Direct edge lookup is faster
                    path_latency += self.graph[path[i]][path[i + 1]]["latency"]

                self.latency_matrix[u, v] = path_latency

    def regenerate_topology(self):
        """Regenerate the network topology (new random graph)."""
        self.node_resources.clear()
        self.original_resources.clear()
        self.link_bandwidth.clear()
        self.original_bandwidth.clear()
        self._generate_topology()

    def reset(self):
        """Reset the network to initial state (full resources)."""
        for node in self.graph.nodes():
            self.node_resources[node] = self.original_resources[node].copy()

            # Reset matrix row
            res = self.node_resources[node]
            self.resource_matrix[node] = [
                res["ram"],
                res["cpu"],
                res["storage"],
                res["security_score"],
            ]

        for edge_key in self.link_bandwidth:
            self.link_bandwidth[edge_key] = self.original_bandwidth[edge_key]

        # Reset bandwidth matrix
        self.bandwidth_matrix.fill(0)
        for (u, v), bw in self.link_bandwidth.items():
            self.bandwidth_matrix[u, v] = bw
            self.bandwidth_matrix[v, u] = bw

        self.active_placements.clear()

    def get_node_resources(self, node_id: int) -> dict:
        """Get current available resources for a node."""
        return self.node_resources[node_id].copy()

    def get_all_node_resources(self) -> np.ndarray:
        """
        Get normalized resource matrix for all nodes.

        Returns:
            Array of shape (num_nodes, 4) with [RAM, CPU, Storage, Security]
        """
        resources = np.zeros((self.num_nodes, 4))
        for node in range(self.num_nodes):
            res = self.node_resources[node]
            # Normalize by original capacity
            orig = self.original_resources[node]
            resources[node] = [
                res["ram"] / orig["ram"],
                res["cpu"] / orig["cpu"],
                res["storage"] / orig["storage"],
                res["security_score"] / self.resource_config["security_score"]["max"],
            ]
        return resources

    def get_nodes_connectivity(self) -> np.ndarray:
        """
        Get connectivity metrics for all nodes (avg available bandwidth).

        Returns:
            Array of shape (num_nodes, 1) normalized [0, 1]
        """
        # Sum of bandwidth for each node's edges
        total_bw_per_node = np.sum(self.bandwidth_matrix, axis=1)

        # Count non-zero entries (neighbors)
        # Note: bandwidth_matrix will have 0s where no edge exists
        # BUT it might also have 0s if bandwidth is fully used.
        # We need the neighbor count (degree), which doesn't change.
        # It's better to store the degree vector or use the adjacency matrix from graph.

        # Using networkx graph degree is cleaner and static per episode
        degrees = np.array(
            [val for (node, val) in self.graph.degree()], dtype=np.float32
        )

        # Avoid division by zero
        degrees = np.maximum(degrees, 1.0)

        avg_bw = total_bw_per_node / degrees

        # Normalize
        max_bw = self.link_config["bandwidth"]["max"]
        connectivity = (avg_bw / max_bw).reshape(-1, 1)

        return connectivity

    def get_feasible_nodes(self, vnf: "VNF", min_security: float) -> np.ndarray:
        """
        Get a boolean mask of nodes that have sufficient resources for the VNF.

        Returns:
            np.ndarray: Shape (num_nodes,) boolean vector
        """
        # Vectorized check: RAM, CPU, Storage >= required AND Security >= required
        # resource_matrix columns: 0=RAM, 1=CPU, 2=Storage, 3=Security

        # Check resources
        ram_check = self.resource_matrix[:, 0] >= vnf.ram
        cpu_check = self.resource_matrix[:, 1] >= vnf.cpu
        storage_check = self.resource_matrix[:, 2] >= vnf.storage
        security_check = self.resource_matrix[:, 3] >= min_security

        return ram_check & cpu_check & storage_check & security_check

    def check_node_feasibility(
        self, node_id: int, vnf: "VNF", min_security: float
    ) -> bool:
        """
        Check if a VNF can be placed on a node.

        Args:
            node_id: Target substrate node
            vnf: VNF to place
            min_security: SFC's minimum security requirement

        Returns:
            True if placement is feasible
        """
        res = self.node_resources[node_id]

        # Check resources
        if res["ram"] < vnf.ram:
            return False
        if res["cpu"] < vnf.cpu:
            return False
        if res["storage"] < vnf.storage:
            return False

        # Check security constraint
        if res["security_score"] < min_security:
            return False

        return True

    def check_bandwidth(self, src_node: int, dst_node: int, required_bw: float) -> bool:
        """
        Check if there's sufficient bandwidth on the path between two nodes.

        If src and dst are the same node, no bandwidth is needed.
        Uses shortest path for bandwidth check.
        """
        if src_node == dst_node:
            return True

        # Use precomputed path if available
        if (
            hasattr(self, "all_paths")
            and src_node in self.all_paths
            and dst_node in self.all_paths[src_node]
        ):
            path = self.all_paths[src_node][dst_node]
        else:
            # Fallback
            try:
                path = nx.shortest_path(self.graph, src_node, dst_node)
            except nx.NetworkXNoPath:
                return False

        # Check bandwidth on each edge of the path
        # Using bandwidth_matrix is much faster than constructing tuple keys
        for i in range(len(path) - 1):
            if self.bandwidth_matrix[path[i], path[i + 1]] < required_bw:
                return False

        return True

    def get_path_latency(self, src_node: int, dst_node: int) -> float:
        """
        Get the latency of the shortest path between two nodes.

        Returns infinity if no path exists.
        """
        if src_node == dst_node:
            return 0.0

        # Use precomputed latency if available
        if (
            hasattr(self, "path_latencies")
            and (src_node, dst_node) in self.path_latencies
        ):
            return self.path_latencies[(src_node, dst_node)]

        # Fallback for safety (though precompute should cover everything)
        try:
            path = nx.shortest_path(self.graph, src_node, dst_node)
        except nx.NetworkXNoPath:
            return float("inf")

        total_latency = 0.0
        for i in range(len(path) - 1):
            total_latency += self.graph[path[i]][path[i + 1]]["latency"]

        return total_latency

    def get_all_path_latencies(self, src_node: int) -> np.ndarray:
        """
        Get latencies from src_node to all other nodes as a vectorized array.

        Returns:
            np.ndarray: Shape (num_nodes,) containing latencies.
                        Unreachable nodes have latency = inf.
        """
        if hasattr(self, "latency_matrix"):
            return self.latency_matrix[src_node]
        elif hasattr(self, "path_latencies"):
            # Fallback to dict reconstruction
            latencies = np.full(self.num_nodes, float("inf"), dtype=np.float32)
            for dst_node in range(self.num_nodes):
                if (src_node, dst_node) in self.path_latencies:
                    latencies[dst_node] = self.path_latencies[(src_node, dst_node)]
            return latencies
        else:
            # Fallback (slow)
            latencies = np.full(self.num_nodes, float("inf"), dtype=np.float32)
            for dst_node in range(self.num_nodes):
                latencies[dst_node] = self.get_path_latency(src_node, dst_node)
            return latencies

    def get_total_latency(self, placement: list[int]) -> float:
        """
        Calculate total end-to-end latency for a placement.

        Args:
            placement: List of node IDs where each VNF is placed (in order)

        Returns:
            Total latency across all consecutive VNF pairs
        """
        total = 0.0
        for i in range(len(placement) - 1):
            total += self.get_path_latency(placement[i], placement[i + 1])
        return total

    def allocate_resources(self, node_id: int, vnf: "VNF"):
        """Allocate resources for a VNF on a node."""
        self.node_resources[node_id]["ram"] -= vnf.ram
        self.node_resources[node_id]["cpu"] -= vnf.cpu
        self.node_resources[node_id]["storage"] -= vnf.storage

        # Update matrix
        self.resource_matrix[node_id, 0] -= vnf.ram
        self.resource_matrix[node_id, 1] -= vnf.cpu
        self.resource_matrix[node_id, 2] -= vnf.storage

    def allocate_bandwidth(self, src_node: int, dst_node: int, bandwidth: float):
        """Allocate bandwidth on the path between two nodes."""
        if src_node == dst_node:
            return

        # Use precomputed path if available
        if (
            hasattr(self, "all_paths")
            and src_node in self.all_paths
            and dst_node in self.all_paths[src_node]
        ):
            path = self.all_paths[src_node][dst_node]
        else:
            try:
                path = nx.shortest_path(self.graph, src_node, dst_node)
            except nx.NetworkXNoPath:
                return

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_key = tuple(sorted([u, v]))
            self.link_bandwidth[edge_key] -= bandwidth

            # Update matrix
            self.bandwidth_matrix[u, v] -= bandwidth
            self.bandwidth_matrix[v, u] -= bandwidth

    def release_resources(self, node_id: int, vnf: "VNF"):
        """Release resources when a VNF placement expires."""
        self.node_resources[node_id]["ram"] += vnf.ram
        self.node_resources[node_id]["cpu"] += vnf.cpu
        self.node_resources[node_id]["storage"] += vnf.storage

        # Update matrix
        self.resource_matrix[node_id, 0] += vnf.ram
        self.resource_matrix[node_id, 1] += vnf.cpu
        self.resource_matrix[node_id, 2] += vnf.storage

    def release_bandwidth(self, src_node: int, dst_node: int, bandwidth: float):
        """Release bandwidth when a placement expires."""
        if src_node == dst_node:
            return

        # Use precomputed path if available
        if (
            hasattr(self, "all_paths")
            and src_node in self.all_paths
            and dst_node in self.all_paths[src_node]
        ):
            path = self.all_paths[src_node][dst_node]
        else:
            try:
                path = nx.shortest_path(self.graph, src_node, dst_node)
            except nx.NetworkXNoPath:
                return

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_key = tuple(sorted([u, v]))
            self.link_bandwidth[edge_key] += bandwidth

            # Update matrix
            self.bandwidth_matrix[u, v] += bandwidth
            self.bandwidth_matrix[v, u] += bandwidth

    def register_placement(self, request: "SFCRequest", placement: list[int]):
        """
        Register a successful placement for TTL tracking.

        Args:
            request: The SFC request that was placed
            placement: List of node IDs where VNFs are placed
        """
        self.active_placements[request.request_id] = {
            "request": request,
            "placement": placement,
            "remaining_ttl": request.ttl,
        }

    def tick(self) -> list[int]:
        """
        Advance time by one step and release expired placements.

        Returns:
            List of request IDs that expired
        """
        expired = []

        for request_id, info in list(self.active_placements.items()):
            info["remaining_ttl"] -= 1

            if info["remaining_ttl"] <= 0:
                # Release resources
                request = info["request"]
                placement = info["placement"]

                for vnf, node_id in zip(request.vnfs, placement):
                    self.release_resources(node_id, vnf)

                # Release bandwidth
                for i in range(len(placement) - 1):
                    self.release_bandwidth(
                        placement[i], placement[i + 1], request.min_bandwidth
                    )

                expired.append(request_id)
                del self.active_placements[request_id]

        return expired
