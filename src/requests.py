"""
Substrate Network and SFC Request Generation for RL-based SFC Placement.

This module provides classes for:
- SubstrateNetwork: Manages the physical network topology and resources
- VNF: Represents a Virtual Network Function with resource requirements
- SFCRequest: Represents a Service Function Chain with constraints
- RequestGenerator: Generates SFC requests sequentially
"""

from dataclasses import dataclass
import random

import networkx as nx
import numpy as np
import yaml


@dataclass
class VNF:
    """Represents a Virtual Network Function with resource requirements."""

    ram: float
    cpu: float
    storage: float
    index: int = 0  # Position in the SFC chain


@dataclass
class SFCRequest:
    """
    Represents a Service Function Chain request.

    Attributes:
        vnfs: List of VNFs in the chain (ordered)
        min_security_score: All hosting nodes must meet this security level
        max_latency: Maximum end-to-end latency allowed
        min_bandwidth: Minimum bandwidth required between consecutive VNFs
        ttl: Time to live (number of timesteps before resources are released)
        request_id: Unique identifier for this request
    """

    vnfs: list[VNF]
    min_security_score: float
    max_latency: float
    min_bandwidth: float
    ttl: int
    request_id: int = 0

    @property
    def num_vnfs(self) -> int:
        return len(self.vnfs)


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

            # Store bandwidth for tracking (use sorted tuple for consistency)
            edge_key = tuple(sorted([u, v]))
            self.link_bandwidth[edge_key] = bandwidth
            self.original_bandwidth[edge_key] = bandwidth

    def reset(self):
        """Reset the network to initial state (full resources)."""
        for node in self.graph.nodes():
            self.node_resources[node] = self.original_resources[node].copy()

        for edge_key in self.link_bandwidth:
            self.link_bandwidth[edge_key] = self.original_bandwidth[edge_key]

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

    def check_node_feasibility(
        self, node_id: int, vnf: VNF, min_security: float
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

        try:
            path = nx.shortest_path(self.graph, src_node, dst_node)
        except nx.NetworkXNoPath:
            return False

        # Check bandwidth on each edge of the path
        for i in range(len(path) - 1):
            edge_key = tuple(sorted([path[i], path[i + 1]]))
            if self.link_bandwidth[edge_key] < required_bw:
                return False

        return True

    def get_path_latency(self, src_node: int, dst_node: int) -> float:
        """
        Get the latency of the shortest path between two nodes.

        Returns infinity if no path exists.
        """
        if src_node == dst_node:
            return 0.0

        try:
            path = nx.shortest_path(self.graph, src_node, dst_node)
        except nx.NetworkXNoPath:
            return float("inf")

        total_latency = 0.0
        for i in range(len(path) - 1):
            total_latency += self.graph[path[i]][path[i + 1]]["latency"]

        return total_latency

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

    def allocate_resources(self, node_id: int, vnf: VNF):
        """Allocate resources for a VNF on a node."""
        self.node_resources[node_id]["ram"] -= vnf.ram
        self.node_resources[node_id]["cpu"] -= vnf.cpu
        self.node_resources[node_id]["storage"] -= vnf.storage

    def allocate_bandwidth(self, src_node: int, dst_node: int, bandwidth: float):
        """Allocate bandwidth on the path between two nodes."""
        if src_node == dst_node:
            return

        try:
            path = nx.shortest_path(self.graph, src_node, dst_node)
        except nx.NetworkXNoPath:
            return

        for i in range(len(path) - 1):
            edge_key = tuple(sorted([path[i], path[i + 1]]))
            self.link_bandwidth[edge_key] -= bandwidth

    def release_resources(self, node_id: int, vnf: VNF):
        """Release resources when a VNF placement expires."""
        self.node_resources[node_id]["ram"] += vnf.ram
        self.node_resources[node_id]["cpu"] += vnf.cpu
        self.node_resources[node_id]["storage"] += vnf.storage

    def release_bandwidth(self, src_node: int, dst_node: int, bandwidth: float):
        """Release bandwidth when a placement expires."""
        if src_node == dst_node:
            return

        try:
            path = nx.shortest_path(self.graph, src_node, dst_node)
        except nx.NetworkXNoPath:
            return

        for i in range(len(path) - 1):
            edge_key = tuple(sorted([path[i], path[i + 1]]))
            self.link_bandwidth[edge_key] += bandwidth

    def register_placement(self, request: SFCRequest, placement: list[int]):
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


class RequestGenerator:
    """Generates SFC requests sequentially."""

    def __init__(self, config: dict):
        """
        Initialize the request generator.

        Args:
            config: Dictionary containing SFC configuration
        """
        self.config = config
        self.vnf_count_range = (config["vnf_count"]["min"], config["vnf_count"]["max"])
        self.vnf_resources = config["vnf_resources"]
        self.constraints = config["constraints"]
        self.ttl_range = (config["ttl"]["min"], config["ttl"]["max"])

        self.next_request_id = 0

    def generate_request(self) -> SFCRequest:
        """Generate a new random SFC request."""
        # Determine number of VNFs
        num_vnfs = random.randint(*self.vnf_count_range)

        # Generate VNFs
        vnfs = []
        for i in range(num_vnfs):
            vnf = VNF(
                ram=random.uniform(
                    self.vnf_resources["ram"]["min"], self.vnf_resources["ram"]["max"]
                ),
                cpu=random.uniform(
                    self.vnf_resources["cpu"]["min"], self.vnf_resources["cpu"]["max"]
                ),
                storage=random.uniform(
                    self.vnf_resources["storage"]["min"],
                    self.vnf_resources["storage"]["max"],
                ),
                index=i,
            )
            vnfs.append(vnf)

        # Generate SFC-level constraints
        min_security = random.uniform(
            self.constraints["security_score"]["min"],
            self.constraints["security_score"]["max"],
        )
        max_latency = random.uniform(
            self.constraints["max_latency"]["min"],
            self.constraints["max_latency"]["max"],
        )
        min_bandwidth = random.uniform(
            self.constraints["min_bandwidth"]["min"],
            self.constraints["min_bandwidth"]["max"],
        )

        # Generate TTL
        ttl = random.randint(*self.ttl_range)

        request = SFCRequest(
            vnfs=vnfs,
            min_security_score=min_security,
            max_latency=max_latency,
            min_bandwidth=min_bandwidth,
            ttl=ttl,
            request_id=self.next_request_id,
        )

        self.next_request_id += 1
        return request

    def reset(self):
        """Reset the generator state."""
        self.next_request_id = 0


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
