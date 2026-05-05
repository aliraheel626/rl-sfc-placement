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

    Each node has RAM, CPU, Storage plus three independent CIA security
    dimensions (Confidentiality, Integrity, Availability) and a trust zone ID.
    Each link has Bandwidth, Latency, and a Link Security score.
    """

    def __init__(self, config: dict):
        """
        Initialize the substrate network from configuration.

        Args:
            config: Dictionary containing substrate network parameters
        """
        self.config = config
        num_nodes_cfg = config["num_nodes"]
        if isinstance(num_nodes_cfg, dict):
            self._num_nodes_range = (num_nodes_cfg["min"], num_nodes_cfg["max"])
            self.max_nodes = config.get("max_nodes", self._num_nodes_range[1])
            self.num_nodes = min(
                random.randint(*self._num_nodes_range), self.max_nodes
            )
        else:
            self._num_nodes_range = None
            self.max_nodes = config.get("max_nodes", num_nodes_cfg)
            self.num_nodes = num_nodes_cfg
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

        # Track nodes that are hard-isolated (cannot be shared with other SFCs)
        # Maps node_id -> request_id of the SFC that isolated it
        self.hard_isolated_nodes: dict[int, int] = {}

        # Topology version counter — incremented every time the graph changes.
        # Used by the GNN edge cache to avoid redundant conversions.
        self.topology_version: int = 0

        self._generate_topology()

    def _generate_topology(self):
        """Generate a random mesh topology with resource attributes."""
        if self._num_nodes_range is not None:
            self.num_nodes = min(
                random.randint(*self._num_nodes_range), self.max_nodes
            )
        # Create random mesh graph
        self.graph = nx.erdos_renyi_graph(self.num_nodes, self.mesh_connectivity)

        # Ensure the graph is connected
        while not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                self.graph.add_edge(node1, node2)

        num_zones = self.resource_config.get("num_zones", 3)

        # Initialize node resources — CIA triad + zone
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
                "confidentiality": random.uniform(
                    self.resource_config["confidentiality"]["min"],
                    self.resource_config["confidentiality"]["max"],
                ),
                "integrity": random.uniform(
                    self.resource_config["integrity"]["min"],
                    self.resource_config["integrity"]["max"],
                ),
                "availability": random.uniform(
                    self.resource_config["availability"]["min"],
                    self.resource_config["availability"]["max"],
                ),
                "zone_id": random.randint(0, num_zones - 1),
            }
            self.node_resources[node] = resources.copy()
            self.original_resources[node] = resources.copy()

        # resource_matrix columns: [RAM, CPU, Storage, Conf, Integ, Avail]
        self.resource_matrix = np.zeros((self.num_nodes, 6), dtype=np.float32)
        for node in self.graph.nodes():
            res = self.node_resources[node]
            self.resource_matrix[node] = [
                res["ram"],
                res["cpu"],
                res["storage"],
                res["confidentiality"],
                res["integrity"],
                res["availability"],
            ]

        # Initialize link attributes (bandwidth, latency, security)
        for u, v in self.graph.edges():
            bandwidth = random.uniform(
                self.link_config["bandwidth"]["min"],
                self.link_config["bandwidth"]["max"],
            )
            latency = random.uniform(
                self.link_config["latency"]["min"], self.link_config["latency"]["max"]
            )
            link_security = random.uniform(
                self.link_config["security"]["min"],
                self.link_config["security"]["max"],
            )
            self.graph[u][v]["bandwidth"] = bandwidth
            self.graph[u][v]["latency"] = latency
            self.graph[u][v]["link_security"] = link_security

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

        # Initialize direct link security matrix
        self.link_security_matrix = np.zeros(
            (self.num_nodes, self.num_nodes), dtype=np.float32
        )
        for u, v in self.graph.edges():
            sec = self.graph[u][v]["link_security"]
            self.link_security_matrix[u, v] = sec
            self.link_security_matrix[v, u] = sec

        self._precompute_latencies()

        # Bump topology version so GNN edge cache knows to refresh
        self.topology_version += 1

    def _precompute_latencies(self):
        """Precompute paths, latencies, and path-level minimum link security."""
        self.latency_matrix = np.full(
            (self.num_nodes, self.num_nodes), float("inf"), dtype=np.float32
        )
        # min link security along shortest path (1.0 where no path or same node)
        self.min_path_link_security_matrix = np.ones(
            (self.num_nodes, self.num_nodes), dtype=np.float32
        )

        self.all_paths = {}

        try:
            self.all_paths = dict(nx.all_pairs_shortest_path(self.graph))
        except Exception:
            return

        for u in self.graph.nodes():
            self.latency_matrix[u, u] = 0.0
            self.min_path_link_security_matrix[u, u] = 1.0

            if u not in self.all_paths:
                continue

            for v, path in self.all_paths[u].items():
                if u == v:
                    continue

                path_latency = 0.0
                min_sec = 1.0
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    path_latency += self.graph[a][b]["latency"]
                    sec = self.graph[a][b]["link_security"]
                    if sec < min_sec:
                        min_sec = sec

                self.latency_matrix[u, v] = path_latency
                self.min_path_link_security_matrix[u, v] = min_sec

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

            res = self.node_resources[node]
            self.resource_matrix[node] = [
                res["ram"],
                res["cpu"],
                res["storage"],
                res["confidentiality"],
                res["integrity"],
                res["availability"],
            ]

        for edge_key in self.link_bandwidth:
            self.link_bandwidth[edge_key] = self.original_bandwidth[edge_key]

        self.bandwidth_matrix.fill(0)
        for (u, v), bw in self.link_bandwidth.items():
            self.bandwidth_matrix[u, v] = bw
            self.bandwidth_matrix[v, u] = bw

        self.active_placements.clear()
        self.hard_isolated_nodes.clear()

    def get_node_resources(self, node_id: int) -> dict:
        """Get current available resources for a node."""
        return self.node_resources[node_id].copy()

    def get_all_node_resources(self) -> np.ndarray:
        """
        Get normalized resource matrix for all nodes.

        Returns:
            Array of shape (num_nodes, 6) with [RAM, CPU, Storage, Conf, Integ, Avail]
            Values are normalized: RAM/CPU/Storage by original capacity, CIA by max config value.
        """
        max_conf = self.resource_config["confidentiality"]["max"]
        max_integ = self.resource_config["integrity"]["max"]
        max_avail = self.resource_config["availability"]["max"]

        resources = np.zeros((self.num_nodes, 6), dtype=np.float32)
        for node in range(self.num_nodes):
            res = self.node_resources[node]
            orig = self.original_resources[node]
            resources[node] = [
                res["ram"] / orig["ram"],
                res["cpu"] / orig["cpu"],
                res["storage"] / orig["storage"],
                res["confidentiality"] / max_conf,
                res["integrity"] / max_integ,
                res["availability"] / max_avail,
            ]
        return resources

    def get_nodes_connectivity(self) -> np.ndarray:
        """
        Get connectivity metrics for all nodes (avg available bandwidth).

        Returns:
            Array of shape (num_nodes, 1) normalized [0, 1]
        """
        total_bw_per_node = np.sum(self.bandwidth_matrix, axis=1)
        degrees = np.array(
            [val for (node, val) in self.graph.degree()], dtype=np.float32
        )
        degrees = np.maximum(degrees, 1.0)
        avg_bw = total_bw_per_node / degrees
        max_bw = self.link_config["bandwidth"]["max"]
        connectivity = (avg_bw / max_bw).reshape(-1, 1)
        return connectivity

    def get_feasible_nodes(self, vnf: "VNF", request: "SFCRequest") -> np.ndarray:
        """
        Get a boolean mask of nodes that have sufficient resources for the VNF
        and meet the SFC's CIA and zone constraints.

        Returns:
            np.ndarray: Shape (num_nodes,) boolean vector
        """
        # resource_matrix columns: 0=RAM, 1=CPU, 2=Storage, 3=Conf, 4=Integ, 5=Avail
        ram_check = self.resource_matrix[:, 0] >= vnf.ram
        cpu_check = self.resource_matrix[:, 1] >= vnf.cpu
        storage_check = self.resource_matrix[:, 2] >= vnf.storage
        conf_check = self.resource_matrix[:, 3] >= request.min_confidentiality
        integ_check = self.resource_matrix[:, 4] >= request.min_integrity
        avail_check = self.resource_matrix[:, 5] >= request.min_availability

        feasible = ram_check & cpu_check & storage_check & conf_check & integ_check & avail_check

        # Zone constraint
        if request.required_zone >= 0:
            for node in range(self.num_nodes):
                if feasible[node] and self.node_resources[node]["zone_id"] != request.required_zone:
                    feasible[node] = False

        return feasible

    def check_node_feasibility(
        self, node_id: int, vnf: "VNF", request: "SFCRequest"
    ) -> bool:
        """
        Check if a VNF can be placed on a node given the SFC constraints.

        Args:
            node_id: Target substrate node
            vnf: VNF to place
            request: SFC request (for CIA and zone constraints)

        Returns:
            True if placement is feasible
        """
        res = self.node_resources[node_id]

        if res["ram"] < vnf.ram:
            return False
        if res["cpu"] < vnf.cpu:
            return False
        if res["storage"] < vnf.storage:
            return False
        if res["confidentiality"] < request.min_confidentiality:
            return False
        if res["integrity"] < request.min_integrity:
            return False
        if res["availability"] < request.min_availability:
            return False
        if request.required_zone >= 0 and res["zone_id"] != request.required_zone:
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
                return False

        for i in range(len(path) - 1):
            if self.bandwidth_matrix[path[i], path[i + 1]] < required_bw:
                return False

        return True

    def check_link_security(
        self, src_node: int, dst_node: int, min_link_security: float
    ) -> bool:
        """
        Check if the minimum link security along the path meets the requirement.

        Args:
            src_node: Source node
            dst_node: Destination node
            min_link_security: Required minimum link security

        Returns:
            True if path link security >= min_link_security
        """
        if src_node == dst_node:
            return True
        if min_link_security <= 0.0:
            return True
        return float(self.min_path_link_security_matrix[src_node, dst_node]) >= min_link_security

    def get_path_latency(self, src_node: int, dst_node: int) -> float:
        """
        Get the latency of the shortest path between two nodes.

        Returns infinity if no path exists.
        """
        if src_node == dst_node:
            return 0.0

        if (
            hasattr(self, "path_latencies")
            and (src_node, dst_node) in self.path_latencies
        ):
            return self.path_latencies[(src_node, dst_node)]

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
            latencies = np.full(self.num_nodes, float("inf"), dtype=np.float32)
            for dst_node in range(self.num_nodes):
                if (src_node, dst_node) in self.path_latencies:
                    latencies[dst_node] = self.path_latencies[(src_node, dst_node)]
            return latencies
        else:
            latencies = np.full(self.num_nodes, float("inf"), dtype=np.float32)
            for dst_node in range(self.num_nodes):
                latencies[dst_node] = self.get_path_latency(src_node, dst_node)
            return latencies

    def get_min_path_link_security_from(self, src_node: int) -> np.ndarray:
        """
        Get the minimum link security along the shortest path from src_node to all nodes.

        Returns:
            np.ndarray: Shape (num_nodes,) with min path link security values.
                        Self-path and unreachable nodes return 1.0.
        """
        if hasattr(self, "min_path_link_security_matrix"):
            return self.min_path_link_security_matrix[src_node]
        return np.ones(self.num_nodes, dtype=np.float32)

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

        self.resource_matrix[node_id, 0] -= vnf.ram
        self.resource_matrix[node_id, 1] -= vnf.cpu
        self.resource_matrix[node_id, 2] -= vnf.storage

    def allocate_bandwidth(self, src_node: int, dst_node: int, bandwidth: float):
        """Allocate bandwidth on the path between two nodes."""
        if src_node == dst_node:
            return

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
            self.bandwidth_matrix[u, v] -= bandwidth
            self.bandwidth_matrix[v, u] -= bandwidth

    def release_resources(self, node_id: int, vnf: "VNF"):
        """Release resources when a VNF placement expires."""
        self.node_resources[node_id]["ram"] += vnf.ram
        self.node_resources[node_id]["cpu"] += vnf.cpu
        self.node_resources[node_id]["storage"] += vnf.storage

        self.resource_matrix[node_id, 0] += vnf.ram
        self.resource_matrix[node_id, 1] += vnf.cpu
        self.resource_matrix[node_id, 2] += vnf.storage

    def release_bandwidth(self, src_node: int, dst_node: int, bandwidth: float):
        """Release bandwidth when a placement expires."""
        if src_node == dst_node:
            return

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

        if request.hard_isolation:
            for node_id in placement:
                self.hard_isolated_nodes[node_id] = request.request_id

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
                request = info["request"]
                placement = info["placement"]

                for vnf, node_id in zip(request.vnfs, placement):
                    self.release_resources(node_id, vnf)

                for i in range(len(placement) - 1):
                    self.release_bandwidth(
                        placement[i], placement[i + 1], request.min_bandwidth
                    )

                if request.hard_isolation:
                    for node_id in placement:
                        if self.hard_isolated_nodes.get(node_id) == request_id:
                            del self.hard_isolated_nodes[node_id]

                expired.append(request_id)
                del self.active_placements[request_id]

        return expired

    def get_sfcs_per_node(self) -> dict[int, int]:
        """
        Get the count of SFCs using each substrate node.

        Returns:
            Dictionary mapping node_id -> number of SFCs using that node
        """
        sfcs_count = {node: 0 for node in self.graph.nodes()}
        for info in self.active_placements.values():
            placed_nodes = set(info["placement"])
            for node in placed_nodes:
                sfcs_count[node] += 1
        return sfcs_count

    def get_vnfs_per_node(self) -> dict[int, int]:
        """
        Get the count of VNFs placed on each substrate node.

        Returns:
            Dictionary mapping node_id -> number of VNFs on that node
        """
        vnfs_count = {node: 0 for node in self.graph.nodes()}
        for info in self.active_placements.values():
            for node in info["placement"]:
                vnfs_count[node] += 1
        return vnfs_count

    def is_node_hard_isolated(self, node_id: int) -> bool:
        """Check if a node is hard-isolated by another SFC."""
        return node_id in self.hard_isolated_nodes

    def get_hard_isolated_nodes_mask(self) -> np.ndarray:
        """
        Get a boolean mask of nodes that are hard-isolated.

        Returns:
            np.ndarray: Shape (num_nodes,) boolean vector where True means isolated
        """
        mask = np.zeros(self.num_nodes, dtype=bool)
        for node_id in self.hard_isolated_nodes:
            mask[node_id] = True
        return mask

    def has_any_sfc_on_node(self, node_id: int) -> bool:
        """Check if any SFC has a VNF placed on this node."""
        for info in self.active_placements.values():
            if node_id in info["placement"]:
                return True
        return False

    def get_nodes_with_sfcs_mask(self) -> np.ndarray:
        """
        Get a boolean mask of nodes that have any SFC placed on them.

        Returns:
            np.ndarray: Shape (num_nodes,) boolean vector where True means occupied
        """
        mask = np.zeros(self.num_nodes, dtype=bool)
        for info in self.active_placements.values():
            for node_id in info["placement"]:
                mask[node_id] = True
        return mask
