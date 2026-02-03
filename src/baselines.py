"""
Baseline Algorithms for SFC Placement.

This module implements baseline placement algorithms for comparison:
- ViterbiPlacement: Optimal dynamic programming approach
- FirstFitPlacement: Simple heuristic (first valid node)
- BestFitPlacement: Greedy heuristic (best resource match)
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.substrate import SubstrateNetwork
from src.requests import SFCRequest, VNF


class BasePlacement(ABC):
    """Abstract base class for placement algorithms."""

    @abstractmethod
    def place(
        self, substrate: SubstrateNetwork, request: SFCRequest
    ) -> Optional[list[int]]:
        """
        Attempt to place an SFC on the substrate network.

        Args:
            substrate: The substrate network
            request: The SFC request to place

        Returns:
            List of node IDs for placement, or None if infeasible
        """
        pass

    def get_valid_nodes(
        self,
        substrate: SubstrateNetwork,
        vnf: VNF,
        request: SFCRequest,
        prev_node: Optional[int] = None,
        current_latency: float = 0.0,
        remaining_vnfs: int = 0,
        min_link_latency: float = 1.0,
    ) -> list[int]:
        """
        Get list of valid nodes for placing a VNF.

        Args:
            substrate: The substrate network
            vnf: The VNF to place
            request: The SFC request (for constraints)
            prev_node: Previous node in the chain (for bandwidth check)
            current_latency: Accumulated latency so far
            remaining_vnfs: Number of VNFs remaining after this one
            min_link_latency: Minimum possible link latency (for optimistic estimate)

        Returns:
            List of valid node IDs
        """
        valid_nodes = []

        # Calculate latency budget (optimistic estimate for remaining hops)
        min_remaining_latency = remaining_vnfs * min_link_latency
        latency_budget = request.max_latency - current_latency - min_remaining_latency

        for node_id in range(substrate.num_nodes):
            # Check resource and security feasibility
            if not substrate.check_node_feasibility(
                node_id, vnf, request.min_security_score
            ):
                continue

            # Check bandwidth from previous node
            if prev_node is not None:
                if not substrate.check_bandwidth(
                    prev_node, node_id, request.min_bandwidth
                ):
                    continue

                # Check latency constraint (early masking)
                hop_latency = substrate.get_path_latency(prev_node, node_id)
                if hop_latency > latency_budget:
                    continue

            valid_nodes.append(node_id)

        return valid_nodes


class ViterbiPlacement(BasePlacement):
    """
    Optimal SFC placement using Viterbi-style dynamic programming.

    This algorithm builds a trellis where:
    - States: (vnf_index, substrate_node)
    - Transitions: Valid placements considering constraints
    - Metric: Minimize total path latency

    The algorithm finds the placement that minimizes total latency
    while satisfying all constraints.
    """

    def place(
        self, substrate: SubstrateNetwork, request: SFCRequest
    ) -> Optional[list[int]]:
        """
        Find optimal placement using dynamic programming.

        Returns placement with minimum latency, or None if infeasible.
        """
        num_vnfs = request.num_vnfs
        num_nodes = substrate.num_nodes

        # dp[vnf_idx][node] = (min_latency, backpointer)
        # Initialize with infinity
        INF = float("inf")
        dp = [[INF for _ in range(num_nodes)] for _ in range(num_vnfs)]
        backpointer = [[None for _ in range(num_nodes)] for _ in range(num_vnfs)]

        # Initialize first VNF - all valid nodes have latency 0
        first_vnf = request.vnfs[0]
        valid_first = self.get_valid_nodes(substrate, first_vnf, request)

        if not valid_first:
            return None

        for node in valid_first:
            dp[0][node] = 0.0

        # Fill the trellis
        min_link_latency = 1.0  # Minimum possible link latency

        for vnf_idx in range(1, num_vnfs):
            vnf = request.vnfs[vnf_idx]
            remaining_vnfs = num_vnfs - vnf_idx - 1
            min_remaining_latency = remaining_vnfs * min_link_latency

            for curr_node in range(num_nodes):
                # Check if current node is valid for this VNF
                if not substrate.check_node_feasibility(
                    curr_node, vnf, request.min_security_score
                ):
                    continue

                # Try all previous nodes
                for prev_node in range(num_nodes):
                    if dp[vnf_idx - 1][prev_node] == INF:
                        continue

                    # Check bandwidth constraint
                    if not substrate.check_bandwidth(
                        prev_node, curr_node, request.min_bandwidth
                    ):
                        continue

                    # Calculate latency for this transition
                    link_latency = substrate.get_path_latency(prev_node, curr_node)
                    total_latency = dp[vnf_idx - 1][prev_node] + link_latency

                    # Early pruning: skip if this path can't possibly meet the constraint
                    if total_latency + min_remaining_latency > request.max_latency:
                        continue

                    # Update if better
                    if total_latency < dp[vnf_idx][curr_node]:
                        dp[vnf_idx][curr_node] = total_latency
                        backpointer[vnf_idx][curr_node] = prev_node

        # Find best final placement that satisfies latency constraint
        best_latency = INF
        best_final_node = None

        for node in range(num_nodes):
            if dp[num_vnfs - 1][node] <= request.max_latency:
                if dp[num_vnfs - 1][node] < best_latency:
                    best_latency = dp[num_vnfs - 1][node]
                    best_final_node = node

        if best_final_node is None:
            return None

        # Backtrack to find placement
        placement = [None] * num_vnfs
        placement[num_vnfs - 1] = best_final_node

        for vnf_idx in range(num_vnfs - 1, 0, -1):
            placement[vnf_idx - 1] = backpointer[vnf_idx][placement[vnf_idx]]

        return placement


class FirstFitPlacement(BasePlacement):
    """
    Simple first-fit heuristic placement.

    For each VNF, selects the first valid node found.
    Fast but may not produce optimal placements.
    """

    def place(
        self, substrate: SubstrateNetwork, request: SFCRequest
    ) -> Optional[list[int]]:
        """
        Place VNFs using first-fit strategy.
        """
        placement = []
        current_latency = 0.0
        min_link_latency = 1.0  # Minimum possible link latency

        for i, vnf in enumerate(request.vnfs):
            prev_node = placement[-1] if placement else None
            remaining_vnfs = request.num_vnfs - i - 1

            valid_nodes = self.get_valid_nodes(
                substrate,
                vnf,
                request,
                prev_node,
                current_latency,
                remaining_vnfs,
                min_link_latency,
            )

            if not valid_nodes:
                return None

            # Select first valid node
            selected_node = valid_nodes[0]
            placement.append(selected_node)

            # Update accumulated latency
            if prev_node is not None:
                current_latency += substrate.get_path_latency(prev_node, selected_node)

        return placement


class BestFitPlacement(BasePlacement):
    """
    Greedy best-fit heuristic placement.

    For each VNF, selects the node with the closest matching
    resources (minimizes wasted capacity).
    """

    def place(
        self, substrate: SubstrateNetwork, request: SFCRequest
    ) -> Optional[list[int]]:
        """
        Place VNFs using best-fit strategy.
        """
        placement = []
        current_latency = 0.0
        min_link_latency = 1.0

        for i, vnf in enumerate(request.vnfs):
            prev_node = placement[-1] if placement else None
            remaining_vnfs = request.num_vnfs - i - 1

            valid_nodes = self.get_valid_nodes(
                substrate,
                vnf,
                request,
                prev_node,
                current_latency,
                remaining_vnfs,
                min_link_latency,
            )

            if not valid_nodes:
                return None

            # Select node with best resource match (minimum waste)
            best_node = None
            best_score = float("inf")

            for node in valid_nodes:
                res = substrate.get_node_resources(node)

                # Calculate waste (excess resources)
                waste = (
                    (res["ram"] - vnf.ram)
                    + (res["cpu"] - vnf.cpu)
                    + (res["storage"] - vnf.storage)
                )

                if waste < best_score:
                    best_score = waste
                    best_node = node

            placement.append(best_node)

            # Update accumulated latency
            if prev_node is not None:
                current_latency += substrate.get_path_latency(prev_node, best_node)

        return placement


def get_placement_algorithm(name: str) -> BasePlacement:
    """
    Factory function to get a placement algorithm by name.

    Args:
        name: Algorithm name (viterbi, first_fit, best_fit)

    Returns:
        Instance of the placement algorithm
    """
    algorithms = {
        "viterbi": ViterbiPlacement,
        "first_fit": FirstFitPlacement,
        "best_fit": BestFitPlacement,
    }

    if name not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}"
        )

    return algorithms[name]()
