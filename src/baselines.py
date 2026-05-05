"""
Baseline Algorithms for SFC Placement.

This module implements BestFitPlacement for comparison with RL:
greedy per-step choice by minimum resource waste (then node id).
CIA security, trust zone, and link security constraints are checked as
hard constraints — identical to the RL agent's action masking — ensuring
a fair comparison.
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
        current_placement: Optional[list[int]] = None,
    ) -> list[int]:
        """
        Get list of valid nodes for placing a VNF.

        Applies the same feasibility checks as the RL agent's action mask:
        CIA security, trust zone, hard isolation, bandwidth, latency budget,
        and link security.

        Args:
            substrate: The substrate network
            vnf: The VNF to place
            request: The SFC request (for all constraints)
            prev_node: Previous node in the chain
            current_latency: Accumulated latency so far
            remaining_vnfs: Number of VNFs remaining after this one
            min_link_latency: Minimum possible link latency (for optimistic estimate)
            current_placement: Nodes already placed in this SFC

        Returns:
            List of valid node IDs
        """
        valid_nodes = []

        # Optimistic latency budget for remaining hops
        min_remaining_latency = remaining_vnfs * min_link_latency
        latency_budget = max(0.0, request.max_latency - current_latency - min_remaining_latency)

        for node_id in range(substrate.num_nodes):
            # CIA + zone feasibility
            if not substrate.check_node_feasibility(node_id, vnf, request):
                continue

            # Hard isolation: skip nodes isolated by other SFCs
            if substrate.is_node_hard_isolated(node_id):
                continue

            # If this SFC requires hard isolation, skip nodes with other SFCs
            if request.hard_isolation:
                if substrate.has_any_sfc_on_node(node_id):
                    if current_placement is None or node_id not in current_placement:
                        continue

            # Bandwidth, latency, and link security from previous node
            if prev_node is not None:
                if not substrate.check_bandwidth(prev_node, node_id, request.min_bandwidth):
                    continue

                hop_latency = substrate.get_path_latency(prev_node, node_id)
                if hop_latency > latency_budget:
                    continue

                if not substrate.check_link_security(prev_node, node_id, request.min_link_security):
                    continue

            valid_nodes.append(node_id)

        return valid_nodes


class BestFitPlacement(BasePlacement):
    """
    Best-fit by resource waste: for each VNF, select the valid node with
    minimum total resource waste (RAM + CPU + Storage slack), breaking ties
    by node ID.  Security, zone, and link-security constraints are enforced
    as hard filters in get_valid_nodes — identical to the RL agent's masking —
    making the comparison fair.
    """

    def place(
        self,
        substrate: SubstrateNetwork,
        request: SFCRequest,
    ) -> Optional[list[int]]:
        """Place VNFs by choosing the valid node with minimum resource waste per step."""
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
                current_placement=placement,
            )

            if not valid_nodes:
                return None

            def key(n: int):
                res = substrate.get_node_resources(n)
                waste = (
                    (res["ram"] - vnf.ram)
                    + (res["cpu"] - vnf.cpu)
                    + (res["storage"] - vnf.storage)
                )
                return (waste, n)

            best_node = min(valid_nodes, key=key)
            placement.append(best_node)

            if prev_node is not None:
                current_latency += substrate.get_path_latency(prev_node, best_node)

        return placement


def get_placement_algorithm(name: str) -> BasePlacement:
    """
    Factory function to get a placement algorithm by name.

    Args:
        name: Algorithm name (best_fit)

    Returns:
        Instance of the placement algorithm
    """
    if name != "best_fit":
        raise ValueError(f"Unknown algorithm: {name}. Available: ['best_fit']")

    return BestFitPlacement()
