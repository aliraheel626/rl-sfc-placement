"""
Baseline Algorithms for SFC Placement.

BestFitPlacement: greedy per-step selection by maximum security margin above the
request minimum, tie-breaking by minimum SFC co-location then minimum resource waste.
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
        valid_nodes = []
        min_remaining_latency = remaining_vnfs * min_link_latency
        latency_budget = max(0.0, request.max_latency - current_latency - min_remaining_latency)

        for node_id in range(substrate.num_nodes):
            if not substrate.check_node_feasibility(node_id, vnf, request.min_security_score):
                continue
            if substrate.is_node_hard_isolated(node_id):
                continue
            if request.hard_isolation:
                if substrate.has_any_sfc_on_node(node_id):
                    if current_placement is None or node_id not in current_placement:
                        continue
            if prev_node is not None:
                if not substrate.check_bandwidth(prev_node, node_id, request.min_bandwidth):
                    continue
                if substrate.get_path_latency(prev_node, node_id) > latency_budget:
                    continue
            valid_nodes.append(node_id)

        return valid_nodes


class BestFitPlacement(BasePlacement):
    """
    Best-fit by maximum security margin: for each VNF, select the valid node with
    the highest security score above the request minimum. Tie-break by minimum SFC
    co-location count, then minimum resource waste, then node index.
    """

    def place(
        self,
        substrate: SubstrateNetwork,
        request: SFCRequest,
    ) -> Optional[list[int]]:
        placement = []
        current_latency = 0.0
        min_link_latency = 1.0
        sfcs_per_node = substrate.get_sfcs_per_node()

        for i, vnf in enumerate(request.vnfs):
            prev_node = placement[-1] if placement else None
            remaining_vnfs = request.num_vnfs - i - 1

            valid_nodes = self.get_valid_nodes(
                substrate, vnf, request, prev_node, current_latency,
                remaining_vnfs, min_link_latency, current_placement=placement,
            )

            if not valid_nodes:
                return None

            def key(n: int):
                sec_margin = -(substrate.node_resources[n]["security_score"] - request.min_security_score)
                coloc = sfcs_per_node.get(n, 0)
                res = substrate.get_node_resources(n)
                waste = (res["ram"] - vnf.ram) + (res["cpu"] - vnf.cpu) + (res["storage"] - vnf.storage)
                return (sec_margin, coloc, waste, n)

            best_node = min(valid_nodes, key=key)
            placement.append(best_node)

            if prev_node is not None:
                current_latency += substrate.get_path_latency(prev_node, best_node)

        return placement


def get_placement_algorithm(name: str) -> BasePlacement:
    if name != "best_fit":
        raise ValueError(f"Unknown algorithm: {name}. Available: ['best_fit']")
    return BestFitPlacement()
