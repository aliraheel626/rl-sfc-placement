"""
Baseline Algorithms for SFC Placement.

BestFitPlacement:  greedy per-step, maximise security margin, tie-break by co-location then waste.
FirstFitPlacement: greedy per-step, first valid node by index.
ViterbiPlacement:  DP over the full chain, globally maximises total security margin.
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
    the highest security score above the request minimum. Tie-break by minimum
    co-location, then minimum resource waste, then node index.
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


class FirstFitPlacement(BasePlacement):
    """Place each VNF on the lowest-indexed valid node (first fit)."""

    def place(
        self,
        substrate: SubstrateNetwork,
        request: SFCRequest,
    ) -> Optional[list[int]]:
        placement = []
        current_latency = 0.0
        min_link_latency = 1.0

        for i, vnf in enumerate(request.vnfs):
            prev_node = placement[-1] if placement else None
            remaining_vnfs = request.num_vnfs - i - 1

            valid_nodes = self.get_valid_nodes(
                substrate, vnf, request, prev_node, current_latency,
                remaining_vnfs, min_link_latency, current_placement=placement,
            )

            if not valid_nodes:
                return None

            best_node = min(valid_nodes)
            placement.append(best_node)

            if prev_node is not None:
                current_latency += substrate.get_path_latency(prev_node, best_node)

        return placement


class ViterbiPlacement(BasePlacement):
    """
    DP/Viterbi placement: finds the globally optimal chain placement that
    maximises total security margin subject to all resource, bandwidth,
    and latency constraints.

    State: (vnf_index, node_id) with tracked accumulated latency.
    Complexity: O(K × N²) where K = VNFs, N = substrate nodes.
    """

    def place(
        self,
        substrate: SubstrateNetwork,
        request: SFCRequest,
    ) -> Optional[list[int]]:
        num_vnfs = request.num_vnfs
        num_nodes = substrate.num_nodes
        min_link_latency = 1.0
        INF = float("inf")

        # backtrack[step] maps node_id → best predecessor node_id (None for step 0)
        backtrack: list[dict[int, Optional[int]]] = []
        prev_score: dict[int, float] = {}   # node → best cumulative security margin
        prev_latency: dict[int, float] = {} # node → accumulated latency on that path

        # --- VNF 0: no predecessor ---
        vnf0 = request.vnfs[0]
        step0: dict[int, Optional[int]] = {}
        for n in range(num_nodes):
            if not substrate.check_node_feasibility(n, vnf0, request.min_security_score):
                continue
            if substrate.is_node_hard_isolated(n):
                continue
            if request.hard_isolation and substrate.has_any_sfc_on_node(n):
                continue
            # Reserve minimum latency for remaining hops
            if (num_vnfs - 1) * min_link_latency > request.max_latency:
                continue
            prev_score[n] = substrate.node_resources[n]["security_score"] - request.min_security_score
            prev_latency[n] = 0.0
            step0[n] = None
        backtrack.append(step0)

        if not prev_score:
            return None

        # --- VNFs 1..K-1 ---
        for i in range(1, num_vnfs):
            vnf = request.vnfs[i]
            remaining = num_vnfs - i - 1
            curr_score: dict[int, float] = {}
            curr_latency: dict[int, float] = {}
            step_bt: dict[int, Optional[int]] = {}

            for n in range(num_nodes):
                if not substrate.check_node_feasibility(n, vnf, request.min_security_score):
                    continue
                if substrate.is_node_hard_isolated(n):
                    continue
                if request.hard_isolation and substrate.has_any_sfc_on_node(n):
                    continue

                node_margin = substrate.node_resources[n]["security_score"] - request.min_security_score
                best_total = -INF
                best_lat = None
                best_prev = None

                for m, m_score in prev_score.items():
                    if not substrate.check_bandwidth(m, n, request.min_bandwidth):
                        continue
                    acc_lat = prev_latency[m] + substrate.get_path_latency(m, n)
                    if acc_lat + remaining * min_link_latency > request.max_latency:
                        continue
                    total = m_score + node_margin
                    if total > best_total:
                        best_total = total
                        best_lat = acc_lat
                        best_prev = m

                if best_prev is not None:
                    curr_score[n] = best_total
                    curr_latency[n] = best_lat
                    step_bt[n] = best_prev

            if not curr_score:
                return None

            backtrack.append(step_bt)
            prev_score = curr_score
            prev_latency = curr_latency

        # --- Backtrack from best final node ---
        best_last = max(prev_score, key=lambda n: prev_score[n])
        placement = [0] * num_vnfs
        placement[-1] = best_last
        for i in range(num_vnfs - 2, -1, -1):
            placement[i] = backtrack[i + 1][placement[i + 1]]

        return placement


def get_placement_algorithm(name: str) -> BasePlacement:
    algos = {"best_fit": BestFitPlacement, "first_fit": FirstFitPlacement, "viterbi": ViterbiPlacement}
    if name not in algos:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algos)}")
    return algos[name]()
