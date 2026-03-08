"""
Baseline Algorithms for SFC Placement.

This module implements baseline placement algorithms for comparison:
- ViterbiPlacement: Optimal dynamic programming approach (latency or risk)
- FirstFitPlacement: Simple heuristic (first valid node, optionally by risk)
- BestFitPlacement: Greedy heuristic (best resource match or lowest risk)
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.substrate import SubstrateNetwork
from src.requests import SFCRequest, VNF


def _robust_ratio(value: float, reference: float) -> float:
    """Bounded ratio value / (value + ref), in [0, 1]."""
    safe_ref = max(reference, 1e-6)
    value = max(value, 0.0)
    return float(min(value / (value + safe_ref), 1.0))


def node_risk_heuristic(
    substrate: SubstrateNetwork,
    node_id: int,
    risk_cfg: dict,
    node_incident_pressure: dict,
    max_security: float,
) -> float:
    """
    Per-node risk score used to rank nodes when optimizing for risk.
    Aligns with risk model: inverse security, tenancy, and incident pressure.
    Lower is better.
    """
    if not risk_cfg.get("enabled", False):
        return 0.0
    raw_security = substrate.node_resources[node_id]["security_score"]
    pressure = node_incident_pressure.get(node_id, 0.0)
    penalty = risk_cfg.get("incident_security_penalty", 0.6)
    multiplier = max(0.0, 1.0 - penalty * pressure)
    effective_security = raw_security * multiplier
    security_norm = min(
        max(effective_security / max(max_security, 1e-6), 0.0), 1.0
    )
    inverse_security_risk = 1.0 - security_norm

    vnfs_per_node = substrate.get_vnfs_per_node()
    sfcs_per_node = substrate.get_sfcs_per_node()
    vnf_count = vnfs_per_node.get(node_id, 0)
    sfc_count = sfcs_per_node.get(node_id, 0)
    vnf_ref = max(risk_cfg.get("load_ref_floor", 1.0), 1.0)
    sfc_ref = max(risk_cfg.get("tenancy_ref_floor", 1.0), 1.0)
    vnf_tenancy_risk = _robust_ratio(float(vnf_count), vnf_ref)
    sfc_tenancy_risk = _robust_ratio(float(sfc_count), sfc_ref)

    w_inv = risk_cfg.get("w_inverse_security", 0.22)
    w_vnf = risk_cfg.get("w_vnf_tenancy", 0.22)
    w_sfc = risk_cfg.get("w_sfc_tenancy", 0.18)
    weight_sum = max(w_inv + w_vnf + w_sfc, 1e-6)
    risk = (
        w_inv * inverse_security_risk
        + w_vnf * vnf_tenancy_risk
        + w_sfc * sfc_tenancy_risk
    ) / weight_sum
    # Add pressure as a proxy for incident propensity (higher pressure -> higher risk)
    risk = risk + 0.1 * pressure
    return min(max(risk, 0.0), 1.0)


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

            # Check hard isolation: skip nodes isolated by other SFCs
            if substrate.is_node_hard_isolated(node_id):
                continue

            # If this SFC requires hard isolation, skip nodes with other SFCs
            if request.hard_isolation:
                if substrate.has_any_sfc_on_node(node_id):
                    # Allow if it's our own placement from earlier in this SFC
                    if current_placement is None or node_id not in current_placement:
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

    Metric: Minimize total path risk only. Latency is enforced as a hard
    constraint (feasibility) and used only as tie-breaker when risk is equal.
    """

    def place(
        self,
        substrate: SubstrateNetwork,
        request: SFCRequest,
        *,
        risk_cfg: Optional[dict] = None,
        node_incident_pressure: Optional[dict] = None,
        max_security: Optional[float] = None,
    ) -> Optional[list[int]]:
        """
        Find placement that minimizes sum of node risk along the path,
        subject to latency <= max_latency. Tie-break by lower latency.
        """
        num_vnfs = request.num_vnfs
        num_nodes = substrate.num_nodes
        use_risk_heuristic = (
            risk_cfg is not None
            and risk_cfg.get("enabled", False)
            and node_incident_pressure is not None
            and max_security is not None
        )
        pressure = node_incident_pressure if node_incident_pressure else {}

        INF = float("inf")
        dp = [[INF for _ in range(num_nodes)] for _ in range(num_vnfs)]
        dp_latency = [[INF for _ in range(num_nodes)] for _ in range(num_vnfs)]
        backpointer = [[None for _ in range(num_nodes)] for _ in range(num_vnfs)]

        first_vnf = request.vnfs[0]
        valid_first = self.get_valid_nodes(substrate, first_vnf, request)
        if not valid_first:
            return None

        for node in valid_first:
            dp[0][node] = (
                node_risk_heuristic(substrate, node, risk_cfg, pressure, max_security)
                if use_risk_heuristic
                else 0.0
            )
            dp_latency[0][node] = 0.0

        min_link_latency = 1.0

        for vnf_idx in range(1, num_vnfs):
            vnf = request.vnfs[vnf_idx]
            remaining_vnfs = num_vnfs - vnf_idx - 1
            min_remaining_latency = remaining_vnfs * min_link_latency

            for curr_node in range(num_nodes):
                if not substrate.check_node_feasibility(
                    curr_node, vnf, request.min_security_score
                ):
                    continue
                if substrate.is_node_hard_isolated(curr_node):
                    continue
                if request.hard_isolation and substrate.has_any_sfc_on_node(curr_node):
                    continue

                curr_risk = (
                    node_risk_heuristic(
                        substrate, curr_node, risk_cfg, pressure, max_security
                    )
                    if use_risk_heuristic
                    else 0.0
                )

                for prev_node in range(num_nodes):
                    if dp[vnf_idx - 1][prev_node] == INF:
                        continue
                    if not substrate.check_bandwidth(
                        prev_node, curr_node, request.min_bandwidth
                    ):
                        continue

                    link_latency = substrate.get_path_latency(prev_node, curr_node)
                    total_latency = dp_latency[vnf_idx - 1][prev_node] + link_latency
                    if total_latency + min_remaining_latency > request.max_latency:
                        continue

                    total_risk = dp[vnf_idx - 1][prev_node] + curr_risk
                    # Update if better risk, or same risk with lower latency (tie-break)
                    better = total_risk < dp[vnf_idx][curr_node] or (
                        total_risk == dp[vnf_idx][curr_node]
                        and total_latency < dp_latency[vnf_idx][curr_node]
                    )
                    if better:
                        dp[vnf_idx][curr_node] = total_risk
                        dp_latency[vnf_idx][curr_node] = total_latency
                        backpointer[vnf_idx][curr_node] = prev_node

        best_final_node = None
        best_risk = INF
        best_latency = INF
        for node in range(num_nodes):
            if dp_latency[num_vnfs - 1][node] <= request.max_latency:
                if dp[num_vnfs - 1][node] < best_risk or (
                    dp[num_vnfs - 1][node] == best_risk
                    and dp_latency[num_vnfs - 1][node] < best_latency
                ):
                    best_risk = dp[num_vnfs - 1][node]
                    best_latency = dp_latency[num_vnfs - 1][node]
                    best_final_node = node

        if best_final_node is None:
            return None

        placement = [None] * num_vnfs
        placement[num_vnfs - 1] = best_final_node
        for vnf_idx in range(num_vnfs - 1, 0, -1):
            placement[vnf_idx - 1] = backpointer[vnf_idx][placement[vnf_idx]]
        return placement


class FirstFitPlacement(BasePlacement):
    """
    First-fit by risk only: for each VNF, select the valid node with lowest risk
    (first when sorted by risk ascending). Tie-break by node id.
    """

    def place(
        self,
        substrate: SubstrateNetwork,
        request: SFCRequest,
        *,
        risk_cfg: Optional[dict] = None,
        node_incident_pressure: Optional[dict] = None,
        max_security: Optional[float] = None,
    ) -> Optional[list[int]]:
        """
        Place VNFs by always choosing the lowest-risk valid node per step.
        """
        placement = []
        current_latency = 0.0
        min_link_latency = 1.0
        pressure = node_incident_pressure if node_incident_pressure else {}
        max_sec = max_security if max_security is not None else 1.0

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

            # Always order by risk (ascending); tie-break by node id
            valid_nodes = sorted(
                valid_nodes,
                key=lambda n: (
                    node_risk_heuristic(substrate, n, risk_cfg or {}, pressure, max_sec),
                    n,
                ),
            )
            selected_node = valid_nodes[0]
            placement.append(selected_node)

            if prev_node is not None:
                current_latency += substrate.get_path_latency(prev_node, selected_node)

        return placement


class BestFitPlacement(BasePlacement):
    """
    Best-fit by risk only: for each VNF, select the valid node with minimum risk.
    Tie-break by resource waste (then node id).
    """

    def place(
        self,
        substrate: SubstrateNetwork,
        request: SFCRequest,
        *,
        risk_cfg: Optional[dict] = None,
        node_incident_pressure: Optional[dict] = None,
        max_security: Optional[float] = None,
    ) -> Optional[list[int]]:
        """
        Place VNFs by always choosing the valid node with minimum risk per step.
        """
        placement = []
        current_latency = 0.0
        min_link_latency = 1.0
        pressure = node_incident_pressure if node_incident_pressure else {}
        max_sec = max_security if max_security is not None else 1.0

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
                risk = node_risk_heuristic(
                    substrate, n, risk_cfg or {}, pressure, max_sec
                )
                res = substrate.get_node_resources(n)
                waste = (
                    (res["ram"] - vnf.ram)
                    + (res["cpu"] - vnf.cpu)
                    + (res["storage"] - vnf.storage)
                )
                return (risk, waste, n)

            best_node = min(valid_nodes, key=key)
            placement.append(best_node)

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
