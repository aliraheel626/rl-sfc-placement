"""
Shared risk computation for SFC placement (baselines, evaluation, and RL environment).

Single source of truth for tenancy / security / incident / exposure risk so PPO and
BestFit use the same formulas.
"""

from __future__ import annotations

import numpy as np

from src.substrate import SubstrateNetwork


def robust_ratio(value: float, reference: float) -> float:
    """Bounded ratio value / (value + ref), in [0, 1]."""
    safe_ref = max(reference, 1e-6)
    value = max(value, 0.0)
    return float(min(value / (value + safe_ref), 1.0))


def effective_node_security(
    substrate: SubstrateNetwork,
    node_id: int,
    max_security: float,
    node_incident_pressure: dict[int, float],
    incident_security_penalty: float,
) -> tuple[float, float]:
    """Return (effective_security, normalized_effective_security in [0, 1])."""
    raw_security = substrate.node_resources[node_id]["security_score"]
    pressure = node_incident_pressure.get(node_id, 0.0)
    multiplier = max(0.0, 1.0 - incident_security_penalty * pressure)
    effective_security = raw_security * multiplier
    security_norm = min(
        max(effective_security / max(max_security, 1e-6), 0.0), 1.0
    )
    return effective_security, security_norm


def decay_incident_pressure(
    node_incident_pressure: dict[int, float], decay: float
) -> None:
    """Decay pressure in-place (same semantics as baseline evaluation)."""
    for node_id in list(node_incident_pressure.keys()):
        node_incident_pressure[node_id] *= decay
        if node_incident_pressure[node_id] < 1e-4:
            node_incident_pressure[node_id] = 0.0


def apply_incident_view(
    substrate: SubstrateNetwork,
    node_incident_pressure: dict[int, float],
    risk_cfg: dict,
    max_security: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporarily project incident pressure into substrate feasibility fields
    so baseline planners see the same temporary node degradation as RL masking.
    Returns backups for restore_incident_view.
    """
    original_resource_matrix = substrate.resource_matrix.copy()
    original_security = np.array(
        [substrate.node_resources[i]["security_score"] for i in range(substrate.num_nodes)],
        dtype=np.float32,
    )
    original_ram = np.array(
        [substrate.node_resources[i]["ram"] for i in range(substrate.num_nodes)],
        dtype=np.float32,
    )
    original_cpu = np.array(
        [substrate.node_resources[i]["cpu"] for i in range(substrate.num_nodes)],
        dtype=np.float32,
    )
    original_storage = np.array(
        [substrate.node_resources[i]["storage"] for i in range(substrate.num_nodes)],
        dtype=np.float32,
    )
    penalty = risk_cfg["incident_security_penalty"]
    block_threshold = risk_cfg["incident_block_threshold"]
    for node_id in range(substrate.num_nodes):
        pressure = node_incident_pressure.get(node_id, 0.0)
        eff_security, _ = effective_node_security(
            substrate,
            node_id,
            max_security,
            node_incident_pressure,
            penalty,
        )
        substrate.node_resources[node_id]["security_score"] = eff_security
        substrate.resource_matrix[node_id, 3] = eff_security
        if pressure >= block_threshold:
            substrate.node_resources[node_id]["ram"] = 0.0
            substrate.node_resources[node_id]["cpu"] = 0.0
            substrate.node_resources[node_id]["storage"] = 0.0
            substrate.resource_matrix[node_id, 0] = 0.0
            substrate.resource_matrix[node_id, 1] = 0.0
            substrate.resource_matrix[node_id, 2] = 0.0
    return (
        original_resource_matrix,
        original_security,
        original_ram,
        original_cpu,
        original_storage,
    )


def restore_incident_view(
    substrate: SubstrateNetwork,
    original_resource_matrix: np.ndarray,
    original_security: np.ndarray,
    original_ram: np.ndarray,
    original_cpu: np.ndarray,
    original_storage: np.ndarray,
) -> None:
    substrate.resource_matrix = original_resource_matrix
    for node_id in range(substrate.num_nodes):
        substrate.node_resources[node_id]["security_score"] = float(
            original_security[node_id]
        )
        substrate.node_resources[node_id]["ram"] = float(original_ram[node_id])
        substrate.node_resources[node_id]["cpu"] = float(original_cpu[node_id])
        substrate.node_resources[node_id]["storage"] = float(original_storage[node_id])


def node_risk_score(
    substrate: SubstrateNetwork,
    node_id: int,
    risk_cfg: dict,
    node_incident_pressure: dict[int, float],
    max_security: float,
    request_ttl: int = 1,
    max_ttl: int = 1,
) -> float:
    """
    Per-node risk score for greedy baselines (deterministic expected incident term).
    Same 5-component weighting as compute_placement_risk. Lower is better.
    """
    if not risk_cfg.get("enabled", False):
        return 0.0

    _, security_norm = effective_node_security(
        substrate,
        node_id,
        max_security,
        node_incident_pressure,
        risk_cfg.get("incident_security_penalty", 0.6),
    )
    inverse_security_risk = 1.0 - security_norm
    pressure = node_incident_pressure.get(node_id, 0.0)

    vnfs_per_node = substrate.get_vnfs_per_node()
    sfcs_per_node = substrate.get_sfcs_per_node()
    occupied_vnf_nodes = [v for v in vnfs_per_node.values() if v > 0]
    occupied_sfc_nodes = [v for v in sfcs_per_node.values() if v > 0]
    num_total_nodes = max(len(sfcs_per_node), 1)

    # Concentration-aware tenancy risk (same formula as compute_placement_risk).
    # substrate_util is shared: any node with SFCs also has VNFs and vice versa.
    num_occupied = len(occupied_sfc_nodes)
    substrate_util = num_occupied / num_total_nodes

    total_active_sfcs = sum(sfcs_per_node.values())
    avg_sfc_tenancy = sum(occupied_sfc_nodes) / num_occupied if occupied_sfc_nodes else 0.0
    sfc_tenancy_risk = (
        avg_sfc_tenancy * (1.0 - substrate_util) / total_active_sfcs
        if total_active_sfcs > 0 else 0.0
    )
    sfc_tenancy_risk = min(max(sfc_tenancy_risk, 0.0), 1.0)

    total_active_vnfs = sum(vnfs_per_node.values())
    avg_vnf_tenancy = sum(occupied_vnf_nodes) / num_occupied if occupied_vnf_nodes else 0.0
    vnf_tenancy_risk = (
        avg_vnf_tenancy * (1.0 - substrate_util) / total_active_vnfs
        if total_active_vnfs > 0 else 0.0
    )
    vnf_tenancy_risk = min(max(vnf_tenancy_risk, 0.0), 1.0)

    vnf_count = float(vnfs_per_node.get(node_id, 0))
    load_reference = max(
        risk_cfg.get("load_ref_floor", 1.0),
        float(np.percentile(list(vnfs_per_node.values()), 75)) if vnfs_per_node else 0.0,
    )
    load_norm = robust_ratio(vnf_count, load_reference)
    incident_base_rate = risk_cfg.get("incident_base_rate", 0.025)
    incident_alpha = risk_cfg.get("incident_alpha", 1.6)
    incident_beta = risk_cfg.get("incident_beta", 0.7)
    p_base = incident_base_rate * (1.0 - security_norm) ** incident_alpha
    p_base *= 1.0 + incident_beta * load_norm
    p_base *= 1.0 + pressure
    incident_risk = min(max(p_base, 0.0), 1.0)

    ttl_exposure_risk = min(request_ttl / max(max_ttl, 1), 1.0)

    w_inv = risk_cfg.get("w_inverse_security", 0.22)
    w_vnf = risk_cfg.get("w_vnf_tenancy", 0.22)
    w_sfc = risk_cfg.get("w_sfc_tenancy", 0.18)
    w_inc = risk_cfg.get("w_incidents", 0.28)
    w_exp = risk_cfg.get("w_exposure", 0.10)
    weight_sum = max(w_inv + w_vnf + w_sfc + w_inc + w_exp, 1e-6)
    risk = (
        w_inv * inverse_security_risk
        + w_vnf * vnf_tenancy_risk
        + w_sfc * sfc_tenancy_risk
        + w_inc * incident_risk
        + w_exp * ttl_exposure_risk
    ) / weight_sum
    return min(max(risk, 0.0), 1.0)


def compute_placement_risk(
    substrate: SubstrateNetwork,
    placement: list[int],
    request_ttl: int,
    risk_cfg: dict,
    max_security: float,
    max_ttl: int,
    node_incident_pressure: dict[int, float],
) -> tuple[float, float, float, float]:
    """
    Stochastic, TTL-aware risk integral and incident metrics for one accepted placement.

    Mutates node_incident_pressure in-place when incidents occur (same as env / compare).

    Returns:
        (risk_integral, realized_incidents, incident_cost, expected_incidents)
    """
    if not risk_cfg.get("enabled", False) or not placement:
        return 0.0, 0.0, 0.0, 0.0

    penalty = risk_cfg["incident_security_penalty"]

    sfcs_per_node = substrate.get_sfcs_per_node()
    vnfs_per_node = substrate.get_vnfs_per_node()
    occupied_sfc_nodes = [v for v in sfcs_per_node.values() if v > 0]
    occupied_vnf_nodes = [v for v in vnfs_per_node.values() if v > 0]
    num_total_nodes = max(len(sfcs_per_node), 1)

    # Concentration-aware tenancy risk:
    #   risk = avg_tenancy_occupied × (1 − substrate_util) / total_active
    # High tenancy on few nodes while many nodes are idle → high risk.
    # High tenancy when most nodes are occupied → lower risk (packing unavoidable).
    # substrate_util is shared: any node with SFCs also has VNFs and vice versa.
    num_occupied = len(occupied_sfc_nodes)
    substrate_util = num_occupied / num_total_nodes

    total_active_sfcs = sum(sfcs_per_node.values())
    avg_sfc_tenancy = sum(occupied_sfc_nodes) / num_occupied if occupied_sfc_nodes else 0.0
    sfc_tenancy_risk = (
        avg_sfc_tenancy * (1.0 - substrate_util) / total_active_sfcs
        if total_active_sfcs > 0 else 0.0
    )
    sfc_tenancy_risk = min(max(sfc_tenancy_risk, 0.0), 1.0)

    total_active_vnfs = sum(vnfs_per_node.values())
    avg_vnf_tenancy = sum(occupied_vnf_nodes) / num_occupied if occupied_vnf_nodes else 0.0
    vnf_tenancy_risk = (
        avg_vnf_tenancy * (1.0 - substrate_util) / total_active_vnfs
        if total_active_vnfs > 0 else 0.0
    )
    vnf_tenancy_risk = min(max(vnf_tenancy_risk, 0.0), 1.0)

    placement_nodes = sorted(set(placement))
    n_nodes = len(placement_nodes)

    security_norms = np.empty(n_nodes, dtype=np.float64)
    for idx, node_id in enumerate(placement_nodes):
        _, security_norms[idx] = effective_node_security(
            substrate,
            node_id,
            max_security,
            node_incident_pressure,
            penalty,
        )
    inverse_security_risk = float(np.mean(1.0 - security_norms))
    inverse_security_risk = min(max(inverse_security_risk, 0.0), 1.0)

    exposure_steps = int(max(1, min(risk_cfg["incident_steps_cap"], request_ttl)))
    ttl_exposure = min(request_ttl / max(max_ttl, 1), 1.0)
    load_reference = max(
        risk_cfg["load_ref_floor"],
        float(np.percentile(list(vnfs_per_node.values()), 75)) if vnfs_per_node else 0.0,
    )

    load_norms = np.empty(n_nodes, dtype=np.float64)
    pressures = np.empty(n_nodes, dtype=np.float64)
    for idx, node_id in enumerate(placement_nodes):
        load_norms[idx] = robust_ratio(vnfs_per_node.get(node_id, 0), load_reference)
        pressures[idx] = node_incident_pressure.get(node_id, 0.0)

    p_base = risk_cfg["incident_base_rate"] * np.power(
        1.0 - security_norms, risk_cfg["incident_alpha"]
    )
    p_base *= 1.0 + risk_cfg["incident_beta"] * load_norms
    p_base *= 1.0 + pressures
    np.clip(p_base, 0.0, 1.0, out=p_base)

    rolls = np.random.random((exposure_steps, n_nodes))
    hits = rolls < p_base[np.newaxis, :]
    expected_incidents = float(p_base.sum() * exposure_steps)
    realized_incidents = float(hits.sum())
    pressure_gain = risk_cfg["incident_pressure_gain"]
    for idx, node_id in enumerate(placement_nodes):
        node_hits = int(hits[:, idx].sum())
        if node_hits > 0:
            node_incident_pressure[node_id] = min(
                1.0,
                node_incident_pressure.get(node_id, 0.0) + pressure_gain * node_hits,
            )

    trials = max(n_nodes * exposure_steps, 1)
    incident_risk = min(realized_incidents / trials, 1.0)
    weight_sum = max(
        risk_cfg["w_vnf_tenancy"]
        + risk_cfg["w_sfc_tenancy"]
        + risk_cfg["w_inverse_security"]
        + risk_cfg["w_incidents"]
        + risk_cfg["w_exposure"],
        1e-6,
    )
    risk_score = (
        risk_cfg["w_vnf_tenancy"] * vnf_tenancy_risk
        + risk_cfg["w_sfc_tenancy"] * sfc_tenancy_risk
        + risk_cfg["w_inverse_security"] * inverse_security_risk
        + risk_cfg["w_incidents"] * incident_risk
        + risk_cfg["w_exposure"] * ttl_exposure
    ) / weight_sum
    risk_integral = min(max(risk_score, 0.0), 1.0)
    incident_cost = realized_incidents * risk_cfg["incident_cost_per_event"]
    return (
        float(risk_integral),
        float(realized_incidents),
        float(incident_cost),
        float(expected_incidents),
    )
