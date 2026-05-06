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
    Temporarily project incident pressure into substrate security fields
    so baseline planners see the same effective security as RL masking.
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
    for node_id in range(substrate.num_nodes):
        eff_security, _ = effective_node_security(
            substrate,
            node_id,
            max_security,
            node_incident_pressure,
            penalty,
        )
        substrate.node_resources[node_id]["security_score"] = eff_security
        substrate.resource_matrix[node_id, 3] = eff_security
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


def compute_node_risk_score(
    substrate: SubstrateNetwork,
    node_id: int,
    risk_cfg: dict,
    node_incident_pressure: dict[int, float],
    max_security: float,
    request_ttl: int = 1,
    max_ttl: int = 1,
) -> float:
    """
    Per-node expected-loss risk score for greedy baselines. Deterministic.

    score = p_base_i × sfcs_i × min(TTL, steps_cap) / T_max

    Symmetrical with compute_placement_risk_score: placement risk = mean of per-node scores.
    Lower is better.
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
    pressure = node_incident_pressure.get(node_id, 0.0)

    vnfs_per_node = substrate.get_vnfs_per_node()
    load_reference = max(
        risk_cfg.get("load_ref_floor", 1.0),
        float(np.percentile(list(vnfs_per_node.values()), 75)) if vnfs_per_node else 0.0,
    )
    load_norm = robust_ratio(float(vnfs_per_node.get(node_id, 0)), load_reference)

    p_base = risk_cfg.get("incident_base_rate", 0.025) * (1.0 - security_norm) ** risk_cfg.get("incident_alpha", 1.6)
    p_base *= 1.0 + risk_cfg.get("incident_beta", 0.7) * load_norm
    p_base *= 1.0 + pressure
    p_base = min(max(p_base, 0.0), 1.0)

    sfcs_per_node = substrate.get_sfcs_per_node()
    sfc_count = float(sfcs_per_node.get(node_id, 0))
    exposure_steps = int(max(1, min(risk_cfg.get("incident_steps_cap", 12), request_ttl)))
    ttl_exposure = exposure_steps / max(max_ttl, 1)

    return min(p_base * sfc_count * ttl_exposure, 1.0)


def compute_placement_risk_score(
    substrate: SubstrateNetwork,
    placement: list[int],
    request_ttl: int,
    risk_cfg: dict,
    max_security: float,
    max_ttl: int,
    node_incident_pressure: dict[int, float],
) -> float:
    """
    TTL-aware deterministic security cost heuristic for one accepted placement.

        rho_r = min(mean_i(p_base_i × sfcs_i) × exposure_steps / T_max, 1.0)

    Symmetrical with compute_node_risk_score: placement cost = mean of per-node scores.
    Lower is better. Does not mutate any state.
    """
    if not risk_cfg.get("enabled", False) or not placement:
        return 0.0

    penalty = risk_cfg["incident_security_penalty"]
    sfcs_per_node = substrate.get_sfcs_per_node()
    vnfs_per_node = substrate.get_vnfs_per_node()

    placement_nodes = sorted(set(placement))
    n_nodes = len(placement_nodes)

    exposure_steps = int(max(1, min(risk_cfg["incident_steps_cap"], request_ttl)))
    load_reference = max(
        risk_cfg["load_ref_floor"],
        float(np.percentile(list(vnfs_per_node.values()), 75)) if vnfs_per_node else 0.0,
    )

    security_norms = np.empty(n_nodes, dtype=np.float64)
    load_norms = np.empty(n_nodes, dtype=np.float64)
    pressures = np.empty(n_nodes, dtype=np.float64)
    sfc_counts = np.empty(n_nodes, dtype=np.float64)
    for idx, node_id in enumerate(placement_nodes):
        _, security_norms[idx] = effective_node_security(
            substrate, node_id, max_security, node_incident_pressure, penalty,
        )
        load_norms[idx] = robust_ratio(vnfs_per_node.get(node_id, 0), load_reference)
        pressures[idx] = node_incident_pressure.get(node_id, 0.0)
        sfc_counts[idx] = float(sfcs_per_node.get(node_id, 0))

    p_base = risk_cfg["incident_base_rate"] * np.power(
        1.0 - security_norms, risk_cfg["incident_alpha"]
    )
    p_base *= 1.0 + risk_cfg["incident_beta"] * load_norms
    p_base *= 1.0 + pressures
    np.clip(p_base, 0.0, 1.0, out=p_base)

    ttl_exposure = exposure_steps / max(max_ttl, 1)
    return float(min(float(np.mean(p_base * sfc_counts)) * ttl_exposure, 1.0))
