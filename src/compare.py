"""
Comparison and Evaluation Script for SFC Placement Algorithms.

This module provides functions to:
- Evaluate individual algorithms on the SFC placement problem
- Compare all algorithms (RL + baselines)
- Generate metrics and visualizations
"""

import argparse
from typing import Optional
from pathlib import Path
import copy
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.substrate import SubstrateNetwork
from src.requests import RequestGenerator, load_config
from src.model import create_masked_env, load_model
from src.baselines import (
    BasePlacement,
    ViterbiPlacement,
    FirstFitPlacement,
    BestFitPlacement,
)


def _build_risk_config(config: dict) -> dict:
    """Build risk configuration with defaults."""
    risk = config.get("risk", {})
    return {
        "enabled": bool(risk.get("enabled", False)),
        "w_vnf_tenancy": float(risk.get("w_vnf_tenancy", 0.22)),
        "w_sfc_tenancy": float(risk.get("w_sfc_tenancy", 0.18)),
        "w_inverse_security": float(risk.get("w_inverse_security", 0.22)),
        "w_incidents": float(risk.get("w_incidents", 0.28)),
        "w_exposure": float(risk.get("w_exposure", 0.10)),
        "tenancy_ref_floor": float(risk.get("tenancy_ref_floor", 1.0)),
        "load_ref_floor": float(risk.get("load_ref_floor", 1.0)),
        "incident_base_rate": float(risk.get("incident_base_rate", 0.025)),
        "incident_alpha": float(risk.get("incident_alpha", 1.6)),
        "incident_beta": float(risk.get("incident_beta", 0.7)),
        "incident_steps_cap": int(risk.get("incident_steps_cap", 12)),
        "incident_pressure_gain": float(risk.get("incident_pressure_gain", 0.20)),
        "incident_pressure_decay": float(risk.get("incident_pressure_decay", 0.92)),
        "incident_security_penalty": float(risk.get("incident_security_penalty", 0.60)),
        "incident_cost_per_event": float(risk.get("incident_cost_per_event", 0.2)),
        "incident_block_threshold": float(risk.get("incident_block_threshold", 0.85)),
        "risk_lambda": float(risk.get("lambda", 0.0)),
    }


def _robust_ratio(value: float, reference: float) -> float:
    safe_ref = max(reference, 1e-6)
    value = max(value, 0.0)
    return float(min(value / (value + safe_ref), 1.0))


def _decay_incident_pressure(node_incident_pressure: dict[int, float], decay: float):
    for node_id in list(node_incident_pressure.keys()):
        node_incident_pressure[node_id] *= decay
        if node_incident_pressure[node_id] < 1e-4:
            node_incident_pressure[node_id] = 0.0


def _effective_security(
    substrate: SubstrateNetwork,
    node_id: int,
    max_security: float,
    node_incident_pressure: dict[int, float],
    risk_cfg: dict,
) -> tuple[float, float]:
    """Return (effective_security, normalized_effective_security)."""
    raw_security = substrate.node_resources[node_id]["security_score"]
    pressure = node_incident_pressure.get(node_id, 0.0)
    multiplier = max(0.0, 1.0 - risk_cfg["incident_security_penalty"] * pressure)
    effective_security = raw_security * multiplier
    security_norm = min(max(effective_security / max(max_security, 1e-6), 0.0), 1.0)
    return effective_security, security_norm


def _apply_incident_view_for_planning(
    substrate: SubstrateNetwork,
    node_incident_pressure: dict[int, float],
    risk_cfg: dict,
    max_security: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporarily project incident pressure into substrate feasibility fields
    so baseline planners see the same temporary node degradation as RL masking.
    Returns backups for restore.
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
    for node_id in range(substrate.num_nodes):
        pressure = node_incident_pressure.get(node_id, 0.0)
        eff_security, _ = _effective_security(
            substrate, node_id, max_security, node_incident_pressure, risk_cfg
        )
        substrate.node_resources[node_id]["security_score"] = eff_security
        substrate.resource_matrix[node_id, 3] = eff_security
        if pressure >= risk_cfg["incident_block_threshold"]:
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


def _restore_incident_view(
    substrate: SubstrateNetwork,
    original_resource_matrix: np.ndarray,
    original_security: np.ndarray,
    original_ram: np.ndarray,
    original_cpu: np.ndarray,
    original_storage: np.ndarray,
):
    substrate.resource_matrix = original_resource_matrix
    for node_id in range(substrate.num_nodes):
        substrate.node_resources[node_id]["security_score"] = float(original_security[node_id])
        substrate.node_resources[node_id]["ram"] = float(original_ram[node_id])
        substrate.node_resources[node_id]["cpu"] = float(original_cpu[node_id])
        substrate.node_resources[node_id]["storage"] = float(original_storage[node_id])


def _compute_risk_metrics(
    substrate: SubstrateNetwork,
    placement: list[int],
    request_ttl: int,
    risk_cfg: dict,
    max_security: float,
    max_ttl: int,
    node_incident_pressure: dict[int, float],
) -> tuple[float, float, float, float]:
    """Compute stochastic, TTL-aware risk integral and incident metrics."""
    if not risk_cfg.get("enabled", False) or not placement:
        return 0.0, 0.0, 0.0, 0.0

    sfcs_per_node = substrate.get_sfcs_per_node()
    vnfs_per_node = substrate.get_vnfs_per_node()
    occupied_sfc_nodes = [v for v in sfcs_per_node.values() if v > 0]
    occupied_vnf_nodes = [v for v in vnfs_per_node.values() if v > 0]

    avg_sfc_tenancy = (
        sum(occupied_sfc_nodes) / len(occupied_sfc_nodes) if occupied_sfc_nodes else 0.0
    )
    avg_vnf_tenancy = (
        sum(occupied_vnf_nodes) / len(occupied_vnf_nodes) if occupied_vnf_nodes else 0.0
    )
    sfc_reference = max(
        risk_cfg["tenancy_ref_floor"],
        float(np.percentile(occupied_sfc_nodes, 75)) if occupied_sfc_nodes else 0.0,
    )
    vnf_reference = max(
        risk_cfg["tenancy_ref_floor"],
        float(np.percentile(occupied_vnf_nodes, 75)) if occupied_vnf_nodes else 0.0,
    )
    sfc_tenancy_risk = _robust_ratio(avg_sfc_tenancy, sfc_reference)
    vnf_tenancy_risk = _robust_ratio(avg_vnf_tenancy, vnf_reference)

    placement_nodes = sorted(set(placement))
    n_nodes = len(placement_nodes)

    security_norms = np.empty(n_nodes, dtype=np.float64)
    for idx, node_id in enumerate(placement_nodes):
        _, security_norms[idx] = _effective_security(
            substrate, node_id, max_security, node_incident_pressure, risk_cfg
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
        load_norms[idx] = _robust_ratio(vnfs_per_node.get(node_id, 0), load_reference)
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
    # TTL exposure is already included via w_exposure * ttl_exposure in risk_score.
    # Do not multiply by ttl_exposure again (prevents quadratic TTL scaling).
    risk_integral = min(max(risk_score, 0.0), 1.0)
    incident_cost = realized_incidents * risk_cfg["incident_cost_per_event"]
    return (
        float(risk_integral),
        float(realized_incidents),
        float(incident_cost),
        float(expected_incidents),
    )


def evaluate_baseline(
    substrate: SubstrateNetwork,
    request_generator: RequestGenerator,
    algorithm: BasePlacement,
    num_requests: int,
    risk_cfg: dict,
    max_security: float,
    max_ttl: int,
    verbose: bool = False,
) -> dict:
    """
    Evaluate a baseline algorithm on a series of requests with sustained load.

    Args:
        substrate: The substrate network (will be copied)
        request_generator: Generator for SFC requests
        algorithm: The placement algorithm to evaluate
        num_requests: Number of requests to process
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics
    """
    # Create a copy of the substrate to not affect original
    substrate_copy = copy.deepcopy(substrate)
    substrate_copy.reset()  # Start with fresh resources
    request_generator.reset()

    accepted = 0
    rejected = 0
    total_latency = 0.0
    latencies = []
    security_margins = []  # Track security margins for each accepted request
    sfc_tenancy_samples = []  # Track SFC tenancy per occupied node over time
    vnf_tenancy_samples = []  # Track VNF tenancy per occupied node over time
    substrate_utilization_samples = []  # Track % of nodes being used over time
    risk_score_samples = []  # Per-request risk score
    realized_incident_samples = []
    incident_cost_samples = []
    total_expected_incidents = 0.0
    total_realized_incidents = 0.0
    total_incident_cost = 0.0
    total_risk_integral = 0.0
    node_incident_pressure = {node_id: 0.0 for node_id in range(substrate_copy.num_nodes)}

    start_time = time.perf_counter()

    iterator = range(num_requests)
    if verbose:
        iterator = tqdm(iterator, desc=f"Evaluating {algorithm.__class__.__name__}")

    for _ in iterator:
        # Advance time first (release expired placements from previous steps)
        substrate_copy.tick()
        _decay_incident_pressure(
            node_incident_pressure,
            risk_cfg["incident_pressure_decay"],
        )

        # Generate a new request
        request = request_generator.generate_request()

        # Try to place
        (
            original_resource_matrix,
            original_security,
            original_ram,
            original_cpu,
            original_storage,
        ) = _apply_incident_view_for_planning(
            substrate_copy,
            node_incident_pressure,
            risk_cfg,
            max_security,
        )
        placement = algorithm.place(
            substrate_copy,
            request,
            risk_cfg=risk_cfg,
            node_incident_pressure=node_incident_pressure,
            max_security=max_security,
        )
        _restore_incident_view(
            substrate_copy,
            original_resource_matrix,
            original_security,
            original_ram,
            original_cpu,
            original_storage,
        )

        if placement is not None:
            # Successful placement
            accepted += 1

            # Calculate and record latency
            latency = substrate_copy.get_total_latency(placement)
            total_latency += latency
            latencies.append(latency)

            # Calculate security margin for this placement
            # Margin = node_security_score - sfc_min_security for each node
            placement_margins = []
            for node_id in placement:
                node_security = substrate_copy.node_resources[node_id]["security_score"]
                margin = node_security - request.min_security_score
                placement_margins.append(margin)
            avg_placement_margin = (
                sum(placement_margins) / len(placement_margins)
                if placement_margins
                else 0.0
            )
            security_margins.append(avg_placement_margin)

            # Allocate resources
            for vnf, node_id in zip(request.vnfs, placement):
                substrate_copy.allocate_resources(node_id, vnf)

            # Allocate bandwidth
            for i in range(len(placement) - 1):
                substrate_copy.allocate_bandwidth(
                    placement[i], placement[i + 1], request.min_bandwidth
                )

            # Register for TTL tracking
            substrate_copy.register_placement(request, placement)
            (
                risk_integral,
                realized_incidents,
                incident_cost,
                expected_incidents,
            ) = _compute_risk_metrics(
                substrate_copy,
                placement,
                request.ttl,
                risk_cfg,
                max_security,
                max_ttl,
                node_incident_pressure,
            )
            risk_score_samples.append(risk_integral)
            realized_incident_samples.append(realized_incidents)
            incident_cost_samples.append(incident_cost)
            total_expected_incidents += expected_incidents
            total_realized_incidents += realized_incidents
            total_incident_cost += incident_cost
            total_risk_integral += risk_integral

        # Sample tenancy metrics after each request (regardless of accept/reject)
        sfcs_per_node = substrate_copy.get_sfcs_per_node()
        vnfs_per_node = substrate_copy.get_vnfs_per_node()

        # Calculate average tenancy only on occupied nodes (nodes with count > 0)
        occupied_sfc_nodes = [v for v in sfcs_per_node.values() if v > 0]
        occupied_vnf_nodes = [v for v in vnfs_per_node.values() if v > 0]

        if occupied_sfc_nodes:
            sfc_tenancy_samples.append(
                sum(occupied_sfc_nodes) / len(occupied_sfc_nodes)
            )
        if occupied_vnf_nodes:
            vnf_tenancy_samples.append(
                sum(occupied_vnf_nodes) / len(occupied_vnf_nodes)
            )

        # Substrate utilization: % of nodes being used
        if sfcs_per_node:
            nodes_in_use = len(occupied_sfc_nodes)
            total_nodes = len(sfcs_per_node)
            substrate_utilization_samples.append(nodes_in_use / total_nodes)

        if placement is None:
            rejected += 1
            # Rejected: no placement; risk 0 (consistent with env and RL eval)
            risk_score_samples.append(0.0)
            realized_incident_samples.append(0.0)
            incident_cost_samples.append(0.0)

    # Compute metrics
    total = accepted + rejected
    elapsed_time = time.perf_counter() - start_time
    acceptance_ratio = accepted / total if total > 0 else 0.0
    avg_latency = total_latency / accepted if accepted > 0 else 0.0
    avg_time_per_request = (elapsed_time / total * 1000) if total > 0 else 0.0  # ms
    avg_sec_margin = (
        sum(security_margins) / len(security_margins) if security_margins else 0.0
    )
    avg_sfc_tenancy = (
        sum(sfc_tenancy_samples) / len(sfc_tenancy_samples)
        if sfc_tenancy_samples
        else 0.0
    )
    avg_vnf_tenancy = (
        sum(vnf_tenancy_samples) / len(vnf_tenancy_samples)
        if vnf_tenancy_samples
        else 0.0
    )
    avg_substrate_utilization = (
        sum(substrate_utilization_samples) / len(substrate_utilization_samples)
        if substrate_utilization_samples
        else 0.0
    )
    avg_risk_score = sum(risk_score_samples) / len(risk_score_samples) if risk_score_samples else 0.0
    avg_expected_incidents = total_expected_incidents / total if total > 0 else 0.0
    avg_realized_incidents = total_realized_incidents / total if total > 0 else 0.0
    avg_incident_cost = total_incident_cost / total if total > 0 else 0.0
    avg_risk_integral = total_risk_integral / total if total > 0 else 0.0

    # Baselines do not track rejection reason; use 0 for latency violation ratio
    latency_violation_ratio = 0.0

    return {
        "algorithm": algorithm.__class__.__name__,
        "total_requests": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_ratio": acceptance_ratio,
        "latency_violation_ratio": latency_violation_ratio,
        "avg_latency": avg_latency,
        "avg_time_ms": avg_time_per_request,
        "avg_sec_margin": avg_sec_margin,
        "avg_sfc_tenancy": avg_sfc_tenancy,
        "avg_vnf_tenancy": avg_vnf_tenancy,
        "avg_substrate_utilization": avg_substrate_utilization,
        "avg_risk_score": avg_risk_score,
        "avg_risk_integral": avg_risk_integral,
        "avg_expected_incidents": avg_expected_incidents,
        "avg_realized_incidents": avg_realized_incidents,
        "avg_incident_cost": avg_incident_cost,
        "latencies": latencies,
        # Per-request samples for rolling average plots
        "sfc_tenancy_samples": sfc_tenancy_samples,
        "vnf_tenancy_samples": vnf_tenancy_samples,
        "substrate_utilization_samples": substrate_utilization_samples,
        "risk_score_samples": risk_score_samples,
        "realized_incident_samples": realized_incident_samples,
        "incident_cost_samples": incident_cost_samples,
    }


def evaluate_rl_agent(
    model_path: str,
    config_path: str,
    num_requests: int,
    verbose: bool = False,
    *,
    model=None,
    substrate: Optional[SubstrateNetwork] = None,
    request_generator: Optional[RequestGenerator] = None,
) -> dict:
    """
    Evaluate a trained RL agent.

    Args:
        model_path: Path to the trained model (ignored if model is provided)
        config_path: Path to configuration file
        num_requests: Number of requests to process
        verbose: Whether to print progress
        model: In-memory model to use (if provided, model_path is optional)
        substrate: Optional substrate (use same as baselines for fair comparison)
        request_generator: Optional request generator (required if substrate provided)

    Returns:
        Dictionary with evaluation metrics
    """
    # Create masked environment (optionally on same substrate as baselines).
    # Use one long episode (no resets) so utilization / VNF-per-node match baseline semantics.
    env = create_masked_env(
        config_path,
        substrate=substrate,
        request_generator=request_generator,
        max_requests_per_episode=num_requests,
    )

    if model is not None:
        pass  # Use provided in-memory model
    else:
        model = load_model(model_path, env)

    accepted = 0
    rejected = 0
    latency_violations = 0  # Rejections due to latency constraint
    total_latency = 0.0
    latencies = []
    security_margins = []  # Track security margins for each accepted request
    sfc_tenancy_samples = []  # Track SFC tenancy per occupied node over time
    vnf_tenancy_samples = []  # Track VNF tenancy per occupied node over time
    substrate_utilization_samples = []  # Track % of nodes being used over time
    risk_score_samples = []  # Per-request integrated risk
    realized_incident_samples = []
    incident_cost_samples = []
    total_expected_incidents = 0.0
    total_realized_incidents = 0.0
    total_incident_cost = 0.0
    total_risk_integral = 0.0

    # Reset environment once
    obs, info = env.reset()

    requests_processed = 0
    pbar = None
    if verbose:
        pbar = tqdm(total=num_requests, desc="Evaluating RL Agent")

    start_time = time.perf_counter()

    while requests_processed < num_requests:
        # Get action mask
        action_mask = env.unwrapped.action_masks()

        # Get action from model
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        if isinstance(action, np.ndarray):
            action = action.item()

        # Take action
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if request processing finished (sfc_complete is True)
        if info.get("sfc_complete", False):
            requests_processed += 1
            if pbar:
                pbar.update(1)

            if info.get("success", False):
                accepted += 1
                # Get latency from placement
                placement = info.get("placement", [])
                if len(placement) > 1:
                    latency = env.unwrapped.substrate.get_total_latency(placement)
                    total_latency += latency
                    latencies.append(latency)

                # Calculate security margin for this placement
                if placement:
                    placement_margins = []
                    min_security = info.get("request_min_security", 0)
                    for node_id in placement:
                        node_security = env.unwrapped.substrate.node_resources[node_id][
                            "security_score"
                        ]
                        margin = node_security - min_security
                        placement_margins.append(margin)
                    avg_placement_margin = (
                        sum(placement_margins) / len(placement_margins)
                        if placement_margins
                        else 0.0
                    )
                    security_margins.append(avg_placement_margin)
                # Use risk from env info (computed before tick()); recomputing after step()
                # would use post-tick substrate state and disagree with training rewards.
                risk_score = info.get("request_risk_integral", info.get("request_risk_score", 0.0))
                expected_incidents = info.get("request_expected_incidents", 0.0)
                realized_incidents = info.get(
                    "request_realized_incidents", info.get("request_incidents", 0.0)
                )
                incident_cost = info.get("request_incident_cost", 0.0)
                risk_score_samples.append(risk_score)
                realized_incident_samples.append(realized_incidents)
                incident_cost_samples.append(incident_cost)
                total_expected_incidents += expected_incidents
                total_realized_incidents += realized_incidents
                total_incident_cost += incident_cost
                total_risk_integral += risk_score
            else:
                rejected += 1
                if info.get("rejection_reason") == "latency_violation":
                    latency_violations += 1
                # Rejected: use env's risk from info (0) for consistency with training.
                risk_score_samples.append(info.get("request_risk_integral", info.get("request_risk_score", 0.0)))
                realized_incident_samples.append(
                    info.get("request_realized_incidents", info.get("request_incidents", 0.0))
                )
                incident_cost_samples.append(info.get("request_incident_cost", 0.0))

            # Sample tenancy metrics after each request
            sfcs_per_node = env.unwrapped.substrate.get_sfcs_per_node()
            vnfs_per_node = env.unwrapped.substrate.get_vnfs_per_node()

            # Calculate average tenancy only on occupied nodes (nodes with count > 0)
            occupied_sfc_nodes = [v for v in sfcs_per_node.values() if v > 0]
            occupied_vnf_nodes = [v for v in vnfs_per_node.values() if v > 0]

            if occupied_sfc_nodes:
                sfc_tenancy_samples.append(
                    sum(occupied_sfc_nodes) / len(occupied_sfc_nodes)
                )
            if occupied_vnf_nodes:
                vnf_tenancy_samples.append(
                    sum(occupied_vnf_nodes) / len(occupied_vnf_nodes)
                )

            # Substrate utilization: % of nodes being used
            if sfcs_per_node:
                nodes_in_use = len(occupied_sfc_nodes)
                total_nodes = len(sfcs_per_node)
                substrate_utilization_samples.append(nodes_in_use / total_nodes)

        # Reset environment if episode terminated
        if terminated or truncated:
            obs, info = env.reset()

    if pbar:
        pbar.close()

    total = accepted + rejected
    elapsed_time = time.perf_counter() - start_time
    acceptance_ratio = accepted / total if total > 0 else 0.0
    latency_violation_ratio = (
        latency_violations / rejected if rejected > 0 else 0.0
    )
    avg_latency = total_latency / accepted if accepted > 0 else 0.0
    avg_time_per_request = (elapsed_time / total * 1000) if total > 0 else 0.0  # ms
    avg_sec_margin = (
        sum(security_margins) / len(security_margins) if security_margins else 0.0
    )
    avg_sfc_tenancy = (
        sum(sfc_tenancy_samples) / len(sfc_tenancy_samples)
        if sfc_tenancy_samples
        else 0.0
    )
    avg_vnf_tenancy = (
        sum(vnf_tenancy_samples) / len(vnf_tenancy_samples)
        if vnf_tenancy_samples
        else 0.0
    )
    avg_substrate_utilization = (
        sum(substrate_utilization_samples) / len(substrate_utilization_samples)
        if substrate_utilization_samples
        else 0.0
    )
    avg_risk_score = sum(risk_score_samples) / len(risk_score_samples) if risk_score_samples else 0.0
    avg_expected_incidents = total_expected_incidents / total if total > 0 else 0.0
    avg_realized_incidents = total_realized_incidents / total if total > 0 else 0.0
    avg_incident_cost = total_incident_cost / total if total > 0 else 0.0
    avg_risk_integral = total_risk_integral / total if total > 0 else 0.0

    return {
        "algorithm": "MaskablePPO",
        "total_requests": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_ratio": acceptance_ratio,
        "latency_violation_ratio": latency_violation_ratio,
        "avg_latency": avg_latency,
        "avg_time_ms": avg_time_per_request,
        "avg_sec_margin": avg_sec_margin,
        "avg_sfc_tenancy": avg_sfc_tenancy,
        "avg_vnf_tenancy": avg_vnf_tenancy,
        "avg_substrate_utilization": avg_substrate_utilization,
        "avg_risk_score": avg_risk_score,
        "avg_risk_integral": avg_risk_integral,
        "avg_expected_incidents": avg_expected_incidents,
        "avg_realized_incidents": avg_realized_incidents,
        "avg_incident_cost": avg_incident_cost,
        "latencies": latencies,
        # Per-request samples for rolling average plots
        "sfc_tenancy_samples": sfc_tenancy_samples,
        "vnf_tenancy_samples": vnf_tenancy_samples,
        "substrate_utilization_samples": substrate_utilization_samples,
        "risk_score_samples": risk_score_samples,
        "realized_incident_samples": realized_incident_samples,
        "incident_cost_samples": incident_cost_samples,
    }

# BOOKMARK, IMPORTANT: THIS IS THE FUNCTION THAT IS USED TO EVALUATE THE RL AGENT.
def run_eval(
    config_path: str,
    model=None,
    num_requests: int = 1000,
    verbose: bool = False,
    *,
    substrate=None,
    request_generator=None,
    model_path: Optional[str] = None,
    save_plot: Optional[str] = None,
    plot_rolling_dir: Optional[str] = None,
) -> dict:
    """
    Run evaluation for baselines and optionally the RL agent on the same substrate/request stream.

    Substrate and request_generator can be provided (e.g. from training env); otherwise they
    are created from config. RL is run if model (in-memory) is provided, or if model_path is
    provided and the file exists; otherwise only baselines are run. Optional printing and
    plotting when verbose, save_plot, or plot_rolling_dir are set.

    Args:
        config_path: Path to configuration file
        model: In-memory RL model (optional; preferred over model_path when both set)
        num_requests: Number of requests per evaluation
        verbose: If True, print "Evaluating X..." per algorithm and the comparison table
        substrate: Optional substrate to use (created from config if None)
        request_generator: Optional request generator (created from config if None)
        model_path: If model is None and this path exists, load model and run RL
        save_plot: If set, call plot_comparison(results, save_plot)
        plot_rolling_dir: If set, call plot_rolling_averages(results, plot_rolling_dir, 50)

    Returns:
        Dictionary mapping algorithm name -> result dict (acceptance_ratio,
        latency_violation_ratio, avg_sfc_tenancy, avg_vnf_tenancy, avg_substrate_utilization, etc.)
    """
    config = load_config(config_path)
    if substrate is None:
        substrate = SubstrateNetwork(config["substrate"])
    if request_generator is None:
        request_generator = RequestGenerator(config["sfc"])
    risk_cfg = _build_risk_config(config)
    max_security = config["substrate"]["resources"]["security_score"]["max"]
    max_ttl = int(config["sfc"]["ttl"]["max"])

    results = {}
    baselines = [
        ViterbiPlacement(),
        FirstFitPlacement(),
        BestFitPlacement(),
    ]

    # Snapshot RNG state so every algorithm sees the same request/incident stream.
    initial_rng_state = random.getstate()
    initial_np_rng_state = np.random.get_state()

    for algorithm in baselines:
        if verbose:
            print(f"\nEvaluating {algorithm.__class__.__name__}...")
        random.setstate(initial_rng_state)
        np.random.set_state(initial_np_rng_state)
        substrate.reset()
        request_generator.reset()
        result = evaluate_baseline(
            substrate,
            request_generator,
            algorithm,
            num_requests=num_requests,
            risk_cfg=risk_cfg,
            max_security=max_security,
            max_ttl=max_ttl,
            verbose=verbose,
        )
        results[result["algorithm"]] = result

    # RL: use in-memory model if provided, else load from model_path if it exists
    run_rl = model is not None or (model_path and Path(model_path).exists())
    if run_rl:
        if verbose:
            print("\nEvaluating RL Agent...")
        random.setstate(initial_rng_state)
        np.random.set_state(initial_np_rng_state)
        substrate.reset()
        request_generator.reset()
        if model is None:
            env = create_masked_env(
                config_path,
                substrate=substrate,
                request_generator=request_generator,
                max_requests_per_episode=num_requests,
            )
            model = load_model(model_path, env)
        result = evaluate_rl_agent(
            model_path="" if model is not None else model_path,
            config_path=config_path,
            num_requests=num_requests,
            verbose=verbose,
            model=model,
            substrate=substrate,
            request_generator=request_generator,
        )
        results[result["algorithm"]] = result
    elif model_path:
        print(f"Warning: Model not found at {model_path}")

    if verbose:
        print("\n" + "=" * 230)
        print("COMPARISON RESULTS")
        print("=" * 230)
        print(
            f"{'Algorithm':<20} {'Accepted':<10} {'Rejected':<10} {'Ratio':<10} {'Avg Latency':<12} {'Avg Sec Margin':<14} {'Avg SFC/Node':<14} {'Avg VNF/Node':<14} {'Substrate Util':<14} {'Risk Integral':<14} {'Real Inc':<10} {'Exp Inc':<10} {'Inc Cost':<10} {'Avg Time (ms)':<14}"
        )
        print("-" * 230)
        for name, result in results.items():
            print(
                f"{name:<20} {result['accepted']:<10} {result['rejected']:<10} "
                f"{result['acceptance_ratio']:.4f}     {result['avg_latency']:<12.2f} {result.get('avg_sec_margin', 0):<14.4f} "
                f"{result.get('avg_sfc_tenancy', 0):<14.2f} {result.get('avg_vnf_tenancy', 0):<14.2f} {result.get('avg_substrate_utilization', 0):<14.2%} "
                f"{result.get('avg_risk_integral', 0):<14.4f} {result.get('avg_realized_incidents', 0):<10.4f} {result.get('avg_expected_incidents', 0):<10.4f} "
                f"{result.get('avg_incident_cost', 0):<10.4f} {result.get('avg_time_ms', 0):.3f}"
            )
        print("=" * 230)

    if save_plot:
        plot_comparison(results, save_plot)
    if plot_rolling_dir:
        plot_rolling_averages(results, plot_rolling_dir, window_size=50)

    return results


def compare_all(
    config_path: str = "config.yaml",
    model_path: Optional[str] = None,
    num_requests: int = 1000,
    save_plot: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Compare all placement algorithms.

    Args:
        config_path: Path to configuration file
        model_path: Path to trained RL model (optional)
        num_episodes: Number of episodes to process per algorithm
        save_plot: Path to save comparison plot (optional)
        verbose: Whether to print results

    Returns:
        Dictionary with results for all algorithms
    """
    return run_eval(
        config_path,
        model=None,
        num_requests=num_requests,
        verbose=verbose,
        model_path=model_path,
        save_plot=save_plot,
        plot_rolling_dir="eval/",
    )


def plot_comparison(results: dict, save_path: str):
    """
    Generate comparison bar charts.

    Args:
        results: Dictionary of algorithm results
        save_path: Path to save the plot
    """
    algorithms = list(results.keys())
    acceptance_ratios = [results[a]["acceptance_ratio"] for a in algorithms]
    avg_latencies = [results[a]["avg_latency"] for a in algorithms]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Acceptance Ratio
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(algorithms)))
    bars1 = axes[0].bar(algorithms, acceptance_ratios, color=colors)
    axes[0].set_ylabel("Acceptance Ratio")
    axes[0].set_title("SFC Acceptance Ratio by Algorithm")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, ratio in zip(bars1, acceptance_ratios):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{ratio:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Average Latency
    bars2 = axes[1].bar(algorithms, avg_latencies, color=colors)
    axes[1].set_ylabel("Average Latency (ms)")
    axes[1].set_title("Average End-to-End Latency by Algorithm")
    axes[1].tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, lat in zip(bars2, avg_latencies):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{lat:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to: {save_path}")
    plt.close()


def plot_rolling_averages(results: dict, output_dir: str, window_size: int = 50):
    """
    Generate rolling window average plots for time-series metrics.

    Args:
        results: Dictionary of algorithm results with per-request samples
        output_dir: Directory to save the plots
        window_size: Size of the rolling window for smoothing
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define metrics to plot
    metrics = [
        (
            "sfc_tenancy_samples",
            "Avg SFCs per Occupied Node",
            "sfc_per_node_rolling.png",
        ),
        (
            "vnf_tenancy_samples",
            "Avg VNFs per Occupied Node",
            "vnf_per_node_rolling.png",
        ),
        (
            "substrate_utilization_samples",
            "Substrate Utilization",
            "substrate_util_rolling.png",
        ),
        (
            "risk_score_samples",
            "Risk Integral",
            "risk_score_rolling.png",
        ),
        (
            "realized_incident_samples",
            "Realized Incidents per Request",
            "realized_incidents_rolling.png",
        ),
        (
            "incident_cost_samples",
            "Incident Cost per Request",
            "incident_cost_rolling.png",
        ),
    ]

    # Color palette for algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for metric_key, metric_title, filename in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        for (alg_name, alg_results), color in zip(results.items(), colors):
            samples = alg_results.get(metric_key, [])
            if not samples:
                continue

            # Convert to numpy array
            samples = np.array(samples)

            # Calculate rolling average
            if len(samples) >= window_size:
                # Use convolution for rolling average
                kernel = np.ones(window_size) / window_size
                rolling_avg = np.convolve(samples, kernel, mode="valid")
                x_values = np.arange(window_size - 1, len(samples))
            else:
                # Not enough data for rolling average, use raw data
                rolling_avg = samples
                x_values = np.arange(len(samples))

            ax.plot(x_values, rolling_avg, label=alg_name, color=color, linewidth=2)

        ax.set_xlabel("Request Number", fontsize=12)
        ax.set_ylabel(metric_title, fontsize=12)
        ax.set_title(
            f"{metric_title} Over Time (Rolling Window = {window_size})", fontsize=14
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage for substrate utilization
        if "utilization" in metric_key:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        plt.tight_layout()
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Rolling average plot saved to: {save_path}")
        plt.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Compare SFC Placement Algorithms")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/sfc_ppo_best.zip",
        help="Path to trained RL model (default: models/sfc_ppo_best.zip, use --no-model to skip)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip RL model evaluation (only evaluate baselines)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Number of requests to evaluate (overrides config)",
    )
    parser.add_argument(
        "--plot", type=str, default=None, help="Path to save comparison plot"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Load config for default num_requests
    config = load_config(args.config)
    num_requests = args.requests or config.get("evaluation", {}).get(
        "num_requests", 1000
    )

    # Handle --no-model flag
    model_path = None if args.no_model else args.model

    compare_all(
        config_path=args.config,
        model_path=model_path,
        num_requests=num_requests,
        save_plot=args.plot,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
