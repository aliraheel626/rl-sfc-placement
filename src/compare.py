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
from src.baselines import BasePlacement, BestFitPlacement
from src.risk import (
    apply_incident_view,
    compute_placement_risk_score,
    decay_incident_pressure,
    restore_incident_view,
)
from src.eval_reporting import (
    COMPARISON_TABLE_ROWS,
    append_eval_result_to_by_algo,
    plot_eval_curves,
)


def _build_risk_config(config: dict) -> dict:
    """Build risk configuration with defaults."""
    risk = config.get("risk", {})
    return {
        "enabled": bool(risk.get("enabled", False)),
        "tenancy_ref_floor": float(risk.get("tenancy_ref_floor", 1.0)),
        "load_ref_floor": float(risk.get("load_ref_floor", 1.0)),
        "incident_base_rate": float(risk.get("incident_base_rate", 0.025)),
        "incident_alpha": float(risk.get("incident_alpha", 1.6)),
        "incident_beta": float(risk.get("incident_beta", 0.7)),
        "incident_steps_cap": int(risk.get("incident_steps_cap", 12)),
        "incident_pressure_decay": float(risk.get("incident_pressure_decay", 0.92)),
        "incident_security_penalty": float(risk.get("incident_security_penalty", 0.60)),
        "revenue_per_ttl_step": float(risk.get("revenue_per_ttl_step", 1.0)),
        "risk_lambda": float(risk.get("lambda", 0.0)),
    }


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
    acceptance_samples = []  # 1 if accepted, 0 if rejected (per request)
    revenue_samples = []  # Per-request revenue (0 if rejected)
    security_margins = []  # Track security margins for each accepted request
    sfc_tenancy_samples = []  # Track SFC tenancy per occupied node over time
    vnf_tenancy_samples = []  # Track VNF tenancy per occupied node over time
    substrate_utilization_samples = []  # Track % of nodes being used over time
    risk_score_samples = []  # Per-request security cost heuristic
    total_risk_integral = 0.0
    total_revenue = 0.0
    node_incident_pressure = {node_id: 0.0 for node_id in range(substrate_copy.num_nodes)}
    node_sfc_placements: dict[int, int] = {}   # cumulative SFC chains per node
    node_vnf_placements: dict[int, int] = {}   # cumulative VNF instances per node

    start_time = time.perf_counter()

    iterator = range(num_requests)
    if verbose:
        iterator = tqdm(iterator, desc=f"Evaluating {algorithm.__class__.__name__}")

    for _ in iterator:
        # Advance time first (release expired placements from previous steps)
        substrate_copy.tick()
        decay_incident_pressure(
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
        ) = apply_incident_view(
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
            max_ttl=max_ttl,
        )
        restore_incident_view(
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
            acceptance_samples.append(1)

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
            # Cumulative per-node placement counts (for distribution analysis)
            for node_id in set(placement):
                node_sfc_placements[node_id] = node_sfc_placements.get(node_id, 0) + 1
            for node_id in placement:
                node_vnf_placements[node_id] = node_vnf_placements.get(node_id, 0) + 1
            risk_integral = compute_placement_risk_score(
                substrate_copy,
                placement,
                request.ttl,
                risk_cfg,
                max_security,
                max_ttl,
                node_incident_pressure,
            )
            risk_score_samples.append(risk_integral)
            total_risk_integral += risk_integral
            req_revenue = risk_cfg.get("revenue_per_ttl_step", 1.0) * request.ttl
            total_revenue += req_revenue
            revenue_samples.append(req_revenue)

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
            acceptance_samples.append(0)
            revenue_samples.append(0.0)
            risk_score_samples.append(0.0)

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
    avg_risk_integral = total_risk_integral / total if total > 0 else 0.0
    avg_revenue_per_request = total_revenue / total if total > 0 else 0.0

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
        "avg_revenue_per_request": avg_revenue_per_request,
        "total_revenue": total_revenue,
        "latencies": latencies,
        # Per-request samples for rolling average plots
        "acceptance_samples": acceptance_samples,
        "revenue_samples": revenue_samples,
        "sfc_tenancy_samples": sfc_tenancy_samples,
        "vnf_tenancy_samples": vnf_tenancy_samples,
        "substrate_utilization_samples": substrate_utilization_samples,
        "risk_score_samples": risk_score_samples,
        "security_margin_samples": security_margins,
        "node_sfc_placements": node_sfc_placements,
        "node_vnf_placements": node_vnf_placements,
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
    elif model_path:
        # load_model → MaskablePPO.load → _setup_model → set_random_seed(saved_seed)
        # which overrides the global RNG.  Snapshot and restore around the load so
        # the caller's carefully-chosen RNG state (for fair comparison) is preserved.
        _rng_before = random.getstate()
        _np_rng_before = np.random.get_state()
        model = load_model(model_path, env)
        random.setstate(_rng_before)
        np.random.set_state(_np_rng_before)
    else:
        raise ValueError("evaluate_rl_agent requires model or model_path")

    accepted = 0
    rejected = 0
    latency_violations = 0  # Rejections due to latency constraint
    total_latency = 0.0
    latencies = []
    acceptance_samples = []  # 1 if accepted, 0 if rejected (per request)
    revenue_samples = []  # Per-request revenue (0 if rejected)
    security_margins = []  # Track security margins for each accepted request
    sfc_tenancy_samples = []  # Track SFC tenancy per occupied node over time
    vnf_tenancy_samples = []  # Track VNF tenancy per occupied node over time
    substrate_utilization_samples = []  # Track % of nodes being used over time
    risk_score_samples = []  # Per-request security cost heuristic
    total_risk_integral = 0.0
    total_revenue = 0.0
    node_sfc_placements: dict[int, int] = {}   # cumulative SFC chains per node
    node_vnf_placements: dict[int, int] = {}   # cumulative VNF instances per node

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
                acceptance_samples.append(1)
                # Get latency from placement
                placement = info.get("placement", [])
                if placement:
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
                # Cumulative per-node placement counts (for distribution analysis)
                for node_id in set(placement):
                    node_sfc_placements[node_id] = node_sfc_placements.get(node_id, 0) + 1
                for node_id in placement:
                    node_vnf_placements[node_id] = node_vnf_placements.get(node_id, 0) + 1
                # Use risk from env info (computed before tick()); recomputing after step()
                # would use post-tick substrate state and disagree with training rewards.
                risk_score = info.get("request_risk_integral", info.get("request_risk_score", 0.0))
                risk_score_samples.append(risk_score)
                total_risk_integral += risk_score
                req_revenue = info.get("request_revenue", 0.0)
                total_revenue += req_revenue
                revenue_samples.append(req_revenue)
            else:
                rejected += 1
                acceptance_samples.append(0)
                revenue_samples.append(0.0)
                if info.get("rejection_reason") == "latency_violation":
                    latency_violations += 1
                risk_score_samples.append(info.get("request_risk_integral", info.get("request_risk_score", 0.0)))

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
    avg_risk_integral = total_risk_integral / total if total > 0 else 0.0
    avg_revenue_per_request = total_revenue / total if total > 0 else 0.0

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
        "avg_revenue_per_request": avg_revenue_per_request,
        "total_revenue": total_revenue,
        "latencies": latencies,
        # Per-request samples for rolling average plots
        "acceptance_samples": acceptance_samples,
        "revenue_samples": revenue_samples,
        "sfc_tenancy_samples": sfc_tenancy_samples,
        "vnf_tenancy_samples": vnf_tenancy_samples,
        "substrate_utilization_samples": substrate_utilization_samples,
        "risk_score_samples": risk_score_samples,
        "security_margin_samples": security_margins,
        "node_sfc_placements": node_sfc_placements,
        "node_vnf_placements": node_vnf_placements,
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
    baselines = [BestFitPlacement()]

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

    # RL: use in-memory model if provided, else load from model_path if it exists.
    # Do not create an env here when model is None: evaluate_rl_agent must create the only
    # env and call reset() once, so the same RNG state produces the same request stream as baselines.
    run_rl = model is not None or (model_path and Path(model_path).exists())
    if run_rl:
        if verbose:
            print("\nEvaluating RL Agent...")
        random.setstate(initial_rng_state)
        np.random.set_state(initial_np_rng_state)
        substrate.reset()
        request_generator.reset()
        result = evaluate_rl_agent(
            model_path=model_path or "",
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
        print("\n" + "=" * 160)
        print("COMPARISON RESULTS")
        print("=" * 160)
        print(
            f"{'Algorithm':<20} {'Accepted':<10} {'Rejected':<10} {'Ratio':<10} {'Avg Latency':<12} {'Avg Sec Margin':<14} {'Avg SFC/Node':<14} {'Avg VNF/Node':<14} {'Substrate Util':<14} {'Sec Cost':<12} {'Avg Time (ms)':<14}"
        )
        print("-" * 160)
        for name, result in results.items():
            print(
                f"{name:<20} {result['accepted']:<10} {result['rejected']:<10} "
                f"{result['acceptance_ratio']:.4f}     {result['avg_latency']:<12.2f} {result.get('avg_sec_margin', 0):<14.4f} "
                f"{result.get('avg_sfc_tenancy', 0):<14.2f} {result.get('avg_vnf_tenancy', 0):<14.2f} {result.get('avg_substrate_utilization', 0):<14.2%} "
                f"{result.get('avg_risk_integral', 0):<12.4f} {result.get('avg_time_ms', 0):.3f}"
            )
        print("=" * 160)

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


def plot_rolling_comparison(
    results: dict,
    output_dir: str = "compare/",
    window_size: int = 50,
) -> None:
    """
    Generate rolling-average line charts from per-request samples for a single
    evaluation episode.  Only the 8 canonical comparison metrics are plotted.

    Produces (all rolling-average line charts):
        sfc_ppo_acceptance_ratio.png  – rolling acceptance ratio
        sfc_risk_training.png         – rolling security cost heuristic
        security_score_training.png   – rolling security margin
        revenue_training.png          – rolling revenue per request

    Args:
        results:     dict mapping algorithm name -> result dict (from run_eval)
        output_dir:  directory to write plots into (created if missing)
        window_size: rolling-window width for the time-series plots
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    algorithms = list(results.keys())
    colors_line = plt.cm.tab10(np.linspace(0, 1, max(len(algorithms), 10)))[: len(algorithms)]
    color_map = dict(zip(algorithms, colors_line))

    # ── helper: rolling average ────────────────────────────────────────────────
    def _rolling(samples: np.ndarray):
        if len(samples) >= window_size:
            kernel = np.ones(window_size) / window_size
            y = np.convolve(samples, kernel, mode="valid")
            x = np.arange(window_size - 1, len(samples))
        else:
            y = samples
            x = np.arange(len(samples))
        return x, y

    # ── helper: rolling time-series chart ─────────────────────────────────────
    def _rolling_chart(
        metric_key: str,
        ylabel: str,
        title: str,
        filename: str,
        as_percent: bool = False,
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algorithms:
            samples = np.array(results[algo].get(metric_key, []))
            if len(samples) == 0:
                continue
            x, y = _rolling(samples)
            ax.plot(x, y, label=algo, color=color_map[algo], linewidth=2)
        ax.set_xlabel("Request Number", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        if as_percent:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        save_path = out / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()

    # 1) Acceptance ratio (rolling)
    _rolling_chart(
        "acceptance_samples",
        "Acceptance Ratio",
        f"Rolling Acceptance Ratio (window={window_size})",
        "sfc_ppo_acceptance_ratio.png",
    )

    # 2) Risk integral (rolling)
    _rolling_chart(
        "risk_score_samples",
        "Risk Integral (lower is better)",
        f"Rolling Avg Risk Integral (window={window_size})",
        "sfc_risk_training.png",
    )

    # 3) Avg security margin (rolling)
    _rolling_chart(
        "security_margin_samples",
        "Avg Security Margin (node score - req min)",
        f"Rolling Avg Security Margin (window={window_size})",
        "security_score_training.png",
    )

    # 4) Revenue per request (rolling)
    _rolling_chart(
        "revenue_samples",
        "Revenue per Request",
        f"Rolling Avg Revenue per Request (window={window_size})",
        "revenue_training.png",
    )


def plot_summary_bar_chart(
    multi_ep_data: dict,
    output_dir: str = "compare/",
) -> None:
    """
    Generate a grouped bar chart with error bars (mean ± std across episodes)
    for the 8 canonical comparison metrics.

    Args:
        multi_ep_data: Output from run_multi_episode_eval()
        output_dir:    Directory to write the chart into
    """
    from src.eval_reporting import COMPARISON_TABLE_ROWS

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_algo = multi_ep_data["by_algo"]
    algos = list(by_algo.keys())

    if len(algos) < 2:
        print("Warning: Need at least 2 algorithms for summary bar chart")
        return

    # Friendly display names
    display_names = []
    for a in algos:
        name = a.replace("Placement", "").replace("Maskable", "")
        display_names.append(name.strip())

    metrics = list(COMPARISON_TABLE_ROWS)
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    bar_colors = plt.cm.tab10(np.linspace(0, 0.3, len(algos)))
    bar_width = 0.35

    for i, row in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(algos))

        means = []
        stds = []
        for algo in algos:
            vals = by_algo[algo].get(row.key, [])
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)

        bars = ax.bar(
            x, means, bar_width,
            yerr=stds, capsize=5,
            color=bar_colors, edgecolor="black", linewidth=0.5,
        )

        ax.set_title(row.label, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            fmt = f"{mean_val:.4f}" if abs(mean_val) < 10 else f"{mean_val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt,
                ha="center", va="bottom", fontsize=7,
            )

    # Hide unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "PPO vs Best-Fit: Mean ± Std Across Evaluation Episodes",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save_path = out / "summary_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Summary bar chart saved to: {save_path}")
    plt.close()


def plot_tenancy_distributions(results: dict, output_dir: str = "compare/") -> None:
    """
    Generate per-node tenancy distribution figures comparing algorithms.

    Uses cumulative per-node SFC/VNF placement counts collected during evaluation
    to show how each algorithm distributes load across substrate nodes.

    Produces three files in output_dir:
        tenancy_grouped_bar.png  – grouped bar chart: count per node per algorithm
        tenancy_histogram.png    – histogram of per-node counts
        tenancy_cdf.png          – empirical CDF of per-node counts
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    algorithms = list(results.keys())
    palette = plt.cm.tab10(np.linspace(0, 0.4, max(len(algorithms), 2)))[: len(algorithms)]
    color_map = dict(zip(algorithms, palette))

    def _to_array(result: dict, key: str) -> np.ndarray:
        d = result.get(key, {})
        if not d:
            return np.zeros(25, dtype=int)
        n = max(d.keys()) + 1
        arr = np.zeros(n, dtype=int)
        for k, v in d.items():
            arr[k] = v
        return arr

    sfc_counts = {a: _to_array(results[a], "node_sfc_placements") for a in algorithms}
    vnf_counts = {a: _to_array(results[a], "node_vnf_placements") for a in algorithms}
    n_nodes = max(len(v) for v in sfc_counts.values())
    node_idx = np.arange(n_nodes)

    # ── 1. Grouped bar chart ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    bw = 0.8 / max(len(algorithms), 1)
    offsets = np.linspace(
        -(bw * (len(algorithms) - 1)) / 2,
        (bw * (len(algorithms) - 1)) / 2,
        len(algorithms),
    )
    for ax, counts_dict, ylabel, title in [
        (axes[0], sfc_counts, "SFC Chains Placed", "Cumulative SFC Placements per Node"),
        (axes[1], vnf_counts, "VNF Instances Placed", "Cumulative VNF Placements per Node"),
    ]:
        for algo, offset in zip(algorithms, offsets):
            arr = counts_dict[algo]
            ax.bar(
                node_idx[: len(arr)] + offset, arr, bw,
                label=algo, color=color_map[algo], alpha=0.82,
                edgecolor="black", linewidth=0.3,
            )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.set_xticks(node_idx[::2])
    axes[1].set_xlabel("Substrate Node Index", fontsize=11)
    fig.suptitle(
        "Per-Node Tenancy Distribution: PPO vs Best-Fit",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = out / "tenancy_grouped_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Tenancy grouped bar chart saved to: {path}")
    plt.close()

    # ── 2. Histogram ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, counts_dict, xlabel, title in [
        (axes[0], sfc_counts, "SFC Count per Node", "SFC Count Distribution per Node"),
        (axes[1], vnf_counts, "VNF Instance Count per Node", "VNF Count Distribution per Node"),
    ]:
        all_vals = np.concatenate(list(counts_dict.values()))
        max_val = int(all_vals.max()) if len(all_vals) > 0 else 1
        bins = np.arange(0, max_val + 2) - 0.5
        for algo in algorithms:
            ax.hist(
                counts_dict[algo], bins=bins, label=algo,
                color=color_map[algo], alpha=0.62, edgecolor="black", linewidth=0.5,
            )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Number of Nodes", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.suptitle(
        "Per-Node Tenancy Histogram: PPO vs Best-Fit",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = out / "tenancy_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Tenancy histogram saved to: {path}")
    plt.close()

    # ── 3. Empirical CDF ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, counts_dict, xlabel, title in [
        (axes[0], sfc_counts, "SFC Count per Node", "Empirical CDF: SFC Count per Node"),
        (axes[1], vnf_counts, "VNF Instance Count per Node", "Empirical CDF: VNF Count per Node"),
    ]:
        for algo in algorithms:
            arr = np.sort(counts_dict[algo].astype(float))
            cdf = np.arange(1, len(arr) + 1) / len(arr)
            ax.step(arr, cdf, where="post", label=algo, color=color_map[algo], linewidth=2)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Cumulative Fraction of Nodes", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylim(0, 1.05)
    fig.suptitle(
        "Per-Node Tenancy CDF: PPO vs Best-Fit",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = out / "tenancy_cdf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Tenancy CDF saved to: {path}")
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
            "Security Cost Heuristic",
            "risk_score_rolling.png",
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


def run_multi_episode_eval(
    config_path: str,
    num_episodes: int,
    num_requests: int,
    model_path: Optional[str] = None,
    substrate=None,
    request_generator: Optional[object] = None,
    verbose: bool = True,
) -> dict:
    """
    Run *num_episodes* independent evaluation rounds, each processing *num_requests*
    requests, and collect per-episode scalar metrics for every algorithm.

    When *substrate* / *request_generator* are provided they are deep-copied before
    each episode so every round starts from the same initial state, mirroring what
    the training callback does.

    Returns:
        {
            "episodes": [1, 2, ..., num_episodes],
            "by_algo": {
                algo_name: {
                    "acceptance_ratio": [...],
                    "latency_violation_ratio": [...],
                    "avg_risk_integral": [...],
                    "avg_sfc_tenancy": [...],
                    "avg_vnf_tenancy": [...],
                    "avg_substrate_utilization": [...],
                }
            }
        }
    """
    import copy

    # Pre-load model once so we don't call load_model (which re-seeds global RNG
    # via set_random_seed) every episode.  A throwaway env is used only for the
    # GNN edge-getter; the real per-episode env is created inside evaluate_rl_agent.
    preloaded_model = None
    if model_path and Path(model_path).exists():
        _tmp_env = create_masked_env(
            config_path,
            substrate=copy.deepcopy(substrate) if substrate is not None else None,
            request_generator=copy.deepcopy(request_generator) if request_generator is not None else None,
            max_requests_per_episode=num_requests,
        )
        preloaded_model = load_model(model_path, _tmp_env)
        del _tmp_env

    by_algo: dict = {}
    episodes = list(range(1, num_episodes + 1))

    for ep in episodes:
        if verbose:
            print(f"\n-- Episode {ep}/{num_episodes} --------------------------")

        # Seed per-episode so every episode uses an equally "mixed" RNG state and
        # episode 1 is not special (avoids the fresh-startup-seed anomaly).
        random.seed(ep * 7919)
        np.random.seed(ep * 7919)

        ep_substrate = copy.deepcopy(substrate) if substrate is not None else None
        ep_rg = copy.deepcopy(request_generator) if request_generator is not None else None

        results = run_eval(
            config_path=config_path,
            model=preloaded_model,
            num_requests=num_requests,
            verbose=verbose,
            substrate=ep_substrate,
            request_generator=ep_rg,
        )

        for algo, res in results.items():
            append_eval_result_to_by_algo(by_algo, algo, res)

    return {"episodes": episodes, "by_algo": by_algo}


def plot_comparison_episodes(
    multi_ep_data: dict,
    output_dir: str = "compare/",
    num_requests: int = 1000,
) -> None:
    """
    Generate the same eval curve plots as the training callback using per-episode
    comparison data from *run_multi_episode_eval*.

    Plot list and filenames are defined in *src.eval_reporting* (single source of truth).
    """
    plot_eval_curves(
        multi_ep_data["episodes"],
        multi_ep_data["by_algo"],
        output_dir,
        num_requests,
        verbose_print=True,
        linewidth=0.8,
    )


def _write_results_table_tex(
    by_algo: dict,
    bf_key: str,
    ppo_key: str,
    output_dir: str,
    n_paired: int,
) -> None:
    """
    Write COMPARISON_TABLE_ROWS as a LaTeX table body to <output_dir>/results_table.tex.

    The file contains only the data rows (no tabular environment, no header, no rules)
    so it can be \input{} inside any table that uses the same five-column layout:
        Metric  &  BF mean  &  PPO mean  &  Delta%  &  Winner  \\
    """
    from src.eval_reporting import COMPARISON_TABLE_ROWS

    _FMT: dict[str, str] = {"total_revenue": ".2f"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results_table.tex"

    lines: list[str] = []
    for row in COMPARISON_TABLE_ROWS:
        bf_vals  = by_algo[bf_key].get(row.key,  [])[:n_paired]
        ppo_vals = by_algo[ppo_key].get(row.key, [])[:n_paired]
        if not bf_vals or not ppo_vals:
            continue
        bf_mean  = float(np.mean(bf_vals))
        ppo_mean = float(np.mean(ppo_vals))

        fmt    = _FMT.get(row.key, ".4f")
        bf_str  = format(bf_mean,  fmt)
        ppo_str = format(ppo_mean, fmt)

        if bf_mean != 0.0:
            pct  = (ppo_mean - bf_mean) / abs(bf_mean) * 100.0
            sign = "+" if pct > 0 else ""
            delta_str = f"${sign}{pct:.2f}$\\%"
            if row.higher_is_better:
                winner = "PPO" if pct > 0 else ("Best-Fit" if pct < 0 else "tie")
            else:
                winner = "PPO" if pct < 0 else ("Best-Fit" if pct > 0 else "tie")
        else:
            delta_str = "---"
            winner    = "---"

        lines.append(
            f"{row.label} & {bf_str} & {ppo_str} & {delta_str} & {winner} \\\\"
        )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"LaTeX results table written to: {path}")


def main():
    """Main entry point with argument parsing."""
    import pickle

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
        help="Number of requests per episode (overrides config evaluation.num_requests)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (overrides config evaluation.num_episodes)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/",
        help="Directory containing sfc_ppo_best.zip and optional pickle files (default: models/)",
    )
    parser.add_argument(
        "--plot", type=str, default=None, help="Path to save single-run comparison bar chart"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()
    verbose = not args.quiet

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})

    num_requests = args.requests or eval_cfg.get("num_requests", 1000)
    num_episodes = args.episodes or eval_cfg.get("num_episodes", 1)

    model_path = None if args.no_model else args.model

    models_dir = Path(args.models_dir)

    # Load saved substrate / request_generator if the config flags are set and the
    # pickles exist (they are written by train.py when the flags are enabled).
    substrate = None
    request_generator = None

    if eval_cfg.get("use_training_substrate", False):
        pkl = models_dir / "substrate.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                substrate = pickle.load(f)
            print(f"Loaded training substrate from: {pkl}")
        else:
            print(f"Warning: use_training_substrate=true but {pkl} not found - using fresh substrate")

    if eval_cfg.get("use_training_request_generator", False):
        pkl = models_dir / "request_generator.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                request_generator = pickle.load(f)
            print(f"Loaded training request generator from: {pkl}")
        else:
            print(f"Warning: use_training_request_generator=true but {pkl} not found - using fresh generator")

    out_dir = "compare/"

    print(f"\nRunning {num_episodes} episode(s) × {num_requests} requests each")
    multi_ep = run_multi_episode_eval(
        config_path=args.config,
        num_episodes=num_episodes,
        num_requests=num_requests,
        model_path=model_path,
        substrate=substrate,
        request_generator=request_generator,
        verbose=verbose,
    )

    # 1) Summary bar chart (mean ± std across episodes)
    plot_summary_bar_chart(multi_ep, output_dir=out_dir)

    # 2) Rolling-average line charts from a single representative episode.
    #    Re-use the first episode's RNG seed for reproducibility.
    print("\nGenerating rolling-average plots from a single evaluation run...")
    random.seed(1 * 7919)
    np.random.seed(1 * 7919)
    ep_substrate = copy.deepcopy(substrate) if substrate is not None else None
    ep_rg = copy.deepcopy(request_generator) if request_generator is not None else None
    rolling_results = run_eval(
        config_path=args.config,
        model=None,
        num_requests=num_requests,
        verbose=verbose,
        substrate=ep_substrate,
        request_generator=ep_rg,
        model_path=model_path,
    )
    plot_rolling_comparison(rolling_results, output_dir=out_dir, window_size=50)

    # 3) Per-node tenancy distribution figures (grouped bar, histogram, CDF)
    plot_tenancy_distributions(rolling_results, output_dir=out_dir)

    if verbose:
        print("\nPer-episode raw values (to verify episode-to-episode variation):")
        for algo, metrics in multi_ep["by_algo"].items():
            util = metrics.get("avg_substrate_utilization", [])
            ar   = metrics.get("acceptance_ratio", [])
            risk = metrics.get("avg_risk_integral", [])
            print(f"  {algo}:")
            print(f"    substrate_util : {[f'{v:.4f}' for v in util]}")
            print(f"    acceptance_ratio: {[f'{v:.4f}' for v in ar]}")
            print(f"    avg_risk_integral: {[f'{v:.4f}' for v in risk]}")

    # Print average-across-episodes performance difference table (PPO vs BestFit).
    by_algo = multi_ep["by_algo"]
    bf_key = next((k for k in by_algo if "BestFit" in k), None)
    ppo_key = next((k for k in by_algo if "PPO" in k or "Maskable" in k), None)

    if bf_key and ppo_key:
        col_w = [32, 14, 14, 12, 10]
        sep = "-" * sum(col_w)
        header = (
            f"{'Metric':<{col_w[0]}}"
            f"{'BestFit mean':>{col_w[1]}}"
            f"{'PPO mean':>{col_w[2]}}"
            f"{'D%':>{col_w[3]}}"
            f"{'Winner':>{col_w[4]}}"
        )
        print(f"\n{'PPO vs BestFit - average across episodes':^{sum(col_w)}}")
        print(sep)
        print(header)
        print(sep)
        bf_ep_count  = len(next(iter(by_algo[bf_key].values()),  []))
        ppo_ep_count = len(next(iter(by_algo[ppo_key].values()), []))
        n_paired = min(bf_ep_count, ppo_ep_count)
        if bf_ep_count != ppo_ep_count:
            print(
                f"  Warning: BestFit ran {bf_ep_count} episode(s) but PPO ran "
                f"{ppo_ep_count} - comparison uses the first {n_paired} episode(s) only."
            )

        for row in COMPARISON_TABLE_ROWS:
            bf_vals  = by_algo[bf_key].get(row.key,  [])[:n_paired]
            ppo_vals = by_algo[ppo_key].get(row.key, [])[:n_paired]
            if not bf_vals or not ppo_vals:
                continue
            bf_mean  = float(np.mean(bf_vals))
            ppo_mean = float(np.mean(ppo_vals))
            if bf_mean != 0:
                pct = (ppo_mean - bf_mean) / abs(bf_mean) * 100
            else:
                pct = float("nan")
            if np.isnan(pct):
                winner = "-"
            elif row.higher_is_better:
                winner = "PPO" if pct > 0 else ("BestFit" if pct < 0 else "tie")
            else:
                winner = "PPO" if pct < 0 else ("BestFit" if pct > 0 else "tie")
            print(
                f"{row.label:<{col_w[0]}}"
                f"{bf_mean:>{col_w[1]}.4f}"
                f"{ppo_mean:>{col_w[2]}.4f}"
                f"{pct:>{col_w[3]}.2f}%"
                f"{winner:>{col_w[4]}}"
            )
        print(sep)
        _write_results_table_tex(by_algo, bf_key, ppo_key, out_dir, n_paired)
    elif not ppo_key:
        print("\n(No PPO results - skipping comparison table; run without --no-model)")


if __name__ == "__main__":
    main()
