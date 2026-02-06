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


def evaluate_baseline(
    substrate: SubstrateNetwork,
    request_generator: RequestGenerator,
    algorithm: BasePlacement,
    num_requests: int,
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

    start_time = time.perf_counter()

    iterator = range(num_requests)
    if verbose:
        iterator = tqdm(iterator, desc=f"Evaluating {algorithm.__class__.__name__}")

    for _ in iterator:
        # Advance time first (release expired placements from previous steps)
        substrate_copy.tick()

        # Generate a new request
        request = request_generator.generate_request()

        # Try to place
        placement = algorithm.place(substrate_copy, request)

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

    return {
        "algorithm": algorithm.__class__.__name__,
        "total_requests": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_ratio": acceptance_ratio,
        "avg_latency": avg_latency,
        "avg_time_ms": avg_time_per_request,
        "avg_sec_margin": avg_sec_margin,
        "avg_sfc_tenancy": avg_sfc_tenancy,
        "avg_vnf_tenancy": avg_vnf_tenancy,
        "avg_substrate_utilization": avg_substrate_utilization,
        "latencies": latencies,
        # Per-request samples for rolling average plots
        "sfc_tenancy_samples": sfc_tenancy_samples,
        "vnf_tenancy_samples": vnf_tenancy_samples,
        "substrate_utilization_samples": substrate_utilization_samples,
    }


def evaluate_rl_agent(
    model_path: str, config_path: str, num_requests: int, verbose: bool = False
) -> dict:
    """
    Evaluate a trained RL agent.

    Args:
        model_path: Path to the trained model
        config_path: Path to configuration file
        num_episodes: Number of episodes to process
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics
    """
    # Create masked environment
    env = create_masked_env(config_path)

    # Load model
    model = load_model(model_path, env)

    accepted = 0
    rejected = 0
    total_latency = 0.0
    latencies = []
    security_margins = []  # Track security margins for each accepted request
    sfc_tenancy_samples = []  # Track SFC tenancy per occupied node over time
    vnf_tenancy_samples = []  # Track VNF tenancy per occupied node over time
    substrate_utilization_samples = []  # Track % of nodes being used over time

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
                    min_security = (
                        env.unwrapped.current_request.min_security_score
                        if hasattr(env.unwrapped, "current_request")
                        and env.unwrapped.current_request
                        else 0
                    )
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
            else:
                rejected += 1

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

    return {
        "algorithm": "MaskablePPO",
        "total_requests": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_ratio": acceptance_ratio,
        "avg_latency": avg_latency,
        "avg_time_ms": avg_time_per_request,
        "avg_sec_margin": avg_sec_margin,
        "avg_sfc_tenancy": avg_sfc_tenancy,
        "avg_vnf_tenancy": avg_vnf_tenancy,
        "avg_substrate_utilization": avg_substrate_utilization,
        "latencies": latencies,
        # Per-request samples for rolling average plots
        "sfc_tenancy_samples": sfc_tenancy_samples,
        "vnf_tenancy_samples": vnf_tenancy_samples,
        "substrate_utilization_samples": substrate_utilization_samples,
    }


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
    config = load_config(config_path)

    # Create substrate network and request generator
    substrate = SubstrateNetwork(config["substrate"])
    request_generator = RequestGenerator(config["sfc"])

    results = {}

    # Evaluate baseline algorithms
    baselines = [
        ViterbiPlacement(),
        FirstFitPlacement(),
        BestFitPlacement(),
    ]

    for algorithm in baselines:
        if verbose:
            print(f"\nEvaluating {algorithm.__class__.__name__}...")

        result = evaluate_baseline(
            substrate,
            request_generator,
            algorithm,
            num_requests=num_requests,
            verbose=verbose,
        )
        results[result["algorithm"]] = result

    # Evaluate RL agent if model path provided
    if model_path and Path(model_path).exists():
        if verbose:
            print("\nEvaluating RL Agent...")

        result = evaluate_rl_agent(
            model_path, config_path, num_requests=num_requests, verbose=verbose
        )
        results[result["algorithm"]] = result
    elif model_path:
        print(f"Warning: Model not found at {model_path}")

    # Print comparison table
    if verbose:
        print("\n" + "=" * 160)
        print("COMPARISON RESULTS")
        print("=" * 160)
        print(
            f"{'Algorithm':<20} {'Accepted':<10} {'Rejected':<10} {'Ratio':<10} {'Avg Latency':<12} {'Avg Sec Margin':<14} {'Avg SFC/Node':<14} {'Avg VNF/Node':<14} {'Substrate Util':<14} {'Avg Time (ms)':<14}"
        )
        print("-" * 160)

        for name, result in results.items():
            print(
                f"{name:<20} {result['accepted']:<10} {result['rejected']:<10} "
                f"{result['acceptance_ratio']:.4f}     {result['avg_latency']:<12.2f} {result.get('avg_sec_margin', 0):<14.4f} "
                f"{result.get('avg_sfc_tenancy', 0):<14.2f} {result.get('avg_vnf_tenancy', 0):<14.2f} {result.get('avg_substrate_utilization', 0):<14.2%} {result.get('avg_time_ms', 0):.3f}"
            )

        print("=" * 160)

    # Generate bar chart plot if requested
    if save_plot:
        plot_comparison(results, save_plot)

    # Always generate rolling average plots to eval/
    plot_rolling_averages(results, "eval/", window_size=50)

    return results


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
