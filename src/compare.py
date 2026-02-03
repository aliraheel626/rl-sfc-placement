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

from src.requests import SubstrateNetwork, RequestGenerator, load_config
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
        else:
            rejected += 1

    # Compute metrics
    total = accepted + rejected
    elapsed_time = time.perf_counter() - start_time
    acceptance_ratio = accepted / total if total > 0 else 0.0
    avg_latency = total_latency / accepted if accepted > 0 else 0.0
    avg_time_per_request = (elapsed_time / total * 1000) if total > 0 else 0.0  # ms

    return {
        "algorithm": algorithm.__class__.__name__,
        "total_requests": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_ratio": acceptance_ratio,
        "avg_latency": avg_latency,
        "avg_time_ms": avg_time_per_request,
        "latencies": latencies,
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
            else:
                rejected += 1

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

    return {
        "algorithm": "MaskablePPO",
        "total_requests": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_ratio": acceptance_ratio,
        "avg_latency": avg_latency,
        "avg_time_ms": avg_time_per_request,
        "latencies": latencies,
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
        print("\n" + "=" * 90)
        print("COMPARISON RESULTS")
        print("=" * 90)
        print(
            f"{'Algorithm':<20} {'Accepted':<12} {'Rejected':<12} {'Ratio':<12} {'Avg Latency':<12} {'Avg Time (ms)':<12}"
        )
        print("-" * 90)

        for name, result in results.items():
            print(
                f"{name:<20} {result['accepted']:<12} {result['rejected']:<12} "
                f"{result['acceptance_ratio']:.4f}       {result['avg_latency']:<12.2f} {result.get('avg_time_ms', 0):.3f}"
            )

        print("=" * 90)

    # Generate plot if requested
    if save_plot:
        plot_comparison(results, save_plot)

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
