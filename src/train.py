"""
Training Script for SFC Placement with Maskable PPO.

This script handles the training loop for the RL agent, including:
- Environment setup with action masking
- Model initialization
- Training with callbacks (episode-based)
- Model saving
"""

# Use non-interactive backend for thread-safe plotting (must be before other imports)
import matplotlib

matplotlib.use("Agg")

import argparse
import shutil
from pathlib import Path

from src.requests import load_config
from src.model import (
    create_masked_env,
    create_maskable_ppo,
    load_model,
    AcceptanceRatioCallback,
    BestModelCallback,
    LatencyViolationCallback,
    SubstrateMetricsCallback,
)
from stable_baselines3.common.callbacks import CallbackList


def train(
    config_path: str = "config.yaml",
    total_timesteps: int = 1_000_000,
    save_path: str = "models/sfc_ppo",
    log_path: str = "logs/",
    plot_freq: int = 1000,
    save_freq: int = 50000,
    seed: int = 42,
    load_path: str = None,
    gnn_type: str = "gcn",
    gnn_hidden_dim: int = 64,
    gnn_features_dim: int = 256,
    num_gnn_layers: int = 3,
):
    """
    Train the Maskable PPO agent for SFC placement.

    Args:
        config_path: Path to the configuration file
        total_timesteps: Total training timesteps
        save_path: Path to save the trained model
        log_path: Path for TensorBoard logs
        plot_freq: Frequency (in steps) to update the acceptance ratio plot
        save_freq: Frequency (in steps) to save the model
        seed: Random seed for reproducibility
        load_path: Path to a checkpoint to load and continue training from
        gnn_type: Type of GNN layer ("gcn", "gat", "sage")
        gnn_hidden_dim: Hidden dimension for GNN layers
        gnn_features_dim: Output dimension of GNN feature extractor
        num_gnn_layers: Number of GNN layers
    """
    # Load configuration
    config = load_config(config_path)
    training_config = config.get("training", {})

    # Create directories
    save_dir = Path(save_path).parent

    if not load_path and save_dir.exists():
        print(f"Cleaning up save directory: {save_dir}")
        # Delete all files and subdirectories in the save directory
        for item in save_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    save_dir.mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("SFC Placement Training with Maskable PPO (GNN compulsory)")
    print("=" * 50)
    print(f"Config: {config_path}")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Save path: {save_path}")
    print(f"Log path: {log_path}")
    print(f"Plot frequency: every {plot_freq} steps")
    print(f"Save frequency: every {save_freq} steps")
    print(f"GNN Type: {gnn_type.upper()}")
    print(
        f"GNN Layers: {num_gnn_layers}, Hidden: {gnn_hidden_dim}, Features: {gnn_features_dim}"
    )
    if load_path:
        print(f"Resuming training from: {load_path}")
    print("=" * 50)

    # Create environment
    print("\nCreating environment...")
    env = create_masked_env(config_path)

    # Create or load model
    reset_timesteps = True
    if load_path:
        print(f"Loading MaskablePPO model from {load_path}...")
        # Start with a dummy learning rate, it will be overwritten by the saved model's schedule
        # unless custom_objects is used, but usually we want to continue with the saved state.
        # However, we must ensure the env is attached.
        model = load_model(load_path, env=env)
        model.tensorboard_log = log_path
        reset_timesteps = False
    else:
        print("Creating MaskablePPO model...")
        n_steps = training_config.get("n_steps", 2048)
        model = create_maskable_ppo(
            env,
            learning_rate=training_config.get("learning_rate", 3e-4),
            n_steps=n_steps,
            batch_size=training_config.get("batch_size", 64),
            n_epochs=training_config.get("n_epochs", 10),
            gamma=training_config.get("gamma", 0.99),
            verbose=1,
            tensorboard_log=log_path,
            seed=seed,
            gnn_type=gnn_type,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_features_dim=gnn_features_dim,
            num_gnn_layers=num_gnn_layers,
        )

    # Setup callbacks
    plot_path = str(Path(save_path).with_suffix("")) + "_acceptance_ratio.png"

    # 1. Acceptance Ratio Tracking and Plotting
    acceptance_callback = AcceptanceRatioCallback(
        save_path=plot_path, plot_freq=plot_freq, verbose=1
    )

    # 2. Save Best Model
    best_model_path = str(Path(save_path).parent / "sfc_ppo_best.zip")
    best_model_callback = BestModelCallback(
        save_path=best_model_path, check_freq=5000, verbose=1
    )

    # 3. Latency Violation Tracking and Plotting
    rejection_plot_path = str(Path(save_path).parent / "sfc_ppo_rejection_ratio.png")
    latency_callback = LatencyViolationCallback(
        save_path=rejection_plot_path, plot_freq=plot_freq, verbose=1
    )

    # 4. Substrate Metrics Tracking and Plotting (SFC/node, VNF/node, utilization)
    substrate_metrics_callback = SubstrateMetricsCallback(
        save_dir=str(Path(save_path).parent), plot_freq=plot_freq, verbose=1
    )

    # Combine callbacks
    callback = CallbackList(
        [
            acceptance_callback,
            best_model_callback,
            latency_callback,
            substrate_metrics_callback,
        ]
    )

    # Start training
    print("\nStarting training...")
    print("-" * 50)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name="MaskablePPO",
        reset_num_timesteps=reset_timesteps,
    )

    # Save final model
    print("\nSaving final model...")
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # Print final statistics
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

    # Get final stats from environment
    base_env = env.unwrapped
    print(f"Total Episodes: {base_env.total_episodes}")
    print(f"Total Requests: {base_env.total_requests}")
    print(f"Total Accepted: {base_env.accepted_requests}")
    print(
        f"Overall Acceptance Ratio: {base_env.accepted_requests / max(1, base_env.total_requests):.4f}"
    )

    # Print rejection breakdown
    print("\nRejection Breakdown:")
    total_rejections = base_env.total_requests - base_env.accepted_requests
    for reason, count in base_env.rejection_reasons.items():
        pct = (count / max(1, total_rejections)) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    return model


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Train Maskable PPO for SFC Placement")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/sfc_ppo",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--log-path", type=str, default="logs/", help="Path for TensorBoard logs"
    )
    parser.add_argument(
        "--plot-freq",
        type=int,
        default=10000,
        help="Frequency (in steps) to update the acceptance ratio plot",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50000,
        help="Frequency (in steps) to save model checkpoints",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="DEFAULT",
        help="Resume training. Defaults to last checkpoint if no path provided.",
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gcn",
        choices=["gcn", "gat", "sage"],
        help="Type of GNN layer: gcn, gat, or sage (default: gcn)",
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for GNN layers (default: 64)",
    )
    parser.add_argument(
        "--gnn-features-dim",
        type=int,
        default=256,
        help="Output dimension of GNN feature extractor (default: 256)",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=3,
        help="Number of GNN layers (default: 3)",
    )

    args = parser.parse_args()

    # Load config to get default timesteps
    config = load_config(args.config)
    timesteps = args.timesteps or config.get("training", {}).get(
        "total_timesteps", 1000000
    )

    # Resolve load path
    load_path = args.resume
    if load_path == "DEFAULT":
        # Default to the last checkpoint (sfc_ppo_last.zip)
        from pathlib import Path

        save_dir = Path(args.save_path).parent
        load_path = str(save_dir / "sfc_ppo_last.zip")

    train(
        config_path=args.config,
        total_timesteps=timesteps,
        save_path=args.save_path,
        log_path=args.log_path,
        plot_freq=args.plot_freq,
        seed=args.seed,
        load_path=load_path,
        gnn_type=args.gnn_type,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_features_dim=args.gnn_features_dim,
        num_gnn_layers=args.gnn_layers,
    )


if __name__ == "__main__":
    main()
