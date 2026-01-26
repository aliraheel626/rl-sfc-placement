"""
Training Script for SFC Placement with Maskable PPO.

This script handles the training loop for the RL agent, including:
- Environment setup with action masking
- Model initialization
- Training with callbacks (episode-based)
- Model saving
"""

import argparse
from pathlib import Path

from src.requests import load_config
from src.model import (
    create_masked_env,
    create_maskable_ppo,
    AcceptanceRatioCallback,
    BestModelCallback,
)
from stable_baselines3.common.callbacks import CallbackList


def train(
    config_path: str = "config.yaml",
    total_timesteps: int = 1_000_000,
    save_path: str = "models/sfc_ppo",
    log_path: str = "logs/",
    plot_freq: int = 10000,
    save_freq: int = 50000,
    seed: int = 42,
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
    """
    # Load configuration
    config = load_config(config_path)
    training_config = config.get("training", {})

    # Create directories
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("SFC Placement Training with Maskable PPO")
    print("=" * 50)
    print(f"Config: {config_path}")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Save path: {save_path}")
    print(f"Log path: {log_path}")
    print(f"Plot frequency: every {plot_freq} steps")
    print(f"Save frequency: every {save_freq} steps")
    print("=" * 50)

    # Create environment
    print("\nCreating environment...")
    env = create_masked_env(config_path)

    # Create model
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
        save_path=best_model_path, check_freq=1000, verbose=1
    )

    # Combine callbacks
    callback = CallbackList([acceptance_callback, best_model_callback])

    # Start training
    print("\nStarting training...")
    print("-" * 50)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name="MaskablePPO",
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

    args = parser.parse_args()

    # Load config to get default timesteps
    config = load_config(args.config)
    timesteps = args.timesteps or config.get("training", {}).get(
        "total_timesteps", 1000000
    )

    train(
        config_path=args.config,
        total_timesteps=timesteps,
        save_path=args.save_path,
        log_path=args.log_path,
        plot_freq=args.plot_freq,
        save_freq=args.save_freq,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
