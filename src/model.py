"""
Maskable PPO Model Creation and Loading for SFC Placement.

This module provides functions to create and configure the Maskable PPO
agent from sb3-contrib for use with the SFC placement environment.
"""

from typing import Optional, Any

import os
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from src.environment import SFCPlacementEnv


def mask_fn(env: SFCPlacementEnv) -> Any:
    """
    Mask function for ActionMasker wrapper.

    Returns the action mask from the environment.
    """
    return env.action_masks()


def create_masked_env(config_path: str = "config.yaml") -> ActionMasker:
    """
    Create an environment wrapped with ActionMasker for MaskablePPO.

    Args:
        config_path: Path to the configuration file

    Returns:
        ActionMasker-wrapped environment
    """
    env = SFCPlacementEnv(config_path=config_path)
    return ActionMasker(env, mask_fn)


def create_maskable_ppo(
    env: ActionMasker,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    verbose: int = 1,
    tensorboard_log: Optional[str] = None,
    **kwargs,
) -> MaskablePPO:
    """
    Create a MaskablePPO agent configured for SFC placement.

    Args:
        env: ActionMasker-wrapped SFC placement environment
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps to run for each update
        batch_size: Minibatch size for updates
        n_epochs: Number of epochs for each update
        gamma: Discount factor
        verbose: Verbosity level
        tensorboard_log: Path for TensorBoard logs
        **kwargs: Additional arguments passed to MaskablePPO

    Returns:
        Configured MaskablePPO model
    """
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )

    return model


def load_model(path: str, env: Optional[ActionMasker] = None) -> MaskablePPO:
    """
    Load a trained MaskablePPO model.

    Args:
        path: Path to the saved model
        env: Optional environment to attach to the model

    Returns:
        Loaded MaskablePPO model
    """
    return MaskablePPO.load(path, env=env)


class AcceptanceRatioCallback(BaseCallback):
    """
    Custom callback to track and log acceptance ratio during training.

    Tracks the per-episode acceptance ratio (accepted/total requests within each episode)
    and plots it against episode number.
    """

    def __init__(
        self,
        save_path: str = "plots/acceptance_ratio.png",
        plot_freq: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.plot_freq = plot_freq
        self.episode_acceptance_ratios = []  # Per-episode acceptance ratios
        self.episodes = []  # Episode numbers

    def _on_step(self) -> bool:
        """Called after each step."""
        # Get info from the environment - check for episode completion
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            # Only record at episode end (when terminated)
            if dones[i] if isinstance(dones, (list, tuple)) else dones:
                if "episode_acceptance_ratio" in info:
                    ratio = info["episode_acceptance_ratio"]
                    episode_num = info.get("total_episodes", len(self.episodes) + 1)

                    self.episode_acceptance_ratios.append(ratio)
                    self.episodes.append(episode_num)

                    # Log to TensorBoard if available
                    if self.logger is not None:
                        self.logger.record("custom/episode_acceptance_ratio", ratio)
                        self.logger.record(
                            "custom/episode_accepted", info.get("episode_accepted", 0)
                        )
                        self.logger.record(
                            "custom/episode_requests", info.get("episode_requests", 0)
                        )

        # Plot and save every plot_freq steps
        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plot()

        return True

    def _save_plot(self):
        """Generates and saves a plot of acceptance ratio per episode."""
        if not self.episode_acceptance_ratios:
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            plt.figure(figsize=(10, 6))
            plt.plot(
                self.episodes,
                self.episode_acceptance_ratios,
                alpha=0.6,
                label="Episode Acceptance Ratio",
                linewidth=0.5,
            )

            # Add moving average for smoother visualization
            if len(self.episode_acceptance_ratios) > 10:
                import numpy as np

                window = min(50, len(self.episode_acceptance_ratios) // 5)
                if window > 1:
                    moving_avg = np.convolve(
                        self.episode_acceptance_ratios,
                        np.ones(window) / window,
                        mode="valid",
                    )
                    plt.plot(
                        self.episodes[window - 1 :],
                        moving_avg,
                        label=f"Moving Avg ({window} episodes)",
                        color="orange",
                        linewidth=2,
                    )

            plt.title("Episode Acceptance Ratio vs Episode Number")
            plt.xlabel("Episode")
            plt.ylabel("Acceptance Ratio (per episode)")
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()

            # Ensure the plot is saved
            plt.tight_layout()
            plt.savefig(self.save_path)
            plt.close()

            if self.verbose > 1:
                print(f"Plot saved to {self.save_path} at step {self.num_timesteps}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save plot: {e}")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self._save_plot()

        if self.episode_acceptance_ratios:
            final_ratio = self.episode_acceptance_ratios[-1]
            avg_ratio = sum(self.episode_acceptance_ratios) / len(
                self.episode_acceptance_ratios
            )

            # Calculate stats over last 10 episodes
            last_n = min(10, len(self.episode_acceptance_ratios))
            recent_avg = sum(self.episode_acceptance_ratios[-last_n:]) / last_n

            if self.verbose > 0:
                print("\nTraining Complete:")
                print(f"  Total Episodes: {len(self.episodes)}")
                print(f"  Final Episode Acceptance Ratio: {final_ratio:.4f}")
                print(f"  Average Acceptance Ratio: {avg_ratio:.4f}")
                print(f"  Last {last_n} Episodes Avg: {recent_avg:.4f}")
                print(f"  Acceptance Ratio Plot saved to: {self.save_path}")


class BestModelCallback(BaseCallback):
    """
    Callback to save the best model based on episode acceptance ratio.
    """

    def __init__(self, save_path: str, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_ratio = 0.0
        self.recent_ratios = []  # Track recent episode ratios

    def _on_step(self) -> bool:
        """Called after each step."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            # Only check at episode end
            if dones[i] if isinstance(dones, (list, tuple)) else dones:
                if "episode_acceptance_ratio" in info:
                    self.recent_ratios.append(info["episode_acceptance_ratio"])

        # Check for best model every check_freq steps
        if self.n_calls % self.check_freq == 0 and self.recent_ratios:
            # Use average of recent episodes as the metric
            avg_ratio = sum(self.recent_ratios) / len(self.recent_ratios)

            if avg_ratio > self.best_ratio:
                self.best_ratio = avg_ratio
                self.model.save(self.save_path)

                if self.verbose > 0:
                    print(
                        f"New best model saved! Avg Ratio: {avg_ratio:.4f} (over {len(self.recent_ratios)} episodes)"
                    )

            # Clear recent ratios for next check period
            self.recent_ratios = []

        return True
