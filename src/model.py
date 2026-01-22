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
    Also saves a plot of the acceptance ratio vs training steps.
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
        self.acceptance_ratios = []
        self.steps = []

    def _on_step(self) -> bool:
        """Called after each step."""
        # Get info from the environment
        infos = self.locals.get("infos", [])

        for info in infos:
            if "acceptance_ratio" in info:
                ratio = info["acceptance_ratio"]
                self.acceptance_ratios.append(ratio)
                self.steps.append(self.num_timesteps)

                # Log to TensorBoard if available
                if self.logger is not None:
                    self.logger.record("custom/acceptance_ratio", ratio)

        # Plot and save every plot_freq steps
        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plot()

        return True

    def _save_plot(self):
        """Generates and saves a plot of acceptance ratio."""
        if not self.acceptance_ratios:
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.acceptance_ratios, label="Acceptance Ratio")

            # Add moving average for smoother visualization
            if len(self.acceptance_ratios) > 100:
                import numpy as np

                window = min(100, len(self.acceptance_ratios) // 10)
                moving_avg = np.convolve(
                    self.acceptance_ratios, np.ones(window) / window, mode="valid"
                )
                plt.plot(
                    self.steps[window - 1 :],
                    moving_avg,
                    label=f"Moving Avg ({window})",
                    color="orange",
                )

            plt.title("Acceptance Ratio vs Training Steps")
            plt.xlabel("Total Timesteps")
            plt.ylabel("Acceptance Ratio")
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

        if self.acceptance_ratios:
            final_ratio = self.acceptance_ratios[-1]
            avg_ratio = sum(self.acceptance_ratios) / len(self.acceptance_ratios)

            if self.verbose > 0:
                print("\nTraining Complete:")
                print(f"  Final Acceptance Ratio: {final_ratio:.4f}")
                print(f"  Average Acceptance Ratio: {avg_ratio:.4f}")
                print(f"  Acceptance Ratio Plot saved to: {self.save_path}")


class BestModelCallback(BaseCallback):
    """
    Callback to save the best model based on acceptance ratio.
    """

    def __init__(self, save_path: str, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_ratio = 0.0

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.check_freq == 0:
            infos = self.locals.get("infos", [])

            for info in infos:
                if "acceptance_ratio" in info:
                    ratio = info["acceptance_ratio"]

                    if ratio > self.best_ratio:
                        self.best_ratio = ratio
                        self.model.save(self.save_path)

                        if self.verbose > 0:
                            print(f"New best model saved! Ratio: {ratio:.4f}")

        return True
