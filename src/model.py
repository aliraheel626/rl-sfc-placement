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
from src.gnn_policy import (
    GNNFeaturesExtractor,
    create_edge_getter,
)


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
    gnn_type: str = "gcn",  # Options: "gcn", "gat", "sage"
    gnn_hidden_dim: int = 64,
    gnn_features_dim: int = 256,
    num_gnn_layers: int = 3,
    **kwargs,
) -> MaskablePPO:
    """
    Create a MaskablePPO agent configured for SFC placement with GNN.

    Args:
        env: ActionMasker-wrapped SFC placement environment
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps to run for each update
        batch_size: Minibatch size for updates
        n_epochs: Number of epochs for each update
        gamma: Discount factor
        verbose: Verbosity level
        tensorboard_log: Path for TensorBoard logs
        gnn_type: Type of GNN layer ("gcn", "gat", "sage")
        gnn_hidden_dim: Hidden dimension for GNN layers
        gnn_features_dim: Output dimension of GNN feature extractor
        num_gnn_layers: Number of GNN layers
        **kwargs: Additional arguments passed to MaskablePPO

    Returns:
        Configured MaskablePPO model with GNN
    """

    policy_kwargs = kwargs.pop("policy_kwargs", {})

    # Create edge getter for dynamic topology support
    edge_getter = create_edge_getter(env)

    policy_kwargs["features_extractor_class"] = GNNFeaturesExtractor
    policy_kwargs["features_extractor_kwargs"] = {
        "edge_getter": edge_getter,
        "node_feat_dim": 6,  # [RAM, CPU, Storage, Security, AvgBW, DistToPrev]
        "hidden_dim": gnn_hidden_dim,
        "features_dim": gnn_features_dim,
        "gnn_type": gnn_type,
        "num_gnn_layers": num_gnn_layers,
        "dropout": 0.1,
    }

    # Smaller MLP on top of GNN features
    if "net_arch" not in policy_kwargs:
        policy_kwargs["net_arch"] = [128, 128]

    base_env = env.unwrapped
    print(
        f"Using PyG GNN Feature Extractor ({gnn_type.upper()}) "
        f"with {base_env.num_nodes} real nodes (max_nodes={base_env.max_nodes})."
    )

    # Use MultiInputPolicy for Dict observation space
    policy_name = "MultiInputPolicy"

    model = MaskablePPO(
        policy_name,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        device="auto",  # Use CUDA if available, else CPU
        **kwargs,
    )

    return model


def load_model(path: str, env: Optional[ActionMasker] = None) -> MaskablePPO:
    """
    Load a trained MaskablePPO model.

    Handles GNN models by injecting a fresh edge_getter from the new env
    (the saved edge_getter closure is not portable across environments).

    Args:
        path: Path to the saved model
        env: Optional environment to attach to the model

    Returns:
        Loaded MaskablePPO model
    """
    import io
    import pickle
    import zipfile

    custom_objects = {}

    if env is not None:
        # Peek at saved data to check if GNN was used
        try:
            model_path = path if path.endswith(".zip") else path + ".zip"
            with zipfile.ZipFile(model_path, "r") as zf:
                if "data" in zf.namelist():
                    with zf.open("data") as f:
                        data = pickle.load(io.BytesIO(f.read()))
                    policy_kwargs = data.get("policy_kwargs", {})
                    extractor_kwargs = policy_kwargs.get(
                        "features_extractor_kwargs", {}
                    )
                    if "edge_getter" in extractor_kwargs:
                        # Inject fresh edge_getter for the new env
                        edge_getter = create_edge_getter(env)
                        extractor_kwargs["edge_getter"] = edge_getter
                        policy_kwargs["features_extractor_kwargs"] = extractor_kwargs
                        custom_objects["policy_kwargs"] = policy_kwargs
        except Exception:
            pass  # Not a GNN model or can't peek -- load normally

    return MaskablePPO.load(
        path, env=env, custom_objects=custom_objects if custom_objects else None
    )


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
        # Derive the last checkpoint path from the best model path
        self.last_checkpoint_path = save_path.replace("_best.zip", "_last.zip")
        if self.last_checkpoint_path == save_path:
            # Fallback if the pattern doesn't match
            self.last_checkpoint_path = save_path.replace(".zip", "_last.zip")
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

            # Always save the latest checkpoint
            self.model.save(self.last_checkpoint_path)
            if self.verbose > 0:
                print(
                    f"Checkpoint saved to {self.last_checkpoint_path} at step {self.num_timesteps} (Avg Ratio: {avg_ratio:.4f})"
                )

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


class LatencyViolationCallback(BaseCallback):
    """
    Custom callback to track and plot latency violation ratio during training.

    Tracks the percentage of rejections caused by latency violations per episode.
    """

    def __init__(
        self,
        save_path: str = "models/sfc_ppo_rejection_ratio.png",
        plot_freq: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.plot_freq = plot_freq
        self.episode_latency_violation_ratios = []  # Per-episode latency violation ratios
        self.episodes = []  # Episode numbers

    def _on_step(self) -> bool:
        """Called after each step."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            # Only record at episode end (when terminated)
            if dones[i] if isinstance(dones, (list, tuple)) else dones:
                if "episode_latency_violation_ratio" in info:
                    ratio = info["episode_latency_violation_ratio"]
                    episode_num = info.get("total_episodes", len(self.episodes) + 1)

                    self.episode_latency_violation_ratios.append(ratio)
                    self.episodes.append(episode_num)

                    # Log to TensorBoard if available
                    if self.logger is not None:
                        self.logger.record(
                            "custom/episode_latency_violation_ratio", ratio
                        )
                        self.logger.record(
                            "custom/episode_latency_violations",
                            info.get("episode_latency_violations", 0),
                        )
                        self.logger.record(
                            "custom/episode_rejections",
                            info.get("episode_rejections", 0),
                        )

        # Plot and save every plot_freq steps
        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plot()

        return True

    def _save_plot(self):
        """Generates and saves a plot of latency violation ratio per episode."""
        if not self.episode_latency_violation_ratios:
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            plt.figure(figsize=(10, 6))
            plt.plot(
                self.episodes,
                self.episode_latency_violation_ratios,
                alpha=0.6,
                label="Latency Violation Ratio",
                linewidth=0.5,
                color="red",
            )

            # Add moving average for smoother visualization
            if len(self.episode_latency_violation_ratios) > 10:
                import numpy as np

                window = min(50, len(self.episode_latency_violation_ratios) // 5)
                if window > 1:
                    moving_avg = np.convolve(
                        self.episode_latency_violation_ratios,
                        np.ones(window) / window,
                        mode="valid",
                    )
                    plt.plot(
                        self.episodes[window - 1 :],
                        moving_avg,
                        label=f"Moving Avg ({window} episodes)",
                        color="darkred",
                        linewidth=2,
                    )

            plt.title("Latency Violation % of Rejections vs Episode")
            plt.xlabel("Episode")
            plt.ylabel("Latency Violations / Total Rejections")
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()

            plt.tight_layout()
            plt.savefig(self.save_path)
            plt.close()

            if self.verbose > 1:
                print(
                    f"Rejection ratio plot saved to {self.save_path} at step {self.num_timesteps}"
                )
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save rejection ratio plot: {e}")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self._save_plot()

        if self.episode_latency_violation_ratios:
            avg_ratio = sum(self.episode_latency_violation_ratios) / len(
                self.episode_latency_violation_ratios
            )

            last_n = min(10, len(self.episode_latency_violation_ratios))
            recent_avg = sum(self.episode_latency_violation_ratios[-last_n:]) / last_n

            if self.verbose > 0:
                print("\nLatency Violation Stats:")
                print(f"  Average Latency Violation Ratio: {avg_ratio:.4f}")
                print(f"  Last {last_n} Episodes Avg: {recent_avg:.4f}")
                print(f"  Rejection Ratio Plot saved to: {self.save_path}")


class SubstrateMetricsCallback(BaseCallback):
    """
    Custom callback to track and plot substrate metrics during training.

    Tracks per-episode averages of:
    - SFC tenancy (avg SFCs per occupied node)
    - VNF tenancy (avg VNFs per occupied node)
    - Substrate utilization (% of nodes being used)
    """

    def __init__(
        self,
        save_dir: str = "models/",
        plot_freq: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.plot_freq = plot_freq

        # Per-episode averages
        self.episode_sfc_tenancy = []
        self.episode_vnf_tenancy = []
        self.episode_substrate_util = []
        self.episodes = []

    def _on_step(self) -> bool:
        """Called after each step."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            # Only record at episode end (when terminated)
            if dones[i] if isinstance(dones, (list, tuple)) else dones:
                episode_num = info.get("total_episodes", len(self.episodes) + 1)

                # Get episode-level substrate metrics
                if "episode_avg_sfc_tenancy" in info:
                    self.episode_sfc_tenancy.append(info["episode_avg_sfc_tenancy"])
                    self.episode_vnf_tenancy.append(
                        info.get("episode_avg_vnf_tenancy", 0)
                    )
                    self.episode_substrate_util.append(
                        info.get("episode_avg_substrate_util", 0)
                    )
                    self.episodes.append(episode_num)

                    # Log to TensorBoard if available
                    if self.logger is not None:
                        self.logger.record(
                            "custom/episode_avg_sfc_tenancy",
                            info["episode_avg_sfc_tenancy"],
                        )
                        self.logger.record(
                            "custom/episode_avg_vnf_tenancy",
                            info.get("episode_avg_vnf_tenancy", 0),
                        )
                        self.logger.record(
                            "custom/episode_avg_substrate_util",
                            info.get("episode_avg_substrate_util", 0),
                        )

        # Plot and save every plot_freq steps
        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plots()

        return True

    def _save_plots(self):
        """Generates and saves plots of substrate metrics per episode."""
        if not self.episode_sfc_tenancy:
            return

        try:
            import numpy as np

            os.makedirs(self.save_dir, exist_ok=True)

            # Define metrics to plot
            metrics = [
                (
                    self.episode_sfc_tenancy,
                    "Avg SFCs per Occupied Node",
                    "sfc_per_node_training.png",
                    "blue",
                    "darkblue",
                ),
                (
                    self.episode_vnf_tenancy,
                    "Avg VNFs per Occupied Node",
                    "vnf_per_node_training.png",
                    "green",
                    "darkgreen",
                ),
                (
                    self.episode_substrate_util,
                    "Substrate Utilization",
                    "substrate_util_training.png",
                    "purple",
                    "indigo",
                ),
            ]

            for data, title, filename, color, avg_color in metrics:
                if not data:
                    continue

                plt.figure(figsize=(10, 6))
                plt.plot(
                    self.episodes,
                    data,
                    alpha=0.6,
                    label=title,
                    linewidth=0.5,
                    color=color,
                )

                # Add moving average for smoother visualization
                if len(data) > 10:
                    window = min(50, len(data) // 5)
                    if window > 1:
                        moving_avg = np.convolve(
                            data, np.ones(window) / window, mode="valid"
                        )
                        plt.plot(
                            self.episodes[window - 1 :],
                            moving_avg,
                            label=f"Moving Avg ({window} episodes)",
                            color=avg_color,
                            linewidth=2,
                        )

                plt.title(f"{title} vs Episode Number")
                plt.xlabel("Episode")
                plt.ylabel(title)

                # Format y-axis as percentage for substrate utilization
                if "Utilization" in title:
                    plt.ylim(0, 1.05)
                    plt.gca().yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
                    )

                plt.grid(True, linestyle="--", alpha=0.7)
                plt.legend()
                plt.tight_layout()

                save_path = os.path.join(self.save_dir, filename)
                plt.savefig(save_path)
                plt.close()

            if self.verbose > 1:
                print(
                    f"Substrate metrics plots saved to {self.save_dir} at step {self.num_timesteps}"
                )
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save substrate metrics plots: {e}")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self._save_plots()

        if self.episode_sfc_tenancy:
            import numpy as np

            avg_sfc = np.mean(self.episode_sfc_tenancy)
            avg_vnf = np.mean(self.episode_vnf_tenancy)
            avg_util = np.mean(self.episode_substrate_util)

            last_n = min(10, len(self.episode_sfc_tenancy))
            recent_sfc = np.mean(self.episode_sfc_tenancy[-last_n:])
            recent_vnf = np.mean(self.episode_vnf_tenancy[-last_n:])
            recent_util = np.mean(self.episode_substrate_util[-last_n:])

            if self.verbose > 0:
                print("\nSubstrate Metrics Stats:")
                print(
                    f"  Avg SFC/Node: {avg_sfc:.2f} (Last {last_n}: {recent_sfc:.2f})"
                )
                print(
                    f"  Avg VNF/Node: {avg_vnf:.2f} (Last {last_n}: {recent_vnf:.2f})"
                )
                print(
                    f"  Avg Substrate Util: {avg_util:.2%} (Last {last_n}: {recent_util:.2%})"
                )
                print(f"  Substrate metrics plots saved to: {self.save_dir}")
