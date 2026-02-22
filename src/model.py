"""
Maskable PPO Model Creation and Loading for SFC Placement.

This module provides functions to create and configure the Maskable PPO
agent from sb3-contrib for use with the SFC placement environment.
"""

from typing import Optional

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
from src.requests import RequestGenerator
from src.substrate import SubstrateNetwork


def create_masked_env(
    config_path: str = "config.yaml",
    *,
    substrate: Optional[SubstrateNetwork] = None,
    request_generator: Optional[RequestGenerator] = None,
    max_requests_per_episode: Optional[int] = None,
) -> ActionMasker:
    """
    Create an environment wrapped with ActionMasker for MaskablePPO.

    Args:
        config_path: Path to the configuration file
        substrate: Optional substrate to use (for fair comparison with baselines)
        request_generator: Optional request generator (required if substrate is provided)
        max_requests_per_episode: Override episode length (e.g. for eval: one long episode)

    Returns:
        ActionMasker-wrapped environment
    """
    env = SFCPlacementEnv(
        config_path=config_path,
        substrate=substrate,
        request_generator=request_generator,
        max_requests_per_episode=max_requests_per_episode,
    )
    return ActionMasker(env, lambda e: e.action_masks())


def create_maskable_ppo(
    env: ActionMasker,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    verbose: int = 1,
    tensorboard_log: Optional[str] = None,
    gnn_type: str = "sage",  # Options: "gcn", "gat", "sage"
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
    SB3 stores "data" as JSON (with cloudpickle for non-JSON values), so we
    parse JSON and optionally deserialize policy_kwargs to replace edge_getter.

    Args:
        path: Path to the saved model
        env: Optional environment to attach to the model

    Returns:
        Loaded MaskablePPO model
    """
    import base64
    import json
    import zipfile

    try:
        import cloudpickle
    except ImportError:
        cloudpickle = None

    custom_objects = {}

    if env is not None:
        # Peek at saved data (JSON format) to check if GNN was used and inject edge_getter
        try:
            model_path = path if path.endswith(".zip") else path + ".zip"
            with zipfile.ZipFile(model_path, "r") as zf:
                if "data" not in zf.namelist():
                    raise ValueError("No 'data' in zip")
                json_str = zf.read("data").decode()
            data = json.loads(json_str)
            policy_kwargs_item = data.get("policy_kwargs")
            if not isinstance(policy_kwargs_item, dict):
                raise ValueError("policy_kwargs not a dict")
            if ":serialized:" in policy_kwargs_item:
                if cloudpickle is None:
                    raise ImportError("cloudpickle required to load GNN model")
                raw = base64.b64decode(policy_kwargs_item[":serialized:"].encode())
                policy_kwargs = cloudpickle.loads(raw)
            else:
                policy_kwargs = policy_kwargs_item
            extractor_kwargs = policy_kwargs.get("features_extractor_kwargs", {})
            if "edge_getter" in extractor_kwargs:
                edge_getter = create_edge_getter(env)
                extractor_kwargs = dict(extractor_kwargs)
                extractor_kwargs["edge_getter"] = edge_getter
                policy_kwargs = dict(policy_kwargs)
                policy_kwargs["features_extractor_kwargs"] = extractor_kwargs
                custom_objects["policy_kwargs"] = policy_kwargs
        except Exception:
            pass  # Not a GNN model or can't peek -- load normally

    return MaskablePPO.load(
        path, env=env, custom_objects=custom_objects if custom_objects else None
    )


class RewardPerStepCallback(BaseCallback):
    """
    Track reward per environment step and plot reward (aggregated per unit steps) vs step.
    """

    def __init__(
        self,
        save_path: str = "models/reward_per_step.png",
        plot_freq: int = 10000,
        step_interval: int = 200,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.plot_freq = plot_freq
        self.step_interval = step_interval
        self.steps = []
        self.reward_means = []
        self._reward_buffer = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", 0)
        if hasattr(rewards, "__len__") and not isinstance(rewards, (str, bytes)):
            try:
                for r in rewards:
                    self._reward_buffer.append(float(r))
            except TypeError:
                self._reward_buffer.append(float(rewards))
        else:
            self._reward_buffer.append(float(rewards))

        while len(self._reward_buffer) >= self.step_interval:
            chunk = self._reward_buffer[: self.step_interval]
            self._reward_buffer = self._reward_buffer[self.step_interval :]
            # Step at end of this bucket (current timestep)
            step = self.num_timesteps
            self.steps.append(step)
            self.reward_means.append(sum(chunk) / len(chunk))

        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plot()
        return True

    def _save_plot(self):
        if not self.steps:
            return
        try:
            import numpy as np

            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            steps = np.array(self.steps)
            rewards = np.array(self.reward_means)

            plt.figure(figsize=(10, 6))
            plt.plot(steps, rewards, alpha=0.6, label="Reward (mean per step)", linewidth=0.8)

            if len(rewards) > 20:
                window = min(50, len(rewards) // 5)
                if window > 1:
                    k = np.ones(window) / window
                    smooth = np.convolve(rewards, k, mode="valid")
                    plt.plot(
                        steps[window - 1 :],
                        smooth,
                        color="orange",
                        linewidth=2,
                        label=f"Moving Avg ({window} bins)",
                    )

            plt.title("Reward per Unit Step vs Training Step")
            plt.xlabel("Training Step")
            plt.ylabel("Reward (mean per {} steps)".format(self.step_interval))
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.save_path)
            plt.close()
            if self.verbose > 1:
                print(f"Reward-per-step plot saved to {self.save_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save reward-per-step plot: {e}")

    def _on_training_end(self) -> None:
        if self._reward_buffer:
            step = self.steps[-1] + self.step_interval if self.steps else self.step_interval
            self.steps.append(step)
            self.reward_means.append(sum(self._reward_buffer) / len(self._reward_buffer))
        self._save_plot()


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


class TrainingEvalCallback(BaseCallback):
    """
    At the end of each episode, run a 1000-request evaluation for the current
    model and all baselines (like src.compare). Each plotted point is from this
    eval. Plots include lines for all baselines.
    """

    def __init__(
        self,
        config_path: str,
        save_dir: str = "models/",
        num_requests: int = 1000,
        plot_freq: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.config_path = config_path
        self.save_dir = save_dir
        self.num_requests = num_requests
        self.plot_freq = plot_freq
        self.episodes = []
        # algorithm_name -> list of values per episode
        self.by_algo = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if dones[i] if isinstance(dones, (list, tuple)) else dones:
                if "episode_acceptance_ratio" not in info:
                    continue
                episode_num = info.get("total_episodes", len(self.episodes) + 1)

                from src.compare import run_episode_eval

                results = run_episode_eval(
                    self.config_path,
                    self.model,
                    num_requests=self.num_requests,
                    verbose=False,
                )

                self.episodes.append(episode_num)
                for algo_name, res in results.items():
                    if algo_name not in self.by_algo:
                        self.by_algo[algo_name] = {
                            "acceptance_ratio": [],
                            "latency_violation_ratio": [],
                            "avg_sfc_tenancy": [],
                            "avg_vnf_tenancy": [],
                            "avg_substrate_utilization": [],
                        }
                    self.by_algo[algo_name]["acceptance_ratio"].append(
                        res["acceptance_ratio"]
                    )
                    self.by_algo[algo_name]["latency_violation_ratio"].append(
                        res.get("latency_violation_ratio", 0.0)
                    )
                    self.by_algo[algo_name]["avg_sfc_tenancy"].append(
                        res.get("avg_sfc_tenancy", 0.0)
                    )
                    self.by_algo[algo_name]["avg_vnf_tenancy"].append(
                        res.get("avg_vnf_tenancy", 0.0)
                    )
                    self.by_algo[algo_name]["avg_substrate_utilization"].append(
                        res.get("avg_substrate_utilization", 0.0)
                    )

                if self.logger is not None:
                    ppo = results.get("MaskablePPO", {})
                    if ppo:
                        self.logger.record(
                            "custom/eval_acceptance_ratio", ppo["acceptance_ratio"]
                        )

        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plots()
        return True

    def _save_plots(self):
        if not self.episodes or not self.by_algo:
            return
        import numpy as np

        os.makedirs(self.save_dir, exist_ok=True)
        algos = list(self.by_algo.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(algos), 10)))[: len(algos)]
        color_map = dict(zip(algos, colors))

        def moving_avg(y, window):
            if len(y) < window or window < 2:
                return None, None
            k = np.ones(window) / window
            ma = np.convolve(y, k, mode="valid")
            x = self.episodes[window - 1 :]
            return x, ma

        # 1) Acceptance ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["acceptance_ratio"]
            ax.plot(
                self.episodes,
                vals,
                alpha=0.6,
                label=algo,
                linewidth=0.5,
                color=color_map[algo],
            )
            if algo == "MaskablePPO" and len(vals) > 10:
                window = min(50, len(vals) // 5)
                if window > 1:
                    x_ma, y_ma = moving_avg(vals, window)
                    if x_ma is not None:
                        ax.plot(
                            x_ma,
                            y_ma,
                            color="orange",
                            linewidth=2,
                            label=f"Moving Avg ({window} ep)",
                        )
        ax.set_title("Episode Acceptance Ratio vs Episode Number (eval 1000 req/episode)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Acceptance Ratio (per episode)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "sfc_ppo_acceptance_ratio.png"))
        plt.close()

        # 2) Latency violation ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["latency_violation_ratio"]
            ax.plot(
                self.episodes,
                vals,
                alpha=0.6,
                label=algo,
                linewidth=0.5,
                color=color_map[algo],
            )
            if algo == "MaskablePPO" and len(vals) > 10:
                window = min(50, len(vals) // 5)
                if window > 1:
                    x_ma, y_ma = moving_avg(vals, window)
                    if x_ma is not None:
                        ax.plot(
                            x_ma,
                            y_ma,
                            color="darkred",
                            linewidth=2,
                            label=f"Moving Avg ({window} ep)",
                        )
        ax.set_title(
            "Latency Violation % of Rejections vs Episode (eval 1000 req/episode)"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Latency Violations / Total Rejections")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "sfc_ppo_rejection_ratio.png"))
        plt.close()

        # 3) Substrate utilization
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["avg_substrate_utilization"]
            ax.plot(
                self.episodes,
                vals,
                alpha=0.6,
                label=algo,
                linewidth=0.5,
                color=color_map[algo],
            )
            if algo == "MaskablePPO" and len(vals) > 10:
                window = min(50, len(vals) // 5)
                if window > 1:
                    x_ma, y_ma = moving_avg(vals, window)
                    if x_ma is not None:
                        ax.plot(
                            x_ma,
                            y_ma,
                            color="indigo",
                            linewidth=2,
                            label=f"Moving Avg ({window} ep)",
                        )
        ax.set_title(
            "Substrate Utilization vs Episode Number (eval 1000 req/episode)"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Substrate Utilization")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "substrate_util_training.png"))
        plt.close()

        # 4) SFC per node
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["avg_sfc_tenancy"]
            ax.plot(
                self.episodes,
                vals,
                alpha=0.6,
                label=algo,
                linewidth=0.5,
                color=color_map[algo],
            )
            if algo == "MaskablePPO" and len(vals) > 10:
                window = min(50, len(vals) // 5)
                if window > 1:
                    x_ma, y_ma = moving_avg(vals, window)
                    if x_ma is not None:
                        ax.plot(
                            x_ma,
                            y_ma,
                            color="darkblue",
                            linewidth=2,
                            label=f"Moving Avg ({window} ep)",
                        )
        ax.set_title(
            "Avg SFCs per Occupied Node vs Episode Number (eval 1000 req/episode)"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg SFCs per Occupied Node")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "sfc_per_node_training.png"))
        plt.close()

        # 5) VNF per node
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["avg_vnf_tenancy"]
            ax.plot(
                self.episodes,
                vals,
                alpha=0.6,
                label=algo,
                linewidth=0.5,
                color=color_map[algo],
            )
            if algo == "MaskablePPO" and len(vals) > 10:
                window = min(50, len(vals) // 5)
                if window > 1:
                    x_ma, y_ma = moving_avg(vals, window)
                    if x_ma is not None:
                        ax.plot(
                            x_ma,
                            y_ma,
                            color="darkgreen",
                            linewidth=2,
                            label=f"Moving Avg ({window} ep)",
                        )
        ax.set_title(
            "Avg VNFs per Occupied Node vs Episode Number (eval 1000 req/episode)"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg VNFs per Occupied Node")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "vnf_per_node_training.png"))
        plt.close()

        if self.verbose > 1:
            print(
                f"Eval plots saved to {self.save_dir} at step {self.num_timesteps}"
            )

    def _on_training_end(self) -> None:
        self._save_plots()
