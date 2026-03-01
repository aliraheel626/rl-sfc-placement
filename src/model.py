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
    requests_per_graph: Optional[int] = None,
) -> ActionMasker:
    """
    Create an environment wrapped with ActionMasker for MaskablePPO.

    Args:
        config_path: Path to the configuration file
        substrate: Optional substrate to use (for fair comparison with baselines)
        request_generator: Optional request generator (required if substrate is provided)
        requests_per_graph: Override requests per graph (e.g. for eval: one long graph)

    Returns:
        ActionMasker-wrapped environment
    """
    env = SFCPlacementEnv(
        config_path=config_path,
        substrate=substrate,
        request_generator=request_generator,
        requests_per_graph=requests_per_graph,
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
    Track per-graph cumulative reward and plot it vs graph number.
    One graph = one topology over requests_per_graph requests.
    """

    def __init__(
        self,
        save_path: str = "models/reward_per_graph.png",
        plot_freq: int = 10000,
        rolling_window: int = 50,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.plot_freq = plot_freq
        self.rolling_window = rolling_window
        self.graphs = []
        self.graph_rewards = []
        self._current_episode_rewards = {}
        self._graph_reward_sum = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        if hasattr(rewards, "__len__") and not isinstance(rewards, (str, bytes)):
            reward_list = list(rewards)
        else:
            reward_list = [float(rewards)]

        if hasattr(dones, "__len__") and not isinstance(dones, (str, bytes)):
            done_list = list(dones)
        else:
            done_list = [bool(dones)]

        if not isinstance(infos, list):
            info_list = [infos]
        else:
            info_list = infos

        n_envs = max(len(reward_list), len(done_list), len(info_list), 1)

        for i in range(n_envs):
            reward_i = float(reward_list[i]) if i < len(reward_list) else 0.0
            done_i = bool(done_list[i]) if i < len(done_list) else False
            info_i = info_list[i] if i < len(info_list) and isinstance(info_list[i], dict) else {}

            self._current_episode_rewards[i] = (
                self._current_episode_rewards.get(i, 0.0) + reward_i
            )

            if done_i:
                episode_reward = float(self._current_episode_rewards.get(i, 0.0))
                self._graph_reward_sum += episode_reward
                del self._current_episode_rewards[i]

                if info_i.get("is_graph_complete", False):
                    graph_id = int(info_i.get("total_graphs", len(self.graphs) + 1))
                    self.graphs.append(graph_id)
                    self.graph_rewards.append(self._graph_reward_sum)
                    self._graph_reward_sum = 0.0

        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plot()
        return True

    def _save_plot(self):
        if not self.graphs:
            return
        try:
            import numpy as np

            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            graphs_arr = np.array(self.graphs)
            rewards = np.array(self.graph_rewards)

            plt.figure(figsize=(10, 6))
            plt.plot(
                graphs_arr,
                rewards,
                alpha=0.35,
                label="Cumulative reward (per graph)",
                linewidth=0.8,
            )

            if len(rewards) > 10:
                window = min(max(int(self.rolling_window), 2), len(rewards))
                if window > 1:
                    k = np.ones(window) / window
                    smooth = np.convolve(rewards, k, mode="valid")
                    plt.plot(
                        graphs_arr[window - 1 :],
                        smooth,
                        color="orange",
                        linewidth=2,
                        label=f"Moving Avg ({window} graphs)",
                    )

            plt.title("Cumulative Reward vs Graph")
            plt.xlabel("Graph")
            plt.ylabel("Cumulative Reward (per graph)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.save_path)
            plt.close()
            if self.verbose > 1:
                print(f"Reward plot saved to {self.save_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save reward plot: {e}")

    def _on_training_end(self) -> None:
        next_graph = (max(self.graphs) + 1) if self.graphs else 1
        for cum_reward in self._current_episode_rewards.values():
            self._graph_reward_sum += float(cum_reward)
        self._current_episode_rewards.clear()
        if self._graph_reward_sum != 0.0:
            self.graphs.append(next_graph)
            self.graph_rewards.append(self._graph_reward_sum)
        self._save_plot()


class AcceptanceRatioCallback(BaseCallback):
    """
    Custom callback to track and log acceptance ratio during training.

    Tracks the per-graph acceptance ratio (accepted/total requests within each graph)
    and plots it against graph number.
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
        self.graph_acceptance_ratios = []
        self.graphs = []

    def _on_step(self) -> bool:
        """Called after each step."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if isinstance(dones, (list, tuple)) or (
                hasattr(dones, "__getitem__") and hasattr(dones, "__len__")
            ):
                done_i = dones[i] if i < len(dones) else False
            else:
                done_i = dones if i == 0 else False
            if done_i and info.get("is_graph_complete", False):
                if "graph_acceptance_ratio" in info:
                    ratio = info["graph_acceptance_ratio"]
                    graph_id = info.get("total_graphs", len(self.graphs) + 1)
                    self.graph_acceptance_ratios.append(ratio)
                    self.graphs.append(graph_id)
                    if self.logger is not None:
                        self.logger.record("custom/graph_acceptance_ratio", ratio)

        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plot()
        return True

    def _save_plot(self):
        """Generates and saves a plot of acceptance ratio per graph."""
        if not self.graph_acceptance_ratios:
            return
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.graphs,
                self.graph_acceptance_ratios,
                alpha=0.6,
                label="Acceptance Ratio (per graph)",
                linewidth=0.5,
            )
            if len(self.graph_acceptance_ratios) > 10:
                import numpy as np
                window = min(50, len(self.graph_acceptance_ratios) // 5)
                if window > 1:
                    moving_avg = np.convolve(
                        self.graph_acceptance_ratios,
                        np.ones(window) / window,
                        mode="valid",
                    )
                    plt.plot(
                        self.graphs[window - 1 :],
                        moving_avg,
                        label=f"Moving Avg ({window} graphs)",
                        color="orange",
                        linewidth=2,
                    )
            plt.title("Acceptance Ratio vs Graph")
            plt.xlabel("Graph")
            plt.ylabel("Acceptance Ratio (per graph)")
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.save_path)
            plt.close()
            if self.verbose > 1:
                print(f"Plot saved to {self.save_path} at step {self.num_timesteps}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save plot: {e}")

    def _on_training_end(self) -> None:
        self._save_plot()
        if self.graph_acceptance_ratios:
            final_ratio = self.graph_acceptance_ratios[-1]
            avg_ratio = sum(self.graph_acceptance_ratios) / len(self.graph_acceptance_ratios)
            last_n = min(10, len(self.graph_acceptance_ratios))
            recent_avg = sum(self.graph_acceptance_ratios[-last_n:]) / last_n
            if self.verbose > 0:
                print("\nTraining Complete:")
                print(f"  Total Graphs: {len(self.graphs)}")
                print(f"  Final Graph Acceptance Ratio: {final_ratio:.4f}")
                print(f"  Average Acceptance Ratio: {avg_ratio:.4f}")
                print(f"  Last {last_n} Graphs Avg: {recent_avg:.4f}")
                print(f"  Acceptance Ratio Plot saved to: {self.save_path}")


class BestModelCallback(BaseCallback):
    """
    Callback to save the best model based on graph acceptance ratio.
    """

    def __init__(self, save_path: str, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.last_checkpoint_path = save_path.replace("_best.zip", "_last.zip")
        if self.last_checkpoint_path == save_path:
            self.last_checkpoint_path = save_path.replace(".zip", "_last.zip")
        self.check_freq = check_freq
        self.best_ratio = 0.0
        self.recent_ratios = []  # Track recent graph acceptance ratios

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if isinstance(dones, (list, tuple)) or (
                hasattr(dones, "__getitem__") and hasattr(dones, "__len__")
            ):
                done_i = dones[i] if i < len(dones) else False
            else:
                done_i = dones if i == 0 else False
            if done_i and info.get("is_graph_complete", False):
                if "graph_acceptance_ratio" in info:
                    self.recent_ratios.append(info["graph_acceptance_ratio"])

        if self.n_calls % self.check_freq == 0 and self.recent_ratios:
            avg_ratio = sum(self.recent_ratios) / len(self.recent_ratios)
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
                        f"New best model saved! Avg Ratio: {avg_ratio:.4f} (over {len(self.recent_ratios)} graphs)"
                    )
            self.recent_ratios = []
        return True


class LatencyViolationCallback(BaseCallback):
    """
    Custom callback to track and plot latency violation ratio during training.

    Tracks the percentage of rejections caused by latency violations per graph.
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
        self.graph_latency_violation_ratios = []
        self.graphs = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if isinstance(dones, (list, tuple)) or (
                hasattr(dones, "__getitem__") and hasattr(dones, "__len__")
            ):
                done_i = dones[i] if i < len(dones) else False
            else:
                done_i = dones if i == 0 else False
            if done_i and info.get("is_graph_complete", False):
                if "graph_latency_violation_ratio" in info:
                    ratio = info["graph_latency_violation_ratio"]
                    graph_id = info.get("total_graphs", len(self.graphs) + 1)
                    self.graph_latency_violation_ratios.append(ratio)
                    self.graphs.append(graph_id)
                    if self.logger is not None:
                        self.logger.record(
                            "custom/graph_latency_violation_ratio", ratio
                        )

        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plot()
        return True

    def _save_plot(self):
        """Generates and saves a plot of latency violation ratio per graph."""
        if not self.graph_latency_violation_ratios:
            return
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.graphs,
                self.graph_latency_violation_ratios,
                alpha=0.6,
                label="Latency Violation Ratio",
                linewidth=0.5,
                color="red",
            )
            if len(self.graph_latency_violation_ratios) > 10:
                import numpy as np
                window = min(50, len(self.graph_latency_violation_ratios) // 5)
                if window > 1:
                    moving_avg = np.convolve(
                        self.graph_latency_violation_ratios,
                        np.ones(window) / window,
                        mode="valid",
                    )
                    plt.plot(
                        self.graphs[window - 1 :],
                        moving_avg,
                        label=f"Moving Avg ({window} graphs)",
                        color="darkred",
                        linewidth=2,
                    )
            plt.title("Latency Violation % of Rejections vs Graph")
            plt.xlabel("Graph")
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
        self._save_plot()
        if self.graph_latency_violation_ratios:
            avg_ratio = sum(self.graph_latency_violation_ratios) / len(
                self.graph_latency_violation_ratios
            )
            last_n = min(10, len(self.graph_latency_violation_ratios))
            recent_avg = sum(self.graph_latency_violation_ratios[-last_n:]) / last_n
            if self.verbose > 0:
                print("\nLatency Violation Stats:")
                print(f"  Average Latency Violation Ratio: {avg_ratio:.4f}")
                print(f"  Last {last_n} Graphs Avg: {recent_avg:.4f}")
                print(f"  Rejection Ratio Plot saved to: {self.save_path}")


class SubstrateMetricsCallback(BaseCallback):
    """
    Custom callback to track and plot substrate metrics during training.

    Tracks per-graph averages of:
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
        self.graph_sfc_tenancy = []
        self.graph_vnf_tenancy = []
        self.graph_substrate_util = []
        self.graphs = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if isinstance(dones, (list, tuple)) or (
                hasattr(dones, "__getitem__") and hasattr(dones, "__len__")
            ):
                done_i = dones[i] if i < len(dones) else False
            else:
                done_i = dones if i == 0 else False
            if done_i and info.get("is_graph_complete", False):
                if "graph_avg_sfc_tenancy" in info:
                    graph_id = info.get("total_graphs", len(self.graphs) + 1)
                    self.graph_sfc_tenancy.append(info["graph_avg_sfc_tenancy"])
                    self.graph_vnf_tenancy.append(
                        info.get("graph_avg_vnf_tenancy", 0)
                    )
                    self.graph_substrate_util.append(
                        info.get("graph_avg_substrate_util", 0)
                    )
                    self.graphs.append(graph_id)
                    if self.logger is not None:
                        self.logger.record(
                            "custom/graph_avg_sfc_tenancy",
                            info["graph_avg_sfc_tenancy"],
                        )
                        self.logger.record(
                            "custom/graph_avg_vnf_tenancy",
                            info.get("graph_avg_vnf_tenancy", 0),
                        )
                        self.logger.record(
                            "custom/graph_avg_substrate_util",
                            info.get("graph_avg_substrate_util", 0),
                        )

        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plots()
        return True

    def _save_plots(self):
        """Generates and saves plots of substrate metrics per graph."""
        if not self.graph_sfc_tenancy:
            return
        try:
            import numpy as np
            os.makedirs(self.save_dir, exist_ok=True)
            metrics = [
                (
                    self.graph_sfc_tenancy,
                    "Avg SFCs per Occupied Node",
                    "sfc_per_node_training.png",
                    "blue",
                    "darkblue",
                ),
                (
                    self.graph_vnf_tenancy,
                    "Avg VNFs per Occupied Node",
                    "vnf_per_node_training.png",
                    "green",
                    "darkgreen",
                ),
                (
                    self.graph_substrate_util,
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
                    self.graphs,
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
                            self.graphs[window - 1 :],
                            moving_avg,
                            label=f"Moving Avg ({window} graphs)",
                            color=avg_color,
                            linewidth=2,
                        )

                plt.title(f"{title} vs Graph")
                plt.xlabel("Graph")
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

        if self.graph_sfc_tenancy:
            import numpy as np

            avg_sfc = np.mean(self.graph_sfc_tenancy)
            avg_vnf = np.mean(self.graph_vnf_tenancy)
            avg_util = np.mean(self.graph_substrate_util)

            last_n = min(10, len(self.graph_sfc_tenancy))
            recent_sfc = np.mean(self.graph_sfc_tenancy[-last_n:])
            recent_vnf = np.mean(self.graph_vnf_tenancy[-last_n:])
            recent_util = np.mean(self.graph_substrate_util[-last_n:])

            if self.verbose > 0:
                print("\nSubstrate Metrics Stats:")
                print(
                    f"  Avg SFC/Node: {avg_sfc:.2f} (Last {last_n} graphs: {recent_sfc:.2f})"
                )
                print(
                    f"  Avg VNF/Node: {avg_vnf:.2f} (Last {last_n} graphs: {recent_vnf:.2f})"
                )
                print(
                    f"  Avg Substrate Util: {avg_util:.2%} (Last {last_n} graphs: {recent_util:.2%})"
                )
                print(f"  Substrate metrics plots saved to: {self.save_dir}")


class TrainingEvalCallback(BaseCallback):
    """
    When a graph completes, run a 1000-request evaluation for the current
    model and all baselines (like src.compare). Each plotted point is from this
    eval. Plots include lines for all baselines; x-axis is graph number.
    """

    def __init__(
        self,
        config_path: str,
        save_dir: str = "models/",
        num_requests: int = 1000,
        plot_freq: int = 10000,
        eval_freq: int = 20000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.config_path = config_path
        self.save_dir = save_dir
        self.num_requests = num_requests
        self.plot_freq = plot_freq
        self.eval_freq = eval_freq
        self._last_eval_step = 0
        self.graphs = []
        self.by_algo = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if isinstance(dones, (list, tuple)) or (
                hasattr(dones, "__getitem__") and hasattr(dones, "__len__")
            ):
                done_i = dones[i] if i < len(dones) else False
            else:
                done_i = dones if i == 0 else False
            if done_i and info.get("is_graph_complete", False):
                if (self.n_calls - self._last_eval_step) < self.eval_freq:
                    continue
                self._last_eval_step = self.n_calls

                graph_id = info.get("total_graphs", len(self.graphs) + 1)

                from src.compare import run_episode_eval

                results = run_episode_eval(
                    self.config_path,
                    self.model,
                    num_requests=self.num_requests,
                    verbose=False,
                )

                self.graphs.append(graph_id)
                for algo_name, res in results.items():
                    if algo_name not in self.by_algo:
                        self.by_algo[algo_name] = {
                            "acceptance_ratio": [],
                            "latency_violation_ratio": [],
                            "avg_sfc_tenancy": [],
                            "avg_vnf_tenancy": [],
                            "avg_substrate_utilization": [],
                            "avg_risk_integral": [],
                            "avg_risk_integral_accepted": [],
                            "avg_realized_incidents": [],
                            "avg_incident_cost": [],
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
                    self.by_algo[algo_name]["avg_risk_integral"].append(
                        res.get("avg_risk_integral", res.get("avg_risk_score", 0.0))
                    )
                    self.by_algo[algo_name]["avg_risk_integral_accepted"].append(
                        res.get("avg_risk_integral_accepted", res.get("avg_risk_integral", 0.0))
                    )
                    self.by_algo[algo_name]["avg_realized_incidents"].append(
                        res.get("avg_realized_incidents", 0.0)
                    )
                    self.by_algo[algo_name]["avg_incident_cost"].append(
                        res.get("avg_incident_cost", 0.0)
                    )

                if self.logger is not None:
                    ppo = results.get("MaskablePPO", {})
                    if ppo:
                        self.logger.record(
                            "custom/eval_acceptance_ratio", ppo["acceptance_ratio"]
                        )
                        self.logger.record(
                            "custom/eval_risk_integral",
                            ppo.get("avg_risk_integral", ppo.get("avg_risk_score", 0.0)),
                        )
                        self.logger.record(
                            "custom/eval_realized_incidents",
                            ppo.get("avg_realized_incidents", 0.0),
                        )
                        self.logger.record(
                            "custom/eval_incident_cost",
                            ppo.get("avg_incident_cost", 0.0),
                        )

        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._save_plots()
        return True

    def _save_plots(self):
        if not self.graphs or not self.by_algo:
            return
        import numpy as np

        os.makedirs(self.save_dir, exist_ok=True)
        algos = list(self.by_algo.keys())
        default_colors = ["#C0392B", "#2980B9", "#E67E22", "#27AE60"]
        color_map = {
            a: default_colors[i % len(default_colors)]
            for i, a in enumerate(algos)
        }
        ls_map = {a: "-" if a == "MaskablePPO" else "--" for a in algos}

        def moving_avg(y, window):
            if len(y) < window or window < 2:
                return None, None
            k = np.ones(window) / window
            ma = np.convolve(y, k, mode="valid")
            x = self.graphs[window - 1 :]
            return x, ma

        # 1) Acceptance ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["acceptance_ratio"]
            ax.plot(
                self.graphs,
                vals,
                alpha=0.7,
                label=algo,
                linewidth=1.0,
                color=color_map[algo],
                linestyle=ls_map[algo],
                marker="o",
                markersize=3,
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
                            linestyle="-",
                            label=f"MaskablePPO Moving Avg ({window} graphs)",
                        )
        ax.set_title("Acceptance Ratio vs Graph (eval 1000 req/graph)")
        ax.set_xlabel("Graph")
        ax.set_ylabel("Acceptance Ratio (per graph)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "sfc_ppo_acceptance_ratio.png"))
        plt.close()

        # 2) Risk integral (per request; rejections = 0)
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["avg_risk_integral"]
            ax.plot(
                self.graphs,
                vals,
                alpha=0.7,
                label=algo,
                linewidth=1.0,
                color=color_map[algo],
                linestyle=ls_map[algo],
                marker="o",
                markersize=3,
            )
            if algo == "MaskablePPO" and len(vals) > 10:
                window = min(50, len(vals) // 5)
                if window > 1:
                    x_ma, y_ma = moving_avg(vals, window)
                    if x_ma is not None:
                        ax.plot(
                            x_ma,
                            y_ma,
                            color="purple",
                            linewidth=2,
                            linestyle="-",
                            label=f"MaskablePPO Moving Avg ({window} graphs)",
                        )
        ax.set_title("Avg Risk Integral vs Graph (eval 1000 req/graph)")
        fig.suptitle("Per request (rejections count as 0 risk)", fontsize=9, style="italic", y=1.02)
        ax.set_xlabel("Graph")
        ax.set_ylabel("Risk Integral (lower is better)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "sfc_risk_training.png"))
        plt.close()

        # 2b) Risk integral on accepted requests only (comparable across acceptance levels)
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo].get("avg_risk_integral_accepted", [])
            if not vals:
                continue
            ax.plot(
                self.graphs[: len(vals)],
                vals,
                alpha=0.7,
                label=algo,
                linewidth=1.0,
                color=color_map[algo],
                linestyle=ls_map[algo],
                marker="o",
                markersize=3,
            )
            if algo == "MaskablePPO" and len(vals) > 10:
                window = min(50, len(vals) // 5)
                if window > 1:
                    x_ma, y_ma = moving_avg(vals, window)
                    if x_ma is not None:
                        ax.plot(
                            x_ma,
                            y_ma,
                            color="purple",
                            linewidth=2,
                            linestyle="-",
                            label=f"MaskablePPO Moving Avg ({window} graphs)",
                        )
        ax.set_title("Avg Risk Integral (accepted requests only) vs Graph")
        ax.set_xlabel("Graph")
        ax.set_ylabel("Risk Integral (lower is better)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "sfc_risk_accepted_training.png"))
        plt.close()

        # 3) Latency violation ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["latency_violation_ratio"]
            ax.plot(
                self.graphs,
                vals,
                alpha=0.7,
                label=algo,
                linewidth=1.0,
                color=color_map[algo],
                linestyle=ls_map[algo],
                marker="o",
                markersize=3,
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
                            linestyle="-",
                            label=f"MaskablePPO Moving Avg ({window} graphs)",
                        )
        ax.set_title(
            "Latency Violation % of Rejections vs Graph (eval 1000 req/graph)"
        )
        ax.set_xlabel("Graph")
        ax.set_ylabel("Latency Violations / Total Rejections")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "sfc_ppo_rejection_ratio.png"))
        plt.close()

        # 4) Substrate utilization
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["avg_substrate_utilization"]
            ax.plot(
                self.graphs,
                vals,
                alpha=0.7,
                label=algo,
                linewidth=1.0,
                color=color_map[algo],
                linestyle=ls_map[algo],
                marker="o",
                markersize=3,
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
                            linestyle="-",
                            label=f"MaskablePPO Moving Avg ({window} graphs)",
                        )
        ax.set_title(
            "Substrate Utilization vs Graph (eval 1000 req/graph)"
        )
        ax.set_xlabel("Graph")
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

        # 5) SFC per node
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["avg_sfc_tenancy"]
            ax.plot(
                self.graphs,
                vals,
                alpha=0.7,
                label=algo,
                linewidth=1.0,
                color=color_map[algo],
                linestyle=ls_map[algo],
                marker="o",
                markersize=3,
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
                            linestyle="-",
                            label=f"MaskablePPO Moving Avg ({window} graphs)",
                        )
        ax.set_title(
            "Avg SFCs per Occupied Node vs Graph (eval 1000 req/graph)"
        )
        ax.set_xlabel("Graph")
        ax.set_ylabel("Avg SFCs per Occupied Node")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "sfc_per_node_training.png"))
        plt.close()

        # 6) VNF per node
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            vals = self.by_algo[algo]["avg_vnf_tenancy"]
            ax.plot(
                self.graphs,
                vals,
                alpha=0.7,
                label=algo,
                linewidth=1.0,
                color=color_map[algo],
                linestyle=ls_map[algo],
                marker="o",
                markersize=3,
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
                            linestyle="-",
                            label=f"MaskablePPO Moving Avg ({window} graphs)",
                        )
        ax.set_title(
            "Avg VNFs per Occupied Node vs Graph (eval 1000 req/graph)"
        )
        ax.set_xlabel("Graph")
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
