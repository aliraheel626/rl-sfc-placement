"""
Custom Gymnasium Environment for SFC Placement.

This environment implements the Service Function Chain placement problem
where an agent must place VNFs onto substrate nodes while respecting
resource, security, bandwidth, and latency constraints.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.requests import (
    SubstrateNetwork,
    RequestGenerator,
    SFCRequest,
    load_config,
)


class SFCPlacementEnv(gym.Env):
    """
    Gymnasium environment for SFC placement using Maskable PPO.

    The agent places VNFs one at a time onto substrate nodes. After each
    placement, resources are allocated. When all VNFs are placed, the
    latency constraint is validated.

    Observation Space:
        - Substrate node resources: (num_nodes, 4) - normalized [RAM, CPU, Storage, Security]
        - Current VNF requirements: (3,) - normalized [RAM, CPU, Storage]
        - SFC constraints: (3,) - [min_security, max_latency, min_bandwidth] (normalized)
        - Current VNF index: (1,) - progress indicator
        - Placement mask: (num_nodes,) - which nodes have been used

    Action Space:
        - Discrete(num_nodes): Select a substrate node for the current VNF
        - Masked to prevent invalid placements

    Reward:
        - +acceptance_reward for successfully placing entire SFC
        - +rejection_penalty (negative) for failed placement
        - +0.5 per step for successful VNF placement (dense reward)
        - -0.1 * (normalized_latency) per step to encourage locality
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, config_path: str = "config.yaml", render_mode: Optional[str] = None
    ):
        super().__init__()

        # Load configuration
        self.config = load_config(config_path)
        self.render_mode = render_mode

        # Initialize substrate network and request generator
        self.substrate = SubstrateNetwork(self.config["substrate"])
        self.request_generator = RequestGenerator(self.config["sfc"])

        # Topology configuration
        self.randomize_topology = self.config.get("training", {}).get(
            "randomize_topology", False
        )
        if self.randomize_topology:
            print("INFO: Topology randomization enabled (new graph per episode).")

        # Reward configuration
        self.acceptance_reward = self.config["rewards"]["acceptance"]
        self.rejection_penalty = self.config["rewards"]["rejection"]

        # Episode configuration - how many requests per episode
        training_config = self.config.get("training", {})
        self.max_requests_per_episode = training_config.get(
            "max_requests_per_episode", 100
        )

        # Environment state
        self.current_request: Optional[SFCRequest] = None
        self.current_vnf_index: int = 0
        self.current_placement: list[int] = []
        self.episode_step: int = 0

        # Episode-level statistics (reset each episode)
        self.episode_requests = 0
        self.episode_accepted = 0

        # Global statistics tracking (across all episodes)
        self.total_requests = 0
        self.accepted_requests = 0
        self.total_episodes = 0

        # Define action and observation spaces
        self.num_nodes = self.substrate.num_nodes
        self.action_space = spaces.Discrete(self.num_nodes)

        # Observation space components
        # 1. Node resources: (num_nodes, 6) -> [RAM, CPU, Storage, Security, AvgBW, DistToPrev]
        # 2. Current VNF: (3,)
        # 3. SFC constraints: (3,)
        # 4. VNF index: (1,)
        # 5. Placement mask: (num_nodes,)
        obs_dim = (
            self.num_nodes * 6  # Node resources + Connectivity + Distance
            + 3  # Current VNF requirements
            + 3  # SFC constraints
            + 1  # Current VNF index (normalized)
            + self.num_nodes  # Placement mask
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Max VNFs for normalization
        self.max_vnfs = self.config["sfc"]["vnf_count"]["max"]

        # Normalization constants for SFC constraints
        self.max_security = self.config["substrate"]["resources"]["security_score"][
            "max"
        ]
        self.max_latency = self.config["sfc"]["constraints"]["max_latency"]["max"]
        self.max_bandwidth = self.config["sfc"]["constraints"]["min_bandwidth"]["max"]

        # Normalization constants for VNF resources
        self.max_vnf_ram = self.config["sfc"]["vnf_resources"]["ram"]["max"]
        self.max_vnf_cpu = self.config["sfc"]["vnf_resources"]["cpu"]["max"]
        self.max_vnf_storage = self.config["sfc"]["vnf_resources"]["storage"]["max"]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Each episode handles max_requests_per_episode SFC requests.
        The substrate network is reset to full capacity at the start of each episode.
        """
        super().reset(seed=seed)

        # Reset substrate network to full capacity for new episode
        if self.randomize_topology:
            self.substrate.regenerate_topology()
        else:
            self.substrate.reset()

        self.request_generator.reset()

        # Reset episode-level counters
        self.episode_requests = 0
        self.episode_accepted = 0
        self.total_episodes += 1

        # Generate first SFC request for this episode
        self._start_new_request()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _start_new_request(self):
        """Start processing a new SFC request within the current episode."""

        # Generate new SFC request
        self.current_request = self.request_generator.generate_request()
        self.current_vnf_index = 0
        self.current_placement = []
        self.episode_step = 0

        self.episode_requests += 1
        self.total_requests += 1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step: place current VNF on the selected node.

        Args:
            action: Index of substrate node to place current VNF

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.episode_step += 1

        # Advance time and release expired placements (TTL is per step)
        self.substrate.tick()

        current_vnf = self.current_request.vnfs[self.current_vnf_index]

        # Validate the action (should be valid due to masking, but double-check)
        if not self._is_valid_action(action):
            # Invalid action - reject the SFC and release any partial resources
            return self._handle_rejection("invalid_action", release_resources=True)

        # Allocate resources for this VNF
        self.substrate.allocate_resources(action, current_vnf)

        # If not the first VNF, allocate bandwidth from previous node
        if self.current_placement:
            prev_node = self.current_placement[-1]
            self.substrate.allocate_bandwidth(
                prev_node, action, self.current_request.min_bandwidth
            )

        # Record placement
        self.current_placement.append(action)
        self.current_vnf_index += 1

        # Check if all VNFs are placed
        if self.current_vnf_index >= self.current_request.num_vnfs:
            # Validate latency constraint
            total_latency = self.substrate.get_total_latency(self.current_placement)

            if total_latency <= self.current_request.max_latency:
                # Success! Register placement for TTL tracking
                return self._handle_acceptance()
            else:
                # Latency violation - need to release allocated resources
                return self._handle_rejection(
                    "latency_violation", release_resources=True
                )

        # Episode continues - move to next VNF
        observation = self._get_observation()
        info = self._get_info()

        # Intermediate reward for progress
        reward = 0.5

        # Penalty for latency/distance (encourage locality)
        if self.current_placement and len(self.current_placement) > 1:
            prev_node = self.current_placement[-2]
            curr_node = self.current_placement[-1]
            latency = self.substrate.get_path_latency(prev_node, curr_node)
            norm_latency = min(latency / self.max_latency, 1.0)
            reward -= 0.1 * norm_latency

        return observation, reward, False, False, info

    def _handle_acceptance(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Handle successful SFC placement."""
        self.substrate.register_placement(self.current_request, self.current_placement)
        self.accepted_requests += 1
        self.episode_accepted += 1

        # Check if episode is complete
        terminated = self.episode_requests >= self.max_requests_per_episode

        if not terminated:
            # Move to next request within the same episode
            self._start_new_request()

        observation = self._get_observation()
        info = self._get_info()
        info["success"] = True
        info["placement"] = self.current_placement.copy()
        info["sfc_complete"] = True  # This SFC was completed

        return observation, self.acceptance_reward, terminated, False, info

    def _handle_rejection(
        self, reason: str, release_resources: bool = False
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Handle SFC rejection."""
        if release_resources:
            # Release all resources allocated so far
            for i, node_id in enumerate(self.current_placement):
                vnf = self.current_request.vnfs[i]
                self.substrate.release_resources(node_id, vnf)

            # Release bandwidth
            for i in range(len(self.current_placement) - 1):
                self.substrate.release_bandwidth(
                    self.current_placement[i],
                    self.current_placement[i + 1],
                    self.current_request.min_bandwidth,
                )

        # Check if episode is complete
        terminated = self.episode_requests >= self.max_requests_per_episode

        if not terminated:
            # Move to next request within the same episode
            self._start_new_request()

        observation = self._get_observation()
        info = self._get_info()
        info["success"] = False
        info["rejection_reason"] = reason
        info["sfc_complete"] = True  # This SFC was completed (rejected)

        return observation, self.rejection_penalty, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        # 1. Node Feature Matrix Construction
        # a. Basic Resources: (num_nodes, 4) [RAM, CPU, Storage, Security]
        base_resources = self.substrate.get_all_node_resources()

        # b. Connectivity: (num_nodes, 1) [AvgBandwidth]
        connectivity = self.substrate.get_nodes_connectivity()

        # c. Distance from previous node: (num_nodes, 1)
        if self.current_placement:
            prev_node = self.current_placement[-1]

            # Vectorized retrieval (O(1) with matrix)
            raw_distances = self.substrate.get_all_path_latencies(prev_node)

            # Handle unreachable nodes (inf)
            # Replace inf with penalty (max_latency * 2)
            raw_distances = np.where(
                np.isinf(raw_distances), self.max_latency * 2, raw_distances
            )

            # Normalize
            distances = np.minimum(raw_distances / self.max_latency, 1.0).reshape(-1, 1)
        else:
            distances = np.zeros((self.num_nodes, 1), dtype=np.float32)

        # Combine node features: (num_nodes, 6)
        node_features = np.concatenate(
            [base_resources, connectivity, distances], axis=1
        )
        node_features_flat = node_features.flatten()

        # 2. Current VNF requirements (normalized)
        if self.current_vnf_index < self.current_request.num_vnfs:
            current_vnf = self.current_request.vnfs[self.current_vnf_index]
            vnf_obs = np.array(
                [
                    current_vnf.ram / self.max_vnf_ram,
                    current_vnf.cpu / self.max_vnf_cpu,
                    current_vnf.storage / self.max_vnf_storage,
                ],
                dtype=np.float32,
            )
        else:
            vnf_obs = np.zeros(3, dtype=np.float32)

        # 3. SFC constraints (normalized)
        sfc_constraints = np.array(
            [
                self.current_request.min_security_score / self.max_security,
                self.current_request.max_latency / self.max_latency,
                self.current_request.min_bandwidth / self.max_bandwidth,
            ],
            dtype=np.float32,
        )

        # 4. Current VNF index (normalized)
        vnf_progress = np.array(
            [self.current_vnf_index / self.max_vnfs], dtype=np.float32
        )

        # 5. Placement mask (which nodes are currently used in this SFC)
        placement_mask = np.zeros(self.num_nodes, dtype=np.float32)
        for node_id in self.current_placement:
            placement_mask[node_id] = 1.0

        # Concatenate all components
        observation = np.concatenate(
            [node_features_flat, vnf_obs, sfc_constraints, vnf_progress, placement_mask]
        ).astype(np.float32)

        # Clip to [0, 1] range
        observation = np.clip(observation, 0.0, 1.0)

        return observation

    def _get_info(self) -> dict:
        """Get additional info about the current state."""
        # Episode-level acceptance ratio (this is what we want to track for learning)
        episode_acceptance_ratio = (
            self.episode_accepted / self.episode_requests
            if self.episode_requests > 0
            else 0.0
        )

        return {
            "current_vnf_index": self.current_vnf_index,
            "total_vnfs": self.current_request.num_vnfs if self.current_request else 0,
            "current_placement": self.current_placement.copy(),
            # Episode-level stats
            "episode_requests": self.episode_requests,
            "episode_accepted": self.episode_accepted,
            "episode_acceptance_ratio": episode_acceptance_ratio,
            # Global stats
            "total_requests": self.total_requests,
            "accepted_requests": self.accepted_requests,
            "total_episodes": self.total_episodes,
            "acceptance_ratio": (
                self.accepted_requests / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
        }

    def _is_valid_action(self, action: int) -> bool:
        """Check if an action is valid for the current state."""
        if action < 0 or action >= self.num_nodes:
            return False

        current_vnf = self.current_request.vnfs[self.current_vnf_index]

        # Check node feasibility (resources + security)
        if not self.substrate.check_node_feasibility(
            action, current_vnf, self.current_request.min_security_score
        ):
            return False

        # Check bandwidth if not first VNF
        if self.current_placement:
            prev_node = self.current_placement[-1]
            if not self.substrate.check_bandwidth(
                prev_node, action, self.current_request.min_bandwidth
            ):
                return False

        return True

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask indicating valid actions.

        This is used by MaskablePPO to prevent selecting invalid actions.
        """
        mask = np.zeros(self.num_nodes, dtype=bool)

        if self.current_request is None:
            return mask

        if self.current_vnf_index >= self.current_request.num_vnfs:
            return mask

        current_vnf = self.current_request.vnfs[self.current_vnf_index]
        min_security = self.current_request.min_security_score

        # 1. Vectorized Resource Feasibility Check (O(1) relative to Python loop)
        feasible_mask = self.substrate.get_feasible_nodes(current_vnf, min_security)

        # 2. Bandwidth Check (only for resource-feasible nodes)
        if self.current_placement:
            prev_node = self.current_placement[-1]
            min_bw = self.current_request.min_bandwidth

            # Get indices of feasible nodes
            feasible_indices = np.where(feasible_mask)[0]

            for node_id in feasible_indices:
                # We still need to check bandwidth for these specific paths.
                # Since we cached paths, this is fast (O(PathLen)).
                if self.substrate.check_bandwidth(prev_node, node_id, min_bw):
                    mask[node_id] = True
        else:
            # First VNF: no bandwidth constraints yet, so mask is just resource feasibility
            mask = feasible_mask

        return mask

        return mask

    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            print(f"\n=== SFC Placement Environment ===")
            print(f"Request ID: {self.current_request.request_id}")
            print(f"VNFs: {self.current_request.num_vnfs}")
            print(
                f"Current VNF: {self.current_vnf_index + 1}/{self.current_request.num_vnfs}"
            )
            print(f"Placement so far: {self.current_placement}")
            print(f"Constraints:")
            print(f"  Min Security: {self.current_request.min_security_score:.2f}")
            print(f"  Max Latency: {self.current_request.max_latency:.2f}")
            print(f"  Min Bandwidth: {self.current_request.min_bandwidth:.2f}")
            print(f"Stats: {self.accepted_requests}/{self.total_requests} accepted")
            print(f"Valid actions: {np.sum(self.action_masks())} nodes")

    def close(self):
        """Clean up resources."""
        pass


# Register the environment with Gymnasium
gym.register(
    id="SFCPlacement-v0",
    entry_point="src.environment:SFCPlacementEnv",
)
