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

from src.substrate import SubstrateNetwork
from src.requests import (
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

    Observation Space (Dict):
        - node_features: (max_nodes, 6) - [RAM, CPU, Storage, Security, AvgBW, DistToPrev]
        - global_context: (9,) - [VNF_RAM, VNF_CPU, VNF_Storage, SFC_Sec, SFC_Lat, SFC_BW, VNF_Progress, HardIsolation, LatencyProgress]
        - placement_mask: (max_nodes,) - which nodes are used in current SFC
        - node_mask: (max_nodes,) - 1.0 for real nodes, 0.0 for padding

    Action Space:
        - Discrete(max_nodes): Select a substrate node for the current VNF
        - Masked to prevent invalid placements (padded nodes always masked)

    Reward:
        - +0.5 per intermediate VNF step (minus latency penalty for locality)
        - +acceptance_reward for successfully placing entire SFC
        - +rejection_penalty (negative) for failed placement
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

        # Link latency bounds for latency-aware masking
        self.min_link_latency = self.config["substrate"]["links"]["latency"]["min"]

        # Episode configuration - how many requests per episode
        training_config = self.config.get("training", {})
        self.max_requests_per_episode = training_config.get(
            "max_requests_per_episode", 100
        )

        # Environment state
        self.current_request: Optional[SFCRequest] = None
        self.current_vnf_index: int = 0
        self.current_placement: list[int] = []
        self.current_latency: float = 0.0  # Accumulated latency for current SFC
        self.episode_step: int = 0

        # Episode-level statistics (reset each episode)
        self.episode_requests = 0
        self.episode_accepted = 0
        self.episode_latency_violations = 0
        self.episode_rejections = 0

        # Episode-level substrate metrics (sampled per request, reset each episode)
        self.episode_sfc_tenancy_samples: list[float] = []
        self.episode_vnf_tenancy_samples: list[float] = []
        self.episode_substrate_utilization_samples: list[float] = []

        # Global statistics tracking (across all episodes)
        self.total_requests = 0
        self.accepted_requests = 0
        self.total_episodes = 0

        # Rejection reason tracking
        self.rejection_reasons = {
            "latency_violation": 0,  # Final check after all VNFs placed
            "latency_constraint": 0,  # Early rejection via action masking
            "invalid_action": 0,
            "no_valid_actions": 0,  # Resources/bandwidth exhausted
        }

        # Define action and observation spaces
        self.num_nodes = self.substrate.num_nodes
        self.max_nodes = self.config["substrate"].get("max_nodes", self.num_nodes)
        assert self.num_nodes <= self.max_nodes, (
            f"num_nodes ({self.num_nodes}) must be <= max_nodes ({self.max_nodes})"
        )
        self.action_space = spaces.Discrete(self.max_nodes)

        # Dict observation space — shapes are fixed at max_nodes for topology agnosticism
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    0.0, 1.0, (self.max_nodes, 6), dtype=np.float32
                ),
                "global_context": spaces.Box(0.0, 1.0, (9,), dtype=np.float32),
                "placement_mask": spaces.Box(
                    0.0, 1.0, (self.max_nodes,), dtype=np.float32
                ),
                "node_mask": spaces.Box(0.0, 1.0, (self.max_nodes,), dtype=np.float32),
            }
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
            self.num_nodes = self.substrate.num_nodes
        else:
            self.substrate.reset()

        self.request_generator.reset()

        # Reset episode-level counters
        self.episode_requests = 0
        self.episode_accepted = 0
        self.episode_latency_violations = 0
        self.episode_rejections = 0
        self.episode_sfc_tenancy_samples = []
        self.episode_vnf_tenancy_samples = []
        self.episode_substrate_utilization_samples = []
        self.total_episodes += 1

        # Generate first SFC request for this episode
        self._start_new_request()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _start_new_request(self):
        """Start processing a new SFC request within the current episode."""

        # Advance time and release expired placements (TTL is per request)
        self.substrate.tick()

        # Generate new SFC request
        self.current_request = self.request_generator.generate_request()
        self.current_vnf_index = 0
        self.current_placement = []
        self.current_latency = 0.0
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

        current_vnf = self.current_request.vnfs[self.current_vnf_index]

        # Validate the action (should be valid due to masking, but double-check)
        if not self._is_valid_action(action):
            # Invalid action - reject the SFC and release any partial resources
            return self._handle_rejection("invalid_action", release_resources=True)

        # Allocate resources for this VNF
        self.substrate.allocate_resources(action, current_vnf)

        # If not the first VNF, allocate bandwidth from previous node and track latency
        if self.current_placement:
            prev_node = self.current_placement[-1]
            self.substrate.allocate_bandwidth(
                prev_node, action, self.current_request.min_bandwidth
            )
            # Accumulate latency for this hop
            self.current_latency += self.substrate.get_path_latency(prev_node, action)

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

        # Intermediate reward: small positive for each successful VNF placement
        reward = 0.5

        # Penalty for latency consumption (encourage locality)
        if len(self.current_placement) > 1:
            prev_node = self.current_placement[-2]
            curr_node = self.current_placement[-1]
            hop_latency = self.substrate.get_path_latency(prev_node, curr_node)
            norm_latency = min(hop_latency / self.current_request.max_latency, 1.0)
            reward -= 0.5 * norm_latency

        return observation, reward, False, False, info

    def _handle_acceptance(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Handle successful SFC placement."""
        self.substrate.register_placement(self.current_request, self.current_placement)
        self.accepted_requests += 1
        self.episode_accepted += 1

        # Sample substrate metrics after placement
        self._sample_substrate_metrics()

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
        # Track rejection reason
        self.episode_rejections += 1
        if reason in self.rejection_reasons:
            self.rejection_reasons[reason] += 1
        if reason == "latency_violation":
            self.episode_latency_violations += 1

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

        # Sample substrate metrics after rejection
        self._sample_substrate_metrics()

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

    def _sample_substrate_metrics(self):
        """Sample substrate tenancy and utilization metrics after each request."""
        sfcs_per_node = self.substrate.get_sfcs_per_node()
        vnfs_per_node = self.substrate.get_vnfs_per_node()

        # Calculate average tenancy only on occupied nodes (nodes with count > 0)
        occupied_sfc_nodes = [v for v in sfcs_per_node.values() if v > 0]
        occupied_vnf_nodes = [v for v in vnfs_per_node.values() if v > 0]

        if occupied_sfc_nodes:
            self.episode_sfc_tenancy_samples.append(
                sum(occupied_sfc_nodes) / len(occupied_sfc_nodes)
            )
        if occupied_vnf_nodes:
            self.episode_vnf_tenancy_samples.append(
                sum(occupied_vnf_nodes) / len(occupied_vnf_nodes)
            )

        # Substrate utilization: % of nodes being used
        if sfcs_per_node:
            nodes_in_use = len(occupied_sfc_nodes)
            total_nodes = len(sfcs_per_node)
            self.episode_substrate_utilization_samples.append(
                nodes_in_use / total_nodes
            )

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Construct the Dict observation with max_nodes padding."""
        N = self.num_nodes
        M = self.max_nodes

        # 1. Node Feature Matrix: (num_nodes, 6) → padded to (max_nodes, 6)
        # a. Basic Resources: (num_nodes, 4) [RAM, CPU, Storage, Security]
        base_resources = self.substrate.get_all_node_resources()

        # b. Connectivity: (num_nodes, 1) [AvgBandwidth]
        connectivity = self.substrate.get_nodes_connectivity()

        # c. Distance from previous node: (num_nodes, 1)
        if self.current_placement:
            prev_node = self.current_placement[-1]
            raw_distances = self.substrate.get_all_path_latencies(prev_node)
            raw_distances = np.where(
                np.isinf(raw_distances), self.max_latency * 2, raw_distances
            )
            distances = np.minimum(raw_distances / self.max_latency, 1.0).reshape(-1, 1)
        else:
            distances = np.zeros((N, 1), dtype=np.float32)

        # Combine real node features: (num_nodes, 6)
        real_node_features = np.concatenate(
            [base_resources, connectivity, distances], axis=1
        )

        # Pad to (max_nodes, 6)
        node_features = np.zeros((M, 6), dtype=np.float32)
        node_features[:N] = np.clip(real_node_features, 0.0, 1.0)

        # 2. Global context: (7,)
        if self.current_vnf_index < self.current_request.num_vnfs:
            current_vnf = self.current_request.vnfs[self.current_vnf_index]
            vnf_obs = [
                current_vnf.ram / self.max_vnf_ram,
                current_vnf.cpu / self.max_vnf_cpu,
                current_vnf.storage / self.max_vnf_storage,
            ]
        else:
            vnf_obs = [0.0, 0.0, 0.0]

        # Latency progress: fraction of budget consumed so far
        latency_progress = (
            self.current_latency / self.current_request.max_latency
            if self.current_request.max_latency > 0
            else 0.0
        )

        global_context = np.array(
            vnf_obs
            + [
                self.current_request.min_security_score / self.max_security,
                self.current_request.max_latency / self.max_latency,
                self.current_request.min_bandwidth / self.max_bandwidth,
                self.current_vnf_index / self.max_vnfs,
                1.0 if self.current_request.hard_isolation else 0.0,
                latency_progress,
            ],
            dtype=np.float32,
        )
        global_context = np.clip(global_context, 0.0, 1.0)

        # 3. Placement mask: (max_nodes,)
        placement_mask = np.zeros(M, dtype=np.float32)
        for node_id in self.current_placement:
            placement_mask[node_id] = 1.0

        # 4. Node mask: 1.0 for real nodes, 0.0 for padding
        node_mask = np.zeros(M, dtype=np.float32)
        node_mask[:N] = 1.0

        return {
            "node_features": node_features,
            "global_context": global_context,
            "placement_mask": placement_mask,
            "node_mask": node_mask,
        }

    def _get_info(self) -> dict:
        """Get additional info about the current state."""
        # Episode-level acceptance ratio (this is what we want to track for learning)
        episode_acceptance_ratio = (
            self.episode_accepted / self.episode_requests
            if self.episode_requests > 0
            else 0.0
        )

        # Calculate episode-level substrate metrics averages
        episode_avg_sfc_tenancy = (
            sum(self.episode_sfc_tenancy_samples)
            / len(self.episode_sfc_tenancy_samples)
            if self.episode_sfc_tenancy_samples
            else 0.0
        )
        episode_avg_vnf_tenancy = (
            sum(self.episode_vnf_tenancy_samples)
            / len(self.episode_vnf_tenancy_samples)
            if self.episode_vnf_tenancy_samples
            else 0.0
        )
        episode_avg_substrate_util = (
            sum(self.episode_substrate_utilization_samples)
            / len(self.episode_substrate_utilization_samples)
            if self.episode_substrate_utilization_samples
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
            "episode_rejections": self.episode_rejections,
            "episode_latency_violations": self.episode_latency_violations,
            "episode_latency_violation_ratio": (
                self.episode_latency_violations / self.episode_rejections
                if self.episode_rejections > 0
                else 0.0
            ),
            # Episode-level substrate metrics
            "episode_avg_sfc_tenancy": episode_avg_sfc_tenancy,
            "episode_avg_vnf_tenancy": episode_avg_vnf_tenancy,
            "episode_avg_substrate_util": episode_avg_substrate_util,
            # Global stats
            "total_requests": self.total_requests,
            "accepted_requests": self.accepted_requests,
            "total_episodes": self.total_episodes,
            "acceptance_ratio": (
                self.accepted_requests / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "rejection_reasons": self.rejection_reasons.copy(),
        }

    def _is_valid_action(self, action: int) -> bool:
        """Check if an action is valid for the current state."""
        if action < 0 or action >= self.num_nodes:
            return False  # Padded nodes or out-of-range

        current_vnf = self.current_request.vnfs[self.current_vnf_index]

        # Check node feasibility (resources + security)
        if not self.substrate.check_node_feasibility(
            action, current_vnf, self.current_request.min_security_score
        ):
            return False

        # Check hard isolation constraints
        # 1. If node is hard-isolated by another SFC, we cannot use it
        if self.substrate.is_node_hard_isolated(action):
            return False

        # 2. If this SFC requires hard isolation, node must not have any other SFC
        if self.current_request.hard_isolation:
            if self.substrate.has_any_sfc_on_node(action):
                # Allow if it's our own placement from earlier in this SFC
                if action not in self.current_placement:
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
        Get mask of valid actions for current state.

        An action is valid if:
        1. Node has sufficient resources (RAM, CPU, storage) for the VNF
        2. Node meets security requirements
        3. Path from previous node has sufficient bandwidth (if not first VNF)
        4. Placing the VNF won't make latency violation inevitable
        5. Node is not hard-isolated by another SFC
        6. If this SFC requires hard isolation, node has no other SFCs

        This is used by MaskablePPO to prevent selecting invalid actions.
        """
        # Mask is max_nodes long — padded positions are always False
        mask = np.zeros(self.max_nodes, dtype=bool)

        if self.current_request is None:
            return mask

        if self.current_vnf_index >= self.current_request.num_vnfs:
            return mask

        current_vnf = self.current_request.vnfs[self.current_vnf_index]
        min_security = self.current_request.min_security_score

        # 1. Vectorized Resource Feasibility Check (only real nodes)
        feasible_mask = self.substrate.get_feasible_nodes(current_vnf, min_security)

        # 2. Exclude hard-isolated nodes (nodes reserved by other SFCs)
        hard_isolated_mask = self.substrate.get_hard_isolated_nodes_mask()
        feasible_mask = feasible_mask & ~hard_isolated_mask

        # 3. If this SFC requires hard isolation, exclude nodes with any other SFC
        if self.current_request.hard_isolation:
            occupied_mask = self.substrate.get_nodes_with_sfcs_mask()
            # Allow nodes we've already placed VNFs on in current SFC
            for node_id in self.current_placement:
                occupied_mask[node_id] = False
            feasible_mask = feasible_mask & ~occupied_mask

        # Calculate latency budget for remaining VNFs.
        # min_remaining_latency is 0 because co-located VNFs have 0 hop latency,
        # so we must not pessimistically reserve budget for future hops.
        latency_budget = self.current_request.max_latency - self.current_latency

        # 4. Bandwidth Check and Latency Check (only for resource-feasible nodes)
        if self.current_placement:
            prev_node = self.current_placement[-1]
            min_bw = self.current_request.min_bandwidth

            feasible_indices = np.where(feasible_mask)[0]

            for node_id in feasible_indices:
                if not self.substrate.check_bandwidth(prev_node, node_id, min_bw):
                    continue

                hop_latency = self.substrate.get_path_latency(prev_node, node_id)
                if hop_latency <= latency_budget:
                    mask[node_id] = True
        else:
            # First VNF: no bandwidth/latency constraints yet
            mask[: self.num_nodes] = feasible_mask

        return mask

    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            print("\n=== SFC Placement Environment ===")
            print(f"Request ID: {self.current_request.request_id}")
            print(f"VNFs: {self.current_request.num_vnfs}")
            print(
                f"Current VNF: {self.current_vnf_index + 1}/{self.current_request.num_vnfs}"
            )
            print(f"Placement so far: {self.current_placement}")
            print("Constraints:")
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
