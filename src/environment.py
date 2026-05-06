"""
Custom Gymnasium Environment for SFC Placement.

This environment implements the Service Function Chain placement problem
where an agent must place VNFs onto substrate nodes while respecting
resource, security, bandwidth, and latency constraints.
"""

import random
from collections import deque
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
from src.risk import (
    compute_node_risk_score,
    compute_placement_risk_score,
    decay_incident_pressure,
    effective_node_security,
    robust_ratio,
)


class SFCPlacementEnv(gym.Env):
    """
    Gymnasium environment for SFC placement using Maskable PPO.

    The agent places VNFs one at a time onto substrate nodes. After each
    placement, resources are allocated. When all VNFs are placed, the
    latency constraint is validated.

    Observation Space (Dict):
        - node_features: (max_nodes, 20) - [RAM, CPU, Storage, Security, AvgBW, DistToPrev,
          RAMGlobalShare, CPUGlobalShare, StorageGlobalShare, VNFGlobalShare, SFCTenancy,
          FitRAM, FitCPU, FitStorage, IncidentPressure, LoadNorm, PBase, ExpectedIncidents,
          NodeRiskScore (=PBase), ExpectedLostRevenue]
        - global_context: (15,) - [VNF_RAM, VNF_CPU, VNF_Storage, SFC_Sec, SFC_Lat, SFC_BW,
          VNF_Progress, HardIsolation, LatencyProgress, TTL,
          RemainingVNFs, RemainingRAM, RemainingCPU, RemainingStorage, WindowedAR]
        - placement_mask: (max_nodes,) - which nodes are used in current SFC
        - node_mask: (max_nodes,) - 1.0 for real nodes, 0.0 for padding

    Action Space:
        - Discrete(max_nodes): Select a substrate node for the current VNF
        - Masked to prevent invalid placements (padded nodes always masked)

    Reward:
        - +0.1 per intermediate VNF step (minus latency penalty for locality)
        - +acceptance_reward for successfully placing entire SFC
        - +rejection_penalty (negative) for failed placement
        - Terminal rewards are scaled by a windowed acceptance-ratio multiplier
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str = "config.yaml",
        render_mode: Optional[str] = None,
        *,
        substrate: Optional[SubstrateNetwork] = None,
        request_generator: Optional[RequestGenerator] = None,
        max_requests_per_episode: Optional[int] = None,
    ):
        super().__init__()

        # Load configuration
        self.config = load_config(config_path)
        self.render_mode = render_mode

        # Use provided substrate/request_generator (e.g. for fair comparison) or create new
        if substrate is not None and request_generator is not None:
            self.substrate = substrate
            self.request_generator = request_generator
            self.randomize_topology = False
            self.randomize_request_generator = False
            self._request_gen_rng_state = None
            self._request_gen_np_rng_state = None
        else:
            self.substrate = SubstrateNetwork(self.config["substrate"])
            self.request_generator = RequestGenerator(self.config["sfc"])
            self.randomize_topology = self.config.get("training", {}).get(
                "randomize_topology", False
            )
            self.randomize_request_generator = self.config.get("training", {}).get(
                "randomize_request_generator", False
            )
            self._request_gen_rng_state = None
            self._request_gen_np_rng_state = None
            if self.randomize_topology:
                print("INFO: Topology randomization enabled (new graph per episode).")
            if not self.randomize_request_generator:
                print(
                    "INFO: Same request sequence per episode (randomize_request_generator=false)."
                )

        # Reward configuration
        self.acceptance_reward = self.config["rewards"]["acceptance"]
        self.rejection_penalty = self.config["rewards"]["rejection"]
        self.ar_window_size = self.config["rewards"].get("acceptance_ratio_window", 10)
        self.ar_reward_weight = float(
            self.config["rewards"].get("acceptance_ratio_reward_weight", 0.5)
        )
        self.ar_flip_rejection = self.config["rewards"].get(
            "acceptance_ratio_flip_rejection", True
        )
        self.recent_outcomes: deque[float] = deque(maxlen=self.ar_window_size)
        risk_cfg = self.config.get("risk", {})
        self.risk_enabled = risk_cfg.get("enabled", False)
        self.risk_lambda = float(risk_cfg.get("lambda", 0.0))
        self.tenancy_ref_floor = float(risk_cfg.get("tenancy_ref_floor", 1.0))
        self.load_ref_floor = float(risk_cfg.get("load_ref_floor", 1.0))
        self.incident_base_rate = float(risk_cfg.get("incident_base_rate", 0.025))
        self.incident_alpha = float(risk_cfg.get("incident_alpha", 1.6))
        self.incident_beta = float(risk_cfg.get("incident_beta", 0.7))
        self.incident_steps_cap = int(risk_cfg.get("incident_steps_cap", 12))
        self.incident_pressure_decay = float(risk_cfg.get("incident_pressure_decay", 0.92))
        self.incident_security_penalty = float(
            risk_cfg.get("incident_security_penalty", 0.60)
        )
        self.revenue_per_ttl_step = float(risk_cfg.get("revenue_per_ttl_step", 1.0))

        # Link latency bounds for latency-aware masking
        self.min_link_latency = self.config["substrate"]["links"]["latency"]["min"]

        # Episode configuration - how many requests per episode (override for eval: single long episode)
        training_config = self.config.get("training", {})
        self.max_requests_per_episode = (
            max_requests_per_episode
            if max_requests_per_episode is not None
            else training_config.get("max_requests_per_episode", 100)
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
        self.episode_risk_scores: list[float] = []
        self.episode_risk_integrals: list[float] = []
        self.episode_revenues: list[float] = []

        # Global statistics tracking (across all episodes)
        self.total_requests = 0
        self.accepted_requests = 0
        self.total_episodes = 0
        self.total_risk_integral: float = 0.0
        self.total_revenue: float = 0.0
        self.node_incident_pressure: dict[int, float] = {}

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
                    0.0, 1.0, (self.max_nodes, 20), dtype=np.float32
                ),
                "global_context": spaces.Box(0.0, 1.0, (15,), dtype=np.float32),
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
        self.max_ttl = self.config["sfc"]["ttl"]["max"]
        self.min_ttl = self.config["sfc"]["ttl"]["min"]

        # Normalization constants for VNF resources
        self.max_vnf_ram = self.config["sfc"]["vnf_resources"]["ram"]["max"]
        self.max_vnf_cpu = self.config["sfc"]["vnf_resources"]["cpu"]["max"]
        self.max_vnf_storage = self.config["sfc"]["vnf_resources"]["storage"]["max"]
        self.min_vnf_storage = self.config["sfc"]["vnf_resources"]["storage"]["min"]

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

        # Same request sequence every episode when randomize_request_generator is False
        if not self.randomize_request_generator:
            if self._request_gen_rng_state is None:
                self._request_gen_rng_state = random.getstate()
                self._request_gen_np_rng_state = np.random.get_state()
            else:
                random.setstate(self._request_gen_rng_state)
                np.random.set_state(self._request_gen_np_rng_state)
        self.request_generator.reset()

        # Reset episode-level counters
        self.recent_outcomes.clear()
        self.episode_requests = 0
        self.episode_accepted = 0
        self.episode_latency_violations = 0
        self.episode_rejections = 0
        self.episode_sfc_tenancy_samples = []
        self.episode_vnf_tenancy_samples = []
        self.episode_substrate_utilization_samples = []
        self.episode_risk_scores = []
        self.episode_risk_integrals = []
        self.episode_revenues = []
        self.node_incident_pressure = {node_id: 0.0 for node_id in range(self.num_nodes)}
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
        decay_incident_pressure(
            self.node_incident_pressure, self.incident_pressure_decay
        )

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

        # Intermediate reward shaping: small signal per VNF step so the agent
        # can learn which placement decisions are good before the terminal
        # accept/reject arrives.  Kept smaller than terminal reward (+1 / -1).
        reward = 0.1  # progress bonus for each feasible VNF placement

        # Latency penalty: encourage locality / co-location
        if len(self.current_placement) > 1:
            prev_node = self.current_placement[-2]
            curr_node = self.current_placement[-1]
            hop_latency = self.substrate.get_path_latency(prev_node, curr_node)
            norm_latency = min(hop_latency / self.current_request.max_latency, 1.0)
            reward -= 0.3 * norm_latency

        return observation, reward, False, False, info

    def _handle_acceptance(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Handle successful SFC placement."""
        # Capture completed-request data before potentially advancing to the next request.
        completed_placement = self.current_placement.copy()
        completed_min_security = self.current_request.min_security_score

        self.substrate.register_placement(self.current_request, self.current_placement)
        self.accepted_requests += 1
        self.episode_accepted += 1

        # Sample substrate metrics after placement
        self._sample_substrate_metrics()

        # Compute security cost before tick(): with TTL=1, tick() can release this
        # placement, so cost must be computed while the placement is still registered.
        risk_integral = self._compute_request_risk(completed_placement)
        self.episode_risk_scores.append(risk_integral)
        self.episode_risk_integrals.append(risk_integral)
        self.total_risk_integral += risk_integral
        request_revenue = self.revenue_per_ttl_step * (self.current_request.ttl if self.current_request is not None else 1)
        self.episode_revenues.append(request_revenue)
        self.total_revenue += request_revenue

        self.recent_outcomes.append(1.0)
        windowed_ar = sum(self.recent_outcomes) / len(self.recent_outcomes)
        reward = (self.acceptance_reward - self.risk_lambda * risk_integral) * (
            1.0 + self.ar_reward_weight * windowed_ar
        )

        # Check if episode is complete
        terminated = self.episode_requests >= self.max_requests_per_episode

        if not terminated:
            # Move to next request within the same episode (calls substrate.tick())
            self._start_new_request()

        observation = self._get_observation()
        info = self._get_info()
        info["success"] = True
        info["placement"] = completed_placement
        info["request_min_security"] = completed_min_security
        info["sfc_complete"] = True  # This SFC was completed
        info["request_risk_score"] = risk_integral
        info["request_risk_integral"] = risk_integral
        info["request_revenue"] = request_revenue
        return observation, reward, terminated, False, info

    def _handle_rejection(
        self, reason: str, release_resources: bool = False
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Handle SFC rejection."""
        risk_integral = 0.0
        self.episode_risk_scores.append(risk_integral)
        self.episode_risk_integrals.append(risk_integral)
        self.episode_revenues.append(0.0)

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

        self.recent_outcomes.append(0.0)
        windowed_ar = sum(self.recent_outcomes) / len(self.recent_outcomes)
        if self.ar_flip_rejection:
            reward = self.rejection_penalty * (
                1.0 + self.ar_reward_weight * (1.0 - windowed_ar)
            )
        else:
            reward = self.rejection_penalty * (
                1.0 + self.ar_reward_weight * windowed_ar
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
        info["request_risk_score"] = risk_integral
        info["request_risk_integral"] = risk_integral

        return observation, reward, terminated, False, info

    def _placement_risk_cfg(self) -> dict:
        """Security cost parameters for compute_placement_risk_score (shared with baselines / compare)."""
        return {
            "enabled": self.risk_enabled,
            "tenancy_ref_floor": self.tenancy_ref_floor,
            "load_ref_floor": self.load_ref_floor,
            "incident_steps_cap": self.incident_steps_cap,
            "incident_base_rate": self.incident_base_rate,
            "incident_alpha": self.incident_alpha,
            "incident_beta": self.incident_beta,
            "incident_security_penalty": self.incident_security_penalty,
        }

    def _compute_request_risk(self, placement: list[int]) -> float:
        """Compute deterministic security cost heuristic for one accepted placement."""
        ttl = self.current_request.ttl if self.current_request is not None else 1
        return compute_placement_risk_score(
            self.substrate,
            placement,
            ttl,
            self._placement_risk_cfg(),
            self.max_security,
            self.max_ttl,
            self.node_incident_pressure,
        )

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

        # 1. Node Feature Matrix: (num_nodes, 20) → padded to (max_nodes, 20)
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

        # d. Global resource shares + tenancy: (num_nodes, 5)
        #    [RAMGlobalShare, CPUGlobalShare, StorageGlobalShare, VNFGlobalShare, SFCTenancy]
        # Resource global shares: used_i / Σ_j used_j  ("what fraction of all consumed
        #   RAM/CPU/Storage is on this node?") — orthogonal to the local remaining-fraction
        #   features which answer "how much capacity is left on this node."
        # VNFGlobalShare: vnfs_i / Σ_j vnfs_j — instance-count concentration.
        # SFCTenancy:     sfcs_i / max(vnfs_i, 1) — chain diversity / blast-radius proxy.
        vnfs_per_node = self.substrate.get_vnfs_per_node()
        sfcs_per_node = self.substrate.get_sfcs_per_node()

        used_ram = np.array(
            [self.substrate.original_resources[i]["ram"] - self.substrate.node_resources[i]["ram"]
             for i in range(N)], dtype=np.float32,
        )
        used_cpu = np.array(
            [self.substrate.original_resources[i]["cpu"] - self.substrate.node_resources[i]["cpu"]
             for i in range(N)], dtype=np.float32,
        )
        used_storage = np.array(
            [self.substrate.original_resources[i]["storage"] - self.substrate.node_resources[i]["storage"]
             for i in range(N)], dtype=np.float32,
        )
        ram_global_share = (used_ram / max(used_ram.sum(), 1e-6)).reshape(-1, 1)
        cpu_global_share = (used_cpu / max(used_cpu.sum(), 1e-6)).reshape(-1, 1)
        storage_global_share = (used_storage / max(used_storage.sum(), 1e-6)).reshape(-1, 1)

        total_vnfs = max(sum(vnfs_per_node.values()), 1)
        vnf_global_share = np.array(
            [vnfs_per_node.get(i, 0) / total_vnfs for i in range(N)],
            dtype=np.float32,
        ).reshape(-1, 1)
        sfc_tenancy = np.array(
            [sfcs_per_node.get(i, 0) / max(vnfs_per_node.get(i, 0), 1) for i in range(N)],
            dtype=np.float32,
        ).reshape(-1, 1)

        # e. VNF fit ratios: (num_nodes, 3) [FitRAM, FitCPU, FitStorage]
        # vnf_demand / remaining_capacity — how much of the node's remaining
        # headroom this VNF would consume. Near 0 = barely dents it, 1 = fills it.
        if self.current_vnf_index < self.current_request.num_vnfs:
            cur_vnf = self.current_request.vnfs[self.current_vnf_index]
            fit_ram = np.array(
                [cur_vnf.ram / max(self.substrate.node_resources[i]["ram"], 1e-6)
                 for i in range(N)], dtype=np.float32,
            ).reshape(-1, 1)
            fit_cpu = np.array(
                [cur_vnf.cpu / max(self.substrate.node_resources[i]["cpu"], 1e-6)
                 for i in range(N)], dtype=np.float32,
            ).reshape(-1, 1)
            fit_storage = np.array(
                [cur_vnf.storage / max(self.substrate.node_resources[i]["storage"], 1e-6)
                 for i in range(N)], dtype=np.float32,
            ).reshape(-1, 1)
        else:
            fit_ram = fit_cpu = fit_storage = np.zeros((N, 1), dtype=np.float32)

        # f. Incident features: (num_nodes, 4)
        #    [IncidentPressure, LoadNorm, PBase, ExpectedIncidents]
        #    These are the direct drivers of p_base — giving the agent the same
        #    signals the risk formula and baselines use, closing the obs asymmetry.
        load_reference = max(
            self.load_ref_floor,
            float(np.percentile(list(vnfs_per_node.values()), 75)) if vnfs_per_node else 0.0,
        )
        exposure_steps_obs = int(max(1, min(self.incident_steps_cap, self.current_request.ttl)))
        incident_pressure_arr = np.array(
            [self.node_incident_pressure.get(i, 0.0) for i in range(N)],
            dtype=np.float32,
        ).reshape(-1, 1)
        load_norm_arr = np.array(
            [robust_ratio(float(vnfs_per_node.get(i, 0)), load_reference) for i in range(N)],
            dtype=np.float32,
        ).reshape(-1, 1)
        # p_base computed using effective (incident-degraded) security norms from base_resources col 3
        # (overwritten below); compute here using the same effective_node_security call.
        p_base_arr = np.zeros(N, dtype=np.float32)
        for i in range(N):
            _, sec_norm = effective_node_security(
                self.substrate, i, self.max_security,
                self.node_incident_pressure, self.incident_security_penalty,
            )
            p_i = self.incident_base_rate * (1.0 - sec_norm) ** self.incident_alpha
            p_i *= 1.0 + self.incident_beta * float(load_norm_arr[i, 0])
            p_i *= 1.0 + self.node_incident_pressure.get(i, 0.0)
            p_base_arr[i] = min(max(p_i, 0.0), 1.0)
            # Reuse the security norm computed here to overwrite col 3.
            base_resources[i, 3] = sec_norm
        p_base_col = p_base_arr.reshape(-1, 1)
        # ExpectedIncidents per node normalized by steps_cap so it stays in [0, 1].
        expected_inc_arr = np.clip(
            p_base_arr * exposure_steps_obs / max(self.incident_steps_cap, 1),
            0.0, 1.0,
        ).reshape(-1, 1)

        # g. Node risk score: (num_nodes, 1) [NodeRiskScore = PBase]
        #    p_base is the primary risk signal — sfcs × TTL scaling is already
        #    captured by other features, so the agent just needs raw p_base here.
        node_risk_arr = p_base_col  # same as index 16, explicit risk slot

        # h. Expected loss proxy per node: (num_nodes, 1)
        #    Blast-radius-weighted risk: 1 - 1/(1 + q_n × SFC(n))
        sfc_counts_arr = np.array(
            [float(sfcs_per_node.get(i, 0)) for i in range(N)], dtype=np.float32
        )
        elr_raw = p_base_arr * sfc_counts_arr
        elr_arr = (1.0 - 1.0 / (1.0 + elr_raw)).reshape(-1, 1).astype(np.float32)

        # Combine real node features (num_nodes, 20)
        real_node_features = np.concatenate(
            [base_resources, connectivity, distances,
             ram_global_share, cpu_global_share, storage_global_share,
             vnf_global_share, sfc_tenancy,
             fit_ram, fit_cpu, fit_storage,
             incident_pressure_arr, load_norm_arr, p_base_col, expected_inc_arr,
             node_risk_arr, elr_arr], axis=1
        )

        # Pad to (max_nodes, 20)
        node_features = np.zeros((M, 20), dtype=np.float32)
        node_features[:N] = np.clip(real_node_features, 0.0, 1.0)

        # 2. Global context: (15,)
        # [0-2]  Current VNF demands (RAM, CPU, Storage)
        if self.current_vnf_index < self.current_request.num_vnfs:
            current_vnf = self.current_request.vnfs[self.current_vnf_index]
            vnf_obs = [
                current_vnf.ram / self.max_vnf_ram,
                current_vnf.cpu / self.max_vnf_cpu,
                current_vnf.storage / self.max_vnf_storage,
            ]
        else:
            vnf_obs = [0.0, 0.0, 0.0]

        # [10-13] Future-chain features: remaining VNF count + aggregate demand
        remaining_start = self.current_vnf_index + 1
        remaining_vnfs_list = self.current_request.vnfs[remaining_start:]
        remaining_count = len(remaining_vnfs_list)
        remaining_count_norm = remaining_count / self.max_vnfs if self.max_vnfs > 0 else 0.0
        if remaining_vnfs_list:
            remaining_ram = sum(v.ram for v in remaining_vnfs_list)
            remaining_cpu = sum(v.cpu for v in remaining_vnfs_list)
            remaining_storage = sum(v.storage for v in remaining_vnfs_list)
        else:
            remaining_ram = remaining_cpu = remaining_storage = 0.0
        denom_ram = self.max_vnfs * self.max_vnf_ram
        denom_cpu = self.max_vnfs * self.max_vnf_cpu
        denom_storage = self.max_vnfs * self.max_vnf_storage
        remaining_ram_norm = remaining_ram / denom_ram if denom_ram > 0 else 0.0
        remaining_cpu_norm = remaining_cpu / denom_cpu if denom_cpu > 0 else 0.0
        remaining_storage_norm = remaining_storage / denom_storage if denom_storage > 0 else 0.0

        # Latency progress: fraction of budget consumed so far
        latency_progress = (
            self.current_latency / self.current_request.max_latency
            if self.current_request.max_latency > 0
            else 0.0
        )

        # TTL normalized to [0, 1] (short TTL -> 0, long TTL -> 1)
        ttl_span = self.max_ttl - self.min_ttl
        ttl_norm = (
            (self.current_request.ttl - self.min_ttl) / ttl_span
            if ttl_span > 0
            else 0.0
        )

        windowed_ar = (
            sum(self.recent_outcomes) / len(self.recent_outcomes)
            if self.recent_outcomes
            else 0.0
        )

        global_context = np.array(
            vnf_obs  # [0-2] current VNF demands
            + [
                self.current_request.min_security_score / self.max_security,  # [3]
                self.current_request.max_latency / self.max_latency,          # [4]
                self.current_request.min_bandwidth / self.max_bandwidth,      # [5]
                self.current_vnf_index / self.max_vnfs,                       # [6]
                1.0 if self.current_request.hard_isolation else 0.0,          # [7]
                latency_progress,                                             # [8]
                ttl_norm,                                                     # [9]
                remaining_count_norm,                                         # [10]
                remaining_ram_norm,                                           # [11]
                remaining_cpu_norm,                                           # [12]
                remaining_storage_norm,                                       # [13]
                windowed_ar,                                                  # [14]
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
        episode_avg_risk_score = (
            sum(self.episode_risk_scores) / len(self.episode_risk_scores)
            if self.episode_risk_scores
            else 0.0
        )
        episode_total_revenue = sum(self.episode_revenues)
        episode_avg_risk_integral = (
            sum(self.episode_risk_integrals) / len(self.episode_risk_integrals)
            if self.episode_risk_integrals
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
            "episode_avg_risk_score": episode_avg_risk_score,
            "episode_total_revenue": episode_total_revenue,
            "episode_avg_risk_integral": episode_avg_risk_integral,
            # Global stats
            "total_requests": self.total_requests,
            "accepted_requests": self.accepted_requests,
            "total_episodes": self.total_episodes,
            "total_revenue": self.total_revenue,
            "total_risk_integral": self.total_risk_integral,
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
        eff_sec, _ = effective_node_security(
            self.substrate,
            action,
            self.max_security,
            self.node_incident_pressure,
            self.incident_security_penalty,
        )
        if eff_sec < self.current_request.min_security_score:
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
        feasible_indices = np.where(feasible_mask)[0]
        for node_id in feasible_indices:
            eff_sec, _ = effective_node_security(
                self.substrate,
                node_id,
                self.max_security,
                self.node_incident_pressure,
                self.incident_security_penalty,
            )
            if eff_sec < min_security:
                feasible_mask[node_id] = False

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

        # Reserve latency for remaining hops (matching baseline pruning).
        remaining_vnfs = self.current_request.num_vnfs - self.current_vnf_index - 1
        min_remaining_latency = remaining_vnfs * self.min_link_latency
        raw_budget = (
            self.current_request.max_latency
            - self.current_latency
            - min_remaining_latency
        )
        # Clamp to non-negative: if we're over budget, only zero-latency hops are
        # allowed; a negative budget would make hop_latency <= budget impossible
        # and incorrectly mask out all actions, forcing the fallback to action 0.
        latency_budget = max(0.0, raw_budget)

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

        # Safety: if no valid action exists, enable action 0 so MaskablePPO
        # can still sample.  step() will catch it via _is_valid_action and
        # auto-reject the SFC.
        if not mask.any():
            mask[0] = True

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
