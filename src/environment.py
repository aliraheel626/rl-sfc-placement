"""
Custom Gymnasium Environment for SFC Placement.

This environment implements the Service Function Chain placement problem
where an agent must place VNFs onto substrate nodes while respecting
resource, CIA security, trust-zone, bandwidth, latency, and link-security
constraints.
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


class SFCPlacementEnv(gym.Env):
    """
    Gymnasium environment for SFC placement using Maskable PPO.

    The agent places VNFs one at a time onto substrate nodes. After each
    placement, resources are allocated. When all VNFs are placed, the
    latency constraint is validated.

    Observation Space (Dict):
        - node_features: (max_nodes, 20) — per-node features:
            [RAM, CPU, Storage, Conf, Integ, Avail, AvgBW, DistToPrev,
             RAMShare, CPUShare, StorageShare, VNFShare, SFCTenancy,
             FitRAM, FitCPU, FitStorage, ZonePublic, ZonePrivate, ZoneDMZ,
             LinkSecurityFromPrev]
        - global_context: (20,) — SFC/VNF-level context:
            [VNF_RAM, VNF_CPU, VNF_Storage,
             SFC_MinConf, SFC_MinInteg, SFC_MinAvail,
             ZoneRequiredFlag, ZoneIdNorm, MinLinkSecurity,
             SFC_MaxLatency, SFC_MinBW, VNF_Progress, HardIsolation,
             LatencyProgress, TTL, RemainingVNFs, RemainingRAM,
             RemainingCPU, RemainingStorage, WindowedAR]
        - placement_mask: (max_nodes,) — which nodes are used in current SFC
        - node_mask: (max_nodes,) — 1.0 for real nodes, 0.0 for padding

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

        # Use provided substrate/request_generator or create new
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
        self.cia_margin_weight = float(self.config["rewards"].get("cia_margin_weight", 0.10))
        self.cia_acceptance_weight = float(self.config["rewards"].get("cia_acceptance_weight", 0.30))
        self.link_security_weight = float(self.config["rewards"].get("link_security_weight", 0.05))

        # Link latency bounds for latency-aware masking
        self.min_link_latency = self.config["substrate"]["links"]["latency"]["min"]

        # Episode length
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
        self.current_latency: float = 0.0
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

        # Episode-level CIA security metrics (per request, accepted only contributes non-zero)
        self.episode_conf_margins: list[float] = []
        self.episode_integ_margins: list[float] = []
        self.episode_avail_margins: list[float] = []
        self.episode_zone_matches: list[float] = []
        self.episode_link_security_margins: list[float] = []

        # Global statistics tracking (across all episodes)
        self.total_requests = 0
        self.accepted_requests = 0
        self.total_episodes = 0

        # Rejection reason tracking
        self.rejection_reasons = {
            "latency_violation": 0,
            "latency_constraint": 0,
            "invalid_action": 0,
            "no_valid_actions": 0,
        }

        # Action and observation spaces
        self.num_nodes = self.substrate.num_nodes
        self.max_nodes = self.config["substrate"].get("max_nodes", self.num_nodes)
        assert self.num_nodes <= self.max_nodes, (
            f"num_nodes ({self.num_nodes}) must be <= max_nodes ({self.max_nodes})"
        )
        self.action_space = spaces.Discrete(self.max_nodes)

        # node_features: 20 cols; global_context: 20 elements
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    0.0, 1.0, (self.max_nodes, 20), dtype=np.float32
                ),
                "global_context": spaces.Box(0.0, 1.0, (20,), dtype=np.float32),
                "placement_mask": spaces.Box(
                    0.0, 1.0, (self.max_nodes,), dtype=np.float32
                ),
                "node_mask": spaces.Box(0.0, 1.0, (self.max_nodes,), dtype=np.float32),
            }
        )

        # Normalization constants
        self.max_vnfs = self.config["sfc"]["vnf_count"]["max"]
        res_cfg = self.config["substrate"]["resources"]
        self.max_conf = res_cfg["confidentiality"]["max"]
        self.max_integ = res_cfg["integrity"]["max"]
        self.max_avail = res_cfg["availability"]["max"]
        self.num_zones = res_cfg.get("num_zones", 3)
        # Max possible CIA margin = max node score minus lowest possible requirement (= 1 from config)
        sfc_cia_min = self.config["sfc"]["constraints"]["confidentiality"]["min"]
        self._max_cia_margin = max(self.max_conf - sfc_cia_min, 1.0)
        self.max_latency = self.config["sfc"]["constraints"]["max_latency"]["max"]
        self.max_bandwidth = self.config["sfc"]["constraints"]["min_bandwidth"]["max"]
        self.max_ttl = self.config["sfc"]["ttl"]["max"]
        self.min_ttl = self.config["sfc"]["ttl"]["min"]
        self.max_vnf_ram = self.config["sfc"]["vnf_resources"]["ram"]["max"]
        self.max_vnf_cpu = self.config["sfc"]["vnf_resources"]["cpu"]["max"]
        self.max_vnf_storage = self.config["sfc"]["vnf_resources"]["storage"]["max"]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if self.randomize_topology:
            self.substrate.regenerate_topology()
            self.num_nodes = self.substrate.num_nodes
        else:
            self.substrate.reset()

        if not self.randomize_request_generator:
            if self._request_gen_rng_state is None:
                self._request_gen_rng_state = random.getstate()
                self._request_gen_np_rng_state = np.random.get_state()
            else:
                random.setstate(self._request_gen_rng_state)
                np.random.set_state(self._request_gen_np_rng_state)
        self.request_generator.reset()

        # Reset episode counters
        self.recent_outcomes.clear()
        self.episode_requests = 0
        self.episode_accepted = 0
        self.episode_latency_violations = 0
        self.episode_rejections = 0
        self.episode_sfc_tenancy_samples = []
        self.episode_vnf_tenancy_samples = []
        self.episode_substrate_utilization_samples = []
        self.episode_conf_margins = []
        self.episode_integ_margins = []
        self.episode_avail_margins = []
        self.episode_zone_matches = []
        self.episode_link_security_margins = []
        self.total_episodes += 1

        self._start_new_request()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _start_new_request(self):
        """Start processing a new SFC request within the current episode."""
        self.substrate.tick()

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

        if not self._is_valid_action(action):
            return self._handle_rejection("invalid_action", release_resources=True)

        self.substrate.allocate_resources(action, current_vnf)

        if self.current_placement:
            prev_node = self.current_placement[-1]
            self.substrate.allocate_bandwidth(
                prev_node, action, self.current_request.min_bandwidth
            )
            self.current_latency += self.substrate.get_path_latency(prev_node, action)

        self.current_placement.append(action)
        self.current_vnf_index += 1

        if self.current_vnf_index >= self.current_request.num_vnfs:
            total_latency = self.substrate.get_total_latency(self.current_placement)
            if total_latency <= self.current_request.max_latency:
                return self._handle_acceptance()
            else:
                return self._handle_rejection(
                    "latency_violation", release_resources=True
                )

        observation = self._get_observation()
        info = self._get_info()

        reward = 0.1  # progress bonus per VNF placement

        # Latency penalty and link-security reward for each hop
        if len(self.current_placement) > 1:
            prev_node = self.current_placement[-2]
            curr_node = self.current_placement[-1]
            hop_latency = self.substrate.get_path_latency(prev_node, curr_node)
            norm_latency = min(hop_latency / self.current_request.max_latency, 1.0)
            reward -= 0.3 * norm_latency

            path_sec = float(self.substrate.get_min_path_link_security_from(prev_node)[curr_node])
            link_margin = max(0.0, path_sec - self.current_request.min_link_security)
            reward += self.link_security_weight * link_margin

        # CIA security margin reward: incentivise picking nodes with security headroom
        res = self.substrate.node_resources[action]
        conf_surplus = max(0.0, res["confidentiality"] - self.current_request.min_confidentiality)
        integ_surplus = max(0.0, res["integrity"]       - self.current_request.min_integrity)
        avail_surplus = max(0.0, res["availability"]    - self.current_request.min_availability)
        norm_cia_margin = (conf_surplus + integ_surplus + avail_surplus) / (3.0 * self._max_cia_margin)
        reward += self.cia_margin_weight * norm_cia_margin

        return observation, reward, False, False, info

    def _handle_acceptance(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Handle successful SFC placement."""
        completed_placement = self.current_placement.copy()
        completed_request = self.current_request

        self.substrate.register_placement(self.current_request, self.current_placement)
        self.accepted_requests += 1
        self.episode_accepted += 1
        self._sample_substrate_metrics()

        # Compute CIA security margins over placement
        conf_m = integ_m = avail_m = 0.0
        if completed_placement:
            for node_id in completed_placement:
                res = self.substrate.node_resources[node_id]
                conf_m += res["confidentiality"] - completed_request.min_confidentiality
                integ_m += res["integrity"] - completed_request.min_integrity
                avail_m += res["availability"] - completed_request.min_availability
            n = len(completed_placement)
            conf_m /= n
            integ_m /= n
            avail_m /= n

        # Zone compliance
        if completed_request.required_zone >= 0:
            zone_ok = all(
                self.substrate.node_resources[nid]["zone_id"] == completed_request.required_zone
                for nid in completed_placement
            )
        else:
            zone_ok = True

        # Link security margin (average over hops)
        link_sec_m = 0.0
        n_hops = len(completed_placement) - 1
        if n_hops > 0:
            for i in range(n_hops):
                src, dst = completed_placement[i], completed_placement[i + 1]
                path_sec = float(self.substrate.get_min_path_link_security_from(src)[dst])
                link_sec_m += path_sec - completed_request.min_link_security
            link_sec_m /= n_hops

        self.episode_conf_margins.append(conf_m)
        self.episode_integ_margins.append(integ_m)
        self.episode_avail_margins.append(avail_m)
        self.episode_zone_matches.append(1.0 if zone_ok else 0.0)
        self.episode_link_security_margins.append(link_sec_m)

        # CIA acceptance bonus: normalised mean margin across all three dimensions
        norm_conf_m = max(0.0, conf_m) / self._max_cia_margin
        norm_integ_m = max(0.0, integ_m) / self._max_cia_margin
        norm_avail_m = max(0.0, avail_m) / self._max_cia_margin
        cia_bonus = self.cia_acceptance_weight * (norm_conf_m + norm_integ_m + norm_avail_m) / 3.0

        reward = float(self.acceptance_reward) + cia_bonus
        self.recent_outcomes.append(1.0)
        windowed_ar = sum(self.recent_outcomes) / len(self.recent_outcomes)
        reward *= 1.0 + self.ar_reward_weight * windowed_ar

        terminated = self.episode_requests >= self.max_requests_per_episode
        if not terminated:
            self._start_new_request()

        observation = self._get_observation()
        info = self._get_info()
        info["success"] = True
        info["placement"] = completed_placement
        info["sfc_complete"] = True
        info["request_conf_margin"] = conf_m
        info["request_integ_margin"] = integ_m
        info["request_avail_margin"] = avail_m
        info["request_zone_matched"] = 1.0 if zone_ok else 0.0
        info["request_link_security_margin"] = link_sec_m
        return observation, reward, terminated, False, info

    def _handle_rejection(
        self, reason: str, release_resources: bool = False
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Handle SFC rejection."""
        self.episode_rejections += 1
        if reason in self.rejection_reasons:
            self.rejection_reasons[reason] += 1
        if reason == "latency_violation":
            self.episode_latency_violations += 1

        # Rejected SFCs contribute 0 to CIA metrics
        self.episode_conf_margins.append(0.0)
        self.episode_integ_margins.append(0.0)
        self.episode_avail_margins.append(0.0)
        self.episode_zone_matches.append(0.0)
        self.episode_link_security_margins.append(0.0)

        if release_resources:
            for i, node_id in enumerate(self.current_placement):
                vnf = self.current_request.vnfs[i]
                self.substrate.release_resources(node_id, vnf)
            for i in range(len(self.current_placement) - 1):
                self.substrate.release_bandwidth(
                    self.current_placement[i],
                    self.current_placement[i + 1],
                    self.current_request.min_bandwidth,
                )

        self._sample_substrate_metrics()

        reward = float(self.rejection_penalty)
        self.recent_outcomes.append(0.0)
        windowed_ar = sum(self.recent_outcomes) / len(self.recent_outcomes)
        if self.ar_flip_rejection:
            reward *= 1.0 + self.ar_reward_weight * (1.0 - windowed_ar)
        else:
            reward *= 1.0 + self.ar_reward_weight * windowed_ar

        terminated = self.episode_requests >= self.max_requests_per_episode
        if not terminated:
            self._start_new_request()

        observation = self._get_observation()
        info = self._get_info()
        info["success"] = False
        info["rejection_reason"] = reason
        info["sfc_complete"] = True
        return observation, reward, terminated, False, info

    def _sample_substrate_metrics(self):
        """Sample substrate tenancy and utilization metrics after each request."""
        sfcs_per_node = self.substrate.get_sfcs_per_node()
        vnfs_per_node = self.substrate.get_vnfs_per_node()

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
        if sfcs_per_node:
            self.episode_substrate_utilization_samples.append(
                len(occupied_sfc_nodes) / len(sfcs_per_node)
            )

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Construct the Dict observation with max_nodes padding."""
        N = self.num_nodes
        M = self.max_nodes

        # ── Node Features (20 cols) ─────────────────────────────────────────
        # a. Base resources: (N, 6) [RAM, CPU, Storage, Conf, Integ, Avail] normalized
        base_resources = self.substrate.get_all_node_resources()

        # b. Connectivity: (N, 1) [AvgBW normalized]
        connectivity = self.substrate.get_nodes_connectivity()

        # c. Distance from previous node: (N, 1) [latency normalized]
        if self.current_placement:
            prev_node = self.current_placement[-1]
            raw_distances = self.substrate.get_all_path_latencies(prev_node)
            raw_distances = np.where(
                np.isinf(raw_distances), self.max_latency * 2, raw_distances
            )
            distances = np.minimum(raw_distances / self.max_latency, 1.0).reshape(-1, 1)
        else:
            distances = np.zeros((N, 1), dtype=np.float32)

        # d. Global resource shares + tenancy: (N, 5)
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

        # e. VNF fit ratios: (N, 3) [FitRAM, FitCPU, FitStorage]
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

        # f. Zone one-hot encoding: (N, num_zones)
        zone_onehot = np.zeros((N, self.num_zones), dtype=np.float32)
        for i in range(N):
            z = self.substrate.node_resources[i]["zone_id"]
            if 0 <= z < self.num_zones:
                zone_onehot[i, z] = 1.0

        # g. Min path link security from previous node: (N, 1)
        if self.current_placement:
            prev_node = self.current_placement[-1]
            link_sec_arr = self.substrate.get_min_path_link_security_from(prev_node)
            link_sec_col = link_sec_arr.reshape(-1, 1)
        else:
            link_sec_col = np.ones((N, 1), dtype=np.float32)

        # Combine real node features: (N, 20)
        # 6 + 1 + 1 + 3 + 2 + 3 + 3 + 1 = 20
        real_node_features = np.concatenate(
            [base_resources,         # 6
             connectivity,           # 1
             distances,              # 1
             ram_global_share, cpu_global_share, storage_global_share,  # 3
             vnf_global_share, sfc_tenancy,                             # 2
             fit_ram, fit_cpu, fit_storage,                             # 3
             zone_onehot,            # 3
             link_sec_col],          # 1
            axis=1
        )

        # Pad to (max_nodes, 20)
        node_features = np.zeros((M, 20), dtype=np.float32)
        node_features[:N] = np.clip(real_node_features, 0.0, 1.0)

        # ── Global Context (20 elements) ────────────────────────────────────
        # [0-2] Current VNF demands
        if self.current_vnf_index < self.current_request.num_vnfs:
            current_vnf = self.current_request.vnfs[self.current_vnf_index]
            vnf_obs = [
                current_vnf.ram / self.max_vnf_ram,
                current_vnf.cpu / self.max_vnf_cpu,
                current_vnf.storage / self.max_vnf_storage,
            ]
        else:
            vnf_obs = [0.0, 0.0, 0.0]

        # [15-18] Future-chain features
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

        latency_progress = (
            self.current_latency / self.current_request.max_latency
            if self.current_request.max_latency > 0
            else 0.0
        )

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

        req = self.current_request
        zone_required_flag = 1.0 if req.required_zone >= 0 else 0.0
        zone_id_norm = (
            req.required_zone / max(self.num_zones - 1, 1)
            if req.required_zone >= 0
            else 0.0
        )

        global_context = np.array(
            vnf_obs  # [0-2]
            + [
                req.min_confidentiality / self.max_conf,  # [3]
                req.min_integrity / self.max_integ,       # [4]
                req.min_availability / self.max_avail,    # [5]
                zone_required_flag,                       # [6]
                zone_id_norm,                             # [7]
                req.min_link_security,                    # [8] already in [0,1]
                req.max_latency / self.max_latency,       # [9]
                req.min_bandwidth / self.max_bandwidth,   # [10]
                self.current_vnf_index / self.max_vnfs,  # [11]
                1.0 if req.hard_isolation else 0.0,       # [12]
                latency_progress,                         # [13]
                ttl_norm,                                 # [14]
                remaining_count_norm,                     # [15]
                remaining_ram_norm,                       # [16]
                remaining_cpu_norm,                       # [17]
                remaining_storage_norm,                   # [18]
                windowed_ar,                              # [19]
            ],
            dtype=np.float32,
        )
        global_context = np.clip(global_context, 0.0, 1.0)

        # Placement mask and node mask
        placement_mask = np.zeros(M, dtype=np.float32)
        for node_id in self.current_placement:
            placement_mask[node_id] = 1.0

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
        episode_acceptance_ratio = (
            self.episode_accepted / self.episode_requests
            if self.episode_requests > 0
            else 0.0
        )
        episode_avg_sfc_tenancy = (
            sum(self.episode_sfc_tenancy_samples) / len(self.episode_sfc_tenancy_samples)
            if self.episode_sfc_tenancy_samples
            else 0.0
        )
        episode_avg_vnf_tenancy = (
            sum(self.episode_vnf_tenancy_samples) / len(self.episode_vnf_tenancy_samples)
            if self.episode_vnf_tenancy_samples
            else 0.0
        )
        episode_avg_substrate_util = (
            sum(self.episode_substrate_utilization_samples)
            / len(self.episode_substrate_utilization_samples)
            if self.episode_substrate_utilization_samples
            else 0.0
        )

        # CIA security metrics
        n_req = len(self.episode_conf_margins)
        episode_avg_conf_margin = (
            sum(self.episode_conf_margins) / n_req if n_req > 0 else 0.0
        )
        episode_avg_integ_margin = (
            sum(self.episode_integ_margins) / n_req if n_req > 0 else 0.0
        )
        episode_avg_avail_margin = (
            sum(self.episode_avail_margins) / n_req if n_req > 0 else 0.0
        )
        episode_zone_compliance = (
            sum(self.episode_zone_matches) / n_req if n_req > 0 else 1.0
        )
        episode_avg_link_sec_margin = (
            sum(self.episode_link_security_margins) / n_req if n_req > 0 else 0.0
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
            # Episode substrate metrics
            "episode_avg_sfc_tenancy": episode_avg_sfc_tenancy,
            "episode_avg_vnf_tenancy": episode_avg_vnf_tenancy,
            "episode_avg_substrate_util": episode_avg_substrate_util,
            # Episode CIA security metrics
            "episode_avg_conf_margin": episode_avg_conf_margin,
            "episode_avg_integ_margin": episode_avg_integ_margin,
            "episode_avg_avail_margin": episode_avg_avail_margin,
            "episode_zone_compliance_ratio": episode_zone_compliance,
            "episode_avg_link_security_margin": episode_avg_link_sec_margin,
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
            return False

        current_vnf = self.current_request.vnfs[self.current_vnf_index]

        # CIA + zone feasibility
        if not self.substrate.check_node_feasibility(
            action, current_vnf, self.current_request
        ):
            return False

        # Hard isolation constraints
        if self.substrate.is_node_hard_isolated(action):
            return False
        if self.current_request.hard_isolation:
            if self.substrate.has_any_sfc_on_node(action):
                if action not in self.current_placement:
                    return False

        # Bandwidth + link security
        if self.current_placement:
            prev_node = self.current_placement[-1]
            if not self.substrate.check_bandwidth(
                prev_node, action, self.current_request.min_bandwidth
            ):
                return False
            if not self.substrate.check_link_security(
                prev_node, action, self.current_request.min_link_security
            ):
                return False

        return True

    def action_masks(self) -> np.ndarray:
        """
        Get mask of valid actions for current state.

        An action is valid if:
        1. Node has sufficient resources (RAM, CPU, Storage) for the VNF
        2. Node meets CIA security requirements and zone constraint
        3. Path from previous node has sufficient bandwidth (if not first VNF)
        4. Path from previous node meets link security requirement
        5. Placing the VNF won't make latency violation inevitable
        6. Node is not hard-isolated by another SFC
        7. If this SFC requires hard isolation, node has no other SFCs
        """
        mask = np.zeros(self.max_nodes, dtype=bool)

        if self.current_request is None:
            return mask
        if self.current_vnf_index >= self.current_request.num_vnfs:
            return mask

        current_vnf = self.current_request.vnfs[self.current_vnf_index]

        # 1. Vectorized CIA + zone feasibility (only real nodes)
        feasible_mask = self.substrate.get_feasible_nodes(current_vnf, self.current_request)

        # 2. Exclude hard-isolated nodes
        hard_isolated_mask = self.substrate.get_hard_isolated_nodes_mask()
        feasible_mask = feasible_mask & ~hard_isolated_mask

        # 3. Hard isolation for this SFC
        if self.current_request.hard_isolation:
            occupied_mask = self.substrate.get_nodes_with_sfcs_mask()
            for node_id in self.current_placement:
                occupied_mask[node_id] = False
            feasible_mask = feasible_mask & ~occupied_mask

        # Latency budget
        remaining_vnfs = self.current_request.num_vnfs - self.current_vnf_index - 1
        min_remaining_latency = remaining_vnfs * self.min_link_latency
        raw_budget = (
            self.current_request.max_latency
            - self.current_latency
            - min_remaining_latency
        )
        latency_budget = max(0.0, raw_budget)

        # 4. Bandwidth, latency, and link security checks
        if self.current_placement:
            prev_node = self.current_placement[-1]
            min_bw = self.current_request.min_bandwidth
            min_link_sec = self.current_request.min_link_security

            for node_id in np.where(feasible_mask)[0]:
                if not self.substrate.check_bandwidth(prev_node, node_id, min_bw):
                    continue
                hop_latency = self.substrate.get_path_latency(prev_node, node_id)
                if hop_latency > latency_budget:
                    continue
                if not self.substrate.check_link_security(prev_node, node_id, min_link_sec):
                    continue
                mask[node_id] = True
        else:
            # First VNF: no bandwidth/latency/link-security constraints yet
            mask[: self.num_nodes] = feasible_mask

        # Safety: if no valid action, enable node 0 so MaskablePPO can sample;
        # step() catches it via _is_valid_action and auto-rejects.
        if not mask.any():
            mask[0] = True

        return mask

    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            req = self.current_request
            print("\n=== SFC Placement Environment ===")
            print(f"Request ID: {req.request_id}")
            print(f"VNFs: {req.num_vnfs}")
            print(f"Current VNF: {self.current_vnf_index + 1}/{req.num_vnfs}")
            print(f"Placement so far: {self.current_placement}")
            print("Constraints:")
            print(f"  Min Confidentiality: {req.min_confidentiality:.2f}")
            print(f"  Min Integrity:       {req.min_integrity:.2f}")
            print(f"  Min Availability:    {req.min_availability:.2f}")
            print(f"  Required Zone:       {req.required_zone}")
            print(f"  Min Link Security:   {req.min_link_security:.2f}")
            print(f"  Max Latency:         {req.max_latency:.2f}")
            print(f"  Min Bandwidth:       {req.min_bandwidth:.2f}")
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
