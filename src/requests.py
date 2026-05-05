"""
SFC Request Generation for RL-based SFC Placement.

This module provides classes for:
- VNF: Represents a Virtual Network Function with resource requirements
- SFCRequest: Represents a Service Function Chain with constraints
- RequestGenerator: Generates SFC requests sequentially
"""

from dataclasses import dataclass
import math
import random

import yaml

# Re-export SubstrateNetwork for backward compatibility
# Saved models contain serialized references to src.requests.SubstrateNetwork
# that need to resolve during model unpickling/loading
from src.substrate import SubstrateNetwork  # noqa: F401


def _truncated_exponential_inverse_cdf(mean: float, low: float, high: float) -> float:
    """
    Sample from exponential(mean) truncated to [low, high] via inverse CDF.

    If U ~ Uniform(0,1), then X = F^{-1}(U) has CDF F (truncated exponential).
    F(x) = (exp(-λ*low) - exp(-λ*x)) / (exp(-λ*low) - exp(-λ*high)),  λ = 1/mean.
    Solving F(X) = U:  X = -mean * ln( exp(-low/mean)*(1-U) + exp(-high/mean)*U ).
    """
    if low >= high:
        return low
    if not math.isfinite(mean) or mean <= 0:
        return random.uniform(low, high)
    lam = 1.0 / mean
    u = random.random()
    exp_neg_la = math.exp(-lam * low)
    exp_neg_lb = math.exp(-lam * high)
    inner = exp_neg_la * (1.0 - u) + exp_neg_lb * u
    if inner <= 0:
        return high
    x = -(1.0 / lam) * math.log(inner)
    return max(low, min(high, x))


@dataclass
class VNF:
    """Represents a Virtual Network Function with resource requirements."""

    ram: float
    cpu: float
    storage: float
    index: int = 0  # Position in the SFC chain


@dataclass
class SFCRequest:
    """
    Represents a Service Function Chain request.

    Attributes:
        vnfs: List of VNFs in the chain (ordered)
        min_confidentiality: All hosting nodes must meet this confidentiality level
        min_integrity: All hosting nodes must meet this integrity level
        min_availability: All hosting nodes must meet this availability level
        required_zone: Trust zone all VNFs must be in (-1 = any zone allowed)
        min_link_security: Minimum security level on every VNF-to-VNF path
        max_latency: Maximum end-to-end latency allowed
        min_bandwidth: Minimum bandwidth required between consecutive VNFs
        ttl: Time to live (number of timesteps before resources are released)
        request_id: Unique identifier for this request
        hard_isolation: If True, this SFC cannot share substrate nodes with
            other SFCs, and other SFCs cannot share nodes with it
    """

    vnfs: list[VNF]
    min_confidentiality: float
    min_integrity: float
    min_availability: float
    required_zone: int        # -1 = any zone; 0/1/2/... = specific zone required
    min_link_security: float
    max_latency: float
    min_bandwidth: float
    ttl: int
    request_id: int = 0
    hard_isolation: bool = False

    @property
    def num_vnfs(self) -> int:
        return len(self.vnfs)


class RequestGenerator:
    """Generates SFC requests sequentially."""

    def __init__(self, config: dict):
        """
        Initialize the request generator.

        Args:
            config: Dictionary containing SFC configuration
        """
        self.config = config
        self.vnf_count_range = (config["vnf_count"]["min"], config["vnf_count"]["max"])
        self.vnf_resources = config["vnf_resources"]
        self.constraints = config["constraints"]
        ttl_cfg = config["ttl"]
        self.ttl_distribution = ttl_cfg.get("distribution", "uniform")
        if self.ttl_distribution == "exponential":
            self.ttl_min = float(ttl_cfg["min"])
            self.ttl_max = float(ttl_cfg["max"])
            if "mean" in ttl_cfg:
                self.ttl_mean = float(ttl_cfg["mean"])
            else:
                self.ttl_mean = (self.ttl_min + self.ttl_max) / 2.0
        else:
            self.ttl_range = (ttl_cfg["min"], ttl_cfg["max"])
        self.hard_isolation_probability = config.get("hard_isolation_probability", 0.0)

        # Zone configuration
        self.num_zones = self.constraints.get("num_zones", 3)
        self.required_zone_probability = self.constraints.get("required_zone_probability", 0.0)

        self.next_request_id = 0

    def generate_request(self) -> SFCRequest:
        """Generate a new random SFC request."""
        num_vnfs = random.randint(*self.vnf_count_range)

        vnfs = []
        for i in range(num_vnfs):
            vnf = VNF(
                ram=random.uniform(
                    self.vnf_resources["ram"]["min"], self.vnf_resources["ram"]["max"]
                ),
                cpu=random.uniform(
                    self.vnf_resources["cpu"]["min"], self.vnf_resources["cpu"]["max"]
                ),
                storage=random.uniform(
                    self.vnf_resources["storage"]["min"],
                    self.vnf_resources["storage"]["max"],
                ),
                index=i,
            )
            vnfs.append(vnf)

        # CIA security requirements
        min_confidentiality = random.uniform(
            self.constraints["confidentiality"]["min"],
            self.constraints["confidentiality"]["max"],
        )
        min_integrity = random.uniform(
            self.constraints["integrity"]["min"],
            self.constraints["integrity"]["max"],
        )
        min_availability = random.uniform(
            self.constraints["availability"]["min"],
            self.constraints["availability"]["max"],
        )

        # Zone requirement
        if random.random() < self.required_zone_probability and self.num_zones > 0:
            required_zone = random.randint(0, self.num_zones - 1)
        else:
            required_zone = -1

        # Link security requirement
        min_link_security = random.uniform(
            self.constraints["min_link_security"]["min"],
            self.constraints["min_link_security"]["max"],
        )

        max_latency = random.uniform(
            self.constraints["max_latency"]["min"],
            self.constraints["max_latency"]["max"],
        )
        min_bandwidth = random.uniform(
            self.constraints["min_bandwidth"]["min"],
            self.constraints["min_bandwidth"]["max"],
        )

        if self.ttl_distribution == "exponential":
            ttl = int(round(_truncated_exponential_inverse_cdf(
                self.ttl_mean, self.ttl_min, self.ttl_max
            )))
        else:
            ttl = random.randint(*self.ttl_range)

        hard_isolation = random.random() < self.hard_isolation_probability

        request = SFCRequest(
            vnfs=vnfs,
            min_confidentiality=min_confidentiality,
            min_integrity=min_integrity,
            min_availability=min_availability,
            required_zone=required_zone,
            min_link_security=min_link_security,
            max_latency=max_latency,
            min_bandwidth=min_bandwidth,
            ttl=ttl,
            request_id=self.next_request_id,
            hard_isolation=hard_isolation,
        )

        self.next_request_id += 1
        return request

    def reset(self):
        """Reset the generator state."""
        self.next_request_id = 0


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
