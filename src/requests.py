"""
SFC Request Generation for RL-based SFC Placement.

This module provides classes for:
- VNF: Represents a Virtual Network Function with resource requirements
- SFCRequest: Represents a Service Function Chain with constraints
- RequestGenerator: Generates SFC requests sequentially
"""

from dataclasses import dataclass
import random

import yaml


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
        min_security_score: All hosting nodes must meet this security level
        max_latency: Maximum end-to-end latency allowed
        min_bandwidth: Minimum bandwidth required between consecutive VNFs
        ttl: Time to live (number of timesteps before resources are released)
        request_id: Unique identifier for this request
    """

    vnfs: list[VNF]
    min_security_score: float
    max_latency: float
    min_bandwidth: float
    ttl: int
    request_id: int = 0

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
        self.ttl_range = (config["ttl"]["min"], config["ttl"]["max"])

        self.next_request_id = 0

    def generate_request(self) -> SFCRequest:
        """Generate a new random SFC request."""
        # Determine number of VNFs
        num_vnfs = random.randint(*self.vnf_count_range)

        # Generate VNFs
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

        # Generate SFC-level constraints
        min_security = random.uniform(
            self.constraints["security_score"]["min"],
            self.constraints["security_score"]["max"],
        )
        max_latency = random.uniform(
            self.constraints["max_latency"]["min"],
            self.constraints["max_latency"]["max"],
        )
        min_bandwidth = random.uniform(
            self.constraints["min_bandwidth"]["min"],
            self.constraints["min_bandwidth"]["max"],
        )

        # Generate TTL
        ttl = random.randint(*self.ttl_range)

        request = SFCRequest(
            vnfs=vnfs,
            min_security_score=min_security,
            max_latency=max_latency,
            min_bandwidth=min_bandwidth,
            ttl=ttl,
            request_id=self.next_request_id,
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
