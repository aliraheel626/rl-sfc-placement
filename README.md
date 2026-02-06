# RL-SFC-Placement

> **Reinforcement Learning-based Service Function Chain Placement using Maskable PPO**

A deep reinforcement learning framework for optimal Virtual Network Function (VNF) placement in Service Function Chains (SFC). This project implements a Maskable PPO agent with optional Graph Neural Network (GNN) policies to intelligently place VNFs on substrate network nodes while respecting resource, security, bandwidth, and latency constraints.

---

## 🎯 Features

- **Maskable PPO Agent**: Uses action masking to prevent invalid placements, ensuring the agent only considers feasible nodes
- **GNN-based Policies**: Optional Graph Neural Network feature extractors (GCN, GAT, GraphSAGE) for topology-aware decision making
- **Custom Gymnasium Environment**: Fully configurable SFC placement environment with realistic constraints
- **Dynamic Topology Support**: Supports randomized network topologies per training episode
- **Baseline Algorithms**: Compare RL agent against classical algorithms:
  - **Viterbi (Optimal)**: Dynamic programming approach for minimum-latency placement
  - **First-Fit**: Simple heuristic selecting the first valid node
  - **Best-Fit**: Greedy heuristic minimizing wasted capacity
- **Comprehensive Evaluation**: Metrics include acceptance ratio, latency, resource utilization, and more
- **Visualization**: Rolling window plots for metric evolution and comparison bar charts

---

## 📁 Project Structure

```
rl-sfc-placement/
├── config.yaml           # Configuration for environment, training, and evaluation
├── main.py               # Entry point
├── pyproject.toml        # Project dependencies (uv/pip)
├── src/
│   ├── environment.py    # Gymnasium environment for SFC placement
│   ├── substrate.py      # Substrate network topology management
│   ├── requests.py       # SFC request and VNF generation
│   ├── model.py          # Maskable PPO model utilities
│   ├── gnn_policy.py     # GNN-based Actor-Critic policies
│   ├── baselines.py      # Baseline placement algorithms
│   ├── train.py          # Training script
│   └── compare.py        # Evaluation and comparison script
├── models/               # Saved trained models
├── eval/                 # Evaluation plots and results
└── logs/                 # TensorBoard training logs
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

Using **uv** (recommended):

```bash
# Clone the repository
git clone https://github.com/aliraheel626/rl-sfc-placement.git
cd rl-sfc-placement

# Install dependencies
uv sync
```

Using **pip**:

```bash
# Clone the repository
git clone https://github.com/aliraheel626/rl-sfc-placement.git
cd rl-sfc-placement

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

---

## 🏋️ Training

Train the Maskable PPO agent with default configuration:

```bash
uv run python -m src.train
```

### Training Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to configuration file | `config.yaml` |
| `--timesteps` | Total training timesteps | `100000` |
| `--save-path` | Path to save trained model | `models/sfc_ppo` |
| `--load-path` | Path to load existing model (for continued training) | `None` |
| `--seed` | Random seed | `42` |
| `--use-gnn` | Enable GNN-based policy | `False` |
| `--gnn-type` | GNN architecture (`gcn`, `gat`, `sage`) | `gcn` |
| `--gnn-hidden-dim` | GNN hidden dimension | `64` |
| `--gnn-features-dim` | GNN output features dimension | `256` |
| `--num-gnn-layers` | Number of GNN layers | `3` |

### Example with GNN Policy

```bash
uv run python -m src.train --use-gnn --gnn-type gat --timesteps 200000
```

### Monitoring Training

Use TensorBoard to monitor training progress:

```bash
uv run tensorboard --logdir logs/
```

---

## 📊 Evaluation & Comparison

Evaluate and compare all algorithms:

```bash
uv run python -m src.compare
```

### Comparison Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to configuration file | `config.yaml` |
| `--model-path` | Path to trained RL model | `models/sfc_ppo.zip` |
| `--num-requests` | Number of SFC requests per evaluation | `1000` |
| `--save-plot` | Path to save comparison plot | `None` |
| `--no-rl` | Skip RL agent evaluation | `False` |

### Example

```bash
uv run python -m src.compare --model-path models/sfc_ppo.zip --num-requests 500
```

---

## ⚙️ Configuration

The `config.yaml` file controls all aspects of the environment:

### Substrate Network

```yaml
substrate:
  num_nodes: 200                    # Number of substrate nodes
  mesh_connectivity: 0.3            # Edge probability between nodes
  resources:                        # Node resource ranges
    ram: { min: 8, max: 64 }
    cpu: { min: 4, max: 32 }
    storage: { min: 100, max: 1000 }
    security_score: { min: 1, max: 5 }
  links:                           # Link attribute ranges
    bandwidth: { min: 100, max: 1000 }
    latency: { min: 1, max: 50 }
```

### SFC Requests

```yaml
sfc:
  vnf_count: { min: 2, max: 6 }    # VNFs per chain
  vnf_resources:                   # VNF resource demands
    ram: { min: 1, max: 16 }
    cpu: { min: 1, max: 8 }
    storage: { min: 10, max: 200 }
  constraints:                     # SFC-level constraints
    security_score: { min: 1, max: 5 }
    max_latency: { min: 50, max: 200 }
    min_bandwidth: { min: 50, max: 500 }
  ttl: { min: 10, max: 100 }       # Time to live
```

### Training

```yaml
training:
  total_timesteps: 100000
  learning_rate: 0.0003
  n_steps: 4096
  batch_size: 256
  n_epochs: 10
  gamma: 0.99
  max_requests_per_episode: 250
  randomize_topology: true

rewards:
  acceptance: 10.0
  rejection: -5.0
```

---

## 📈 Metrics

The framework tracks the following metrics:

| Metric | Description |
|--------|-------------|
| **Acceptance Ratio** | Percentage of SFC requests successfully placed |
| **Average Latency** | Mean end-to-end latency of accepted SFCs |
| **SFC/Node** | Average number of SFCs using each substrate node |
| **VNF/Node** | Average number of VNFs placed per substrate node |
| **Substrate Utilization** | Overall resource utilization of the substrate network |
| **Avg Security Margin** | Average difference between SFC security requirement and node security |

---

## 🧠 Technical Details

### Environment

The `SFCPlacementEnv` is a custom Gymnasium environment where:
- **State**: Includes substrate node resources, current VNF requirements, placement history, and connectivity metrics
- **Action**: Select a substrate node to place the current VNF
- **Reward**: Positive reward for successful SFC placement, negative for rejection
- **Termination**: Episode ends after processing `max_requests_per_episode` SFC requests

### Action Masking

The environment implements action masking via `action_masks()` to ensure the agent only considers valid placements. A node is valid if:
1. Sufficient resources (RAM, CPU, storage)
2. Meets security requirements  
3. Path from previous node has sufficient bandwidth
4. Placement won't cause latency violation

### GNN Feature Extractor

The optional GNN-based policy processes the substrate network topology to learn structural patterns for better placement decisions. Supported architectures:
- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)
- **GraphSAGE** (Graph Sample and Aggregate)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ali Raheel**

---

## 🙏 Acknowledgments

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for the RL framework
- [SB3-Contrib](https://sb3-contrib.readthedocs.io/) for Maskable PPO implementation
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN layers
- [Gymnasium](https://gymnasium.farama.org/) for the environment API
- [NetworkX](https://networkx.org/) for graph operations
