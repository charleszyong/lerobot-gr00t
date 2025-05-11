# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LeRobot is a Hugging Face project that provides models, datasets, and tools for real-world robotics in PyTorch. It focuses on lowering the barrier to entry to robotics through shared datasets and pretrained models, with an emphasis on imitation learning and reinforcement learning.

## Development Environment Setup

1. Create a virtual environment with Python 3.10:
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

2. Install ffmpeg:
```bash
conda install ffmpeg -c conda-forge
```

3. Install LeRobot:
```bash
pip install -e .
```

4. For specific simulation environments, install the relevant extras:
```bash
# For example, to install aloha and pusht environments
pip install -e ".[aloha, pusht]"
```

5. For development, use `uv` or `poetry`:
```bash
# Using uv
uv venv --python 3.10 && source .venv/bin/activate
uv sync --extra dev --extra test

# Using poetry
poetry sync --extras "dev test"
```

## Common Commands

### Running Tests

Run all tests:
```bash
python -m pytest -sv ./tests
```

Run specific tests:
```bash
# Run a specific test file
pytest tests/test_available.py

# Run a specific test
pytest tests/test_available.py::test_available_policies
```

### Code Formatting and Linting

Format code using pre-commit:
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on staged files
pre-commit

# Run on all files
pre-commit run --all-files
```

### Building Docker Images

```bash
# For CPU-only
make build-cpu

# For GPU support
make build-gpu
```

### End-to-End Testing

Test the full training and evaluation pipeline for different models:
```bash
# Test ACT policy
make test-act-ete-train
make test-act-ete-eval

# Test Diffusion policy
make test-diffusion-ete-train
make test-diffusion-ete-eval

# Test TDMPC policy
make test-tdmpc-ete-train
make test-tdmpc-ete-eval
```

## Key Workflows

### Dataset Visualization

Visualize a dataset from Hugging Face Hub:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

Visualize a local dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --local-files-only 1 \
    --episode-index 0
```

### Evaluate a Pretrained Policy

```bash
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

### Train a Policy

```bash
python lerobot/scripts/train.py \
    --policy.type=diffusion \
    --env.type=pusht \
    --dataset.repo_id=lerobot/pusht \
    --batch_size=64 \
    --steps=50000 \
    --eval_freq=1000 \
    --eval.n_episodes=10 \
    --policy.device=cuda
```

To use Weights and Biases for logging:
```bash
# First, login to wandb (one-time setup)
wandb login

# Then add wandb.enable=true to your training command
python lerobot/scripts/train.py --wandb.enable=true [other args]
```

## Code Architecture and Structure

### Key Components

1. **Datasets**: The `LeRobotDataset` class is the main data interface, providing access to datasets stored on the Hugging Face hub or locally. It is built on top of the Hugging Face datasets library but adds functionality for temporal relationships through `delta_timestamps`.

2. **Policies**: The core machine learning models that learn from the datasets and can be deployed on robots:
   - ACT (Action Chunking Transformer)
   - Diffusion (Diffusion-based policy learning)
   - TDMPC (Temporal Difference Model Predictive Control)
   - VQBeT (Vector Quantized Behavior Transformer)

3. **Environments**: Simulation environments that implement the Gymnasium interface:
   - aloha: Bimanual manipulation using ALOHA robots
   - pusht: Pushing tasks
   - xarm: Single-arm manipulation using XArm

4. **Robot Devices**: Hardware interfaces for real robots:
   - Dynamixel motors
   - OpenCV cameras
   - Various robot implementations (SO-100, SO-101, LeKiwi, etc.)

### Directory Structure

```
.
├── examples/         # Example code and tutorials
├── lerobot/
│   ├── configs/      # Configuration classes with CLI options
│   ├── common/       # Core components
│   │   ├── datasets/   # Dataset implementations and utilities
│   │   ├── envs/       # Environment interfaces
│   │   ├── policies/   # Policy implementations
│   │   ├── robot_devices/ # Hardware interfaces
│   │   └── utils/      # Various utilities
│   └── scripts/      # Command-line tools
└── tests/            # Test suite
```

## Working with Datasets

LeRobot datasets are based on the Hugging Face Datasets library but add special functionality for robotics:
- Temporal relationships through `delta_timestamps`
- Automatic downloading from Hugging Face hub
- Serialization using parquet for metadata and MP4 for videos

Standard dataset creation:
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Basic usage
dataset = LeRobotDataset("lerobot/aloha_static_coffee")

# Loading specific episodes
dataset = LeRobotDataset("lerobot/aloha_static_coffee", episodes=[0, 1, 2])

# With temporal relationships
delta_timestamps = {
    "observation.images.cam_high": [-1, -0.5, -0.2, 0],  # Past frames
    "action": [0, 1/30, 2/30]  # Current and future actions
}
dataset = LeRobotDataset("lerobot/aloha_static_coffee", delta_timestamps=delta_timestamps)
```

## Working with Policies

Policies can be loaded from the Hugging Face hub or created from scratch:

```python
from lerobot.common.policies.factory import create_policy

# Load a pretrained policy
policy = create_policy(path="lerobot/diffusion_pusht", device="cuda")

# Create a new policy
from lerobot.configs.policies import DiffusionConfig
config = DiffusionConfig(device="cuda")
policy = create_policy(config=config)
```

## Working with Environments

Environments follow the Gymnasium API:

```python
from lerobot.common.envs.factory import create_env

# Create an environment
env = create_env(type="pusht", render_mode="human")

# Reset the environment
obs, info = env.reset()

# Take a step
action = policy(obs)
next_obs, reward, terminated, truncated, info = env.step(action)
```