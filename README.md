# Py-Gym Framework

An object-oriented reinforcement learning framework focused on modularity and clear separation of concerns.

## Project Goals

- **Architecture Scalability**: Leveraging OOP principles to build a scalable hierarchy for diverse RL algorithms.
- **Algorithmic Clarity**: Implementing core RL methods (Policy Gradient, Actor-Critic) with readable, well-structured code.

## Architecture & Module Responsibilities

The framework follows a strict modular design where each component has a single responsibility:

- **[Agent](agent/)**: Orchestrates decision-making and learning.
    - **[Policy](agent/policy)**: Neural network architectures and action selection logic.
    - **[Buffer](agent/buffer)**: Experience replay or rollout storage management.
    - **[Encoder](agent/encoder)**: State representation learning and feature extraction.
- **[Envs](envs/)**: Standardized interaction wrappers for Gymnasium environments.
- **[Utils](utils/)**: Training utilities including the decoupled `TrainingMonitor` and logging systems.



## Roadmap

- [x] Base Infrastructure (OOP Core & Logging)
- [x] REINFORCE Implementation
- [ ] Actor-Critic (Advantage-based learning)
- [ ] PPO (Proximal Policy Optimization)

## Setup

```bash
poetry install
poetry run python envs/lunar/main.py # example: lunar lander
```

## [Dependencies](pyproject.toml)

- Python 3.10+
- PyTorch
- Gymnasium
- Poetry
