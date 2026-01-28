"""Team RL: Reinforcement learning with learned embeddings.

Team Members: Pushpita Joardar, Hun Tae Kim

This module contains:
- Reacher-v5 environment wrappers
- RL agents (using stable-baselines3)
- Embedding integration with Team Estimation's model
- Benchmarking utilities

The goal is to compare RL performance using:
1. Raw joint states
2. Embeddings from Team Estimation's trained model
"""

from articulated.rl.environment import ReacherWithEmbedding
from articulated.rl.agent import RLAgent

__all__ = ["ReacherWithEmbedding", "RLAgent"]
