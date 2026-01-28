"""RL agents for Reacher benchmark.

Tasks:
1. Set up RL training with stable-baselines3
2. Implement training loop
3. Compare raw vs embedded observations
"""

from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO, SAC


class RLAgent:
    """RL agent for Reacher benchmark.

    Supports training with:
    - Raw observations (baseline)
    - Embedded observations (from Team Estimation)
    """

    def __init__(
        self,
        algorithm: str = "ppo",
        use_embedding: bool = False,
        embedding_model_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        seed: Optional[int] = None,
    ):
        """Initialize the RL agent.

        Args:
            algorithm: RL algorithm ('ppo' or 'sac').
            use_embedding: Whether to use embeddings.
            embedding_model_path: Path to Team Estimation's checkpoint.
            learning_rate: Learning rate.
            seed: Random seed.
        """
        self.algorithm = algorithm
        self.use_embedding = use_embedding
        self.embedding_model_path = embedding_model_path
        self.learning_rate = learning_rate
        self.seed = seed

        self.env: Optional[gym.Env] = None
        self.model: Optional[PPO | SAC] = None

    def setup(self) -> None:
        """Set up the environment and model.

        TODO: Implement environment and model setup:
        1. Load embedding model if use_embedding is True
        2. Create appropriate environment wrapper
        3. Initialize RL algorithm (PPO or SAC)
        """
        raise NotImplementedError("Agent setup not yet implemented")

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_freq: int = 10_000,
        save_path: Optional[str] = None,
    ) -> dict:
        """Train the RL agent.

        TODO: Implement training loop:
        1. Set up evaluation callback
        2. Call model.learn()
        3. Save trained model

        Args:
            total_timesteps: Total training timesteps.
            eval_freq: Evaluation frequency.
            save_path: Path to save the trained model.

        Returns:
            Training statistics.
        """
        raise NotImplementedError("Training not yet implemented")

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Evaluate the trained agent.

        TODO: Implement evaluation:
        1. Run n_episodes with deterministic policy
        2. Compute mean reward and episode length

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Evaluation metrics.
        """
        raise NotImplementedError("Evaluation not yet implemented")
