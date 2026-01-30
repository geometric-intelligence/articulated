"""RL agents for Reacher benchmark.

Tasks:
1. Set up RL training with stable-baselines3
2. Implement training loop
3. Compare raw vs embedded observations
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from articulated.rl.environment import ReacherWithEmbedding


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
        n_steps: int = 2048,
        batch_size: int = 64,
        policy: str = "MlpPolicy",
        device: str = "auto",
        history_length: int = 10,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the RL agent.

        Args:
            algorithm: RL algorithm ('ppo' or 'sac').
            use_embedding: Whether to use embeddings.
            embedding_model_path: Path to Team Estimation's checkpoint.
            learning_rate: Learning rate.
            n_steps: Number of steps per rollout (PPO only).
            batch_size: Mini-batch size (PPO/SAC).
            policy: Policy network type for SB3.
            device: Torch device for SB3 ("cpu", "cuda", "auto").
            history_length: Timesteps of velocity history for embeddings.
            render_mode: Optional render mode for the env.
            seed: Random seed.
        """
        self.algorithm = algorithm.lower()
        self.use_embedding = use_embedding
        self.embedding_model_path = embedding_model_path
        self.learning_rate = (
            float(learning_rate) if isinstance(learning_rate, str) else learning_rate
        )
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.policy = policy
        self.device = device
        self.history_length = history_length
        self.render_mode = render_mode
        self.seed = seed

        self.env: Optional[gym.Env] = None
        self.model: Optional[PPO | SAC] = None
        self.embedding_model = None

    def setup(self) -> None:
        """Set up the environment and model.

        Loads the embedding model if needed, creates the env, and initializes
        the SB3 algorithm.
        """
        if self.use_embedding:
            if not self.embedding_model_path:
                raise ValueError(
                    "embedding_model_path is required when use_embedding is True."
                )
            from articulated.estimation.model import StateEstimationModel

            self.embedding_model = StateEstimationModel.load_for_embedding(
                self.embedding_model_path
            )

        self.env = self._make_env(render_mode=self.render_mode)

        if self.seed is not None:
            self.env.reset(seed=self.seed)
            if hasattr(self.env.action_space, "seed"):
                self.env.action_space.seed(self.seed)

        if self.algorithm == "ppo":
            if self.n_steps % self.batch_size != 0:
                raise ValueError(
                    "For PPO, n_steps must be divisible by batch_size."
                )
            self.model = PPO(
                self.policy,
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                seed=self.seed,
                device=self.device,
                verbose=1,
            )
        elif self.algorithm == "sac":
            self.model = SAC(
                self.policy,
                self.env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                seed=self.seed,
                device=self.device,
                verbose=1,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

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
        if self.model is None or self.env is None:
            raise RuntimeError("Call setup() before train().")

        eval_env = None
        callback = None
        if eval_freq and eval_freq > 0:
            eval_env = self._make_env(render_mode=None)
            callback = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                verbose=1,
            )

        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        if save_path:
            self.model.save(save_path)

        if eval_env is not None:
            eval_env.close()

        return {
            "total_timesteps": total_timesteps,
        }

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
        if self.model is None:
            raise RuntimeError("Call setup() and train() before evaluate().")

        eval_env = self._make_env(render_mode=None)
        rewards, lengths = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        eval_env.close()

        rewards_arr = np.asarray(rewards, dtype=np.float32)
        lengths_arr = np.asarray(lengths, dtype=np.float32)

        return {
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "mean_length": float(lengths_arr.mean()),
            "std_length": float(lengths_arr.std()),
        }

    def _make_env(self, render_mode: Optional[str]) -> gym.Env:
        """Create a Reacher-v5 env, optionally wrapped for embeddings."""
        if self.use_embedding:
            return ReacherWithEmbedding(
                embedding_model=self.embedding_model,
                use_embedding=True,
                history_length=self.history_length,
                render_mode=render_mode,
            )
        return gym.make("Reacher-v5", render_mode=render_mode)
