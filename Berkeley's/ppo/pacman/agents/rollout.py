# pacman/agents/rollout.py
"""On-policy rollout buffer for PPO."""
import numpy as np
import torch


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(
        self,
        num_envs: int,
        rollout_steps: int,
        grid_shape: tuple[int, int, int],  # (C, H, W)
        num_scalars: int,
        num_actions: int,
    ):
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.total = num_envs * rollout_steps

        T, N = rollout_steps, num_envs
        self.obs_grids = np.zeros((T, N, *grid_shape), dtype=np.float32)
        self.obs_scalars = np.zeros((T, N, num_scalars), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=bool)
        self.legal_masks = np.zeros((T, N, num_actions), dtype=bool)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

        self._step = 0

    def insert(
        self,
        obs_grid: np.ndarray,
        obs_scalars: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        legal_mask: np.ndarray,
    ) -> None:
        t = self._step
        self.obs_grids[t] = obs_grid
        self.obs_scalars[t] = obs_scalars
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done
        self.legal_masks[t] = legal_mask
        self._step += 1

    def compute_gae(
        self,
        last_value: np.ndarray,  # (N,)
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and returns in-place."""
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def batch_generator(self, minibatch_size: int, device: torch.device):
        """Yield shuffled minibatches as torch tensors."""
        T, N = self.rollout_steps, self.num_envs
        total = T * N
        indices = np.random.permutation(total)

        # Flatten (T, N, ...) -> (T*N, ...)
        flat_grids = self.obs_grids.reshape(total, *self.obs_grids.shape[2:])
        flat_scalars = self.obs_scalars.reshape(total, -1)
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_masks = self.legal_masks.reshape(total, -1)
        flat_values = self.values.reshape(total)

        for start in range(0, total, minibatch_size):
            end = start + minibatch_size
            idx = indices[start:end]
            yield {
                "obs_grid": torch.as_tensor(flat_grids[idx], device=device),
                "obs_scalars": torch.as_tensor(flat_scalars[idx], device=device),
                "actions": torch.as_tensor(flat_actions[idx], device=device),
                "old_log_probs": torch.as_tensor(flat_log_probs[idx], device=device),
                "advantages": torch.as_tensor(flat_advantages[idx], device=device),
                "returns": torch.as_tensor(flat_returns[idx], device=device),
                "legal_masks": torch.as_tensor(flat_masks[idx], device=device),
                "old_values": torch.as_tensor(flat_values[idx], device=device),
            }

    def reset(self) -> None:
        self._step = 0
