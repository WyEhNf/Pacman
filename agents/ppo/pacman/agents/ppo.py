# pacman/agents/ppo.py
"""Proximal Policy Optimization with clipped surrogate objective."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .networks import ActorCritic
from .rollout import RolloutBuffer


class PPO:
    """PPO agent for Pac-Man."""

    def __init__(self, network: ActorCritic, config: dict, device: torch.device):
        self.network = network.to(device)
        self.device = device
        self.config = config["ppo"]

        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=self.config["learning_rate"], eps=1e-5,
        )

    @torch.no_grad()
    def select_action(
        self,
        obs_grid: np.ndarray,     # (N, 8, 31, 28)
        obs_scalars: np.ndarray,  # (N, 4)
        legal_masks: np.ndarray,  # (N, 4) bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for all environments. Returns (actions, log_probs, values)."""
        grid_t = torch.as_tensor(obs_grid, device=self.device)
        scalars_t = torch.as_tensor(obs_scalars, device=self.device)
        mask_t = torch.as_tensor(legal_masks, device=self.device)

        logits, values = self.network(grid_t, scalars_t, mask_t)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
        )

    @torch.no_grad()
    def get_value(self, obs_grid: np.ndarray, obs_scalars: np.ndarray,
                  legal_masks: np.ndarray) -> np.ndarray:
        """Get value estimate for terminal bootstrap."""
        grid_t = torch.as_tensor(obs_grid, device=self.device)
        scalars_t = torch.as_tensor(obs_scalars, device=self.device)
        mask_t = torch.as_tensor(legal_masks, device=self.device)
        _, values = self.network(grid_t, scalars_t, mask_t)
        return values.squeeze(-1).cpu().numpy()

    def update(self, rollout: RolloutBuffer) -> dict:
        """Run PPO update over the rollout buffer. Returns loss metrics."""
        cfg = self.config
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Normalize advantages
        adv = rollout.advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        rollout.advantages = adv.reshape(rollout.rollout_steps, rollout.num_envs)

        for _epoch in range(cfg["num_epochs"]):
            for batch in rollout.batch_generator(cfg["minibatch_size"], self.device):
                logits, values = self.network(
                    batch["obs_grid"], batch["obs_scalars"], batch["legal_masks"],
                )
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy()

                # Policy loss -- clipped surrogate
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                advantages = batch["advantages"]
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - cfg["clip_epsilon"], 1 + cfg["clip_epsilon"]) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss -- clipped
                values_squeezed = values.squeeze(-1)
                value_pred_clipped = batch["old_values"] + torch.clamp(
                    values_squeezed - batch["old_values"],
                    -cfg["clip_epsilon"], cfg["clip_epsilon"],
                )
                value_loss1 = (values_squeezed - batch["returns"]) ** 2
                value_loss2 = (value_pred_clipped - batch["returns"]) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + cfg["value_loss_coef"] * value_loss
                    - self._entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def anneal_lr(self, update: int, total_updates: int) -> None:
        if not self.config.get("lr_anneal", True):
            return
        frac = 1.0 - update / total_updates
        lr = self.config["learning_rate"] * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @property
    def _entropy_coef(self) -> float:
        """Current entropy coefficient (set externally by trainer)."""
        return getattr(self, "_current_entropy_coef", self.config["entropy_coef_start"])

    def set_entropy_coef(self, coef: float) -> None:
        self._current_entropy_coef = coef
