# pacman/training/dream_trainer.py
"""Dream Agent: PPO training entirely inside the learned world model's imagination."""
from __future__ import annotations

import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..world_model.world_model import WorldModel, symexp
from ..env.pacman_env import PacmanEnv


class DreamPolicy(nn.Module):
    """MLP actor-critic that operates on the world model's latent state (h, z).

    Args:
        latent_dim: Dimension of concatenated latent state [h; z].
        hidden: Width of the first hidden layer.
        head_hidden: Width of the second hidden layer.
        num_actions: Number of discrete actions.
    """

    def __init__(
        self,
        latent_dim: int = 2560,
        hidden: int = 512,
        head_hidden: int = 256,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value.

        Args:
            latent: Concatenated [h; z] of shape (B, latent_dim).

        Returns:
            Tuple of (logits (B, num_actions), value (B, 1)).
        """
        return self.actor(latent), self.critic(latent)


class DreamTrainer:
    """Train a DreamPolicy entirely inside the learned world model's imagination.

    The world model is frozen; only the DreamPolicy is updated via PPO on
    imagined rollouts. Periodic evaluation in the real environment tracks
    transfer quality.

    Args:
        world_model: Pretrained WorldModel (will be set to eval and frozen).
        config: Game/env config dict (used for PacmanEnv creation).
        device: Torch device.
        imagination_horizon: Number of steps per imagined rollout.
        num_imaginations: Batch size of parallel imagined trajectories.
        lr: Learning rate for the policy optimizer.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_epsilon: PPO clipped surrogate epsilon.
        entropy_coef_start: Starting entropy bonus coefficient (high for exploration).
        entropy_coef_end: Final entropy bonus coefficient after annealing.
        entropy_anneal_fraction: Fraction of training over which to anneal entropy.
        min_entropy: Minimum entropy threshold; below this the coef is tripled.
        latent_noise: Std of Gaussian noise added to latent states during imagination
            to prevent overfitting to world model imperfections.
        value_coef: Value loss coefficient.
        ppo_epochs: Number of PPO epochs per update.
    """

    def __init__(
        self,
        world_model: WorldModel,
        config: dict,
        device: torch.device,
        imagination_horizon: int = 10,
        num_imaginations: int = 512,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef_start: float = 0.5,
        entropy_coef_end: float = 0.05,
        entropy_anneal_fraction: float = 0.7,
        min_entropy: float = 0.3,
        latent_noise: float = 0.1,
        value_coef: float = 0.5,
        ppo_epochs: int = 2,
    ) -> None:
        self.device = device
        self.config = config
        self.imagination_horizon = imagination_horizon
        self.num_imaginations = num_imaginations
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_anneal_fraction = entropy_anneal_fraction
        self.min_entropy = min_entropy
        self.latent_noise = latent_noise
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs

        # World model is FROZEN -- never trained here
        self.wm = world_model.to(device)
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad_(False)

        # Policy operates on latent state
        self.policy = DreamPolicy(latent_dim=self.wm.rssm.latent_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Cache starting states to avoid regenerating every update
        self._cached_starts: tuple[torch.Tensor, torch.Tensor] | None = None
        self._starts_refresh_every = 20  # regenerate every N updates

    # ------------------------------------------------------------------
    # Starting states
    # ------------------------------------------------------------------

    def _get_starting_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create diverse starting observations by running real envs with random actions.

        Returns:
            Tuple of (grids (B, 8, 31, 28), scalars (B, 5)) on self.device.
        """
        env = PacmanEnv(self.config, difficulty=2)
        grids = []
        scalars = []

        for i in range(self.num_imaginations):
            env.reset(seed=i)
            # Advance a random number of steps (0-200) for diversity
            num_advance = random.randint(0, 200)
            for _ in range(num_advance):
                mask = env.get_legal_mask()
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal))
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    env.reset(seed=i + self.num_imaginations)
                    break

            # Get single-frame (not stacked) observation
            obs = env._build_obs()
            grids.append(obs["grid"])
            scalars.append(obs["scalars"])

        grids_t = torch.as_tensor(np.stack(grids), device=self.device)
        scalars_t = torch.as_tensor(np.stack(scalars), device=self.device)
        return grids_t, scalars_t

    # ------------------------------------------------------------------
    # Imagination rollout
    # ------------------------------------------------------------------

    def _imagine_rollout(
        self,
        start_grids: torch.Tensor,
        start_scalars: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Roll out the policy inside the world model's latent space.

        The policy's forward pass is computed WITH gradients (it is the training
        target). World model operations are under no_grad.

        Args:
            start_grids: (B, 8, 31, 28) starting grid observations.
            start_scalars: (B, 5) starting scalar observations.

        Returns:
            Dictionary with keys: latent, action, log_prob, reward, cont, value,
            bootstrap_value. Shapes are (B, H, ...) for sequence data.
        """
        B = start_grids.shape[0]
        H = self.imagination_horizon

        # 1. Encode starting observations (no_grad -- world model is frozen)
        with torch.no_grad():
            enc = self.wm.encoder(start_grids, start_scalars)  # (B, 512)

        # 2. Initialize RSSM state and get posterior z from encoder output
        with torch.no_grad():
            h, z = self.wm.rssm.initial_state(B)
            h = self.wm.rssm.dynamics(
                h, z, torch.zeros(B, dtype=torch.long, device=self.device)
            )
            z, _ = self.wm.rssm.posterior(h, enc)

        # 3. Imagination rollout
        latents = []
        actions = []
        log_probs = []
        rewards = []
        conts = []
        values = []

        for _ in range(H):
            # Concatenate latent (detach h and z so WM grads don't flow)
            latent = torch.cat([h.detach(), z.detach()], dim=-1)  # (B, latent_dim)

            # Add noise to latent to prevent overfitting to world model quirks
            if self.latent_noise > 0 and self.policy.training:
                latent = latent + self.latent_noise * torch.randn_like(latent)

            # Policy forward -- WITH grad (this is what we train)
            logits, value = self.policy(latent)
            value = value.squeeze(-1)  # (B,)
            dist = Categorical(logits=logits)
            action = dist.sample()  # (B,)
            log_prob = dist.log_prob(action)  # (B,)

            # Advance dynamics (no_grad -- world model is frozen)
            with torch.no_grad():
                h = self.wm.rssm.dynamics(h, z, action)
                z, _ = self.wm.rssm.prior(h)

                # Predict reward and continuation from the new latent
                new_latent = torch.cat([h, z], dim=-1)
                reward = self.wm.reward_head(new_latent).squeeze(-1)  # (B,)
                reward = symexp(reward)  # convert from symlog scale to real scale
                cont = torch.sigmoid(
                    self.wm.continue_head(new_latent).squeeze(-1)
                )  # (B,)

            latents.append(latent)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            conts.append(cont)
            values.append(value)

        # 4. Bootstrap value from final latent (no_grad on policy for bootstrap)
        with torch.no_grad():
            final_latent = torch.cat([h, z], dim=-1)
            _, bootstrap_value = self.policy(final_latent)
            bootstrap_value = bootstrap_value.squeeze(-1)  # (B,)

        return {
            "latent": torch.stack(latents, dim=1),      # (B, H, latent_dim)
            "action": torch.stack(actions, dim=1),       # (B, H)
            "log_prob": torch.stack(log_probs, dim=1),   # (B, H)
            "reward": torch.stack(rewards, dim=1),       # (B, H)
            "cont": torch.stack(conts, dim=1),           # (B, H)
            "value": torch.stack(values, dim=1),         # (B, H)
            "bootstrap_value": bootstrap_value,           # (B,)
        }

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self, rollout: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation on imagined rollout.

        Uses `cont` (continuation probability) as the discount mask, replacing
        the usual gamma * (1 - done) pattern.

        Args:
            rollout: Dictionary from _imagine_rollout.

        Returns:
            Tuple of (advantages (B, H), returns (B, H)).
        """
        rewards = rollout["reward"]     # (B, H)
        values = rollout["value"]       # (B, H)
        conts = rollout["cont"]         # (B, H)
        bootstrap = rollout["bootstrap_value"]  # (B,)

        B, H = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=self.device)

        for t in reversed(range(H)):
            if t == H - 1:
                next_value = bootstrap
            else:
                next_value = values[:, t + 1]

            # cont serves as the discount factor (encodes both gamma and done)
            discount = self.gamma * conts[:, t]
            delta = rewards[:, t] + discount * next_value - values[:, t]
            last_gae = delta + discount * self.gae_lambda * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values.detach()
        return advantages.detach(), returns.detach()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _get_entropy_coef(self, update: int, total_updates: int) -> float:
        """Anneal entropy coefficient from start to end over the anneal fraction."""
        anneal_end = int(total_updates * self.entropy_anneal_fraction)
        if update >= anneal_end:
            return self.entropy_coef_end
        frac = update / max(anneal_end, 1)
        return self.entropy_coef_start + frac * (self.entropy_coef_end - self.entropy_coef_start)

    def _ppo_update(
        self,
        rollout: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        entropy_coef: float = 0.15,
    ) -> dict[str, float]:
        """Run PPO clipped surrogate update on imagination data.

        Args:
            rollout: Dictionary from _imagine_rollout.
            advantages: (B, H) advantage estimates.
            returns: (B, H) return targets.
            entropy_coef: Current entropy bonus coefficient.

        Returns:
            Dictionary of average loss values.
        """
        B, H = advantages.shape

        # Flatten everything to (B*H, ...)
        flat_latent = rollout["latent"].reshape(B * H, -1)
        flat_action = rollout["action"].reshape(B * H)
        flat_old_log_prob = rollout["log_prob"].reshape(B * H).detach()
        flat_advantages = advantages.reshape(B * H)
        flat_returns = returns.reshape(B * H)

        # Normalize advantages
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            # Forward pass
            logits, values = self.policy(flat_latent)
            values = values.squeeze(-1)  # (B*H,)
            dist = Categorical(logits=logits)
            new_log_prob = dist.log_prob(flat_action)
            entropy = dist.entropy().mean()

            # Ratio
            ratio = torch.exp(new_log_prob - flat_old_log_prob)

            # Clipped surrogate loss
            surr1 = ratio * flat_advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * flat_advantages
            )
            pg_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            v_loss = F.mse_loss(values, flat_returns)

            # Entropy floor: if entropy drops below min_entropy, boost penalty
            effective_entropy_coef = entropy_coef
            if entropy.item() < self.min_entropy:
                effective_entropy_coef = entropy_coef * 3.0

            # Total loss
            loss = pg_loss + self.value_coef * v_loss - effective_entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_entropy += entropy.item()

        n = self.ppo_epochs
        return {
            "pg_loss": total_pg_loss / n,
            "value_loss": total_v_loss / n,
            "entropy": total_entropy / n,
        }

    # ------------------------------------------------------------------
    # Real environment evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_in_real_env(self, num_episodes: int) -> dict[str, float]:
        """Deploy the dream-trained policy in the real Pac-Man environment.

        At each step: encode observation with wm.encoder, get posterior z,
        form latent, run policy (greedy argmax), advance RSSM state.

        Args:
            num_episodes: Number of evaluation episodes.

        Returns:
            Dictionary with mean_score and level_clear_rate.
        """
        self.policy.eval()
        env = PacmanEnv(self.config, difficulty=2)

        scores = []
        cleared = 0

        for ep in range(num_episodes):
            env.reset(seed=ep + 10000)
            obs = env._build_obs()

            # Initialize RSSM state
            h, z = self.wm.rssm.initial_state(1)
            h = self.wm.rssm.dynamics(
                h, z, torch.zeros(1, dtype=torch.long, device=self.device)
            )

            done = False
            while not done:
                # Encode single-frame observation
                grid_t = torch.as_tensor(
                    obs["grid"][None], device=self.device
                )  # (1, 8, 31, 28)
                scalars_t = torch.as_tensor(
                    obs["scalars"][None], device=self.device
                )  # (1, 5)
                enc = self.wm.encoder(grid_t, scalars_t)  # (1, 512)

                # Get posterior z from encoder output
                z, _ = self.wm.rssm.posterior(h, enc)

                # Form latent and run policy (greedy)
                latent = torch.cat([h, z], dim=-1)  # (1, latent_dim)
                logits, _ = self.policy(latent)
                action = logits.argmax(dim=-1).item()

                # Advance RSSM deterministic state
                action_t = torch.tensor([action], dtype=torch.long, device=self.device)
                h = self.wm.rssm.dynamics(h, z, action_t)

                # Step real environment
                full_obs, _, terminated, truncated, info = env.step(action)
                obs = env._build_obs()  # single-frame obs for next step
                done = terminated or truncated

            scores.append(info["score"])
            if info.get("winner") == "pacman":
                cleared += 1

        self.policy.train()
        return {
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "level_clear_rate": cleared / max(num_episodes, 1),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save(
        self, save_dir: Path, update: int, is_best: bool = False
    ) -> None:
        """Save policy and optimizer state.

        Args:
            save_dir: Directory to save checkpoints.
            update: Current training update number.
            is_best: Whether this is the best model so far.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update": update,
        }
        torch.save(state, save_dir / "dream_agent_latest.pt")
        if is_best:
            torch.save(state, save_dir / "dream_agent_best.pt")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_updates: int = 5000,
        log_every: int = 10,
        eval_every: int = 50,
        eval_episodes: int = 20,
        save_dir: str | Path | None = None,
        patience: int = 500,
    ) -> dict:
        """Main dream training loop.

        1. Sample diverse starting states from real environment
        2. Imagine rollouts inside the world model
        3. Compute GAE advantages
        4. PPO update on imagined data
        5. Periodically evaluate in real environment
        6. Early stop if real eval doesn't improve for `patience` updates

        Args:
            total_updates: Total number of imagination + PPO update cycles.
            log_every: Print metrics every N updates.
            eval_every: Evaluate in real environment every N updates.
            eval_episodes: Number of episodes per real-env evaluation.
            save_dir: Directory to save checkpoints. If None, saving is skipped.
            patience: Stop if real eval score doesn't improve for this many updates.
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        self.policy.train()
        best_score = -float("inf")
        best_update = 0
        start_time = time.time()

        print(f"Starting dream training: {total_updates} updates")
        print(f"  Imagination horizon: {self.imagination_horizon}")
        print(f"  Num imaginations: {self.num_imaginations}")
        print(f"  PPO epochs: {self.ppo_epochs}")
        print(f"  Device: {self.device}")
        print()

        for update in range(1, total_updates + 1):
            # 1. Get diverse starting states (cached, refreshed periodically)
            if self._cached_starts is None or update % self._starts_refresh_every == 1:
                self._cached_starts = self._get_starting_states()
            start_grids, start_scalars = self._cached_starts

            # 2. Imagine rollout
            rollout = self._imagine_rollout(start_grids, start_scalars)

            # 3. Compute GAE
            advantages, returns = self._compute_gae(rollout)

            # 4. PPO update (with annealed entropy coef)
            entropy_coef = self._get_entropy_coef(update, total_updates)
            losses = self._ppo_update(rollout, advantages, returns, entropy_coef=entropy_coef)

            # 5. Log
            if update % log_every == 0:
                elapsed = time.time() - start_time
                mean_dream_reward = rollout["reward"].sum(dim=1).mean().item()
                print(
                    f"Update {update}/{total_updates} | "
                    f"dream_reward={mean_dream_reward:.2f} | "
                    f"pg_loss={losses['pg_loss']:.4f} "
                    f"v_loss={losses['value_loss']:.4f} "
                    f"entropy={losses['entropy']:.4f} | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

            # 6. Evaluate in real environment
            if update % eval_every == 0:
                eval_result = self._evaluate_in_real_env(eval_episodes)
                print(
                    f"  [EVAL] mean_score={eval_result['mean_score']:.1f} | "
                    f"level_clear_rate={eval_result['level_clear_rate']:.2%}"
                )

                # Save best model + early stopping check
                if save_dir is not None:
                    is_best = eval_result["mean_score"] > best_score
                    if is_best:
                        best_score = eval_result["mean_score"]
                        best_update = update
                        print(f"  New best score: {best_score:.1f}")
                    self._save(save_dir, update, is_best=is_best)

                # Early stopping: if no improvement for `patience` updates
                if update - best_update >= patience and best_update > 0:
                    print(
                        f"\n  Early stopping: no improvement for {patience} updates "
                        f"(best={best_score:.1f} at update {best_update})",
                        flush=True,
                    )
                    break

        # Final evaluation and save
        print("\nFinal evaluation...")
        eval_result = self._evaluate_in_real_env(eval_episodes)
        print(
            f"  [FINAL] mean_score={eval_result['mean_score']:.1f} | "
            f"level_clear_rate={eval_result['level_clear_rate']:.2%}"
        )
        if save_dir is not None:
            is_best = eval_result["mean_score"] > best_score
            if is_best:
                best_score = eval_result["mean_score"]
                best_update = total_updates
            self._save(save_dir, total_updates, is_best=is_best)
            print(f"Final checkpoint saved.")

        best_path = save_dir / "dream_agent_best.pt" if save_dir else Path(".")
        print("Dream training complete.")
        return {
            "best_score": float(best_score),
            "best_update": int(best_update),
            "best_path": best_path,
        }
