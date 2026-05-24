"""PPO behavior distillation: collect latent-action pairs and train via behavioral cloning."""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel
from pacman.agents.networks import ActorCritic
from pacman.training.dream_trainer import DreamPolicy


def collect_distillation_data(
    ppo_network: ActorCritic,
    world_model: WorldModel,
    config: dict,
    device: torch.device,
    num_episodes: int = 500,
    difficulty: int = 2,
) -> dict[str, torch.Tensor]:
    """Run PPO agent in real game while encoding through world model.

    At each step, the PPO agent picks an action from frame-stacked observations,
    and the world model encodes the single-frame observation into latent space.
    Returns (latent_state, ppo_action) pairs for behavioral cloning.

    Args:
        ppo_network: Trained ActorCritic network (eval mode).
        world_model: Trained WorldModel (eval mode, frozen).
        config: Game config dict.
        device: Torch device.
        num_episodes: Number of episodes to collect.
        difficulty: Game difficulty level.

    Returns:
        Dict with "latents" (N, 2560) and "actions" (N,) tensors.
    """
    ppo_network.eval()
    world_model.eval()

    # Use frame_stack=1 env for world model encoding; manage PPO frame stack manually
    env_config = {**config, "env": {**config["env"], "frame_stack": 1}}
    env = PacmanEnv(env_config, difficulty=difficulty)

    all_latents = []
    all_actions = []
    total_steps = 0
    start_time = time.time()

    for ep in range(num_episodes):
        env.reset(seed=ep)
        obs = env._build_obs()  # single-frame: {"grid": (8,31,28), "scalars": (5,)}

        # Manual frame stack for PPO (4 frames)
        frame_stack = deque(maxlen=4)
        for _ in range(4):
            frame_stack.append(obs["grid"])

        # Initialize RSSM state
        h, z = world_model.rssm.initial_state(1)
        h = world_model.rssm.dynamics(
            h, z, torch.zeros(1, dtype=torch.long, device=device)
        )

        done = False
        while not done:
            with torch.no_grad():
                # --- World model encoding ---
                grid_t = torch.as_tensor(obs["grid"][None], device=device)
                scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
                enc = world_model.encoder(grid_t, scalars_t)
                z, _ = world_model.rssm.posterior(h, enc)
                latent = torch.cat([h, z], dim=-1)  # (1, 2560)

                # --- PPO action selection (greedy) ---
                stacked_grid = np.stack(list(frame_stack), axis=0).reshape(
                    -1, obs["grid"].shape[1], obs["grid"].shape[2]
                )  # (32, 31, 28)
                ppo_grid_t = torch.as_tensor(
                    stacked_grid[None], device=device
                )  # (1, 32, 31, 28)
                ppo_scalars_t = torch.as_tensor(
                    obs["scalars"][None], device=device
                )  # (1, 5)
                legal_mask = torch.as_tensor(
                    env.get_legal_mask()[None], device=device
                )  # (1, 4)
                ppo_logits, _ = ppo_network(ppo_grid_t, ppo_scalars_t, legal_mask)
                action = ppo_logits.argmax(dim=-1).item()  # greedy

                # Store pair
                all_latents.append(latent.cpu())
                all_actions.append(action)

                # Advance RSSM
                action_t = torch.tensor([action], dtype=torch.long, device=device)
                h = world_model.rssm.dynamics(h, z, action_t)

            # Step environment
            _, _, terminated, truncated, _ = env.step(action)
            obs = env._build_obs()
            frame_stack.append(obs["grid"])
            done = terminated or truncated
            total_steps += 1

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Episode {ep + 1}/{num_episodes} | "
                f"Steps: {total_steps} | {(ep + 1) / elapsed:.1f} eps/s",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"  Collection complete: {num_episodes} episodes, {total_steps} steps, {elapsed:.1f}s")

    return {
        "latents": torch.cat(all_latents, dim=0),  # (N, 2560)
        "actions": torch.tensor(all_actions, dtype=torch.long),  # (N,)
    }


def train_behavioral_cloning(
    policy: DreamPolicy,
    latents: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 5,
    val_fraction: float = 0.2,
) -> dict:
    """Train DreamPolicy actor via behavioral cloning on (latent, action) pairs.

    Only trains the actor head. The critic is left untouched.

    Args:
        policy: DreamPolicy to train (actor head).
        latents: (N, 2560) latent states.
        actions: (N,) expert action labels.
        device: Torch device.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Initial learning rate.
        patience: Early stopping patience on val loss.
        val_fraction: Fraction of data for validation.

    Returns:
        Dict with train_loss, val_loss, val_accuracy, best_epoch.
    """
    policy.to(device)
    policy.train()

    # Train/val split
    N = latents.shape[0]
    n_val = int(N * val_fraction)
    n_train = N - n_val
    perm = torch.randperm(N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_latents = latents[train_idx].to(device)
    train_actions = actions[train_idx].to(device)
    val_latents = latents[val_idx].to(device)
    val_actions = actions[val_idx].to(device)

    # Only optimize actor parameters (weight decay prevents overfitting)
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # --- Training ---
        policy.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            batch_latents = train_latents[idx]
            batch_actions = train_actions[idx]

            logits = policy.actor(batch_latents)
            loss = F.cross_entropy(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        policy.eval()
        with torch.no_grad():
            val_logits = policy.actor(val_latents)
            val_loss = F.cross_entropy(val_logits, val_actions).item()
            val_preds = val_logits.argmax(dim=-1)
            val_accuracy = (val_preds == val_actions).float().mean().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_accuracy:.1%} lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in policy.actor.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (best={best_epoch})")
                break

    # Restore best weights
    if best_state is not None:
        policy.actor.load_state_dict(best_state)

    print(f"  BC complete: best_epoch={best_epoch} val_loss={best_val_loss:.4f} val_acc={val_accuracy:.1%}")

    return {
        "train_loss": train_loss,
        "val_loss": best_val_loss,
        "val_accuracy": val_accuracy,
        "best_epoch": best_epoch,
    }
