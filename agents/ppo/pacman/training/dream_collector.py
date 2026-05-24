"""Collect gameplay data from a trained dream agent for world model fine-tuning."""
import time

import numpy as np
import torch

from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel
from pacman.world_model.replay_buffer import EpisodeReplayBuffer
from pacman.training.dream_trainer import DreamPolicy


def collect_dream_episodes(
    world_model: WorldModel,
    policy: DreamPolicy,
    config: dict,
    device: torch.device,
    num_episodes: int = 300,
    difficulty: int = 2,
) -> EpisodeReplayBuffer:
    """Collect episodes by deploying a DreamPolicy in the real game.

    Args:
        world_model: Pretrained world model (eval mode, frozen).
        policy: Trained DreamPolicy checkpoint.
        config: Game config dict.
        device: Torch device.
        num_episodes: Number of episodes to collect.
        difficulty: Game difficulty level.

    Returns:
        EpisodeReplayBuffer containing the collected episodes.
    """
    world_model.eval()
    policy.eval()
    env = PacmanEnv(config, difficulty=difficulty)
    buffer = EpisodeReplayBuffer(max_episodes=num_episodes)

    total_steps = 0
    start_time = time.time()

    for ep in range(num_episodes):
        env.reset(seed=ep + 50000)  # different seeds from PPO collection
        obs = env._build_obs()

        # Initialize RSSM state
        h, z = world_model.rssm.initial_state(1)
        h = world_model.rssm.dynamics(
            h, z, torch.zeros(1, dtype=torch.long, device=device)
        )

        grids, scalars, actions, rewards, dones = [], [], [], [], []
        done = False

        while not done:
            grid_t = torch.as_tensor(obs["grid"][None], device=device)
            scalars_t = torch.as_tensor(obs["scalars"][None], device=device)

            with torch.no_grad():
                enc = world_model.encoder(grid_t, scalars_t)
                z, _ = world_model.rssm.posterior(h, enc)

                latent = torch.cat([h, z], dim=-1)
                logits, _ = policy(latent)
                # Stochastic sampling for exploration diversity
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()

                action_t = torch.tensor([action], dtype=torch.long, device=device)
                h = world_model.rssm.dynamics(h, z, action_t)

            grids.append(obs["grid"])
            scalars.append(obs["scalars"])
            actions.append(action)

            _, reward, terminated, truncated, _ = env.step(action)
            obs = env._build_obs()
            rewards.append(reward)
            dones.append(terminated or truncated)
            done = terminated or truncated

        T = len(actions)
        buffer.add_episode({
            "grid": torch.from_numpy(np.stack(grids[:T])),
            "scalars": torch.from_numpy(np.stack(scalars[:T])),
            "action": torch.tensor(actions, dtype=torch.long),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.bool),
        })
        total_steps += T

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed
            print(
                f"  Episode {ep + 1}/{num_episodes} | "
                f"Steps: {total_steps} | {eps_per_sec:.1f} eps/s",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"  Collection complete: {num_episodes} episodes, {total_steps} steps, {elapsed:.1f}s")
    return buffer
