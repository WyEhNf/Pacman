"""Collect gameplay data from a trained PPO agent for world model training."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.training.checkpoint import load_checkpoint
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.replay_buffer import EpisodeReplayBuffer


def main():
    parser = argparse.ArgumentParser(
        description="Collect gameplay data from a trained PPO agent."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint file (e.g., runs/.../checkpoints/best.pt)",
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Where to save buffer. Default: checkpoint's grandparent / replay_buffer.pt",
    )
    parser.add_argument("--difficulty", type=int, default=2)
    args = parser.parse_args()

    # --- Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Config and network ---
    config = load_config()
    env_cfg = config["env"]
    net_cfg = config["network"]
    grid_channels = env_cfg["observation_channels"] * env_cfg.get("frame_stack", 1)

    network = ActorCritic(
        grid_channels=grid_channels,
        cnn_channels=net_cfg["cnn_channels"],
        cnn_kernels=net_cfg["cnn_kernels"],
        cnn_strides=net_cfg["cnn_strides"],
        shared_hidden=net_cfg["shared_hidden"],
        head_hidden=net_cfg["head_hidden"],
    ).to(device)

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    meta = load_checkpoint(ckpt_path.parent, network, filename=ckpt_path.name)
    network.eval()
    print(f"Loaded checkpoint: update={meta['update']}")

    # --- Environment ---
    env = PacmanEnv(config, difficulty=args.difficulty)

    # --- Replay buffer ---
    buffer = EpisodeReplayBuffer(max_episodes=args.episodes)

    # --- Output path ---
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ckpt_path.parent.parent / "replay_buffer.pt"

    # --- Collect episodes ---
    total_steps = 0
    start_time = time.time()

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)

        grids = []    # single-frame (8, H, W)
        scalars = []  # (5,)
        actions = []
        rewards = []
        dones = []

        # Record initial single-frame observation
        single_obs = env._build_obs()
        grids.append(single_obs["grid"])
        scalars.append(single_obs["scalars"])

        done = False
        while not done:
            # Agent uses frame-stacked obs for decisions
            grid_t = torch.from_numpy(obs["grid"]).unsqueeze(0).to(device)
            scalars_t = torch.from_numpy(obs["scalars"]).unsqueeze(0).to(device)
            legal_mask = torch.from_numpy(env.get_legal_mask()).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = network(grid_t, scalars_t, legal_mask)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store single-frame obs (from _build_obs, NOT the stacked version)
            single_obs = env._build_obs()
            grids.append(single_obs["grid"])
            scalars.append(single_obs["scalars"])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        T = len(actions)
        # Trim grids/scalars to match actions length (drop the extra trailing obs)
        episode = {
            "grid": torch.from_numpy(np.stack(grids[:T])),
            "scalars": torch.from_numpy(np.stack(scalars[:T])),
            "action": torch.tensor(actions, dtype=torch.long),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.bool),
        }
        buffer.add_episode(episode)
        total_steps += T

        if (ep + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed
            print(
                f"Episode {ep + 1}/{args.episodes} | "
                f"Steps so far: {total_steps} | "
                f"{eps_per_sec:.1f} eps/s"
            )

    elapsed = time.time() - start_time
    print(f"\nCollection complete!")
    print(f"  Episodes: {args.episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saving to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer.save(str(output_path))
    print("Done.")


if __name__ == "__main__":
    main()
