# scripts/watch.py
"""Watch a trained Pac-Man agent play in Pygame."""
import argparse
from pathlib import Path

import numpy as np
import torch

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.env.pacman_env import PacmanEnv
from pacman.training.checkpoint import load_checkpoint
from pacman.training.trainer import get_device
from pacman.viz.renderer import GameRenderer


def _build_network_from_config(config):
    """Build ActorCritic with architecture matching the config."""
    env_cfg = config["env"]
    net_cfg = config["network"]
    frame_stack = env_cfg.get("frame_stack", 1)
    grid_channels = env_cfg["observation_channels"] * frame_stack
    return ActorCritic(
        grid_channels=grid_channels,
        num_scalars=env_cfg.get("num_scalar_features", 5),
        cnn_channels=net_cfg.get("cnn_channels", [32, 64, 64]),
        cnn_kernels=net_cfg.get("cnn_kernels", [3, 3, 3]),
        cnn_strides=net_cfg.get("cnn_strides", [1, 2, 2]),
        shared_hidden=net_cfg.get("shared_hidden", 512),
        head_hidden=net_cfg.get("head_hidden", 128),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    device = get_device(config)

    network = None
    if args.run_dir:
        ckpt_path = Path(args.run_dir) / "checkpoints"
        # Load checkpoint to get saved config for architecture matching
        ckpt_data = torch.load(
            ckpt_path / args.checkpoint, map_location="cpu", weights_only=False,
        )
        saved_config = ckpt_data.get("config", config)
        network = _build_network_from_config(saved_config)
        network.load_state_dict(ckpt_data["model_state_dict"])
        network.to(device).eval()
        # Use saved config for env too (frame stacking etc.)
        config = saved_config

    env = PacmanEnv(config, difficulty=2)
    renderer = GameRenderer(config)
    obs, _ = env.reset(seed=args.seed)
    paused = False

    while True:
        action_probs = None
        value = None

        if network is not None:
            with torch.no_grad():
                grid_t = torch.as_tensor(obs["grid"][None], device=device)
                scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
                mask = env.get_legal_mask()
                mask_t = torch.as_tensor(mask[None], device=device)
                logits, val = network(grid_t, scalars_t, mask_t)
                probs = torch.softmax(logits, dim=-1)
                action_probs = probs[0].cpu().numpy()
                value = val[0].item()
                action = logits.argmax(dim=-1).item()
        else:
            mask = env.get_legal_mask()
            action = np.random.choice(np.where(mask)[0])

        agent_info = {"action_probs": action_probs, "value": value}
        running, step_requested = renderer.render(env.state, agent_info)
        if not running:
            break

        if not paused or step_requested:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                old_score = env.state.score
                old_lives = env.state.pac_lives
                obs, _ = env.reset()
                # Carry over score and lives if level was cleared
                if old_lives > 0:
                    env.state.score = old_score
                    env.state.pac_lives = old_lives

        renderer.tick(args.fps)

    renderer.close()


if __name__ == "__main__":
    main()
