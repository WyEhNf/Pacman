# scripts/collect_dream_data.py
"""CLI entry point for collecting gameplay data from a trained dream agent."""
import argparse
from pathlib import Path

import torch

from pacman.utils.config import load_config
from pacman.world_model.world_model import WorldModel
from pacman.training.dream_trainer import DreamPolicy
from pacman.training.dream_collector import collect_dream_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Collect gameplay data from a trained dream agent."
    )
    parser.add_argument("--world-model", type=str, required=True)
    parser.add_argument("--dream-agent", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    config = load_config()

    # Load world model
    wm = WorldModel().to(device)
    wm_ckpt = torch.load(args.world_model, map_location=device, weights_only=True)
    wm.load_state_dict(wm_ckpt["model_state_dict"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    print(f"Loaded world model from {args.world_model}")

    # Load dream policy
    policy = DreamPolicy(latent_dim=wm.rssm.latent_dim).to(device)
    da_ckpt = torch.load(args.dream_agent, map_location=device, weights_only=True)
    policy.load_state_dict(da_ckpt["policy_state_dict"])
    policy.eval()
    print(f"Loaded dream agent from {args.dream_agent}")

    # Collect
    buffer = collect_dream_episodes(
        world_model=wm,
        policy=policy,
        config=config,
        device=device,
        num_episodes=args.episodes,
        difficulty=args.difficulty,
    )

    # Save
    output_path = args.output or str(Path(args.dream_agent).parent.parent / "dream_replay_buffer.pt")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    buffer.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
