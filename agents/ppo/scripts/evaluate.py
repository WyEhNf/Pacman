# scripts/evaluate.py
"""Evaluate a trained Pac-Man agent from checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.training.checkpoint import load_checkpoint
from pacman.training.evaluator import Evaluator
from pacman.training.trainer import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to training run directory")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    args = parser.parse_args()

    config = load_config()
    device = get_device(config)
    network = ActorCritic()
    ckpt_path = Path(args.run_dir) / "checkpoints"
    meta = load_checkpoint(ckpt_path, network, filename=args.checkpoint)
    print(f"Loaded checkpoint: update={meta['update']}")

    evaluator = Evaluator(config)
    results = evaluator.evaluate(network, args.episodes, device)

    print(f"\nResults ({args.episodes} episodes):")
    print(f"  Level clear rate: {results['level_clear_rate']:.1%}")
    print(f"  Mean score:       {results['mean_score']:.0f}")
    print(f"  Mean steps:       {results['mean_steps']:.0f}")
    print(f"  Mean ghosts eaten:{results['mean_ghosts_eaten']:.1f}")

    out_path = Path(args.run_dir) / "eval" / f"eval_{meta['update']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
