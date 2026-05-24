# scripts/train.py
"""Train Pac-Man PPO agent."""
import argparse
from datetime import datetime
from pathlib import Path

from pacman.utils.config import load_config
from pacman.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Pac-Man PPO agent")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--run-dir", type=str, default=None, help="Run output directory")
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_envs:
        config["env"]["num_envs"] = args.num_envs
    if args.device:
        config["device"] = args.device

    run_dir = args.run_dir or f"runs/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    trainer = Trainer(config, Path(run_dir), resume=args.resume)

    try:
        trainer.train(total_updates=args.total_updates)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer._save(trainer.current_update, is_best=False)
        print("Checkpoint saved.")


if __name__ == "__main__":
    main()
